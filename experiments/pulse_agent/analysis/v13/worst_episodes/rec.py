"""One Full episode recorded for the autopsy: the agent's play, and clairvoyant plans that respect "base load first".

uv run python outputs/recovery_tune/tools/with_lock.py uv run python outputs/w13a/rec.py --eps="777:26,777:8,2026:3"

Per episode, in outputs/w13a/data/<task>-e<root>/:
  ep<n>.pkl        the play of --agent in process, as the scorer plays it: every week's StepRecord, the marks, J
  ep<n>_bf.npz     the clairvoyant LP plus the hull of "base load before fabs" (a valid bound for any agent)
  ep<n>_f1.npz     the same with every grid's burn of every fuel capped by ours over the episode (fuel logistics kept)
  ep<n>_f2.npz     the same with every grid's fab energy capped by ours week by week (our calendar kept)
  ep<n>_free.npz   the clairvoyant LP itself (only with --free)
Each stage is skipped when its file exists.
"""
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def hull_rows(inst, marks, model, subsets=False):
    """Rows (coo lists, rhs) of the base-first hull: valid for any agent, whatever it knows.

    Per week, grid with fabs and fuel k (S_k its full segment, margin = G-bar - y-bar, G_k its output): the fabs'
    energy is at most margin - c (S_k - G_k), c = min(margin, S_k) / S_k (the chord of max(0, margin - shortfall)).
    ``subsets``: the same chord for every subset A of the grid's fabs, whose energy is min(draw_A, max(0, margin -
    shortfall)): flat at draw_A up to a shortfall of a = margin - draw_A, then down to v = min(draw_A, max(0, margin -
    S_k)) at S_k. It stops the plan from giving one fab the energy of weeks that are whole for none.
    """
    from itertools import combinations

    T = model.T
    nc = len(model.columns)
    col = {key: j for j, key in enumerate(model.columns)}
    rows, cols, vals, rhs = [], [], [], []
    r = 0
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if ga.priority != "base_first" or not fabs:
            continue
        groups = [fabs]
        if subsets:
            groups = [list(c) for size in range(1, len(fabs) + 1) for c in combinations(fabs, size)]
        for t in range(T):
            Gbar, ybar = float(marks.G_bar[t, go]), float(marks.y_bar[t, go])
            margin = Gbar - ybar
            for k in ga.fuels:
                S = ga.shares[k] * Gbar
                for A in groups:
                    full = len(A) == len(fabs)
                    draw = sum(inst.nodes[inst.fabs[fo]].fab.e * float(marks.alpha_bar[t, fo]) * inst.nodes[inst.fabs[fo]].fab.cap0 for fo in A)
                    if margin <= 0 or S <= 0:
                        if not full:
                            continue
                        c, b = 0.0, max(margin, 0.0)
                    elif full:
                        c = min(margin, S) / S
                        b = margin - c * S
                    else:
                        a = max(0.0, margin - draw)
                        v = min(draw, max(0.0, margin - S))
                        if a >= S - 1e-9 or v >= draw - 1e-9:
                            continue
                        c = (draw - v) / (S - a)  # E_A <= draw - c (S - G - a)
                        b = draw - c * (S - a)
                    for fo in A:
                        rows.append(r), cols.append(t * nc + col[("E", fo)]), vals.append(1.0)
                    if c:
                        rows.append(r), cols.append(t * nc + col[("G", go, k)]), vals.append(-c)
                    rhs.append(b)
                    r += 1
    return rows, cols, vals, rhs


def with_rows(model, rows, cols, vals, rhs):
    from scipy.sparse import csr_matrix, vstack

    extra = csr_matrix((vals, (rows, cols)), shape=(len(rhs), model.A_ub.shape[1]))
    return replace(model, A_ub=vstack([model.A_ub, extra]).tocsr(), b_ub=np.concatenate([model.b_ub, np.asarray(rhs)]))


def solve(model, target, label):
    from shockbench_flow.oracle.lp import ORACLE_METHOD, solve_oracle

    start = time.perf_counter()
    res = solve_oracle(model, method=ORACLE_METHOD)
    nc = len(model.columns)
    x = None if res.x is None else res.x.reshape(model.T, nc).astype(np.float32)
    np.savez_compressed(target, x=x, J=np.float64(res.J_usd if res.J_usd is not None else np.nan), status=res.status)
    print(f"  {label}: status {res.status} J {res.J_usd / 1e9 if res.J_usd else float('nan'):,.1f} bn"
          f" ({time.perf_counter() - start:.0f} s)", flush=True)


def one(agent, task, root, n, stages):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.oracle.lp import build_lp
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    out = HERE / "data" / f"{task}-e{root}"
    out.mkdir(parents=True, exist_ok=True)
    census = ROOT / "outputs" / "s13" / "s13_census" / f"{task}-e{root}" / f"ep{n}.pkl"  # v12's play, to the cent
    play = out / f"ep{n}.pkl"
    inst, omega, marks, fallback = _world(task, root, n, REPLICATIONS, str(default_cache_dir()))
    fields = ("u", "c", "o", "kappa", "supply", "G_bar", "y_bar", "R", "alpha_bar", "sigma_scr", "R_osat", "demand",
              "prohibited", "tariff", "u_now", "o_now", "kappa_now", "supply_now", "G_bar_now", "y_bar_now", "R_now")
    if not (out / f"ep{n}_marks.npz").exists():
        np.savez_compressed(out / f"ep{n}_marks.npz", **{f: np.asarray(getattr(marks, f)) for f in fields})
    if census.exists():
        with open(census, "rb") as f:
            doc = pickle.load(f)
        segment, energy = doc["segment"], np.asarray(doc["energy"])
    else:
        if not play.exists():
            start = time.perf_counter()
            shim = _metered_shim(load_agent_class(str(agent), f"w13a_{root}_{n}"), None)
            traj = rollout(inst, shim, omega, "standard", _policy_seed(root, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            errors = [(int(w), str(e)) for w, e in shim.errors]
            unload_agent()
            doc = {"episode": n, "root": root, "J": traj.J_cents / 100, "salvage": traj.salvage, "errors": errors, "records": traj.records}
            with open(play, "wb") as f:
                pickle.dump(doc, f, protocol=4)
            print(f"episode {root}/{n}: played J {doc['J'] / 1e9:,.1f} bn, errors {errors[:2]} ({time.perf_counter() - start:.0f} s)", flush=True)
        with open(play, "rb") as f:
            doc = pickle.load(f)
        segment = [dict(r.segment) for r in doc["records"]]
        energy = np.array([r.energy for r in doc["records"]])
    todo = [s for s in stages if not (out / f"ep{n}_{s}.npz").exists()]
    if not todo:
        return
    model = build_lp(inst, marks)
    if not (out / "columns.pkl").exists():
        with open(out / "columns.pkl", "wb") as f:
            pickle.dump(model.columns, f)
    nc = len(model.columns)
    col = {key: j for j, key in enumerate(model.columns)}
    T = model.T
    hulls = {False: hull_rows(inst, marks, model), True: hull_rows(inst, marks, model, subsets=True)}
    for s in todo:
        rows, cols, vals, rhs = (list(a) for a in hulls[s in ('b2', 'g1', 'g2')])
        if s == "free":
            solve(model, out / f"ep{n}_free.npz", f"{root}/{n} free")
            continue
        if s in ("f1", "g1"):  # every grid's burn of every fuel over the episode at most ours
            r = len(rhs)
            for go, g in enumerate(inst.grids):
                for k in inst.nodes[g].grid.fuels:
                    ours = sum(w.get((go, k), 0.0) for w in segment)
                    for t in range(T):
                        rows.append(r), cols.append(t * nc + col[("G", go, k)]), vals.append(1.0)
                    rhs.append(ours * (1 + 1e-9) + 1e-6)
                    r += 1
        if s in ("f2", "g2"):  # every grid's fab energy each week at most ours
            r = len(rhs)
            for go, g in enumerate(inst.grids):
                fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
                if not fabs:
                    continue
                for t in range(T):
                    ours = sum(float(energy[t, fo]) for fo in fabs)
                    for fo in fabs:
                        rows.append(r), cols.append(t * nc + col[("E", fo)]), vals.append(1.0)
                    rhs.append(ours * (1 + 1e-9) + 1e-6)
                    r += 1
        solve(with_rows(model, rows, cols, vals, rhs), out / f"ep{n}_{s}.npz", f"{root}/{n} {s}")


def main(eps, agent="agents/mpc_pulse_rec_v13w", task="full", stages="bf,f1,f2", n_jobs=2, dry=False):
    """eps: "777:26,777:8,2026:3" (root:episode pairs)."""
    from joblib import Parallel, delayed

    agent = (ROOT / agent).resolve()
    pairs = [tuple(int(v) for v in item.split(":")) for item in str(eps).split(",")]
    stages = [s for s in (stages.split(",") if isinstance(stages, str) else list(stages)) if s]
    if dry:  # build the model and the extra rows of the first pair, nothing played or solved
        from shockbench_flow.evaluation.cache import default_cache_dir
        from shockbench_flow.oracle.lp import build_lp
        from shockbench_flow.policies.naive_fq import REPLICATIONS
        from shockbench_flow_agent.scoring import _world

        inst, omega, marks, fallback = _world(task, pairs[0][0], pairs[0][1], REPLICATIONS, str(default_cache_dir()))
        model = build_lp(inst, marks)
        m2 = with_rows(model, *hull_rows(inst, marks, model, subsets=True))
        print(model.A_ub.shape, m2.A_ub.shape, m2.b_ub.shape, len(model.columns), model.T)
        return
    Parallel(n_jobs=n_jobs)(delayed(one)(agent, task, root, n, stages) for root, n in pairs)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
