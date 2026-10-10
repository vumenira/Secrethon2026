"""Play episodes of an agent folder in-process and keep each one's records, pulse log and what the pulse saw.

uv run python outputs/recovery_tune/tools/with_lock.py uv run python <this> --name=r13rec_off --task=full --entropy=12345 --episodes=16
    [--params='{"schedule_ends": false}'] [--only=3,7] [--n_jobs=2]

Writes outputs/recovery_tune/r13/rec/<task>-e<entropy>-n<episodes>/<name>/ep<n>.pkl.gz and rows.json (the costs, as
tune.py caches them). The play itself is the agent's own: the wrapper around Pulse.adjust only reads.
"""

import gzip
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"
BASE = "mpc_pulse_rec_v13r"


def folder(name, base, params):
    out = OUT / "agents" / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "agents" / base, out, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    (out / "params.json").write_text(json.dumps(params))
    return out


def plan_part(plan, inst, chip_slots):
    """The plan's lots, fab energy, lost sales, served demand, shed load and chip stocks, per window week."""
    if plan is None:
        return None
    model, x = plan
    tmpl, nc, H = model.meta["template"], int(model.meta["nc"]), int(model.T)
    x = np.asarray(x, dtype=float)
    ub = np.asarray(model.ub, dtype=float)

    def col(key, src=x):
        j = tmpl.get(key)
        return np.full(H, np.nan) if j is None else src[j : H * nc : nc].copy()

    F, D, G = len(inst.fabs), len(inst.demands), len(inst.grids)
    return {
        "H": H,
        "p": np.array([col(("p", fo)) for fo in range(F)], dtype=np.float32),
        "pub": np.array([col(("p", fo), ub) for fo in range(F)], dtype=np.float32),
        "E": np.array([col(("E", fo)) for fo in range(F)], dtype=np.float32),
        "U": np.array([col(("U", do)) for do in range(D)], dtype=np.float32),
        "D": np.array([col(("D", do)) for do in range(D)], dtype=np.float32),
        "ysh": np.array([col(("ysh", go)) for go in range(G)], dtype=np.float32),
        "I": np.array([col(("I", s)) for s in chip_slots], dtype=np.float32),
    }


def play(root, spec, episodes, dest):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        K = inst.commodities
        chip_slots = [s for s, st in enumerate(inst.stock_slots) if K[st.k].id.startswith("chip") and st.node not in set(inst.chokepoints)]
        trace = []
        with without_secret_like():
            factory = load_agent_class(root, f"submission_{Path(root).stem}")
            pm = sys.modules["pulse"]
            orig = pm.Pulse.adjust

            def adjust(self, obs, values, prohibited, action, plan=None, _orig=orig, _trace=trace, _slots=chip_slots):
                t = int(obs["week"])
                ins = self.inst
                rec = {"t": t, "held": dict(self.held), "shorted": dict(self.shorted), "n_log": len(self.log)}
                stock = {(int(a), int(b)): float(q) for a, b, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
                arriving = {}
                pp = obs["pipeline"]
                for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
                    if int(week) == t:
                        key = (ins.edges[int(e)].head, int(k))
                        arriving[key] = arriving.get(key, 0.0) + float(q)
                before = dict(zip(action["flows"]["slot"], action["flows"]["qty"])) if action.get("flows") else {}
                grids = {}
                for go, g, fabs, fuels, shares in self.grids:
                    fu = {}
                    for k, (share, thr, feeders, worth) in fuels.items():
                        tank = inflow = 0.0
                        ss_all = []
                        for e, tail, storage, ss in feeders:
                            tank += stock.get((tail, k), 0.0)
                            inflow += arriving.get((tail, k), 0.0)
                            ss_all += [s for s in ss if s in before]
                        fu[k] = {
                            "here": stock.get((g, k), 0.0), "arr": arriving.get((g, k), 0.0), "tank": tank, "inflow": inflow,
                            "planned": sum(before[s] for s in ss_all), "slots": ss_all,
                        }
                    fb = {}
                    for fo in fabs:
                        f = ins.fabs[fo]
                        fa = ins.nodes[f].fab
                        into = [s for s in self.tail_slots_into(f, fa.input) if s in before]
                        fb[fo] = {
                            "wafers": stock.get((f, fa.input), 0.0), "warr": arriving.get((f, fa.input), 0.0),
                            "chips": stock.get((f, fa.product), 0.0), "alpha": float(values["fab_alpha"][fo]),
                            "R": float(values["fab_R"][fo]), "planned": sum(before[s] for s in into), "slots": into,
                        }
                    grids[go] = {"G": float(values["grid_G"][go]), "y": float(values["grid_y"][go]), "fuels": fu, "fabs": fb}
                rec["grids"] = grids
                rec["plan"] = plan_part(plan, ins, _slots)
                out = _orig(self, obs, values, prohibited, action, plan)
                after = dict(zip(out["flows"]["slot"], out["flows"]["qty"])) if out.get("flows") else {}
                for go, gr in grids.items():
                    for k, fu in gr["fuels"].items():
                        fu["sent"] = sum(after.get(s, 0.0) for s in fu.pop("slots"))
                    for fo, fb in gr["fabs"].items():
                        fb["sent"] = sum(after.get(s, 0.0) for s in fb.pop("slots"))
                rec["rate"] = dict(self.rate)
                rec["held_after"] = dict(self.held)
                rec["log"] = list(self.log[rec["n_log"] :])
                _trace.append(rec)
                return out

            pm.Pulse.adjust = adjust
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        agent = shim.agent
        log = list(getattr(getattr(agent, "pulse", None), "log", []) or [])
        failed = (getattr(agent, "pulse_failed", None), getattr(getattr(agent, "planner", None), "failed", None))
        unload_agent()
        row = {
            "episode": n,
            "J_policy_cents": traj.J_cents,
            "fallback_weeks": sum(took_fallback(r) for r in traj.records),
            "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
            "failed": failed,
            "seconds": round(seconds, 3),
            "omega_hash": omega.hash,
        }
        keep = {
            "row": row,
            "records": traj.records,
            "log": log,
            "trace": trace,
            "chip_slots": chip_slots,
            "marks": {f: np.asarray(getattr(marks, f)) for f in ("G_bar", "y_bar", "alpha_bar", "R", "demand")},
        }
        with gzip.open(Path(dest) / f"ep{n}.pkl.gz", "wb", compresslevel=3) as fh:
            pickle.dump(keep, fh, protocol=pickle.HIGHEST_PROTOCOL)
        rows.append(row)
        print(f"  episode {n}: J {traj.J_cents / 1e11:.2f} bn, {seconds:.0f} s, errors {row['first_error']}", flush=True)
    return rows


def main(name, task="full", entropy=12345, episodes=16, params="{}", only="", n_jobs=2, base=BASE, variants=""):
    from joblib import Parallel, delayed
    from shockbench_flow_agent import EpisodeSet

    if variants:
        spec = json.loads(Path(variants).read_text())[name]
        params, base = spec.get("params", {}), spec.get("base", base)
    params = params if isinstance(params, dict) else json.loads(params)
    params = {"strict": True, **params}
    es = EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=True)
    dest = OUT / "r13" / "rec" / f"{task}-e{entropy}-n{episodes}" / name
    dest.mkdir(parents=True, exist_ok=True)
    root = folder(name, base, params)
    ns = list(es.episodes)
    if only != "":
        chosen = {int(v) for v in (only if isinstance(only, (tuple, list)) else str(only).split(","))}
        ns = [n for n in ns if n in chosen]
    chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
    start = time.perf_counter()
    parts = Parallel(n_jobs=n_jobs)(delayed(play)(str(root), es._spec, [int(n) for n in c], str(dest)) for c in chunks)
    rows = sorted((r for part in parts for r in part), key=lambda r: r["episode"])
    (dest / "rows.json").write_text(json.dumps({"params": params, "rows": rows}, indent=1))
    print(f"{name}: {len(rows)} episodes in {time.perf_counter() - start:.0f} s -> {dest}")
    ref = OUT / f"{task}-e{entropy}-n{episodes}" / "v12_all.json"
    if ref.exists():
        v12 = {r["episode"]: r["J_policy_cents"] for r in json.loads(ref.read_text())["rows"]}
        diff = [(r["episode"], round((r["J_policy_cents"] - v12[r["episode"]]) / 1e11, 3)) for r in rows if r["J_policy_cents"] != v12[r["episode"]]]
        print(f"against v12_all: {'equal to the cent' if not diff else diff}")
    if any(r["first_error"] or r["fallback_weeks"] for r in rows):
        print("ERRORS:", [(r["episode"], r["first_error"], r["fallback_weeks"]) for r in rows if r["first_error"] or r["fallback_weeks"]][:5])
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
