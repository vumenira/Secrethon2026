"""Play an agent folder on episodes of a root and keep, per week: the grids' segments, shed load, fab energy, lots,
stocks, the planner's window plan at the grids (segment outputs, shortfalls, lots) and the pulse's log.

The folder is loaded as the scorer loads it and never edited; ``params`` are put into the loaded module's ``PARAMS``
(with ``strict`` true) before the episode starts. Each episode's cost is compared with a cached row to the cent.

uv run python play.py run --task=small --tag=base --n_jobs=2 [--only=5,17] [--params="{'open_hold': True}"]
uv run python play.py show --task=small --tag=base --n=5 --weeks=16
"""

import json
import pickle
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "recovery_tune"
TOTAL = {"small": 64, "full": 16}


def world(task, n, entropy=12345):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))


def run_dir(task, entropy=12345, episodes=None):
    return OUT / f"{task}-e{entropy}-n{episodes or TOTAL[task]}"


def references(task, entropy=12345, episodes=None):
    return {r["episode"]: r for r in json.loads((run_dir(task, entropy, episodes) / "references.json").read_text())}


def cached(task, name, entropy=12345, episodes=None):
    rows = json.loads((run_dir(task, entropy, episodes) / f"{name}.json").read_text())["rows"]
    return {r["episode"]: r["J_policy_cents"] for r in rows}


def path(task, tag, n):
    return HERE / "traj" / tag / f"{task}_{n}.pkl"


def one(task, n, agent, params, tag, entropy=12345, plans=True):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = world(task, n, entropy)
    start = time.perf_counter()
    kept = {}
    box = [0]
    with without_secret_like():
        factory = load_agent_class(str(ROOT / "agents" / agent), "submission_v8play")
        module = sys.modules[factory.__module__]
        module.PARAMS.update({"strict": True})
        module.PARAMS.update(params or {})
        orig_act, orig_record = module.Planner.act, module.Planner._record

        def act(self, obs):
            box[0] = int(obs["week"])
            return orig_act(self, obs)

        def record(self, model, res):
            if plans and res.ok:
                tmpl, nc = model.meta["template"], model.meta["nc"]
                x = np.asarray(res.x, dtype=float).reshape(model.T, nc)
                ub = np.asarray(model.ub, dtype=float).reshape(model.T, nc)
                pinst = self._inst
                week = {"G": {}, "cap": {}, "short": {}, "lam": {}, "ysh": {}, "p": {}, "pcap": {}, "E": {}}
                for go, g in enumerate(pinst.grids):
                    ga = pinst.nodes[g].grid
                    for k in ga.fuels:
                        j = tmpl.get(("G", go, k))
                        if j is not None:
                            week["G"][(go, k)] = x[:, j].astype(np.float32)
                            week["cap"][(go, k)] = ub[:, j].astype(np.float32)
                        j = tmpl.get(("short", go, k))
                        if j is not None:
                            week["short"][(go, k)] = x[:, j].astype(np.float32)
                    j = tmpl.get(("lam", go))
                    if j is not None:
                        week["lam"][go] = x[:, j].astype(np.float32)
                    week["ysh"][go] = x[:, tmpl[("ysh", go)]].astype(np.float32)
                for fo in range(len(pinst.fabs)):
                    week["p"][fo] = x[:, tmpl[("p", fo)]].astype(np.float32)
                    week["pcap"][fo] = ub[:, tmpl[("p", fo)]].astype(np.float32)
                    j = tmpl.get(("E", fo))
                    if j is not None:
                        week["E"][fo] = x[:, j].astype(np.float32)
                kept[box[0]] = week
            return orig_record(self, model, res)

        module.Planner.act = act
        module.Planner._record = record
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent_obj = shim.agent
    log = list(agent_obj.pulse.log) if agent_obj.pulse is not None else []
    failed = (getattr(agent_obj.planner, "forecast_failed", None), getattr(agent_obj.planner, "reserve_failed", None), agent_obj.pulse_failed)
    unload_agent()
    recs = traj.records
    dump = {
        "episode": n,
        "task": task,
        "J_cents": int(traj.J_cents),
        "segment": [dict(r.segment) for r in recs],
        "shed": np.array([r.shed for r in recs]),
        "served_load": np.array([r.served_load for r in recs]),
        "energy": np.array([r.energy for r in recs]),
        "lots": np.array([r.lots_started for r in recs]),
        "stock": np.array([r.stock for r in recs]),
        "disposal": np.array([r.disposal for r in recs]),
        "served": np.array([r.served for r in recs]),
        "lost": np.array([r.lost for r in recs]),
        "demand": np.array([r.demand for r in recs]),
        "requested": [dict(r.requested) for r in recs],
        "executed": [dict(r.executed) for r in recs],
        "costs": [r.costs.as_dict() for r in recs],
        "cost_cents": [int(r.cost_cents) for r in recs],
        "G_bar": np.asarray(marks.G_bar),
        "y_bar": np.asarray(marks.y_bar),
        "R": np.asarray(marks.R),
        "alpha_bar": np.asarray(marks.alpha_bar),
        "pulse_log": log,
        "plans": kept,
        "errors": errors,
        "failed": failed,
        "seconds": time.perf_counter() - start,
    }
    p = path(task, tag, n)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(dump))
    return n, int(traj.J_cents), errors[:1], failed, round(dump["seconds"], 1)


def run(task="small", tag="base", agent="mpc_pulse_rec_v8", params=None, only=None, n_jobs=2, compare="t_v7", entropy=12345, plans=True):
    from joblib import Parallel, delayed

    assert entropy == 12345, "single episodes are looked at on the tuning root only"
    refs = references(task)
    base = cached(task, compare) if compare else {}
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    results = Parallel(n_jobs=min(n_jobs, len(ns)))(delayed(one)(task, n, agent, dict(params or {}), tag, entropy, plans) for n in ns)
    same = 0
    for n, J, errors, failed, seconds in results:
        r = refs[n]
        D = r["J_naive_cents"] - r["J_oracle_cents"]
        ok = base.get(n) == J
        same += ok
        d = (base[n] - J) / D if n in base else float("nan")
        print(
            f"episode {n:2d} level {r['stratum']}: score {(r['J_naive_cents'] - J) / D:.4f}; against {compare}: {d:+.4f}"
            f"{' (same to the cent)' if ok else ''}; failures {failed}{' ERRORS ' + str(errors) if errors else ''} ({seconds} s)",
            flush=True,
        )
    print(f"{same} of {len(results)} episodes cost the same as {compare} to the cent; {time.perf_counter() - start:.0f} s")


def names(inst):
    return [nd.id for nd in inst.nodes], [c.id for c in inst.commodities]


def show(task="small", tag="base", n=0, weeks=16, first=1, grids=None, other=None):
    """Per grid with fabs and week: stock, deliveries, segments, shed, fab energy, lots, the plan's view, the log."""
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node, K = names(inst)
    d = pickle.loads(path(task, tag, n).read_bytes())
    o = pickle.loads(path(task, other, n).read_bytes()) if other else None
    refs = references(task)
    D = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
    print(f"{task} episode {n} level {refs[n]['stratum']} tag {tag}: J {d['J_cents']}; score {(refs[n]['J_naive_cents'] - d['J_cents']) / D:.4f}; gap {D / 1e11:.1f} bn")
    if o:
        print(f"   against tag {other}: {(o['J_cents'] - d['J_cents']) / D:+.4f} of the gap ({(o['J_cents'] - d['J_cents']) / 1e11:+.2f} bn)")
        cum = np.cumsum(np.array(o["cost_cents"]) - np.array(d["cost_cents"])) / 1e11
        print("   cumulative saving against it by week (bn): " + " ".join(f"{w + 1}:{cum[w]:+.1f}" for w in range(0, len(cum), 4)))
    slot_of = {(s.node, s.k): i for i, s in enumerate(inst.stock_slots)}
    T = inst.T
    chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
    logs = {}
    for entry in d["pulse_log"]:
        logs.setdefault((entry[0], entry[1]), []).append(entry[2])
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if not fabs or (grids and node[g] not in grids):
            continue
        draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
        print(f"  {node[g]}: base {ga.deliverable - draw:.0f} of {ga.deliverable:.0f}, fabs' full draw {draw:.0f}; fuels " + ", ".join(f"{K[k]} burn {ga.shares[k] * ga.deliverable:.0f}" for k in ga.fuels) + f"; rationed {K[ga.rationed] if ga.rationed is not None else None}")
        feeders = {}
        for k in ga.fuels:
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if k in ed.K and ed.tau == 0 and ed.tail not in chk and ed.tail not in supply and (ed.tail, k) in slot_of:
                    feeders[k] = (e, ed.tail, [s for s, (ee, kk, _l) in enumerate(inst.action_slots) if ee == e and kk == k])
        fuels = [k for k in ga.fuels if k in feeders]
        head = "  wk | " + " | ".join(f"{K[k][:5]:>5s}: grid term sent  seg plan4-12" for k in fuels) + " | shed%  G/G0 | " + " ".join(f"{node[inst.fabs[fo]][4:14]:>10s}: lots waf chips disp planp" for fo in fabs) + " | E/draw | log"
        print(head)
        for t in range(first, min(weeks, T) + 1):
            i = t - 1
            cells = []
            for k in fuels:
                burn = ga.shares[k] * ga.deliverable
                e, tail, ss = feeders[k]
                sent = sum(d["executed"][i].get(s, 0.0) for s in ss)
                seg = d["segment"][i].get((go, k), 0.0)
                plan = d["plans"].get(t)
                if plan is not None and (go, k) in plan["short"]:
                    sh, cap = plan["short"][(go, k)], plan["cap"][(go, k)]
                    span = slice(3, min(12, len(sh)))
                    pl = float(sh[span].sum() / max(cap[span].sum(), 1e-9)) if len(sh) > 3 else float("nan")
                else:
                    pl = float("nan")
                cells.append(f"{K[k][:5]:>5s}: {d['stock'][i][slot_of[(g, k)]] / burn:4.2f} {d['stock'][i][slot_of[(tail, k)]] / burn:4.2f} {sent / burn:4.2f} {seg / burn:4.2f} {pl:8.2f}")
            fcells = []
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                plan = d["plans"].get(t)
                pp = float(plan["p"][fo][1:5].mean() / fa.cap0) if plan is not None and len(plan["p"][fo]) > 4 else float("nan")
                st = inst.stock_slots[slot_of[(f, fa.product)]].storage
                fcells.append(f"{'':>10s}  {d['lots'][i][fo] / fa.cap0:4.2f} {d['stock'][i][slot_of[(f, fa.input)]] / fa.cap0:3.1f} {d['stock'][i][slot_of[(f, fa.product)]] / (st or fa.cap0):5.2f} {d['disposal'][i][slot_of[(f, fa.product)]] / fa.cap0:4.2f} {pp:5.2f}")
            E = sum(d["energy"][i][fo] for fo in fabs)
            y = d["y_bar"][i][go]
            print(f"  {t:3d} | " + " | ".join(cells) + f" | {100 * d['shed'][i][go] / max(y, 1e-9):5.1f} {d['G_bar'][i][go] / ga.deliverable:5.2f} | " + " ".join(fcells) + f" | {E / draw:6.2f} | {','.join(logs.get((t, go), []))}")


def week1(task="small", tag="base"):
    """What the week-1 plan expects at each grid with fabs, per episode: the shortfall share of each fuel, weeks 4-12."""
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    node, K = names(inst)
    refs = references(task)
    for n in sorted(refs):
        p = path(task, tag, n)
        if not p.exists():
            continue
        d = pickle.loads(p.read_bytes())
        plan = d["plans"][1]
        cells = []
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if not fabs:
                continue
            parts = []
            for k in ga.fuels:
                if (go, k) in plan["short"]:
                    sh, cap = plan["short"][(go, k)], plan["cap"][(go, k)]
                    parts.append(f"{K[k][:3]} {sh[3:12].sum() / max(cap[3:12].sum(), 1e-9):.2f}")
            cells.append(f"{node[g][5:]}: " + " ".join(parts))
        print(f"  {n:2d} L{refs[n]['stratum']} | " + " | ".join(cells))


if __name__ == "__main__":
    fire.Fire({"run": run, "show": show, "week1": week1})
