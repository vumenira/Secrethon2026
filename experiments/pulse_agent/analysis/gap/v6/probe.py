"""Play single episodes of an agent folder with entries of PARAMS replaced in memory, and compare with a cached row.

The folder is loaded as the scorer loads it and never edited. Each episode prints its score, the difference to the
cached base row in score units of that episode, the pulse's log counts and any failure; ``--grid`` adds one grid week
by week, ``--dump`` keeps the records.

uv run python probe.py --task=small --only=25 --params='{"ration_trim": true}' --grid=grid_tw
"""

import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402


def one(task, n, folder, params, entropy, grid, dump, timing):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    start = time.perf_counter()
    cpu = []
    with without_secret_like():
        factory = load_agent_class(str(folder), "submission_probe")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params)
        if timing:
            orig = module.Agent.act

            def act(self, observation):
                t0 = time.process_time()
                out = orig(self, observation)
                cpu.append(time.process_time() - t0)
                return out

            module.Agent.act = act
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    log = list(agent.pulse.log) if agent.pulse is not None else []
    failed = (getattr(agent.planner, "forecast_failed", None), getattr(agent.planner, "reserve_failed", None), agent.pulse_failed)
    unload_agent()
    seconds = time.perf_counter() - start
    out = {"episode": n, "J": int(traj.J_cents), "errors": errors[:2], "failed": failed, "log": log, "seconds": seconds, "cpu": cpu}
    inst = inst.at_digest(marks.instance_digest)
    name = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    lines = []
    if grid is not None:
        go = next(i for i, g in enumerate(inst.grids) if name[g] == grid)
        g = inst.grids[go]
        ga = inst.nodes[g].grid
        fabs = inst.grid_fabs[go]
        feeders = {}
        for k in ga.fuels:
            for e in inst.in_edges[g]:
                ed = inst.edges[e]
                if k in ed.K and ed.tau == 0 and (ed.tail, k) in inst.slot_index:
                    feeders[k] = (e, inst.slot_index[(ed.tail, k)])
        events = {}
        for w, g2, what, moved in log:
            if g2 == go:
                events.setdefault(w, []).append(what)
        head = f"{'wk':>3s} {'Gbar':>6s} {'shed':>6s} {'load':>5s} {'fabE':>5s} {'draw':>5s}"
        for k in ga.fuels:
            head += f" | {K[k][:5]:>5s} burn {'cap':>6s} {'grid':>7s} {'term':>7s} {'deliv':>7s}"
        lines.append(f"{task} episode {n} {grid}: base {ga.base_load:.0f} deliverable {ga.deliverable:.0f}")
        lines.append(head + "  pulse")
        for ti, r in enumerate(traj.records):
            Gb, yb = float(marks.G_bar[ti, go]), float(marks.y_bar[ti, go])
            null = r.segment.get((go, None), 0.0)
            null_cap = ga.shares.get(None, 0.0) * Gb
            load = null / null_cap if null_cap > 0 else float("nan")
            E = sum(float(r.energy[fo]) for fo in fabs)
            draw = 0.0
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                R, al = float(marks.R[ti, fo]), float(marks.alpha_bar[ti, fo])
                wafers = float(r.stock[inst.slot_index[(f, fa.input)]]) + float(r.lots_started[fo])
                draw += fa.e * min(al * R * fa.cap0, wafers) / R if R > 0 else 0.0
            line = f"{ti + 1:3d} {Gb:6.0f} {float(r.shed[go]):6.0f} {load:5.2f} {E:5.0f} {draw:5.0f}"
            for k in ga.fuels:
                s = inst.slot_index[(g, k)]
                line += f" | {r.segment.get((go, k), 0.0):10.0f} {ga.shares[k] * Gb:6.0f} {float(r.stock[s]):7.0f}"
                if k in feeders:
                    e, ts = feeders[k]
                    deliv = sum(q for (e2, k2, _lane), q in r.x.items() if e2 == e and k2 == k)
                    line += f" {float(r.stock[ts]):7.0f} {deliv:7.0f}"
                else:
                    line += f" {'':7s} {'':7s}"
            lines.append(line + "  " + ",".join(events.get(ti + 1, [])))
    out["lines"] = lines
    # weeks after a trim in which the ration held the rationed segment below what the week asked for
    psi = inst.params.psi
    lost_e = lost_y = 0.0
    weeks_missed = 0
    for w, go, what, _moved in log:
        if what != "trim" or w >= inst.T:
            continue
        g = inst.grids[go]
        ga = inst.nodes[g].grid
        r_, nx = traj.records[w - 1], traj.records[w]
        thr = psi * ga.ibar[ga.rationed]
        ration = min(1.0, float(r_.stock[inst.slot_index[(g, ga.rationed)]]) / thr)
        room = ga.shares[ga.rationed] * float(marks.G_bar[w, go]) * (1.0 - ration)
        draw = 0.0
        for fo in inst.grid_fabs[go]:
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            R, al = float(marks.R[w, fo]), float(marks.alpha_bar[w, fo])
            wafers = float(nx.stock[inst.slot_index[(f, fa.input)]]) + float(nx.lots_started[fo])
            draw += fa.e * min(al * R * fa.cap0, wafers) / R if R > 0 else 0.0
        E = sum(float(nx.energy[fo]) for fo in inst.grid_fabs[go])
        short_e, short_y = max(draw - E, 0.0), float(nx.shed[go])
        if room > 1e-6 and short_e + short_y > 1e-6:
            weeks_missed += 1
            a = min(short_y, room)
            lost_y += a
            lost_e += min(short_e, room - a)
    out["miss"] = (weeks_missed, lost_e, lost_y)
    out["shed"] = float(sum(float(r.shed.sum()) for r in traj.records))
    out["energy"] = float(sum(float(r.energy.sum()) for r in traj.records))
    out["disposal"] = float(sum(float(r.disposal.sum()) for r in traj.records))
    parts = Counter()
    for r in traj.records:
        for key, v in r.costs.as_dict().items():
            if isinstance(v, (int, float)):
                parts[key] += float(v)
    out["parts"] = dict(parts)
    out["salvage"] = float(traj.salvage)
    if dump:
        path = HERE / "traj" / f"{dump}_{task}_{entropy}_{n}.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"J": int(traj.J_cents), "records": traj.records, "log": log}))
    return out


def parse(params):
    """A dict, a JSON file, or "name=value,name=value" with JSON values (PowerShell eats the quotes of inline JSON)."""
    if isinstance(params, dict):
        return dict(params)
    if isinstance(params, (tuple, list)):
        params = ",".join(str(v) for v in params)
    params = str(params).strip()
    if params in ("", "{}"):
        return {}
    if Path(params).is_file():
        return json.loads(Path(params).read_text())
    out = {}
    for item in params.split(","):
        key, value = item.split("=", 1)
        try:
            out[key.strip()] = json.loads(value.strip().lower() if value.strip().lower() in ("true", "false", "null") else value)
        except json.JSONDecodeError:
            out[key.strip()] = value.strip()
    return out


def main(task="small", only=0, folder="mpc_pulse_rec_v6", params="{}", base="v_r1", entropy=12345, episodes=None, grid=None, dump=None, n_jobs=1, timing=False, parts=False):
    from joblib import Parallel, delayed

    params = parse(params)
    params.setdefault("strict", True)
    root = Path(folder) if Path(folder).is_absolute() else C.ROOT / "agents" / folder
    episodes = episodes or C.TOTAL[task]
    out_dir = C.OUT / f"{task}-e{entropy}-n{episodes}"
    refs = {r["episode"]: r for r in json.loads((out_dir / "references.json").read_text())}
    cached = {r["episode"]: r["J_policy_cents"] for r in json.loads((out_dir / f"{base}.json").read_text())["rows"]} if base else {}
    ns = [only] if isinstance(only, int) else [int(v) for v in only]
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, root, params, entropy, grid, dump, timing) for n in ns)
    total = 0.0
    for r in results:
        n = r["episode"]
        ref = refs[n]
        gap = ref["J_naive_cents"] - ref["J_oracle_cents"]
        score = (ref["J_naive_cents"] - r["J"]) / gap
        diff = (cached[n] - r["J"]) / gap if n in cached else float("nan")
        cents = r["J"] - cached[n] if n in cached else None
        total += diff
        kinds = Counter(what for _w, _g, what, _m in r["log"])
        print(
            f"episode {n} level {ref['stratum']}: {score:.4f} ({diff:+.4f} against {base}; {cents} cents) shed {r['shed']:,.0f} fab energy {r['energy']:,.0f}"
            f" disposed {r['disposal']:,.0f} log {dict(kinds)} failed {r['failed']} errors {r['errors']} ({r['seconds']:.0f} s)"
            + (f" | after a trim: {r['miss'][0]} weeks short, fab energy {r['miss'][1]:,.0f}, base load {r['miss'][2]:,.0f} GWh" if r["miss"][0] else ""),
            flush=True,
        )
        if parts:
            print("   parts (bn USD): " + " ".join(f"{k} {v / 1e9:,.2f}" for k, v in sorted(r["parts"].items())) + f" salvage {r['salvage'] / 1e9:,.2f}")
        if timing and r["cpu"]:
            print(f"   act CPU per week: mean {np.mean(r['cpu']):.3f} s, slowest {np.max(r['cpu']):.3f} s")
        for line in r["lines"]:
            print(line)
    if len(results) > 1:
        print(f"mean difference over {len(results)} episodes {total / len(results):+.4f} (unweighted, each in its own episode's units)")


if __name__ == "__main__":
    fire.Fire(main)
