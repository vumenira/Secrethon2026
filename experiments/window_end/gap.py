"""Where an agent's cost sits against the clairvoyant plan's: per episode, week, cost component and grid.

Plays the agent (in process, as the scorer does) and solves the clairvoyant LP of every episode, then saves per week:
the cost components, base load shed per grid (units and USD), lots started per fab, for both. One JSON per episode in
outputs/gap/<task>-e<root>/<agent>/.

uv run python experiments/window_end/gap.py run --agent=agents/mpc_pulse_rec_v8_on --task=small --entropy=12345 \
    --episodes=64
uv run python experiments/window_end/gap.py report --agent=mpc_pulse_rec_v8_on --task=small --entropy=12345
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
COMPS = ("freight", "war_risk", "tariff", "holding", "queue_holding", "shortage", "disposal", "shed")


def one(agent, task, entropy, n, out):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.oracle.lp import ORACLE_METHOD, _terms, build_lp, solve_oracle
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    target = out / f"{n}.json"
    if target.exists():
        return
    inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))
    shim = _metered_shim(load_agent_class(str(agent), f"gap_{n}"), None)
    traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, ""), marks=marks, fallback=fallback)
    unload_agent()
    G, F = len(inst.grids), len(inst.fabs)
    voll = [inst.nodes[g].grid.voll for g in inst.grids]
    wafer_slots = [
        next(i for i, sl in enumerate(inst.stock_slots) if sl.node == f and sl.k == inst.nodes[f].fab.input)
        for f in inst.fabs
    ]
    ours = {
        "parts": [[getattr(r.costs, c) for c in COMPS] for r in traj.records],
        "shed": [r.shed.tolist() for r in traj.records],
        "lots": [r.lots_started.tolist() for r in traj.records],
        "energy": [r.energy.tolist() for r in traj.records],
        "lost": [r.lost.tolist() for r in traj.records],  # demand not served, per demand ordinal
        # end-of-week stock per slot summed over the episode (holding = slot holding rate x stock, every week)
        "stock_sum": np.sum([r.stock for r in traj.records], axis=0).tolist(),
        "demand": [r.demand.tolist() for r in traj.records],
        "wafers": [[float(r.stock[s]) for s in wafer_slots] for r in traj.records],  # end of week, at each fab
        "J": traj.J_cents / 100,
    }
    model = build_lp(inst, marks)
    res = solve_oracle(model, method=ORACLE_METHOD)
    weekly, _, _, _ = _terms(model, res.x)
    T = model.T
    shed, lots, energy = np.zeros((T, G)), np.zeros((T, F)), np.zeros((T, F))
    lost = np.zeros((T, len(inst.demands)))
    stock_sum = np.zeros(len(inst.stock_slots))
    for j, key in enumerate(model.keys):
        if key[0] == "ysh":
            shed[key[1] - 1, key[2]] += res.x[j]
        elif key[0] == "p":
            lots[key[1] - 1, key[2]] += res.x[j]
        elif key[0] == "E":
            energy[key[1] - 1, key[2]] += res.x[j]
        elif key[0] == "U":
            lost[key[1] - 1, key[2]] += res.x[j]
        elif key[0] == "I":
            stock_sum[key[2]] += res.x[j]
    oracle = {
        "parts": [[float(np.sum(w.get(c, []))) for c in COMPS] for w in weekly],
        "shed": shed.tolist(),
        "lots": lots.tolist(),
        "energy": energy.tolist(),
        "lost": lost.tolist(),
        "stock_sum": stock_sum.tolist(),
        "J": res.J_cents / 100 if res.J_cents is not None else None,
    }
    events = [
        {"type": str(getattr(ev, "event_type", getattr(ev, "kind", ""))), "onset": int(getattr(ev, "onset", -1))}
        for ev in getattr(omega, "events", ())
    ]
    cap = np.asarray(marks.R, dtype=float).tolist()  # the fabs' capacity each week (lots), outages included
    target.write_text(
        json.dumps({"episode": n, "cap": cap, "voll": voll, "ours": ours, "oracle": oracle, "events": events})
    )


def run(agent="agents/mpc_pulse_rec_v8_on", task="small", entropy=12345, episodes=64, n_jobs=3):
    from joblib import Parallel, delayed

    agent = (ROOT / agent).resolve()
    out = ROOT / "outputs" / "gap" / f"{task}-e{entropy}" / agent.name
    out.mkdir(parents=True, exist_ok=True)
    Parallel(n_jobs=n_jobs, verbose=5)(delayed(one)(agent, task, entropy, n, out) for n in range(episodes))


def report(agent="mpc_pulse_rec_v8_on", task="small", entropy=12345):
    out = ROOT / "outputs" / "gap" / f"{task}-e{entropy}" / agent
    eps = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"), key=lambda p: int(p.stem))]
    refs = (
        {
            r["episode"]: r
            for r in json.loads(
                (ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{len(list(out.glob('*.json')))}" / "references.json").read_text()
            )
        }
        if (ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{len(list(out.glob('*.json')))}" / "references.json").exists()
        else {}
    )
    ours = np.array([np.sum(e["ours"]["parts"], axis=0) for e in eps])
    orc = np.array([np.sum(e["oracle"]["parts"], axis=0) for e in eps])
    gap = ours - orc
    total = gap.sum()
    print(
        f"{task} root {entropy}: {len(eps)} episodes; our cost above the clairvoyant plan's, bn USD (sum over episodes)"
    )
    print(f"  total (components, salvage left out) {total / 1e9:,.0f}")
    for i, c in enumerate(COMPS):
        print(
            f"  {c:14s} ours {ours[:, i].sum() / 1e9:9,.0f}  clairvoyant {orc[:, i].sum() / 1e9:9,.0f}"
            f"  gap {gap[:, i].sum() / 1e9:+9,.0f}"
        )
    if refs:
        naive = sum(refs[e["episode"]]["J_naive_cents"] for e in eps) / 100
        oj = sum(refs[e["episode"]]["J_oracle_cents"] for e in eps) / 100
        closed = 1 - sum(e["ours"]["J"] - e["oracle"]["J"] for e in eps) / (naive - oj)
        print(
            f"  the naive-to-clairvoyant gap is {(naive - oj) / 1e9:,.0f} bn; ours closes {closed:.4f} (episodes alike)"
        )
    # by week of the episode
    T = len(eps[0]["ours"]["parts"])
    by_week = np.zeros((T, len(COMPS)))
    for e in eps:
        by_week += np.array(e["ours"]["parts"]) - np.array(e["oracle"]["parts"])
    print("gap by period (bn USD): shortage, shed, other")
    step = max(1, T // 13)
    for a in range(0, T, step):
        b = by_week[a : a + step]
        short, shed = b[:, 5].sum(), b[:, 7].sum()
        rest = b.sum() - short - shed
        print(f"  weeks {a + 1:3d}-{min(T, a + step):3d}: {short / 1e9:+8,.0f} {shed / 1e9:+8,.0f} {rest / 1e9:+8,.0f}")
    # shed by grid
    G = len(eps[0]["voll"])
    so = np.zeros(G)
    sc = np.zeros(G)
    for e in eps:
        v = np.array(e["voll"])
        so += (np.array(e["ours"]["shed"]) * v).sum(axis=0)
        sc += (np.array(e["oracle"]["shed"]) * v).sum(axis=0)
    print("shed by grid (bn USD, VOLL x units): ours, clairvoyant, gap")
    for g in range(G):
        print(f"  grid {g}: {so[g] / 1e9:9,.0f} {sc[g] / 1e9:9,.0f} {(so[g] - sc[g]) / 1e9:+9,.0f}")
    lo = sum(np.sum(e["ours"]["lots"]) for e in eps)
    lc = sum(np.sum(e["oracle"]["lots"]) for e in eps)
    print(f"lots started: ours {lo:,.0f}, clairvoyant {lc:,.0f} ({lo / lc:.3f})")
    # per-episode share of the gap by harm level
    if refs:
        print("per harm level: gap above clairvoyant split shortage / shed / other (bn USD)")
        for s in (1, 2, 3, 4):
            idx = [i for i, e in enumerate(eps) if refs[e["episode"]]["stratum"] == s]
            if idx:
                g = gap[idx]
                short, shed = g[:, 5].sum(), g[:, 7].sum()
                rest = g.sum() - short - shed
                print(
                    f"  level {s} ({len(idx)} episodes): {short / 1e9:+8,.0f} {shed / 1e9:+8,.0f} {rest / 1e9:+8,.0f}"
                )


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})
