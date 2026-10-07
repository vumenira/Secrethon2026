"""Why fabs lack energy: per grid, the headroom above base load and each fuel's shortfall, against what its fabs need.

uv run python <this> --agent=f_e10_w4 --episodes=0,3 [--weekly=grid_cn]
"""

import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent


def folder(name):
    for base in (ROOT / "agents", ROOT / "outputs" / "recovery_tune" / "agents", HERE / "agents"):
        if (base / name / "agent.py").is_file():
            return str(base / name)
    raise ValueError(name)


def main(agent="f_e10_w4", episodes=(0,), weekly="", task="full", entropy=12345, first=14):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    cache = str(default_cache_dir())
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
        N, K, S = inst.nodes, inst.commodities, inst.stock_slots
        with without_secret_like():
            factory = load_agent_class(folder(agent), f"submission_{agent}")
            shim = _metered_shim(factory, None)
            traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        unload_agent()
        rec = traj.records
        T = inst.T
        psi = inst.params.psi
        print(f"\n=== episode {n}: {agent} J {traj.J_cents / 1e11:.1f} bn; psi {psi}; weeks {first}..{T}")
        print("grid: weeks(share) the grid's capacity leaves no room for its fabs | mean share of the fabs' full need that: capacity allows, fuel allows, fabs got"
              " | mean shortfall by fuel as a share of the fabs' need | base load shed share")
        for go, g in enumerate(inst.grids):
            ga = N[g].grid
            fabs = inst.grid_fabs[go]
            if not fabs:
                continue
            rows = []
            for t in range(T):
                r = rec[t]
                prev = rec[t - 1].stock if t else None
                Gbar, ybar = float(marks.G_bar[t, go]), float(marks.y_bar[t, go])
                need = sum(N[inst.fabs[fo]].fab.e * float(marks.alpha_bar[t, fo]) * N[inst.fabs[fo]].fab.cap0 for fo in fabs)  # at R: energy e p / R, p <= alpha R cap
                want = sum(N[inst.fabs[fo]].fab.e * min(float(marks.alpha_bar[t, fo] * marks.R[t, fo]) * N[inst.fabs[fo]].fab.cap0,
                                                         float(r.stock[inst.slot_index[(inst.fabs[fo], N[inst.fabs[fo]].fab.input)]] + r.lots_started[fo]))
                           / max(float(marks.R[t, fo]), 1e-9) for fo in fabs)  # with the wafers on hand
                head = Gbar - ybar
                short = {}
                for k in ga.fuels:
                    s = inst.slot_index[(g, k)]
                    burn = r.segment.get((go, k), 0.0)
                    cap = ga.shares[k] * Gbar
                    ration = 1.0
                    if k == ga.rationed and psi * ga.ibar[k] > 0:
                        before = float(prev[s]) if prev is not None else float(dict(((a, b), q) for a, b, q in inst.initial_state.stock).get((g, k), 0.0))
                        ration = min(1.0, before / (psi * ga.ibar[k]))
                    av = min(cap * ration, float(r.stock[s]) + burn)  # the simulator's available output of the segment
                    short[k] = max(cap - av, 0.0)
                got = float(sum(r.energy[fo] for fo in fabs))
                total = sum(r.segment.get((go, k), 0.0) for k in list(ga.fuels) + [None])
                rows.append((Gbar / ga.deliverable, ybar, need, want, head, short, got, float(r.shed[go]), total,
                             {k: float(r.stock[inst.slot_index[(g, k)]]) for k in ga.fuels}))
            sel = rows[first - 1:]
            need = np.array([x[2] for x in sel])
            want = np.array([x[3] for x in sel])
            head = np.array([x[4] for x in sel])
            got = np.array([x[6] for x in sel])
            shed = np.array([x[7] for x in sel])
            ybar = np.array([x[1] for x in sel])
            fuel_short = np.array([sum(x[5].values()) for x in sel])
            cap_allows = np.clip(head, 0, need) / np.maximum(need, 1e-9)
            fuel_allows = np.clip(head - fuel_short, 0, need) / np.maximum(need, 1e-9)
            line = (f"  {N[g].id:9s} need {need.mean():6.0f} GWh/wk (with wafers on hand {want.mean():6.0f}); no room {np.mean(head < 1e-6):.2f}"
                    f" | capacity {cap_allows.mean():.2f} fuel {fuel_allows.mean():.2f} got {(got / np.maximum(need, 1e-9)).mean():.2f} |")
            for k in ga.fuels:
                sh = np.array([x[5][k] for x in sel])
                line += f" {K[k].id} {sh.mean() / max(need.mean(), 1e-9):.2f} (weeks short {np.mean(sh > 1e-6):.2f})"
            line += f" | shed {shed.sum() / ybar.sum():.3f}"
            print(line)
            if weekly and N[g].id in (list(weekly) if isinstance(weekly, (list, tuple)) else str(weekly).split(",")):
                print("   week  Gbar/G0  head  need  want  got   shed  total  " + "  ".join(f"{K[k].id}: short stock(wk)" for k in ga.fuels))
                for t, x in enumerate(rows, start=1):
                    fuels = "  ".join(f"{x[5][k]:7.0f} {x[9][k] / (ga.shares[k] * ga.deliverable):5.2f}" for k in ga.fuels)
                    print(f"   {t:4d}  {x[0]:.3f} {x[4]:6.0f} {x[2]:5.0f} {x[3]:5.0f} {x[6]:5.0f} {x[7]:6.0f} {x[8]:7.0f}  {fuels}")


if __name__ == "__main__":
    fire.Fire(main)
