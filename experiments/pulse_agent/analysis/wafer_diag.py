"""Why fabs lack wafers in weeks they have power: wafers on hand, what feeds each fab, and what the window LP planned.

uv run python <this> --agent=ft_buffer_v2 --episode=3 --task=full [--plan=fab_jp_memory_1 --plan_weeks=30,31,32]
"""

import sys
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


def main(agent="ft_buffer_v2", episode=3, task="full", entropy=12345, plan="", plan_weeks=(), first=14, edges=True):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    cache = str(default_cache_dir())
    inst, omega, marks, fallback = _world(task, entropy, episode, REPLICATIONS, cache)
    N, K, S, E = inst.nodes, inst.commodities, inst.stock_slots, inst.edges
    T = inst.T
    plans = {}  # week -> {fab ordinal: (p per window week, wafer arrivals per window week, wafer stock per window week)}
    with without_secret_like():
        factory = load_agent_class(folder(agent), f"submission_{agent}")
        module = sys.modules[factory.__module__]
        week_box = [0]
        orig_act = module.Planner.act

        def act(self, obs):
            week_box[0] = int(obs["week"])
            return orig_act(self, obs)

        def record(self, model, res):
            if not res.ok:
                return
            tmpl, nc, H = model.meta["template"], model.meta["nc"], model.T
            z = res.x.reshape(H, nc)
            out = {}
            for fo, f in enumerate(inst.fabs):
                fa = N[f].fab
                p = z[:, tmpl[("p", fo)]].copy()
                arrive = np.zeros(H)
                for e in inst.in_edges[f]:
                    ed = E[e]
                    if fa.input not in ed.K:
                        continue
                    for key, j in tmpl.items():
                        if key[0] == "x" and key[1] == e and key[2] == fa.input:
                            col = z[:, j]
                            if ed.tau < H:
                                arrive[ed.tau:] += col[: H - ed.tau]
                s = inst.slot_index[(f, fa.input)]
                stock = z[:, tmpl[("I", s)]].copy()
                sp_ = inst.slot_index[(f, fa.product)]
                prod = z[:, tmpl[("I", sp_)]].copy()
                leave = np.zeros(H)
                for e in inst.out_edges[f]:
                    if fa.product in E[e].K:
                        for key, j in tmpl.items():
                            if key[0] == "x" and key[1] == e and key[2] == fa.product:
                                leave += z[:, j]
                go_ = inst.grid_ordinal[fa.grid]
                shed = z[:, tmpl[("ysh", go_)]].copy()
                out[fo] = (p, arrive, stock, prod, leave, shed)
            plans[week_box[0]] = out

        module.Planner.act = act
        module.Planner._record = record
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, episode, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    log = list(getattr(getattr(shim.agent, "pulse", None), "log", []) or [])
    lead = dict(getattr(getattr(shim.agent, "pulse", None), "lead", {}) or {})
    unload_agent()
    rec = traj.records
    print(f"{task} episode {episode} {agent}: J {traj.J_cents / 1e11:.1f} bn")
    wafer = inst.commodity_index["wafer"]
    print("fab: weeks counted (sellable, from week %d) | powered share | in powered weeks: lots/capacity, wafers on hand/capacity, weeks with under half | wafer storage at the fab in weeks of capacity | mean wafer stock (weeks)" % first)
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        s = inst.slot_index[(f, fa.input)]
        last = int(T - lead.get(fo, 14))
        rows = []
        for t in range(first, last + 1):
            r = rec[t - 1]
            cap = float(marks.alpha_bar[t - 1, fo] * marks.R[t - 1, fo]) * fa.cap0
            if cap <= 0:
                continue
            on_hand = float(r.stock[s] + r.lots_started[fo] + r.disposal[s])
            phat = min(cap, on_hand)
            asked = fa.e * phat / max(float(marks.R[t - 1, fo]), 1e-9)
            powered = float(r.energy[fo]) >= asked * (1 - 1e-6) - 1e-9
            rows.append((cap, on_hand, float(r.lots_started[fo]), powered, float(r.stock[s]), float(r.disposal[s])))
        if not rows:
            continue
        a = np.array(rows)
        pw = a[:, 3] > 0
        line = f"  {N[f].id:18s} {N[fa.grid].id:8s} weeks {len(rows):3d} | powered {pw.mean():.2f} |"
        if pw.any():
            line += f" lots {a[pw, 2].sum() / a[pw, 0].sum():.2f} on hand {np.minimum(a[pw, 1], a[pw, 0]).sum() / a[pw, 0].sum():.2f} under half {int((a[pw, 1] < 0.5 * a[pw, 0]).sum()):3d} |"
        line += f" storage {S[s].storage / fa.cap0:.2f} wk | stock {a[:, 4].mean() / fa.cap0:.2f} wk | disposed {a[:, 5].sum() / fa.cap0:.1f} wk of capacity"
        print(line)
    if edges:
        print("\nwafer supply by fab: inbound edges (tail, lead, nominal capacity in weeks of the fab's capacity, mean capacity, share of weeks prohibited, mean tariff, mean flow), weeks %d..%d" % (first, T - 14))
        sl = slice(first - 1, T - 14)
        weeks = T - 14 - first + 1
        flow = {}
        for r in rec[sl]:
            for (e, kk, _lane), q in r.x.items():
                if kk == wafer:
                    flow[e] = flow.get(e, 0.0) + q / weeks
        for fo, f in enumerate(inst.fabs):
            fa = N[f].fab
            print(f"  {N[f].id} (capacity {fa.cap0:.0f}/wk):")
            for e in inst.in_edges[f]:
                ed = E[e]
                if wafer not in ed.K:
                    continue
                ts = inst.slot_index.get((ed.tail, wafer))
                tail_stock = float(np.mean([r.stock[ts] for r in rec[sl]])) if ts is not None else float("nan")
                print(f"     {ed.id:46s} tail {N[ed.tail].type[:5]} tau {ed.tau} u0 {ed.u0 / fa.cap0:5.2f} cap {float(np.mean(marks.u[sl, e])) / fa.cap0:5.2f}"
                      f" banned {float(np.mean(marks.prohibited[sl, e, wafer])):.2f} tariff {float(np.mean(marks.tariff[sl, e, wafer])):.2f}"
                      f" flow {flow.get(e, 0.0) / fa.cap0:5.2f} tail stock {tail_stock / fa.cap0:6.1f} wk")
        print("  material nodes: supply per week (nominal, mean available), lifted, stock")
        for s, st in enumerate(S):
            if st.k == wafer and st.node in inst.supply_nodes:
                print(f"     {N[st.node].id:22s} nominal {st.supply:9.0f} avail {float(np.mean(marks.supply[sl, s])):9.0f} lifted {float(np.mean([r.lift[s] for r in rec[sl]])):9.0f}"
                      f" stock {float(np.mean([r.stock[s] for r in rec[sl]])):10.0f} storage {st.storage:10.0f}")
        pool = inst.commodity_pool[wafer]
        print("  fleet: pool", pool, "measure", inst.params.fleet_measure[pool], "share", inst.params.fleet_share[pool])
    if plan:
        fo = inst.fab_ordinal[inst.node_index[plan]]
        f = inst.fabs[fo]
        fa = N[f].fab
        s = inst.slot_index[(f, fa.input)]
        go = inst.grid_ordinal[fa.grid]
        modes = {t: m for t, gg, m, _mv in log if gg == go}
        ws = [int(w) for w in (plan_weeks if isinstance(plan_weeks, (list, tuple)) else [plan_weeks])]
        print(f"\n{plan}: per week: pulse mode, capacity share available, wafers on hand, lots, energy got/asked, stock end, disposal (all in weeks of nominal capacity)")
        for t in range(max(1, min(ws) - 6), min(T, max(ws) + 8) + 1):
            r = rec[t - 1]
            cap = float(marks.alpha_bar[t - 1, fo] * marks.R[t - 1, fo])
            on_hand = float(r.stock[s] + r.lots_started[fo] + r.disposal[s]) / fa.cap0
            asked = fa.e * min(cap * fa.cap0, on_hand * fa.cap0) / max(float(marks.R[t - 1, fo]), 1e-9)
            print(f"   {t:3d} {modes.get(t, '-'):4s} cap {cap:4.2f} on hand {on_hand:5.2f} lots {r.lots_started[fo] / fa.cap0:5.2f} energy {r.energy[fo]:6.1f}/{asked:6.1f} stock {r.stock[s] / fa.cap0:5.2f} disposed {r.disposal[s] / fa.cap0:5.2f}")
        print(f"   its chips' way out: edges carrying {K[fa.product].id} from the fab (head, lead, capacity nominal and mean in weeks of the fab's capacity, share of weeks prohibited)")
        for e in inst.out_edges[f]:
            ed = E[e]
            if fa.product in ed.K:
                print(f"     {ed.id:46s} head {N[ed.head].type[:5]} tau {ed.tau} u0 {ed.u0 / fa.cap0:5.2f} cap {float(np.mean(marks.u[first - 1:, e])) / fa.cap0:5.2f} banned {float(np.mean(marks.prohibited[first - 1:, e, fa.product])):.2f}")
        for t in ws:
            if t not in plans:
                continue
            p, arrive, stock, prod, leave, shed = plans[t][fo]
            sprod = S[inst.slot_index[(f, fa.product)]]
            print(f"     (its chip stock at the fab now {rec[t - 2].stock[inst.slot_index[(f, fa.product)]] / fa.cap0 if t > 1 else 0:4.2f} wk of capacity, storage {sprod.storage / fa.cap0:4.2f} wk)")
            print("     chips out " + " ".join(f"{x / fa.cap0:4.2f}" for x in leave))
            print("     chip stock " + " ".join(f"{x / fa.cap0:4.2f}" for x in prod))
            print("     base shed (GWh) " + " ".join(f"{x:5.0f}" for x in shed))
            print(f"   the window planned in week {t} (window weeks 1..{len(p)}), in weeks of nominal capacity:")
            print("     lots     " + " ".join(f"{x / fa.cap0:4.2f}" for x in p))
            print("     arrivals " + " ".join(f"{x / fa.cap0:4.2f}" for x in arrive))
            print("     stock    " + " ".join(f"{x / fa.cap0:4.2f}" for x in stock))


if __name__ == "__main__":
    fire.Fire(main)
