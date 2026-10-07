"""Why the flat cap (plan_duty) loses: one episode, v4 without and with the cap, plans and outcomes at fab grids.

uv run python <this> --episode=60 --task=small [--weekly=<grid id>] [--agent=ft_buffer_v4]
"""

import sys
from pathlib import Path

import fire
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


def summarise(inst, marks, res, name, duty_log=None):
    traj, plans, tmpl = res["traj"], res["plans"], res["template"]
    N, S = inst.nodes, inst.stock_slots
    rec, T = traj.records, inst.T
    share, room, draw = lab.calendar_of(inst, marks, traj)
    pulse = getattr(res["agent"], "pulse", None)
    log = list(getattr(pulse, "log", []) or [])
    out = {"share": share, "lots": {}, "plan_next": {}, "wafers_in": {}}
    print(f"--- {name}: J {traj.J_cents / 1e11:.2f} bn; parts (bn): " + " ".join(f"{k} {v:.1f}" for k, v in lab.parts_of(traj).items() if abs(v) > 0.05))
    for go in lab.fab_grids(inst):
        g = inst.grids[go]
        ga = N[g].grid
        sh = share[:, go]
        held = sorted({t for t, gg, mode, _m in log if gg == go and mode == "off"})
        on = sorted({t for t, gg, mode, _m in log if gg == go and mode == "on"})
        print(f"  grid {N[g].id:10s} weeks with full fab power {int((sh >= 0.999).sum()):3d}, none {int((sh <= 0.001).sum()):3d}, partial {int(((sh > 0.001) & (sh < 0.999)).sum()):3d};"
              f" mean share {np.nanmean(sh):.3f}; pulse off-weeks {len(held)}, on-weeks changed {len(on)}; shed {sum(r.shed[go] for r in rec) * ga.voll / 1e9:.1f} bn")
        for fo in inst.grid_fabs[go]:
            f = inst.fabs[fo]
            fa = N[f].fab
            if fa.e <= 0:
                continue
            s = inst.slot_index[(f, fa.input)]
            cap = np.array([float(marks.alpha_bar[t, fo] * marks.R[t, fo]) * fa.cap0 for t in range(T)])
            lots = np.array([float(r.lots_started[fo]) for r in rec])
            on_hand = np.array([float(r.stock[s] + r.lots_started[fo] + r.disposal[s]) for r in rec])
            powered = sh >= 0.999
            arrivals = np.zeros(T)
            for ti, r in enumerate(rec):
                for (e, k, _lane), q in r.x.items():
                    if k == fa.input and inst.edges[e].head == f and ti + inst.edges[e].tau < T:
                        arrivals[ti + inst.edges[e].tau] += q
            # what the plan of week t expected of the next 4 weeks (window weeks 2..5) against what happened
            exp4, got4, exp1 = [], [], []
            for t in range(1, T - 4):
                z = plans.get(t)
                if z is None or z.shape[0] < 5:
                    continue
                p = z[:, tmpl[("p", fo)]]
                exp4.append(p[1:5].sum())
                got4.append(lots[t : t + 4].sum())
                exp1.append(p[1])
            out["lots"][fo] = lots
            out["wafers_in"][fo] = arrivals
            line = (f"    fab {N[f].id:20s} lots {lots.sum() / fa.cap0:6.2f} wk of capacity (possible {cap.sum() / fa.cap0:5.1f});"
                    f" in powered weeks: lots/cap {lots[powered].sum() / max(cap[powered].sum(), 1e-9):.2f}, wafers on hand/cap {np.minimum(on_hand, cap)[powered].sum() / max(cap[powered].sum(), 1e-9):.2f};"
                    f" wafers arrived {arrivals.sum() / fa.cap0:6.2f} wk; mean wafer stock {np.mean([r.stock[s] for r in rec]) / fa.cap0:.2f} wk, disposed {sum(r.disposal[s] for r in rec) / fa.cap0:.2f} wk;"
                    f" plan of weeks +1..+4: {np.sum(exp4) / max(np.sum(got4), 1e-9):.2f} x what was started")
            print(line)
    if duty_log:
        print("  the cap's duty estimate against the share of full power over the next 4 and 12 weeks (by grid: mean estimate | mean next 4 | mean next 12 | mean abs error to next 4 | weeks capped under 0.999)")
        for go in lab.fab_grids(inst):
            est = np.array([duty_log.get((t, go), np.nan) for t in range(1, T + 1)])
            n4 = np.array([np.nanmean(share[t : t + 4, go]) if t < T else np.nan for t in range(1, T + 1)])
            n12 = np.array([np.nanmean(share[t : t + 12, go]) if t < T else np.nan for t in range(1, T + 1)])
            ok = np.isfinite(est) & np.isfinite(n4)
            capped = ok & (est < 0.999)
            if capped.any():
                print(f"    {N[inst.grids[go]].id:10s} all weeks: {np.mean(est[ok]):.3f} | {np.mean(n4[ok]):.3f} | {np.mean(n12[ok]):.3f} | {np.mean(np.abs(est[ok] - n4[ok])):.3f} | {int(capped.sum())}"
                      f"   capped weeks only: est {np.mean(est[capped]):.3f} next4 {np.mean(n4[capped]):.3f} next12 {np.mean(n12[capped]):.3f}; too low by >0.2 in {int((n4[capped] - est[capped] > 0.2).sum())}, too high by >0.2 in {int((est[capped] - n4[capped] > 0.2).sum())}")
    return out


def weekly_table(inst, marks, a, b, grid_id, duty_log):
    N, E = inst.nodes, inst.edges
    g = inst.node_index[grid_id]
    go = inst.grid_ordinal[g]
    ga = N[g].grid
    fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
    sa, _ra, _da = lab.calendar_of(inst, marks, a["traj"])
    sb, _rb, _db = lab.calendar_of(inst, marks, b["traj"])
    la = {t: mode for t, gg, mode, _m in getattr(a["agent"].pulse, "log", []) if gg == go and mode != "waf"}
    lb = {t: mode for t, gg, mode, _m in getattr(b["agent"].pulse, "log", []) if gg == go and mode != "waf"}
    print(f"\n{grid_id}: week | G/G0 | A: pulse, power share, per fab [wafers on hand, lots, plan's next-week lots, wafers sent] (weeks of capacity) | B: duty, the same")
    for t in range(1, inst.T + 1):
        line = f"{t:4d} {marks.G_bar[t - 1, go] / ga.deliverable:5.2f} |"
        for res, sh, lg, tag in ((a, sa, la, "A"), (b, sb, lb, "B")):
            r = res["traj"].records[t - 1]
            z = res["plans"].get(t)
            tmpl = res["template"]
            line += f" {lg.get(t, '-'):3s} {sh[t - 1, go]:4.2f}"
            if tag == "B":
                line += f" d{duty_log.get((t, go), float('nan')):4.2f}"
            for fo in fabs:
                f = inst.fabs[fo]
                fa = N[f].fab
                s = inst.slot_index[(f, fa.input)]
                on_hand = float(r.stock[s] + r.lots_started[fo] + r.disposal[s]) / fa.cap0
                sent = sum(q for (e, k, _l), q in r.x.items() if k == fa.input and E[e].head == f) / fa.cap0
                nxt = z[1, tmpl[("p", fo)]] / fa.cap0 if z is not None and z.shape[0] > 1 else float("nan")
                line += f" [{on_hand:4.2f} {r.lots_started[fo] / fa.cap0:4.2f} {nxt:4.2f} {sent:4.2f}]"
            line += " |"
        print(line)


def main(episode=60, task="small", agent="ft_buffer_v4", weekly=""):
    w = lab.world(task, episode)
    inst, marks = w[0], w[2]
    refs = lab.references(task)
    gap = (refs[episode]["J_naive_cents"] - refs[episode]["J_oracle_cents"]) / 1e11
    duty_log = {}

    def patch(mods):
        res = mods["reserve"]
        orig = res.Builder.duty

        def duty(self, go):
            v = orig(self, go)
            duty_log[(self.settings.week, go)] = v
            return v

        res.Builder.duty = duty

    a = lab.play(task, episode, agent, dict(lab.V4), keep_plans=True, world_=w, tag="a")
    b = lab.play(task, episode, agent, dict(lab.V4) | {"plan_duty": True}, patch=patch, keep_plans=True, world_=w, tag="b")
    ja, jb = a["traj"].J_cents / 1e11, b["traj"].J_cents / 1e11
    print(f"{task} episode {episode} (level {refs[episode]['stratum']}, gap {gap:.1f} bn): A = v4 {ja:.2f} bn, B = v4 + plan_duty {jb:.2f} bn; B - A {jb - ja:+.2f} bn ({(ja - jb) / gap:+.4f} of the gap); errors {len(a['errors'])} {len(b['errors'])}")
    pa, pb = lab.parts_of(a["traj"]), lab.parts_of(b["traj"])
    print("  B - A by cost part (bn): " + " ".join(f"{k} {pb[k] - pa[k]:+.2f}" for k in pa if abs(pb[k] - pa[k]) > 0.005))
    oa = summarise(inst, marks, a, "A (v4)")
    ob = summarise(inst, marks, b, "B (v4 + plan_duty)", duty_log)
    print("  lots B - A by fab (weeks of capacity), wafers arrived B - A:")
    for fo in oa["lots"]:
        fa = inst.nodes[inst.fabs[fo]].fab
        print(f"    {inst.nodes[inst.fabs[fo]].id:20s} lots {(ob['lots'][fo].sum() - oa['lots'][fo].sum()) / fa.cap0:+6.2f}  wafers {(ob['wafers_in'][fo].sum() - oa['wafers_in'][fo].sum()) / fa.cap0:+6.2f}")
    print("  unserved demand by sink, B - A (bn USD):")
    T = inst.T
    for do, d in enumerate(inst.demands):
        ua = sum((r.backlog[do] if d.backlog else r.lost[do]) for r in a["traj"].records) * d.pi / 1e9
        ub = sum((r.backlog[do] if d.backlog else r.lost[do]) for r in b["traj"].records) * d.pi / 1e9
        if abs(ub - ua) > 0.05:
            print(f"    {inst.nodes[d.node].id:16s} {inst.commodities[d.k].id:14s} {ub - ua:+8.2f}  (A {ua:.1f})")
    if weekly:
        weekly_table(inst, marks, a, b, weekly, duty_log)


if __name__ == "__main__":
    fire.Fire(main)
