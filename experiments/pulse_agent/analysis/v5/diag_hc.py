"""One episode: v4 as played against the same agent told the hindsight calendar; outcomes at fab grids and by cost part.

uv run python <this> --episode=25 --task=small --form=energy [--weekly=grid_kr] [--extra='{"short_premium": 0}']
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ceiling  # noqa: E402
import lab  # noqa: E402
from diag_cap import summarise  # noqa: E402


def fuel_table(inst, marks, a, b, grid_id, first=1, last=None):
    N, E, K = inst.nodes, inst.edges, inst.commodities
    g = inst.node_index[grid_id]
    go = inst.grid_ordinal[g]
    ga = N[g].grid
    fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
    feeders = {k: [e for e in inst.in_edges[g] if k in E[e].K and E[e].tau == 0] for k in ga.fuels}
    shares = [lab.calendar_of(inst, marks, r["traj"])[0] for r in (a, b)]
    logs = [{t: mode for t, gg, mode, _m in getattr(r["agent"].pulse, "log", []) if gg == go and mode != "waf"} for r in (a, b)]
    print(f"\n{grid_id}: week G/G0 | for A then B: pulse, power share, lots per fab (wk of capacity), shed (share of base load), per fuel [tank (wk of burn), sent to grid, grid stock, burnt, arrived at tank]")
    for t in range(first, (last or inst.T) + 1):
        line = f"{t:4d} {marks.G_bar[t - 1, go] / ga.deliverable:5.2f} |"
        for res, sh, lg in zip((a, b), shares, logs):
            r = res["traj"].records[t - 1]
            line += f" {lg.get(t, '-'):3s} {sh[t - 1, go]:4.2f} " + " ".join(f"{r.lots_started[fo] / N[inst.fabs[fo]].fab.cap0:4.2f}" for fo in fabs)
            line += f" sh{r.shed[go] / max(float(marks.y_bar[t - 1, go]), 1e-9):5.2f} "
            for k in ga.fuels:
                burn = ga.shares[k] * ga.deliverable
                tails = {E[e].tail for e in feeders[k]}
                tank = sum(r.stock[inst.slot_index[(v, k)]] for v in tails if (v, k) in inst.slot_index) / burn
                sent = sum(x for (e, kk, _l), x in r.x.items() if kk == k and e in feeders[k]) / burn
                prev = res["traj"].records[t - 2] if t > 1 else None
                arr = 0.0
                if prev is not None:
                    before = sum(prev.stock[inst.slot_index[(v, k)]] for v in tails if (v, k) in inst.slot_index) / burn
                    arr = tank + sent - before
                line += f"[{K[k].id[:3]} {tank:5.2f} {sent:4.2f} {r.stock[inst.slot_index[(g, k)]] / burn:5.2f} {r.segment.get((go, k), 0.0) / burn:4.2f} {arr:5.2f}]"
            line += " |"
        print(line)


def main(episode=25, task="small", form="energy", weekly="", extra=None, first=1, last=None, base_extra=None):
    w = lab.world(task, episode)
    inst, marks = w[0], w[2]
    refs = lab.references(task)
    gap = (refs[episode]["J_naive_cents"] - refs[episode]["J_oracle_cents"]) / 1e11
    _j0, share, pulsed, flows = ceiling.base_pass(task, episode, w)
    params, only, pin = ceiling.form_params(form)
    params |= dict(extra or {})
    a = lab.play(task, episode, ceiling.AGENT, dict(lab.V4) | dict(base_extra or {}), keep_plans=True, world_=w, tag="a")
    b = lab.play(task, episode, ceiling.AGENT, params, patch=ceiling.feed(share, pulsed if only else None, flows if pin else None, pin == "wafers"), keep_plans=True, world_=w, tag="b")
    ja, jb = a["traj"].J_cents / 1e11, b["traj"].J_cents / 1e11
    print(f"{task} episode {episode} (level {refs[episode]['stratum']}, gap {gap:.1f} bn): A = v4 {ja:.2f} bn, B = hindsight calendar ({form}) {jb:.2f} bn; B - A {jb - ja:+.2f} bn ({(ja - jb) / gap:+.4f} of the gap); errors {len(a['errors'])} {len(b['errors'])}")
    pa, pb = lab.parts_of(a["traj"]), lab.parts_of(b["traj"])
    print("  B - A by cost part (bn): " + " ".join(f"{k} {pb[k] - pa[k]:+.2f}" for k in pa if abs(pb[k] - pa[k]) > 0.005))
    oa = summarise(inst, marks, a, "A (v4)")
    ob = summarise(inst, marks, b, f"B ({form})")
    print("  lots B - A by fab (weeks of capacity), wafers arrived B - A:")
    for fo in oa["lots"]:
        fa = inst.nodes[inst.fabs[fo]].fab
        print(f"    {inst.nodes[inst.fabs[fo]].id:20s} lots {(ob['lots'][fo].sum() - oa['lots'][fo].sum()) / fa.cap0:+6.2f}  wafers {(ob['wafers_in'][fo].sum() - oa['wafers_in'][fo].sum()) / fa.cap0:+6.2f}")
    print("  unserved demand by sink, B - A (bn USD):")
    for do, d in enumerate(inst.demands):
        ua = sum((r.backlog[do] if d.backlog else r.lost[do]) for r in a["traj"].records) * d.pi / 1e9
        ub = sum((r.backlog[do] if d.backlog else r.lost[do]) for r in b["traj"].records) * d.pi / 1e9
        if abs(ub - ua) > 0.05:
            print(f"    {inst.nodes[d.node].id:16s} {inst.commodities[d.k].id:14s} {ub - ua:+8.2f}  (A {ua:.1f})")
    print("  base load shed by grid, B - A (bn USD):")
    for go, g in enumerate(inst.grids):
        voll = inst.nodes[g].grid.voll
        sa = sum(r.shed[go] for r in a["traj"].records) * voll / 1e9
        sb = sum(r.shed[go] for r in b["traj"].records) * voll / 1e9
        if abs(sb - sa) > 0.05:
            print(f"    {inst.nodes[g].id:12s} {sb - sa:+8.2f}  (A {sa:.1f})")
    if weekly:
        from diag_cap import weekly_table

        weekly_table(inst, marks, a, b, weekly, {})
        fuel_table(inst, marks, a, b, weekly, first, last)


if __name__ == "__main__":
    fire.Fire(main)
