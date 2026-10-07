"""One episode: v4 against v5 with the pulse's forecast told to the window; outcomes at fab grids, forecast accuracy.

uv run python <this> --episode=12 --task=small --variant="e+l+s" [--weekly=grid_kr]
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402
import probe_fc  # noqa: E402
from diag_cap import summarise, weekly_table  # noqa: E402
from diag_hc import fuel_table  # noqa: E402


def main(episode=12, task="small", variant="e+l+s", weekly="", first=1, last=None, extra=None):
    w = lab.world(task, episode)
    inst, marks = w[0], w[2]
    refs = lab.references(task)
    gap = (refs[episode]["J_naive_cents"] - refs[episode]["J_oracle_cents"]) / 1e11
    seen = {}

    def patch(mods):
        P = mods["pulse"].Pulse
        orig = P.forecast

        def forecast(self, obs, values, prohibited, H):
            out = orig(self, obs, values, prohibited, H)
            seen[int(obs["week"])] = out
            return out

        P.forecast = forecast

    a = lab.play(task, episode, "ft_buffer_v5", dict(lab.V4), keep_plans=True, world_=w, tag="a")
    b = lab.play(task, episode, "ft_buffer_v5", probe_fc.params_of(variant) | dict(extra or {}), patch=patch, keep_plans=True, world_=w, tag="b")
    ja, jb = a["traj"].J_cents / 1e11, b["traj"].J_cents / 1e11
    print(f"{task} episode {episode} (level {refs[episode]['stratum']}, gap {gap:.1f} bn): A = v4 {ja:.2f} bn, B = forecast calendar ({variant}) {jb:.2f} bn; B - A {jb - ja:+.2f} bn ({(ja - jb) / gap:+.4f} of the gap); errors {len(a['errors'])} {len(b['errors'])}")
    pa, pb = lab.parts_of(a["traj"]), lab.parts_of(b["traj"])
    print("  B - A by cost part (bn): " + " ".join(f"{k} {pb[k] - pa[k]:+.2f}" for k in pa if abs(pb[k] - pa[k]) > 0.005))
    oa = summarise(inst, marks, a, "A (v4)")
    ob = summarise(inst, marks, b, f"B ({variant})")
    share = ob["share"]
    print("  B's forecast against B's own calendar, by grid: share of weeks called right 1, 2, 4, 8 weeks ahead (ON at 0.5); weeks forecast OFF that were ON, and ON that were OFF, 2 weeks ahead")
    for go in lab.fab_grids(inst):
        line = f"    {inst.nodes[inst.grids[go]].id:10s}"
        for lead in (1, 2, 4, 8):
            hit = [(seen[t][go][lead] >= 0.5) == (share[t - 1 + lead, go] >= 0.5) for t in seen if lead < len(seen[t][go]) and t - 1 + lead < inst.T]
            line += f" {np.mean(hit):.2f}"
        off_on = sum(1 for t in seen if 2 < len(seen[t][go]) and t + 1 < inst.T and seen[t][go][2] < 0.5 <= share[t + 1, go])
        on_off = sum(1 for t in seen if 2 < len(seen[t][go]) and t + 1 < inst.T and seen[t][go][2] >= 0.5 > share[t + 1, go])
        line += f"   forecast OFF but ON: {off_on}, forecast ON but OFF: {on_off}"
        print(line)
    print("  lots B - A by fab (weeks of capacity), wafers arrived B - A:")
    for fo in oa["lots"]:
        fa = inst.nodes[inst.fabs[fo]].fab
        print(f"    {inst.nodes[inst.fabs[fo]].id:20s} lots {(ob['lots'][fo].sum() - oa['lots'][fo].sum()) / fa.cap0:+6.2f}  wafers {(ob['wafers_in'][fo].sum() - oa['wafers_in'][fo].sum()) / fa.cap0:+6.2f}")
    print("  base load shed by grid, B - A (bn USD):")
    for go, g in enumerate(inst.grids):
        voll = inst.nodes[g].grid.voll
        sa = sum(r.shed[go] for r in a["traj"].records) * voll / 1e9
        sb = sum(r.shed[go] for r in b["traj"].records) * voll / 1e9
        if abs(sb - sa) > 0.05:
            print(f"    {inst.nodes[g].id:12s} {sb - sa:+8.2f}  (A {sa:.1f})")
    if weekly:
        go = inst.grid_ordinal[inst.node_index[weekly]]

        def glyph(v):
            return "x" if v >= 0.999 else "." if v <= 0.001 else str(int(v * 10))

        print(f"\n{weekly}: week | B's calendar of the next 12 weeks | B's forecast of them")
        for t in sorted(seen):
            print(f"   {t:3d} {''.join(glyph(share[t - 1 + k, go]) for k in range(min(13, inst.T - t + 1))):13s} {''.join(glyph(v) for v in seen[t][go][:13])}")
        weekly_table(inst, marks, a, b, weekly, {})
        fuel_table(inst, marks, a, b, weekly, first, last)


if __name__ == "__main__":
    fire.Fire(main)
