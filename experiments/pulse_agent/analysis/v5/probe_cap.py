"""Which fault of the flat cap (plan_duty) costs what: the cap per fab or per grid, on its own duty or the true one.

uv run python <this> --episodes=60,25,50,32,57 --task=small

Four variants of v4 + plan_duty, each against v4 as cached:
  cap           the cap as built: each fab's lots at most duty x capacity, the duty estimated from the fuel cover
  cap_grid      the same duty as one row per grid (the fabs' energy at most duty x their full draw)
  true          the cap per fab with the true duty: the mean share of fab power over the next ``weeks`` weeks of v4's play
  true_grid     the true duty as one row per grid
"""

import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ceiling  # noqa: E402
import lab  # noqa: E402


def patch_of(variant, share, weeks):
    def patch(mods):
        B = mods["reserve"].Builder
        if variant.startswith("true"):

            def duty(self, go):
                t = self.settings.week  # the window's second week is episode week t + 1: row t of the calendar
                ahead = share[t : t + weeks, go]
                ahead = ahead[np.isfinite(ahead)]
                return float(ahead.mean()) if ahead.size else 1.0

            B.duty = duty
        if variant.endswith("grid"):

            def add_duty(self, where):
                inst = self.inst
                later = self.t >= 2
                for go in range(len(inst.grids)):
                    fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
                    if not fabs or inst.nodes[inst.grids[go]].grid.priority != "base_first":
                        continue
                    d = self.duty(go)
                    draw = np.zeros(self.T)
                    for fo in fabs:
                        fa = inst.nodes[inst.fabs[fo]].fab
                        draw = draw + fa.e * self.marks.alpha_bar[:, fo] * fa.cap0
                    r = self.ub.row(("duty", go), np.where(later, d, 1.0) * draw)
                    for fo in fabs:
                        self.ub.add(r, where[("E", fo)], 1.0)

            B.add_duty = add_duty

    return patch


def main(episodes, task="small", variants="cap,cap_grid,true,true_grid", weeks=4):
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    names = list(variants) if isinstance(variants, (list, tuple)) else [v for v in str(variants).split(",") if v]
    refs = lab.references(task)
    total = {name: 0.0 for name in names}
    gaps = 0.0
    print(f"{task}: v4 + plan_duty in four forms against v4, bn USD (share of the episode's gap); positive: better")
    for n in ns:
        w = lab.world(task, n)
        j0, share, _pulsed, _flows = ceiling.base_pass(task, n, w)
        gap = (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 1e11
        gaps += gap
        line = f"  episode {n:3d} (level {refs[n]['stratum']}, gap {gap:7.1f})"
        for name in names:
            res = lab.play(task, n, "ft_buffer_v5", dict(lab.V4) | {"plan_duty": True}, patch=patch_of(name, share, int(weeks)), world_=w, tag="pc")
            d = (j0 - res["traj"].J_cents) / 1e11
            total[name] += d
            line += f" | {name} {d:+7.2f} ({d / gap:+.4f}){' ERR' if res['errors'] else ''}"
        print(line, flush=True)
    print("  sum: " + " | ".join(f"{name} {total[name]:+.1f} bn ({total[name] / gaps:+.4f})" for name in names))


if __name__ == "__main__":
    fire.Fire(main)
