"""A compact table of the persistence checks (persist.py).

uv run python persistrep.py --task=small
"""

import json

import fire
import numpy as np

import common as C


def main(task="small"):
    refs = C.references(task)
    rows = []
    for p in sorted((C.HERE / "persist").glob(f"{task}_*.json")):
        rows.append(json.loads(p.read_text()))
    print(f"{task}: {len(rows)} checks. Scores of the true world are shares of the episode's gap; 'in the world' is the cost in the persisting world, as a share of the same gap, relative to our calendar there")
    tot = {"differ": 0, "agree": 0}
    for r in sorted(rows, key=lambda r: (r["week"], r["episode"])):
        n, D = r["episode"], r["D"]
        naive = refs[n]["J_naive_cents"] / 100
        sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
        rel = lambda a, b: float("nan") if a is None or b is None else (a - b) / D  # noqa: E731
        jo, jp, jw = r.get("J_ours_in_world"), r.get("J_plan_in_world"), r.get("J_world")
        dif = sum(g.get("differ") or 0 for g in r.get("grids", []))
        agr = sum(g.get("world_as_plan") or 0 for g in r.get("grids", []))
        tot["differ"] += dif
        tot["agree"] += agr
        b4 = {lab: sum(g[lab]["burn4"].get("lng", 0.0) for g in r.get("grids", []) if lab in g) for lab in ("ours", "world", "plan")}
        sh = {lab: sum(g[lab]["shed"] for g in r.get("grids", []) if lab in g) for lab in ("ours", "world", "plan")}
        print(
            f"  episode {n:2d} week {r['week']}: {r.get('status')}, start {r.get('start')}; in the world: the plan's calendar {'not allowed' if jp is None else f'{rel(jo, jp):+.4f}' if jo is not None else 'allowed, ours not'}, the world's own {('%+.4f' % rel(jo, jw)) if jo is not None and jw is not None else ('ours not allowed' if jo is None else 'none')};"
            f" first 12 weeks: the plan differs from ours in {dif} grid-weeks, the world agrees with the plan in {agr};"
            f" gas burnt in the first 4 weeks, all grids: ours {b4['ours']:.0f} world {b4['world']:.0f} plan {b4['plan']:.0f};"
            f" true world: ours {sc(r.get('J_true_ours')):.4f}, plan {sc(r.get('J_true_plan')):.4f}, world's first 4 weeks then plan {sc(r.get('J_true_4_plan')):.4f}, first 8 then plan {sc(r.get('J_true_8_plan')):.4f}, first 8 then ours {sc(r.get('J_true_8_ours')):.4f}"
        )
    print(f"  in all: the plan differs from ours in {tot['differ']} grid-weeks of the first 12; the persisting world's optimum agrees with the plan in {tot['agree']} ({tot['agree'] / max(tot['differ'], 1):.0%})")
    for key, label in (("J_true_ours", "our calendar"), ("J_true_plan", "the realizable plan's"), ("J_true_4_plan", "the persisting world's first 4 weeks, then the plan's"), ("J_true_8_plan", "its first 8 weeks, then the plan's"), ("J_true_8_ours", "its first 8 weeks, then ours")):
        vals = [(refs[r["episode"]]["J_naive_cents"] / 100 - r[key]) / r["D"] for r in rows if r.get(key) is not None]
        print(f"    {label:56s} mean score {np.mean(vals) if vals else float('nan'):.4f} over {len(vals)} checks")


if __name__ == "__main__":
    fire.Fire(main)
