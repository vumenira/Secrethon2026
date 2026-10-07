"""The table of Question 2: the clusters of calendar differences (diff.py) by what makes them possible, in pooled
score units.

A cluster's size is its step when the clusters are added to our calendar in the order they begin. Its extra fab
energy (plan minus ours over its span) is traced to its sources and the step is split in their proportion:
  more base load shed inside the span (the same fuel put into fewer, whole weeks); gas or crude burnt more in the
  span, which came from stock we still hold at the span's end, from burning less before at this grid (in weeks 1..5 or
  later), or from more of it arriving by the span's end (more into all grids together, or at another grid's expense).

uv run python diffrep.py --task=small
"""

import json

import fire
import numpy as np

import common as C
from report import weights


def split(c, z_ours):
    """{class: share} of one cluster's extra fab energy by source (shares sum to 1), and tags."""
    dE = c["plan"]["E"] - c["ours"]["E"]
    parts = {}
    parts["shed more in the span"] = max(0.0, c["plan"]["shed"] - c["ours"]["shed"])
    for k in ("lng", "crude"):
        if k not in c["fuel"]:
            continue
        f = c["fuel"][k]
        o, p = f["ours"], f["plan"]
        dG = p["G"] - o["G"]
        if dG <= 0:
            continue
        saved = max(0.0, o["G_before"] - p["G_before"])
        saved_open = min(saved, max(0.0, o["G_open"] - p["G_open"]))
        arrived = max(0.0, p["arr_to_b"] - o["arr_to_b"])
        held = max(0.0, o["stock_end"] - p["stock_end"])
        tot = saved + arrived + held
        if tot <= 0:
            parts[f"{k}: unexplained"] = dG
            continue
        sc = dG / tot
        parts[f"{k}: still held by us at the span's end"] = held * sc
        parts[f"{k}: burnt less before, weeks 1-5"] = saved_open * sc
        parts[f"{k}: burnt less before, later weeks"] = (saved - saved_open) * sc
        allg = c["arr_all"].get(k, 0.0)
        new = arrived if allg >= arrived else max(0.0, allg)
        other = arrived - new
        d = f.get("sent_from")
        gi = c["gi"]
        seen = d is not None and (d <= 2 or min(z_ours[gi][max(d - 5, 0) : d]) == 0)
        parts[f"{k}: more arrived, more into all grids, sent when we were already short"] = new * sc if seen else 0.0
        parts[f"{k}: more arrived, more into all grids, sent before any shortage of ours"] = 0.0 if seen else new * sc
        parts[f"{k}: more arrived, at another grid's expense"] = other * sc
    tot = sum(parts.values())
    if tot <= 0:
        return {"no extra energy source found": 1.0}, dE
    return {k: v / tot for k, v in parts.items() if v > 0}, dE


def main(task="small", only=None, top=12):
    refs = C.references(task)
    res = {}
    for n in sorted(refs):
        p = C.HERE / "diff" / f"{task}_{n}.json"
        if p.exists():
            res[n] = json.loads(p.read_text())
    if only is not None:
        only = [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
        res = {n: r for n, r in res.items() if n in only}
    ns = sorted(res)
    w = weights(task, sorted(refs))  # pooled units of the whole set
    src = {s: sum(res[n].get("src") == s for n in ns) for s in ("new", "old")}
    print(f"{task}: {len(ns)} episodes (plans from this run's MILP: {src['new']}, the earlier incumbent: {src['old']})")
    for sub, label in ((ns, "these episodes"),):
        J = {k: {n: res[n][k] * 100 for n in sub} for k in ("J_played", "J_cal", "J_plan", "J_final")}
        sc = {k: C.pooled(refs, v)[0] for k, v in J.items()}
        print(f"  on {label}: played {sc['J_played']:.4f}; our calendar optimised {sc['J_cal']:.4f}; the realizable plan {sc['J_plan']:.4f}; after putting back what does not pay {sc['J_final']:.4f}")
    tot = sum(w[n] * (res[n]["J_cal"] - res[n]["J_final"]) for n in ns)
    raw = sum(res[n].get("weeks_differ_raw", 0) for n in ns)
    phys = sum(res[n].get("weeks_differ_physical", 0) for n in ns)
    fin = sum(res[n].get("weeks_differ_final", 0) for n in ns)
    ncl = sum(len(res[n].get("clusters", [])) for n in ns)
    print(f"  the difference to explain {tot:.4f} pooled; grid-weeks that differ: {raw} as solved, {phys} where the plan sheds or powers, {fin} that pay, in {ncl} clusters")
    left = sum(res[n].get("step_left", 0) for n in ns)
    steps = sum(w[n] * c.get("step", 0.0) for n in ns for c in res[n].get("clusters", []))
    print(f"  steps sum to {steps:.4f}; clusters whose step could not be solved at the end of an episode: {left}")
    cls, band, grid, kind = {}, {}, {}, {}
    rows = []
    for n in ns:
        zo = res[n].get("z_ours")
        for c in res[n].get("clusters", []):
            val = w[n] * c.get("step", 0.0)
            shares, dE = split(c, zo)
            n_on, n_off = len(c["on"]), len(c["off"])
            kd = "it runs where we shed" if n_off == 0 else ("it sheds where we run" if n_on == 0 else "both (weeks moved)")
            kind.setdefault(kd, [0.0, 0])
            kind[kd][0] += val
            kind[kd][1] += 1
            b = "weeks 1-4" if c["a"] <= 4 else "weeks 5-13" if c["a"] <= 13 else "weeks 14-26" if c["a"] <= 26 else "weeks 27-39" if c["a"] <= 39 else "weeks 40-52"
            band[b] = band.get(b, 0.0) + val
            grid[c["grid"]] = grid.get(c["grid"], 0.0) + val
            if dE <= 1e-6 * max(c["emax_on"], 1.0):
                key = "no more fab energy in the span (weeks moved, or fewer weeks run)"
                cls[key] = cls.get(key, 0.0) + val
            else:
                for k, s in shares.items():
                    cls[k] = cls.get(k, 0.0) + val * s
            part = c["partial_on"] / c["plan"]["E_on"] if c["plan"]["E_on"] > 0 else 0.0
            rows.append((val, n, c["grid"], c["a"], c["b"], n_on, n_off, dE, part, shares))
    print("\n  by kind of cluster (pooled units; number):")
    for k, (v, m) in sorted(kind.items(), key=lambda kv: -kv[1][0]):
        print(f"    {k:28s} {v:8.4f}  ({m})")
    print("  by the week the cluster begins:")
    for k in ("weeks 1-4", "weeks 5-13", "weeks 14-26", "weeks 27-39", "weeks 40-52"):
        print(f"    {k:12s} {band.get(k, 0.0):8.4f}")
    print("  by grid:")
    for k, v in sorted(grid.items(), key=lambda kv: -kv[1]):
        print(f"    {k:8s} {v:8.4f}")
    print("  by the source of the extra fab energy (pooled units; share):")
    for k, v in sorted(cls.items(), key=lambda kv: -kv[1]):
        print(f"    {k:78s} {v:8.4f}  {v / tot:5.0%}")
    part_val = sum(r[0] * r[8] for r in rows)
    print(f"  of all of it, in weeks the plan runs its fabs below 90 % of their draw: {part_val:.4f} ({part_val / tot:.0%})")
    print(f"\n  the {top} largest clusters:")
    rows.sort(key=lambda r: -r[0])
    for val, n, g, a, b, n_on, n_off, dE, part, shares in rows[:top]:
        main_src = sorted(shares.items(), key=lambda kv: -kv[1])[:2]
        print(f"    episode {n:2d} {g} weeks {a}-{b}: {val:.4f} pooled; it runs {n_on} weeks we shed, sheds {n_off} we run; extra fab energy {dE:.0f}; " + "; ".join(f"{k} {s:.0%}" for k, s in main_src))
    (C.HERE / f"diffrep_{task}.json").write_text(json.dumps({"total": tot, "classes": cls, "band": band, "grid": grid}, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
