"""Summary of a t3b_weekcpu.py output: CPU per week by setting, against the first setting."""

import json
import sys

import numpy as np


data = json.loads(open(sys.argv[1]).read())
names = list(data["settings"])
rows = data["rows"]
stat = {}
for name in names:
    rs = [r for r in rows if r["name"] == name]
    c = np.array([r["cpu"] for r in rs])
    w = np.array([r["wall"] for r in rs])
    it = np.array([r["iterations"] for r in rs])
    stat[name] = {
        "n": len(rs),
        "cpu/episode": c.sum(axis=1).mean(),
        "week 1 mean": c[:, 0].mean(),
        "week 1 max": c[:, 0].max(),
        "max wk 2+": c[:, 1:].max(),
        "weeks 2-12": c[:, 1:12].mean(),
        "weeks 2-20": c[:, 1:20].mean(),
        "wall wk 2-12": w[:, 1:12].mean(),
        "iters 2-12": it[:, 1:12].mean(),
        "lp_fallback": sum(r["lp_fallback"] for r in rs),
        "fallback": sum(r["fallback_weeks"] for r in rs),
        "errors": sum(r["errors"] for r in rs),
        "trail": sorted({t for r in rs for t in r["trail"]}),
    }
keys = ["cpu/episode", "week 1 mean", "week 1 max", "max wk 2+", "weeks 2-12", "weeks 2-20", "wall wk 2-12", "iters 2-12"]
print(f"{'setting':10s} n  " + "  ".join(f"{k:>14s}" for k in keys))
b = stat[names[0]]
for name in names:
    s = stat[name]
    print(f"{name:10s} {s['n']}  " + "  ".join(f"{s[k]:8.3f} x{s[k] / b[k]:4.2f}" for k in keys)
          + f"  lp_fallback {s['lp_fallback']} fallback {s['fallback']} errors {s['errors']}")
for name in names:
    print(name, stat[name]["trail"][:4])
