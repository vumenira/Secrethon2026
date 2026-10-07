"""The two per-week CPU runs together: ratios to the base on the same episodes, and costs against the harness cache."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")
rows = []
for f in ("weekcpu_windows_out.json", "weekcpu_final_out.json"):
    rows += json.loads((HERE / f).read_text())["rows"]
cache = {"base": "mpc_det_h20", "h24": "t3b_h24", "h28": "t3b_h28", "h32": "t3b_h32", "h40": "t3b_h40",
         "d110": "t3b_d110", "h32_d110": "t3b_h32_d110"}
names = list(cache)
base = {r["episode"]: np.array(r["cpu"]) for r in rows if r["name"] == "base"}
base_it = {r["episode"]: np.array(r["iterations"]) for r in rows if r["name"] == "base"}
print(f"{'setting':9s} {'n':>2s} {'cpu/episode':>12s} {'week 1':>12s} {'weeks 2-12':>12s} {'slowest wk':>12s} {'slowest 2+':>12s}"
      f" {'iters 2-12':>11s}  same-as-cache  lp_fallback fallback errors")
for name in names:
    rs = [r for r in rows if r["name"] == name]
    c = np.array([r["cpu"] for r in rs])
    b = np.array([base[r["episode"]] for r in rs])
    it = np.array([r["iterations"] for r in rs])
    bit = np.array([base_it[r["episode"]] for r in rs])
    cached = [x["J_policy_cents"] for x in json.loads((OUT / f"{cache[name]}.json").read_text())["rows"]]
    same = sum(r["J_policy_cents"] == cached[r["episode"]] for r in rs)

    def cell(x, y):
        return f"{x:6.3f} x{x / y:4.2f}"

    print(
        f"{name:9s} {len(rs):2d} {cell(c.sum(axis=1).mean(), b.sum(axis=1).mean())} {cell(c[:, 0].mean(), b[:, 0].mean())}"
        f" {cell(c[:, 1:12].mean(), b[:, 1:12].mean())} {cell(c.max(), b.max())} {cell(c[:, 1:].max(), b[:, 1:].max())}"
        f" {it[:, 1:12].mean():6.0f} x{it[:, 1:12].mean() / bit[:, 1:12].mean():4.2f}  {same}/{len(rs)}"
        f"  {sum(r['lp_fallback'] for r in rs)} {sum(r['fallback_weeks'] for r in rs)} {sum(r['errors'] for r in rs)}"
    )
