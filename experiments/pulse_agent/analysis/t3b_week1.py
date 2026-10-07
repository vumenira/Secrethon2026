"""Week 1 of the per-week CPU runs: simplex iterations and CPU by episode, each setting against the base."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
rows = []
for f in ("weekcpu_windows_out.json", "weekcpu_final_out.json"):
    rows += json.loads((HERE / f).read_text())["rows"]
names = ["base", "h24", "h28", "h32", "h40", "d110", "h32_d110"]
for name in names:
    rs = sorted((r for r in rows if r["name"] == name), key=lambda r: r["episode"])
    it = [r["iterations"][0] for r in rs]
    cpu = [r["cpu"][0] for r in rs]
    top = np.sort(np.concatenate([r["cpu"] for r in rs]))[-3:]
    print(f"{name:9s} week-1 iterations {it} mean {np.mean(it):6.0f}; week-1 cpu {cpu} median {np.median(cpu):.3f};"
          f" 3 slowest weeks {top.round(3).tolist()}")
