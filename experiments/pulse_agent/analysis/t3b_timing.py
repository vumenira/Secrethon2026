"""Seconds per episode of the cached t3b_ variants, and whether a variant's costs equal another's to the cent.

An episode over 5 times its variant's median is dropped from the mean and the maximum (the laptop slept in one run).

uv run python <this file> <run dir> [--same=a,b;c,d] [--bases=t3b_base,t3b_base2]
"""

import json
import sys
from pathlib import Path

import numpy as np


def rows(run, name):
    return json.loads((run / f"{name}.json").read_text())["rows"]


def main():
    run = Path(sys.argv[1])
    opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--"))
    names = sorted((p.stem for p in run.glob("t3b_*.json")), key=lambda n: (run / f"{n}.json").stat().st_mtime)
    if (run / "mpc_det_h20.json").is_file():
        names = ["mpc_det_h20", *names]
    bases = [b for b in opts.get("bases", "t3b_base").split(",") if (run / f"{b}.json").is_file()]
    secs, kept = {}, {}
    for n in names:
        s = np.array([r["seconds"] for r in rows(run, n)])
        secs[n] = s
        kept[n] = s[s <= 5 * np.median(s)]
    print(f"{'variant':12s} {'params':44s} {'written':>8s} {'mean s':>7s} {'median':>7s} {'max s':>7s} drop fallb err  "
          + "  ".join(f"mean,median/{b}" for b in bases))
    for n in names:
        path = run / f"{n}.json"
        data = json.loads(path.read_text())
        rs = data["rows"]
        s, k = secs[n], kept[n]
        fallback = sum(r["fallback_weeks"] for r in rs)
        cpu = sum(r.get("cpu_weeks", 0) for r in rs)
        errors = sum(r.get("first_error") is not None for r in rs)
        invalid = sum(r.get("invalid_entries", 0) for r in rs)
        ratios = "  ".join(f"{k.mean() / kept[b].mean():9.2f} {np.median(s) / np.median(secs[b]):5.2f}" for b in bases)
        params = json.dumps((data.get("spec") or {}).get("params", {}))
        import datetime

        when = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        print(f"{n:12s} {params:44s} {when:>8s} {k.mean():7.2f} {np.median(s):7.2f} {k.max():7.2f} {len(s) - len(k):4d}"
              f" {fallback:5d} {errors:3d}  {ratios}  cpu_weeks {cpu} invalid {invalid} n {len(rs)}")
    for pair in opts.get("same", "").split(";"):
        if not pair:
            continue
        a, b = pair.split(",")
        ja = [r["J_policy_cents"] for r in rows(run, a)]
        jb = [r["J_policy_cents"] for r in rows(run, b)][: len(ja)]
        diff = [i for i, (x, y) in enumerate(zip(ja, jb)) if x != y]
        print(f"{a} against {b}: {len(ja)} episodes compared, {len(diff)} differ to the cent {diff[:10]}")


if __name__ == "__main__":
    main()
