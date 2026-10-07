"""One episode of q1.py, printed (a debugging run)."""

import sys

import q1

n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
out = q1.one("small", n)
D = out["D"]
import common as C

naive = C.references("small")[n]["J_naive_cents"] / 100
for k, v in out["J"].items():
    print(f"{k:10s} J {v / 1e9 if v is not None else float('nan'):12.3f} bn  score {(naive - v) / D if v is not None else float('nan'):.4f}  dist {out['dist'].get(k)}  {out['status'].get(k)}")
print(out["events"], out["carried"], out["seconds"])
