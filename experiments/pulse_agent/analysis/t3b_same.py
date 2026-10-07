"""Costs to the cent: the 3 smoke episodes of the final agent.py against the 64-episode cache and the baseline."""

import json
from pathlib import Path


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")


def costs(run, name):
    return [r["J_policy_cents"] for r in json.loads((OUT / run / f"{name}.json").read_text())["rows"]]


n3, n64 = "small-e12345-n3", "small-e12345-n64"
for a, b in (
    ((n3, "t3b_base"), (n64, "mpc_det_h20")),
    ((n3, "t3b_base"), (n3, "mpc_det_h20")),
    ((n3, "t3b_h32_d110"), (n64, "t3b_h32_d110")),
    ((n64, "t3b_base"), (n64, "mpc_det_h20")),
    ((n64, "t3b_base2"), (n64, "mpc_det_h20")),
    ((n64, "t3b_base3"), (n64, "mpc_det_h20")),
    ((n64, "t3b_base4"), (n64, "mpc_det_h20")),
):
    x, y = costs(*a), costs(*b)
    n = min(len(x), len(y))
    print(f"{a[1]} ({a[0]}) against {b[1]} ({b[0]}): {n} episodes, {sum(p != q for p, q in zip(x[:n], y[:n]))} differ")
