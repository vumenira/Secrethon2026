"""Per-episode difference of two cached variants as a share of each episode's gap (positive: a is better than b).

uv run python <this> --a=ft_s_v4_r123 --b=ft_s_v4 [--task=small]
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
OUT = ROOT / "outputs" / "recovery_tune"


def main(a, b, task="small", entropy=12345, top=10):
    run = OUT / f"{task}-e{entropy}-n{64 if task == 'small' else 16}"
    refs = json.loads((run / "references.json").read_text())
    ra = json.loads((run / f"{a}.json").read_text())["rows"]
    rb = json.loads((run / f"{b}.json").read_text())["rows"]
    rows = []
    for ref, x, y in zip(refs, ra, rb):
        gap = ref["J_naive_cents"] - ref["J_oracle_cents"]
        rows.append((ref["episode"], ref["stratum"], ref["excluded"], gap / 1e11, (y["J_policy_cents"] - x["J_policy_cents"]) / 1e11,
                     (ref["J_naive_cents"] - y["J_policy_cents"]) / gap if gap else float("nan")))
    money = np.array([r[4] for r in rows])
    gaps = np.array([r[3] for r in rows])
    print(f"{task}: {a} against {b}: total {money.sum():+.1f} bn of a total gap of {gaps.sum():.0f} bn ({money.sum() / gaps.sum():+.4f});"
          f" better {int((money > 1e-6).sum())} worse {int((money < -1e-6).sum())} same {int((np.abs(money) <= 1e-6).sum())}")
    order = np.argsort(money)
    print("  worst:")
    for i in order[:top]:
        e, s, ex, gap, d, base = rows[i]
        print(f"    episode {e:3d} level {s} gap {gap:8.1f} bn  {d:+8.2f} bn ({d / gap:+.4f} of its gap)  b scores {base:.3f} {'EXCLUDED' if ex else ''}")
    print("  best:")
    for i in order[::-1][:top]:
        e, s, ex, gap, d, base = rows[i]
        print(f"    episode {e:3d} level {s} gap {gap:8.1f} bn  {d:+8.2f} bn ({d / gap:+.4f} of its gap)  b scores {base:.3f} {'EXCLUDED' if ex else ''}")


if __name__ == "__main__":
    fire.Fire(main)
