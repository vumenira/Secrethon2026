"""Whether two cached plays of tune.py cost the same to the cent in every episode.

uv run python outputs/recovery_tune/tools/same.py --a=<name> --b=<name> --task=full --entropy=12345 --episodes=16
"""
import json
from pathlib import Path

import fire

OUT = Path(__file__).resolve().parents[1]


def same(a, b, task="full", entropy=12345, episodes=16):
    out = OUT / f"{task}-e{entropy}-n{episodes}"
    ra, rb = (json.loads((out / f"{n}.json").read_text())["rows"] for n in (a, b))
    ca = {r["episode"]: r["J_policy_cents"] for r in ra}
    cb = {r["episode"]: r["J_policy_cents"] for r in rb}
    diff = [(n, ca[n] - cb[n]) for n in sorted(ca) if ca[n] != cb.get(n)]
    errs = [(r["episode"], r["first_error"]) for r in ra + rb if r.get("first_error")]
    verdict = "equal to the cent in every episode" if not diff else f"{len(diff)} episodes differ (bn USD): {[(n, round(d / 1e11, 3)) for n, d in diff[:16]]}"
    print(f"{a} vs {b} on {task} root {entropy} ({len(ca)} episodes): {verdict}")
    print(f"   fallback weeks {sum(r['fallback_weeks'] for r in ra)} / {sum(r['fallback_weeks'] for r in rb)}; errors {errs[:3]}")


if __name__ == "__main__":
    fire.Fire(same)
