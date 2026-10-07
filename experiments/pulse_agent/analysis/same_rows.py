"""Whether two cached variants have the same cost on every episode."""

import json
from pathlib import Path

import fire


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")


def main(a, b, task="full", entropy=12345, episodes=16):
    run = OUT / f"{task}-e{entropy}-n{episodes}"
    ra = [r["J_policy_cents"] for r in json.loads((run / f"{a}.json").read_text())["rows"]]
    rb = [r["J_policy_cents"] for r in json.loads((run / f"{b}.json").read_text())["rows"]]
    diff = [i for i, (x, y) in enumerate(zip(ra, rb, strict=True)) if x != y]
    print(f"{task}: {a} and {b}: {len(ra)} episodes, {len(diff)} differ {diff[:10]}")


if __name__ == "__main__":
    fire.Fire(main)
