"""Per-episode cost difference of two cached plays (bn USD saved by a over b), sorted; plus the pooled score difference."""
import json, sys
from pathlib import Path
OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")
def main(a, b, task="full", entropy=12345, episodes=16, top=12):
    d = OUT / f"{task}-e{entropy}-n{episodes}"
    ra = {r["episode"]: r["J_policy_cents"] for r in json.loads((d / f"{a}.json").read_text())["rows"]}
    rb = {r["episode"]: r["J_policy_cents"] for r in json.loads((d / f"{b}.json").read_text())["rows"]}
    refs = {r["episode"]: r for r in json.loads((d / "references.json").read_text())}
    rows = sorted(((rb[n] - ra[n]) / 1e11, n) for n in ra)
    better = sum(1 for v, _ in rows if v > 1e-6); worse = sum(1 for v, _ in rows if v < -1e-6)
    print(f"{a} against {b} on {task} {entropy}: saved {sum(v for v, _ in rows):.1f} bn in all; better {better} worse {worse} equal {len(rows) - better - worse}")
    print("  worst:", [(n, round(v, 1), refs[n]["stratum"]) for v, n in rows[:top] if v < -1e-6])
    print("  best: ", [(n, round(v, 1), refs[n]["stratum"]) for v, n in rows[::-1][:top] if v > 1e-6])
if __name__ == "__main__":
    import fire; fire.Fire(main)
