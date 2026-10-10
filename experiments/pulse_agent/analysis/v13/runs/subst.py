"""ends_needed alone on a set: v12's cached play with the replayed episodes put in (those the pair changed).

An episode the pair (ends_needed + runs_reach) left equal to the cent is one where neither rule changed a flow, so
ends_needed alone leaves it equal too; only the changed ones were replayed (rec.py). Writes <set>/r13_ends2s.json.
"""
import json, sys
from pathlib import Path
OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")

def main(task, entropy, episodes, *names):
    d = OUT / f"{task}-e{entropy}-n{episodes}"
    v12 = json.loads((d / "v12_all.json").read_text())["rows"]
    pair = {r["episode"]: r["J_policy_cents"] for r in json.loads((d / "r13_er.json").read_text())["rows"]}
    played = {}
    for name in names:
        for r in json.loads((OUT / "r13" / "rec" / d.name / name / "rows.json").read_text())["rows"]:
            played[r["episode"]] = r
    changed = [r["episode"] for r in v12 if pair[r["episode"]] != r["J_policy_cents"]]
    missing = [n for n in changed if n not in played]
    rows = [dict(r, J_policy_cents=played[r["episode"]]["J_policy_cents"]) if r["episode"] in played else dict(r) for r in v12]
    (d / "r13_ends2s.json").write_text(json.dumps({"spec": {"note": "ends_needed alone: v12_all with the episodes the pair changed replayed (r13/rec)", "replayed": sorted(played)}, "rows": rows}, indent=1))
    print(f"{d.name}: the pair changed {sorted(changed)}; replayed {sorted(played)}; not replayed {missing}")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), *sys.argv[4:])
