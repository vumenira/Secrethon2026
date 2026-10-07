"""Print the spells of chosen episodes with the cached costs of chosen variants.

uv run python show.py grid|closure 23,54,60 variant [variant ...]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RUN = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")
data = json.loads((HERE / "spells.json").read_text())
kind, eps, names = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3:]
cost = {n: [r["J_policy_cents"] / 1e11 for r in json.loads((RUN / f"{n}.json").read_text())["rows"]] for n in names}
for ep in eps:
    base = cost[names[0]][ep]
    print(f"episode {ep}: " + "  ".join(f"{n} {base - cost[n][ep]:+.1f}" for n in names[1:]) + f"  (bn USD saved over {names[0]})")
    for r in data["rows"]:
        if r["episode"] == ep and r["kind"] == kind:
            path = [r["values"][0]]
            runs = [1]
            for v in r["values"][1:]:
                if v == path[-1]:
                    runs[-1] += 1
                else:
                    path.append(v)
                    runs.append(1)
            ev = [(e["type"][:3], e["onset"], e["end"], e["sev"]) for e in r["events"]]
            print(f"   {r['name']} weeks {r['first']}-{r['last']}: shares x weeks {[(round(p, 3), n) for p, n in zip(path, runs)]} events {ev}")
