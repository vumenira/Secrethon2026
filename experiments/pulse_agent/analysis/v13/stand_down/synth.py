"""Write s13_ends for Full 777 as both less mild_start's own effect, episode by episode (an estimate, not a play)."""
import json
from pathlib import Path

d = Path("outputs/recovery_tune/full-e777-n32")
base = {r["episode"]: r for r in json.loads((d / "v12_all.json").read_text())["rows"]}
mild = {r["episode"]: r for r in json.loads((d / "s13_mild.json").read_text())["rows"]}
both = json.loads((d / "s13_both.json").read_text())
rows = []
for r in both["rows"]:
    n = r["episode"]
    r = dict(r)
    r["J_policy_cents"] = r["J_policy_cents"] - (mild[n]["J_policy_cents"] - base[n]["J_policy_cents"])
    rows.append(r)
(d / "s13_ends_est.json").write_text(json.dumps({"spec": "both less mild_start's own effect (not a play)", "rows": rows}, indent=1))
for s in ("full-e12345-n16", "full-e2026-n16"):
    p = Path("outputs/recovery_tune") / s
    (p / "s13_ends_est.json").write_text((p / "s13_ends.json").read_text())
print("written")
