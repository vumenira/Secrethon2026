import json
from pathlib import Path
here = Path(__file__).parent
for task in ("full", "small"):
    base = json.loads((here / f"milp_{task}.json").read_text())
    warm = json.loads((here / f"milp_{task}_warm.json").read_text())
    for k, v in warm.items():
        if v.get("J_usd") is None:
            continue
        old = base.get(k, {})
        better = old.get("J_usd") is None or v["J_usd"] < old["J_usd"]
        bound = max([b for b in (v.get("bound"), old.get("bound")) if b is not None])
        new = dict(old)
        if better:
            new.update({a: v[a] for a in ("J_usd", "score", "message", "seconds", "status") if a in v})
        new["bound"] = bound
        new["episode"], new["stratum"] = int(k), v["stratum"]
        base[k] = new
    (here / f"milp_{task}_merged.json").write_text(json.dumps(base, indent=1))
    print(task, len(base), "episodes merged")
