"""Per-episode costs of cached variants, pair by pair: how many of the episodes are equal to the cent."""
import json
import sys
from pathlib import Path

OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")


def rows(name):
    return {r["episode"]: r for r in json.loads((OUT / f"{name}.json").read_text())["rows"]}


for pair in sys.argv[1:]:
    a, b = pair.split("=")
    ra, rb = rows(a), rows(b)
    assert sorted(ra) == sorted(rb), "not the same episodes"
    assert all(ra[n]["omega_hash"] == rb[n]["omega_hash"] for n in ra), "not the same scenarios"
    off = [(n, ra[n]["J_policy_cents"] - rb[n]["J_policy_cents"]) for n in sorted(ra) if ra[n]["J_policy_cents"] != rb[n]["J_policy_cents"]]
    total = sum(r["J_policy_cents"] for r in ra.values()), sum(r["J_policy_cents"] for r in rb.values())
    extra = {k: sum((r.get(k) or 0) for r in ra.values()) for k in ("fallback_weeks", "cpu_weeks", "invalid_entries", "forecast_failed")}
    errors = [r["first_error"] for r in ra.values() if r.get("first_error")]
    print(f"{a} vs {b}: {len(ra) - len(off)} of {len(ra)} episodes equal to the cent; sums {total[0]} and {total[1]}")
    print(f"   {a}: {extra}, errors {errors[:2]}, slowest episode {max(r['seconds'] for r in ra.values()):.1f} s")
    if off:
        print("   differing (episode, cents):", off[:10])
