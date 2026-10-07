"""Pooled scores of the warm-started runs: played, our calendar with the best plan, realizable incumbent and bound."""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent


def main(task="small", tags="_warm", entropy=12345):
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    total = {"small": 64, "full": 16}[task]
    refs = {r["episode"]: r for r in json.loads((ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{total}" / "references.json").read_text())}
    res = {}
    for tag in tags.split(",") if isinstance(tags, str) else tags:
        path = HERE / f"milp_{task}{tag}.json"
        if path.exists():
            for k, v in json.loads(path.read_text()).items():
                res[int(k)] = {**res.get(int(k), {}), **{a: b for a, b in v.items() if b is not None}}
    base = HERE / f"milp_{task}.json"
    first = {int(k): v for k, v in json.loads(base.read_text()).items()} if base.exists() else {}
    rows = []
    for n, v in sorted(res.items()):
        r = refs[n]
        naive, oracle = r["J_naive_cents"] / 100, r["J_oracle_cents"] / 100
        J_inc = min([x for x in (v.get("J_usd"), first.get(n, {}).get("J_usd"), v.get("J_calendar"), v["J_played"]) if x is not None])
        bounds = [x for x in (v.get("bound"), first.get(n, {}).get("bound")) if x is not None]
        J_bnd = max(max(bounds), oracle) if bounds else None
        rows.append({"n": n, "s": r["stratum"], "naive": naive, "oracle": oracle, "played": v["J_played"], "calendar": v.get("J_calendar"), "inc": J_inc, "bound": J_bnd})

    def pooled(key, subset):
        by = {}
        for s in (1, 2, 3, 4):
            rs = [x for x in subset if x["s"] == s]
            if rs:
                by[s] = (np.mean([x["naive"] - x[key] for x in rs]), np.mean([x["naive"] - x["oracle"] for x in rs]))
        num = sum(STRATUM_WEIGHTS[s - 1] * by[s][0] for s in by)
        den = sum(STRATUM_WEIGHTS[s - 1] * by[s][1] for s in by)
        return num / den, {s: round(by[s][0] / by[s][1], 3) for s in by}

    print(f"{task}: {len(rows)} episodes {[x['n'] for x in rows]}; levels {[sum(x['s'] == s for x in rows) for s in (1, 2, 3, 4)]}")
    cal = [x for x in rows if x["calendar"] is not None]
    for key, label in (("played", "our agent as played"), ("calendar", "our on/off calendar, the best plan under it")):
        p, by = pooled(key, cal)
        print(f"   {label:46s} {p:.4f}  {by}   ({len(cal)} episodes)")
    full = [x for x in rows if x["bound"] is not None]
    if full:
        for key, label in (("played", "our agent as played"), ("calendar", "our calendar, best plan"), ("inc", "realizable clairvoyant, incumbent"), ("bound", "realizable clairvoyant, proven bound")):
            if all(x[key] is not None for x in full):
                p, by = pooled(key, full)
                print(f"   {label:46s} {p:.4f}  {by}   ({len(full)} episodes with a MILP)")
        for x in full:
            g = x["naive"] - x["oracle"]
            print(f"      episode {x['n']:2d} level {x['s']}: played {(x['naive'] - x['played']) / g:.3f}  calendar {(x['naive'] - x['calendar']) / g if x['calendar'] else float('nan'):.3f}  incumbent {(x['naive'] - x['inc']) / g:.3f}  bound {(x['naive'] - x['bound']) / g:.3f}")


if __name__ == "__main__":
    fire.Fire(main)
