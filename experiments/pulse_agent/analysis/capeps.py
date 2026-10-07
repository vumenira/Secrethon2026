"""The strikes of chosen episodes and the edges they hit, from capcells.pkl."""

import pickle
import sys
from collections import defaultdict
from pathlib import Path


def main(*eps):
    data = pickle.loads((Path(__file__).parent / "capcells.pkl").read_bytes())
    per = {p["episode"]: p for p in data["per_episode"]}
    rows = defaultdict(list)
    for r in data["rows"]:
        if "strike" in r["causes"]:
            rows[r["episode"]].append(r)
    for n in map(int, eps):
        print(f"episode {n}: T {per[n]['T']}")
        for onset, dur, sev, kind, target in per[n]["strikes"]:
            print(f"   strike onset {onset:7.2f} lasts {dur:6.2f} wk (to {onset + dur:6.2f}) severity {sev:.3f} target {kind} {target}")
        spells = defaultdict(list)
        for r in rows[n]:
            spells[(r["edge"], r["first"])].append(r)
        seen = set()
        for (e, first), rs in sorted(spells.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            sig = (first, tuple((r["week"], round(r["share"], 3), r["cls"]) for r in rs if "strike" in r["causes"]))
            weeks = [r["week"] for r in rs]
            shares = sorted({round(r["share"], 3) for r in rs})
            classes = sorted({r["cls"] for r in rs})
            key = (first, min(weeks), max(weeks), tuple(shares), tuple(classes))
            if key in seen:
                continue
            seen.add(key)
            n_same = sum(
                1 for (e2, f2), r2 in spells.items()
                if (f2, min(x["week"] for x in r2), max(x["week"] for x in r2), tuple(sorted({round(x["share"], 3) for x in r2})), tuple(sorted({x["cls"] for x in r2}))) == key
            )
            print(f"   {n_same:3d} edges: impaired from week {first}, strike cells weeks {min(weeks)}..{max(weeks)}, shares {shares}, classed {classes}, u0 {rs[0]['u0']:.3g}")


if __name__ == "__main__":
    main(*sys.argv[1:])
