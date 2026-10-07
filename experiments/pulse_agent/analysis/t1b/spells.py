"""Ground truth of closures and grid shortages on the tuning root: spells as the agent sees them, and their events.

Writes spells.json beside this file: one row per spell (a run of weeks an element is shown impaired).
"""

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent


def runs(mask):
    """(first, last) index pairs of the runs of True in a 1-D mask."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def main(task="small", entropy=12345, start=0, episodes=64, out="spells.json"):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.marks import read_events
    from shockbench_flow.omega import codes
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    cache = str(default_cache_dir())
    rows, meta = [], {}
    for n in range(start, start + episodes):
        inst, omega, marks, _fb = _world(task, entropy, n, REPLICATIONS, cache)
        events = read_events(inst, omega.arrays)
        T = inst.T
        if n == start:
            meta = {
                "T": T,
                "chokepoints": [inst.nodes[c].id for c in inst.chokepoints],
                "chokepoint_nodes": list(inst.chokepoints),
                "grids": [inst.nodes[g].id for g in inst.grids],
                "grid_nodes": list(inst.grids),
                "regions": list(inst.regions) if not hasattr(inst.regions[0], "id") else [r.id for r in inst.regions],
                "adjacency": {inst.nodes[c].id: list(inst.chokepoint_adjacency.get(c, ())) for c in inst.chokepoints},
                "burn_in": int(np.asarray(omega.arrays.get("meta_burn_in", -1))) if "meta_burn_in" in omega.arrays else None,
                "keys": sorted(k for k in omega.arrays.keys() if k.startswith("meta")),
            }
        o_now, wr = np.asarray(marks.o_now), np.asarray(marks.wr_class)
        G_now = np.asarray(marks.G_bar_now)
        G0 = np.array([inst.nodes[g].grid.deliverable for g in inst.grids])
        names = codes.EVENT_TYPES
        ids = [r if isinstance(r, str) else r.id for r in inst.regions]
        pair = {ids.index("CN"), ids.index("TW")}
        dyad_edges = [
            e
            for e, edge in enumerate(inst.edges)
            if not edge.coupling and edge.u0 is not None and {inst.nodes[edge.tail].region, inst.nodes[edge.head].region} == pair
        ]
        u_now = np.asarray(marks.u_now)
        dyad_share = np.max(u_now[:, dyad_edges] / np.array([inst.edges[e].u0 for e in dyad_edges]), axis=1)
        # closures
        for ci, c in enumerate(inst.chokepoints):
            mine = [q for q in events if names[q.type] in ("militarised_closure", "weather_closure") and q.target == c]
            adjacent = inst.chokepoint_adjacency.get(c, ())
            wars = [q for q in events if names[q.type] == "regional_conflict" and q.region in adjacent]
            for a, b in runs(o_now[:, ci] < 1.0):
                # weeks a+1..b+1 are shown impaired (instants a..b)
                act = [q for q in mine if q.onset <= b and q.end > a]
                rows.append(
                    {
                        "kind": "closure",
                        "episode": n,
                        "element": ci,
                        "name": inst.nodes[c].id,
                        "first": a + 1,
                        "last": b + 1,
                        "T": T,
                        "values": [round(float(x), 6) for x in o_now[a : b + 1, ci]],
                        "wr": [int(x) for x in wr[a : b + 1, ci]],
                        "after": None if b + 1 >= T else round(float(o_now[b + 1, ci]), 6),
                        "dyad": [round(float(x), 4) for x in dyad_share[a : b + 1]] if inst.nodes[c].id == "chk_taiwan" else None,
                        "events": [
                            {
                                "type": names[q.type],
                                "onset": round(q.onset, 3),
                                "end": round(q.end, 3),
                                "sev": round(q.severity, 4),
                                "derived": bool(names[q.type] == "militarised_closure" and q.severity >= 1.0),
                            }
                            for q in act
                        ],
                        "wars": [
                            {"onset": round(q.onset, 3), "end": round(q.end, 3), "region": q.region, "cp": q.counterpart}
                            for q in wars
                            if q.onset <= b and q.end > a
                        ],
                    }
                )
        for gi, g in enumerate(inst.grids):
            mine = [q for q in events if names[q.type] == "energy_shock" and q.target == g]
            for a, b in runs(G_now[:, gi] < G0[gi] * (1 - 1e-12)):
                act = [q for q in mine if q.onset <= b and q.end > a]
                rows.append(
                    {
                        "kind": "grid",
                        "episode": n,
                        "element": gi,
                        "name": inst.nodes[g].id,
                        "first": a + 1,
                        "last": b + 1,
                        "T": T,
                        "values": [round(float(x), 6) for x in G_now[a : b + 1, gi] / G0[gi]],
                        "after": None if b + 1 >= T else round(float(G_now[b + 1, gi] / G0[gi]), 6),
                        "events": [
                            {"type": names[q.type], "onset": round(q.onset, 3), "end": round(q.end, 3), "sev": round(q.severity, 4)}
                            for q in act
                        ],
                    }
                )
        # every energy shock and closure event of the episode window, for the counts
        for q in events:
            nm = names[q.type]
            if nm in ("energy_shock", "militarised_closure", "weather_closure", "regional_conflict") and q.end > 0 and q.onset < T:
                rows.append(
                    {
                        "kind": "event",
                        "episode": n,
                        "type": nm,
                        "target_kind": codes.TARGET_KINDS[q.target_kind],
                        "target": q.target,
                        "region": q.region,
                        "cp": q.counterpart,
                        "onset": round(q.onset, 3),
                        "end": round(q.end, 3),
                        "sev": round(q.severity, 4),
                    }
                )
        print(n, len(rows), flush=True)
    (HERE / out).write_text(json.dumps({"meta": meta, "rows": rows}))
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(start=int(a[0]) if a else 0, episodes=int(a[1]) if len(a) > 1 else 64, out=a[2] if len(a) > 2 else "spells.json")
