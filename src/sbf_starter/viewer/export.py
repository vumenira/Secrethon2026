"""An episode record (``dashboard.record_episode``, ``load_record``) -> the compact JSON the viewer page draws.

Week t of the page (1..T) shows the network the agent saw when it decided (the observation of week t: capacities,
edge states, strait open fractions, warnings), the shipments it made that week (requested and executed), and what
followed (the observation of week t + 1: stock, backlog, goods in transit, cargo queued at the straits, served demand)
with the week's costs. Floats keep 4 significant digits; unobserved values are null.
"""

import math
from collections.abc import Mapping

import numpy as np
from shockbench_flow.information.flat import FlatLayout
from shockbench_flow_agent.convert import layout_tables
from shockbench_flow_gym import dashboard


STATE_CODE = {"open": "0", "reduced": "1", "closed": "2", "unobserved": "3"}


def _num(x) -> float | int | None:
    x = float(x)
    if not math.isfinite(x):
        return None
    if x == int(x) and abs(x) < 1e15:
        return int(x)
    return float(f"{x:.4g}")


def _usd(a) -> list | int:
    """Money in whole dollars (an array of any shape as nested lists)."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return int(round(float(a)))
    return [_usd(row) for row in a]


def _nums(a) -> list:
    """An array of any shape as nested lists of ``_num``."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 0:
        return _num(a)
    return [_nums(row) for row in a] if a.ndim > 1 else [_num(x) for x in a]


def _seen(obs: Mapping, key: str, rows: slice) -> np.ndarray:
    v = np.asarray(obs[key], dtype=np.float64)[rows]
    return np.where(np.asarray(obs[f"{key}.observed"])[rows] == 1, v, np.nan)


def _warning_names(static: Mapping, units: list) -> list[str]:
    regions, nodes, dyads = static["regions"], static["nodes"]["id"], static["dyads"]
    names = []
    for kind, i in units:
        if kind == "region":
            names.append(f"region {regions[i]}")
        elif kind == "dyad":
            names.append(f"{regions[dyads['a'][i]]} vs {regions[dyads['b'][i]]}")
        elif kind == "chokepoint":
            names.append(f"strait {nodes[i]}")
        else:
            names.append(f"{kind} {i}")
    return names


def _queues(obs: Mapping, layout: Mapping, straits: list[int], T: int) -> list[list]:
    """(T, straits) units queued at each strait after week t: Tiny's per-lot list or Small's and Full's dense table."""
    out = np.zeros((T, len(straits)))
    col = {node: j for j, node in enumerate(straits)}
    if "lot_keys" in layout:  # (lot keys, T) per week; row i's strait is lot_keys[i][0]
        qty = np.nan_to_num(_seen(obs, "queue_lots.qty", slice(1, T + 1)))  # (T, lot keys, T)
        for i, key in enumerate(layout["lot_keys"]):
            if key[0] in col:
                out[:, col[key[0]]] += qty[:, i, :].sum(axis=-1)
    elif "queue_lots.chokepoint" in obs:
        node = np.asarray(obs["queue_lots.chokepoint"])[1 : T + 1]
        qty = np.nan_to_num(_seen(obs, "queue_lots.qty", slice(1, T + 1)))
        for t in range(T):
            for n, q in zip(node[t], qty[t]):
                if q > 0 and int(n) in col:
                    out[t, col[int(n)]] += q
    return _nums(out)


def _flows(static: Mapping, shipped: np.ndarray) -> list[list]:
    """Per week, ``[edge, commodity, units]`` executed on each edge (a lane slot's units count on every lane edge)."""
    slots, lanes = static["action_slots"], static["lanes"]["edges"]
    weeks = []
    for row in shipped:
        per: dict[tuple[int, int], float] = {}
        for s, q in enumerate(row):
            if q <= 0:
                continue
            lane = slots["lane"][s]
            for e in lanes[lane] if lane is not None else [slots["edge"][s]]:
                key = (int(e), int(slots["k"][s]))
                per[key] = per.get(key, 0.0) + float(q)
        weeks.append([[e, k, _num(q)] for (e, k), q in sorted(per.items())])
    return weeks


def _transit(obs: Mapping, T: int) -> list[list]:
    """Per week, ``[edge, commodity, lane or -1, arrival week, units]`` of the goods on their way after week t."""
    qty, seen = np.asarray(obs["pipeline.qty"]), np.asarray(obs["pipeline.qty.observed"])
    edge, k, arr = (np.asarray(obs[f"pipeline.{f}"]) for f in ("edge", "k", "arrival_week"))
    lane = np.where(np.asarray(obs["pipeline.lane.observed"]) == 1, np.asarray(obs["pipeline.lane"]), -1)
    weeks = []
    for t in range(1, T + 1):
        rows = np.flatnonzero((seen[t] == 1) & (qty[t] > 0))
        weeks.append([[int(edge[t, i]), int(k[t, i]), int(lane[t, i]), int(arr[t, i]), _num(qty[t, i])] for i in rows])
    return weeks


def _alt(a) -> str | None:
    if not a:
        return None
    return ", ".join(f"{k} {v}" for k, v in a.items()) if isinstance(a, Mapping) else str(a)


# columns left to right in the order goods travel; straits go to the gaps between them (``layout_positions``)
KIND_COLUMN = {"source": 0, "terminal": 1, "grid": 2, "material": 2, "fab": 3, "osat": 4, "sink": 5}
SWEEPS = 8  # barycenter passes of the crossing reduction


def layout_positions(static: Mapping) -> list[tuple[float, float]]:
    """Node positions (column, row): kinds on even columns, each strait in the gap next to its routes' median column,
    and the order within each column by barycenter sweeps, the usual heuristic that cuts edge crossings.
    """
    types = list(static["nodes"]["type"])
    n = len(types)
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for a, b in zip(static["edges"]["tail"], static["edges"]["head"]):
        if a != b:
            nbrs[a].add(b)
            nbrs[b].add(a)
    present = sorted({KIND_COLUMN.get(t, 0) for t in types if t != "chokepoint"})
    even = {c: 2 * i for i, c in enumerate(present)}  # absent kinds leave no gap
    col = [even[KIND_COLUMN.get(t, 0)] if t != "chokepoint" else None for t in types]
    last = max((c for c in col if c is not None), default=0)
    for i in range(n):  # a strait sits in the odd gap column nearest the median column of its non-strait neighbours
        if col[i] is None:
            near = sorted(col[j] for j in nbrs[i] if col[j] is not None)
            m = near[len(near) // 2] if near else 1
            col[i] = min(max(1, 2 * round((m - 1) / 2) + 1), max(last - 1, 1))
    columns: dict[int, list[int]] = {}
    for i in range(n):
        columns.setdefault(col[i], []).append(i)
    row = [0.0] * n
    tallest = max(len(v) for v in columns.values())

    def place(c):  # centred; a short column spreads out (up to twice the step) to use the height of the tall ones
        nodes = columns[c]
        step = min(2.0, (tallest - 1) / max(len(nodes) - 1, 1)) if len(nodes) > 1 else 1.0
        for r, i in enumerate(nodes):
            row[i] = (r - (len(nodes) - 1) / 2) * max(step, 1.0)

    for c in columns:
        place(c)
    order = sorted(columns)
    for sweep in range(SWEEPS):
        for c in order if sweep % 2 == 0 else order[::-1]:

            def bary(i):
                ys = [row[j] for j in nbrs[i] if col[j] != c]
                return sum(ys) / len(ys) if ys else row[i]

            columns[c] = sorted(columns[c], key=lambda i: (bary(i), row[i]))
            place(c)
    return [(float(col[i]), float(row[i])) for i in range(n)]


def normal_plan(static: Mapping) -> list[float]:
    """Each action slot's weekly quantity in the normal plan: the week-0 shipments of the initial pipeline."""
    eid, lid, kid = static["edges"]["id"], static["lanes"]["id"], static["commodities"]["id"]
    slots, lanes = static["action_slots"], static["lanes"]["edges"]
    plan: dict[tuple, float] = {}
    for p in static["instance"]["initial_state"]["pipeline"]:
        if p["dispatch_week"] == 0:
            key = (p["edge"], p["k"], p["lane"])
            plan[key] = plan.get(key, 0.0) + float(p["qty"])
    out = []
    for e, k, lane in zip(slots["edge"], slots["k"], slots["lane"]):
        first = lanes[lane][0] if lane is not None else e
        out.append(plan.get((eid[first], kid[k], lid[lane] if lane is not None else None), 0.0))
    return out


def network(static: Mapping) -> dict:
    """The tables the map is drawn from: nodes with their positions, edges, lanes, slots, straits, goods."""
    layout = layout_tables(FlatLayout.from_static(static))
    types = list(static["nodes"]["type"])
    pos = layout_positions(static)
    edges = static["edges"]
    plan = normal_plan(static)
    overrides = static.get("override_slots") or {}
    return {
        "nodes": [
            {
                "id": name,
                "type": types[i],
                "region": static["regions"][static["nodes"]["region"][i]],
                "x": round(pos[i][0], 3),
                "y": round(pos[i][1], 3),
            }
            for i, name in enumerate(static["nodes"]["id"])
        ],
        "edges": [
            {
                "id": edges["id"][e],
                "tail": edges["tail"][e],
                "head": edges["head"][e],
                "mode": edges["mode"][e],
                "u0": _num(edges["u0"][e]) if edges["u0"][e] is not None else None,
                "tau0": edges["tau0"][e],
                "c0": _num(edges["c0"][e]),
                "K": list(edges["K"][e]),
                "alt": _alt(edges["alt_of"][e]),
            }
            for e in range(len(edges["id"]))
        ],
        "goods": list(static["commodities"]["id"]),
        "units": {k: v for k, v in static["units"].items()},
        "lanes": [
            {"id": i, "edges": list(es), "straits": list(cs)}
            for i, es, cs in zip(static["lanes"]["id"], static["lanes"]["edges"], static["lanes"]["chokepoints"])
        ],
        "slots": [
            {"edge": e, "k": k, "lane": lane, "plan": _num(q)}
            for e, k, lane, q in zip(
                static["action_slots"]["edge"], static["action_slots"]["k"], static["action_slots"]["lane"], plan
            )
        ],
        "release_pairs": [list(p) for p in layout["release_pairs"]],
        "override_slots": [list(p) for p in zip(*(overrides.get(c, []) for c in ("chokepoint", "k", "out_edge")))],
        "straits": list(layout["chokepoints"]),
        "stock_slots": [list(s) for s in layout["stock_slots"]],
        "demands": [list(d) for d in layout["demands"]],
        "sinks": [list(p) for p in zip(static["sinks"]["node"], static["sinks"]["k"])],
        "grids": list(layout["grids"]),
        "warning_names": _warning_names(static, layout["warning_units"]),
        "cost_components": list(layout["cost_components"]),
    }


def episode(record, label: str = "", eval_row: Mapping | None = None) -> dict:
    """One episode's JSON (module docstring); ``eval_row``: the evaluation's row of this episode, if any."""
    rec = dashboard.load_record(record) if not isinstance(record, Mapping) else record
    meta, static, obs = rec["meta"], rec["static"], rec["obs"]
    T = int(meta["T"])
    layout = layout_tables(FlatLayout.from_static(static))
    week_obs = [{k: v[t] for k, v in obs.items()} for t in range(T)]
    states = ["".join(STATE_CODE[s] for s in dashboard.edge_states_flat(static, w)) for w in week_obs]
    shipped = dashboard.shipped_units(rec)
    cost = -np.asarray(rec["reward_cents"], dtype=np.float64) / 100  # the week's cost; week T's net of the salvage
    out = {
        "episode": meta.get("episode"),
        "label": label,
        "agent": meta.get("agent"),
        "instance": meta["instance"],
        "regime": meta["regime"],
        "T": T,
        "J": _usd(meta["J_cents"] / 100),
        "salvage": _usd(meta.get("salvage_cents", 0) / 100),
        "weeks": {
            "u": _nums(_seen(obs, "graph_now.u", slice(0, T))),
            "state": states,
            "open": _nums(_seen(obs, "graph_now.open", slice(0, T))),
            "warning": _nums(_seen(obs, "warning.score", slice(0, T))),
            "requested": _nums(np.nan_to_num(_seen(obs, "last_week.clip.requested", slice(1, T + 1)))),
            "executed": _nums(shipped),
            "flows": _flows(static, shipped),
            "transit": _transit(obs, T),
            "queue": _queues(obs, layout, list(layout["chokepoints"]), T),
            "stock": _nums(_seen(obs, "stock.qty", slice(1, T + 1))),
            "backlog": _nums(_seen(obs, "backlog.qty", slice(1, T + 1))),
            "demand": _nums(_seen(obs, "last_week.sinks.demand", slice(1, T + 1))),
            "served": _nums(_seen(obs, "last_week.sinks.served", slice(1, T + 1))),
            "lost": _nums(_seen(obs, "last_week.sinks.lost", slice(1, T + 1))),
            "shed": _nums(_seen(obs, "last_week.shed.qty", slice(1, T + 1))),
            "costs": _usd(rec["costs"]),
            "cost": _usd(cost),
        },
        "events": [
            {k: e.get(k) for k in ("week", "signal", "subject", "before", "after", "note")} for e in rec["events"]
        ],
    }
    action = rec.get("action", {})
    release = np.asarray(action.get("release_mode", np.zeros((T, 0))))
    if release.size and release.any():  # only when the agent took over a strait's release somewhere
        out["release"] = release[:T].astype(int).tolist()
        out["override"] = _nums(np.asarray(action.get("override_qty", np.zeros((T, 0))))[:T])
    if rec.get("naive"):
        out["naive"] = {
            "J": _usd(meta.get("naive_J_cents", -np.sum(rec["naive"]["reward_cents"])) / 100),
            "costs": _usd(rec["naive"]["costs"]),
            "cost": _usd(-np.asarray(rec["naive"]["reward_cents"], dtype=np.float64) / 100),
        }
    if eval_row:
        out["eval"] = dict(eval_row)
    return out
