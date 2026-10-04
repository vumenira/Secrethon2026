"""Rule-based agent: ship what each destination needs, on the best open route, instead of the maximum everywhere.

Every week, for every node that holds goods (sources, terminals, fabs, OSATs), it splits the stock between the places
that stock can go:

1. Need. Each destination (node, good) has a target: enough stock for ``cover`` weeks of its use, at most ``fill`` of
   its storage, plus what it will use while a shipment travels. Its position is its stock plus everything already on
   the way to it (in transit or queued at a strait). The need is the target minus the position. A destination that
   uses none of the good (a market without demand for it) needs nothing.
2. Route. A destination's routes are tried best first, ranked by freight + tariff + a charge per week of lead time
   + a charge for a partly closed strait. A route carries at most the smallest free capacity along its edges (cargo
   already queued at a strait goes first), scaled by its straits' open fraction. It is skipped when a strait is
   nearly closed (``min_open``) or when an announced sanction will block one of its edges before the cargo arrives.
   A partly open strait with no alternative is still used: some cargo gets through.
3. Warnings. A destination whose usual route crosses a strait or region with a high early-warning score gets a
   larger cover (``warn_boost``), so it stocks up before a disruption.
4. Push. What is left after every need is met is pushed towards the destinations with free storage (``push``), so
   that sources, whose unshipped supply is lost, never sit on stock.

Only numpy and the standard library: the server has nothing else. A ``params.json`` beside this file overrides
``PARAMS`` (the same pattern as agents/heuristic). Notes and tuning: experiments/3rule_based.md.
"""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
PARAMS = {
    "cover": 6.0,  # weeks of use a destination aims to hold (capped by storage)
    "fill": 0.95,  # share of a destination's storage it may fill (the rest is a margin against disposal)
    "min_open": 0.05,  # a route through a strait less open than this is not used
    "open_penalty": 0.5,  # route ranking: a strait's closed share costs this share of the good's value
    "open_power": 1.0,  # a usable route's capacity is scaled by (open fraction) ** open_power
    "lead_cost": 0.02,  # cost of one week of lead time, as a share of the good's value (route choice only)
    "push": 1.0,  # share of the stock left after the needs that is pushed to free storage downstream
    "pile": 2.0,  # weeks of exit capacity a plant may pile up of its product before its input is throttled
    "warn_level": 0.5,  # an early-warning score above this counts as a warning
    "warn_boost": 1.0,  # extra cover, as a share of ``cover``, at a warning score of 1
}
if (HERE / "params.json").is_file():
    PARAMS |= json.loads((HERE / "params.json").read_text())


class Agent:
    def __init__(self, config=None):
        self.p = dict(PARAMS)
        static, layout = config["static"], config["layout"]
        inst = static["instance"]
        nodes, edges, lanes, slots = static["nodes"], static["edges"], static["lanes"], static["action_slots"]
        node_ids, com_ids = nodes["id"], static["commodities"]["id"]
        com_index = {k: i for i, k in enumerate(com_ids)}
        edge_index = {e: i for i, e in enumerate(edges["id"])}
        lane_index = {ln: i for i, ln in enumerate(lanes["id"])}
        self.n_edges = len(edges["id"])
        self.u0 = np.array([u if u is not None else 0.0 for u in edges["u0"]], dtype=float)
        self.value = np.array([c["v"] for c in inst["commodities"]], dtype=float)  # customs value per unit
        chk_pos = {node: i for i, node in enumerate(layout["chokepoints"])}  # strait node -> graph_now.open index

        # where a lane ends; a shipment off any lane ends at its edge's head
        self.lane_end = [edges["head"][lane_edges[-1]] for lane_edges in lanes["edges"]]
        self.edge_head = list(edges["head"])

        # stock slots: (node, good) -> index of stock.qty, with storage
        self.stock_slot = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        storage = np.full(len(layout["stock_slots"]), np.inf)
        for i, (n, k) in enumerate(layout["stock_slots"]):
            entry = _node_entry(inst, node_ids[n]).get("stock", {}).get(com_ids[k], {})
            if entry.get("storage") is not None:
                storage[i] = float(entry["storage"])
        self.storage = storage

        # each action slot: origin, destination, good, the route's edges and straits, its lead time
        self.slots = []
        for s, (e, k, ln) in enumerate(zip(slots["edge"], slots["k"], slots["lane"])):
            route = [e] if ln is None else list(lanes["edges"][ln])
            straits = [] if ln is None else [chk_pos[c] for c in lanes["chokepoints"][ln] if c in chk_pos]
            self.slots.append(
                {
                    "s": s,
                    "edge": e,
                    "k": k,
                    "origin": edges["tail"][e],
                    "dest": self.edge_head[e] if ln is None else self.lane_end[ln],
                    "route": route,
                    "straits": straits,
                    "tau": int(sum(edges["tau0"][x] for x in route)),
                }
            )
        # slots grouped by the stock they draw on, (origin, good)
        self.by_origin: dict[tuple[int, int], list[dict]] = {}
        for sl in self.slots:
            self.by_origin.setdefault((sl["origin"], sl["k"]), []).append(sl)

        # how fast each destination uses a good: markets from the forecast, grids from their fuel shares, fabs from
        # their capacity, everything else from the normal plan's inflow
        self.demand_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["demands"])}
        self.grid_use = {}  # (grid node, fuel) -> (grid ordinal, share of generation)
        for gi, g in enumerate(layout["grids"]):
            shares = _node_entry(inst, node_ids[g])["grid"]["shares"]
            for fuel, share in shares.items():
                if fuel in com_index:
                    self.grid_use[(int(g), com_index[fuel])] = (gi, float(share))
        self.fab_use = {}  # (fab node, wafer) -> fab ordinal
        for fi, f in enumerate(layout["fabs"]):
            self.fab_use[(int(f), com_index[_node_entry(inst, node_ids[f])["fab"]["input"]])] = fi
        # plants: a fab makes one product from wafers; an OSAT packages each raw chip into a finished one
        self.fabs = []  # (fab ordinal, node, input, product)
        for fi, f in enumerate(layout["fabs"]):
            fab = _node_entry(inst, node_ids[f])["fab"]
            self.fabs.append((fi, int(f), com_index[fab["input"]], com_index[fab["product"]]))
        self.osats = []  # (OSAT ordinal, node, [(raw, packaged), ...] dearest first)
        for oi, o in enumerate(layout["osats"]):
            packages = _node_entry(inst, node_ids[o])["osat"]["packages"]
            pairs = [(com_index[r], com_index[pk]) for r, pk in packages.items()]
            self.osats.append((oi, int(o), sorted(pairs, key=lambda rp: -self.value[rp[1]])))
        self.nominal_in: dict[tuple[int, int], float] = {}
        for sh in inst["initial_state"]["pipeline"]:
            if sh["dispatch_week"] != 0:
                continue
            ln = sh.get("lane")
            dest = self.lane_end[lane_index[ln]] if ln is not None else self.edge_head[edge_index[sh["edge"]]]
            key = (dest, com_index[sh["k"]])
            self.nominal_in[key] = self.nominal_in.get(key, 0.0) + float(sh["qty"])
        # tau-0 edges (a terminal into its grid) leave nothing in the pipeline: their grid's use stands in

        # warnings: each destination's straits and regions, as indices of warning.score
        warn_index = {(kind, int(i)): w for w, (kind, i) in enumerate(layout["warning_units"])}
        region_of = list(nodes["region"])  # region indices, the same as the warning's region units
        self.dest_warn: dict[tuple[int, int], list[int]] = {}
        for sl in self.slots:
            units = [warn_index.get(("chokepoint", layout["chokepoints"][c])) for c in sl["straits"]]
            units += [warn_index.get(("region", region_of[sl["origin"]]))]
            ws = self.dest_warn.setdefault((sl["dest"], sl["k"]), [])
            ws += [w for w in units if w is not None and w not in ws]

        # Small and Full keep the strait queues as one table; Tiny as per-lot lists
        self.lot_keys = layout.get("lot_keys")

    def act(self, observation):
        p, obs = self.p, observation
        week = int(obs["week"][0])
        stock = np.asarray(obs["stock.qty"], dtype=float)
        mask = np.asarray(obs["action_mask"])
        flows = np.zeros(len(self.slots))

        position = self._position(obs, stock)
        rate = self._rates(obs)
        warning = self._warning(obs)

        # this week's edge capacity, cost and lead time
        u = np.where(obs["graph_now.u.observed"] == 1, obs["graph_now.u"], self.u0)
        u = np.maximum(np.nan_to_num(u, nan=0.0), 0.0)
        cost_e, tariff = obs["graph_now.c"], obs["graph_now.tariff"]
        tau_e = obs["graph_now.tau"]
        open_now = np.where(obs["graph_now.open.observed"] == 1, obs["graph_now.open"], 1.0)
        banned = self._pending_bans(obs)

        # each usable slot: its capacity factor, lead time and cost this week
        usable = {}
        for sl in self.slots:
            s, k = sl["s"], sl["k"]
            if mask[s] != 1:
                continue
            opened = min((float(open_now[c]) for c in sl["straits"]), default=1.0)
            if opened < p["min_open"]:
                continue
            tau = int(sum(tau_e[x] for x in sl["route"]))
            if any(banned.get((x, k), 10**9) <= week + tau for x in sl["route"]):
                continue  # an announced sanction blocks the route before the cargo gets through
            cost = sum(float(cost_e[x]) + float(tariff[x, k]) * self.value[k] for x in sl["route"])
            rank = cost + self.value[k] * (p["lead_cost"] * tau + p["open_penalty"] * (1.0 - opened))
            usable[s] = (opened ** p["open_power"], tau, rank)

        left = u.copy()  # edge capacity not yet used this week, on every edge of a route
        if self.lot_keys is not None:  # cargo queued at a strait leaves first: its next edge has that much less room
            queued = np.asarray(obs["queue_lots.qty"]).sum(axis=1)
            for (_c, _k, _ln, nxt), q in zip(self.lot_keys, queued):
                left[int(nxt)] -= q
            left = np.maximum(left, 0.0)
        self._backpressure(obs, stock, rate, usable, left)

        # each destination's target position: cover (raised by warnings) up to its storage, plus use in transit
        lead: dict[tuple[int, int], int] = {}
        for sl in self.slots:
            if sl["s"] in usable:
                key = (sl["dest"], sl["k"])
                lead[key] = min(lead.get(key, 10**9), usable[sl["s"]][1])
        need, room = {}, {}
        for key in {(sl["dest"], sl["k"]) for sl in self.slots}:
            r = rate.get(key, 0.0)
            i = self.stock_slot.get(key)
            if r <= 0.0 or i is None:
                need[key] = room[key] = 0.0
                continue
            boost = 1.0 + p["warn_boost"] * max((warning[w] for w in self.dest_warn.get(key, [])), default=0.0)
            cap = p["fill"] * self.storage[i]
            in_transit = r * lead.get(key, 0)
            need[key] = max(0.0, min(cap, r * p["cover"] * boost) + in_transit - position.get(key, 0.0))
            room[key] = max(0.0, cap + in_transit - position.get(key, 0.0))

        for (origin, k), out in self.by_origin.items():
            i = self.stock_slot.get((origin, k))
            avail = float(stock[i]) if i is not None else 0.0
            routes = sorted((sl for sl in out if sl["s"] in usable), key=lambda sl: usable[sl["s"]][2])
            if avail <= 0.0 or not routes:
                continue
            dests = sorted({sl["dest"] for sl in routes}, key=lambda d: self._cover_weeks(d, k, position, rate))
            for d in dests:  # the needs, the most urgent destination first
                avail = self._send(routes, d, k, min(need[(d, k)], avail), avail, flows, left, usable, position, need)
            if avail > 0.0 and p["push"] > 0.0:  # the rest, to free storage downstream, in proportion to it
                total = sum(room[(d, k)] for d in dests)
                budget = p["push"] * avail
                for d in dests:
                    if total > 0.0:
                        share = budget * room[(d, k)] / total
                        avail = self._send(routes, d, k, share, avail, flows, left, usable, position, room)
        return {"flows": flows * mask}

    # ----- helpers --------------------------------------------------------------------------------------------------

    def _send(self, routes, dest, k, want, avail, flows, left, usable, position, ledger):
        """Ship up to ``want`` of good k to ``dest`` over its routes, best first; return the stock left."""
        for sl in routes:
            if want <= 1e-9 or avail <= 1e-9:
                break
            if sl["dest"] != dest:
                continue
            factor = usable[sl["s"]][0]
            q = min(want, avail, factor * min(left[x] for x in sl["route"]))
            if q <= 0.0:
                continue
            flows[sl["s"]] += q
            for x in sl["route"]:
                left[x] -= q
            avail -= q
            want -= q
            position[(dest, k)] = position.get((dest, k), 0.0) + q
            ledger[(dest, k)] = max(0.0, ledger[(dest, k)] - q)
        return avail

    def _backpressure(self, obs, stock, rate, usable, left):
        """Feed a plant only what its product can leave by: an OSAT's raw chips, then a fab's wafers.

        A plant's exit is the capacity of its usable routes to places that use its product. Its input rate is at most
        that exit, scaled down to 0 as its product piles up from ``pile`` to 2 ``pile`` weeks of exit. An OSAT's
        throughput goes to its dearest finished chip first, since packaging splits it pro rata over the raw stock.
        """
        pile = self.p["pile"]

        def exit_capacity(node, product):
            total = 0.0
            for sl in self.by_origin.get((node, product), []):
                if sl["s"] in usable and rate.get((sl["dest"], product), 0.0) > 0.0:
                    total += usable[sl["s"]][0] * min(left[x] for x in sl["route"])
            return total

        def throttled(base, exit_cap, node, product):
            i = self.stock_slot.get((node, product))
            piled = float(stock[i]) if i is not None else 0.0
            scale = 1.0 if exit_cap <= 0.0 else float(np.clip(2.0 - piled / (pile * exit_cap), 0.0, 1.0))
            return min(base, exit_cap) * scale

        thr = obs["graph_now.osat.thr_eff"]
        for oi, o, pairs in self.osats:
            budget = float(thr[oi])
            for raw, pk in pairs:
                want = throttled(rate.get((o, raw), 0.0), exit_capacity(o, pk), o, pk)
                rate[(o, raw)] = min(want, budget)
                budget -= rate[(o, raw)]
        for _fi, f, wafer, product in self.fabs:
            rate[(f, wafer)] = throttled(rate.get((f, wafer), 0.0), exit_capacity(f, product), f, product)

    def _position(self, obs, stock):
        """Stock on hand plus everything on its way, per (node, good)."""
        pos = {key: float(stock[i]) for key, i in self.stock_slot.items()}
        live = obs["pipeline.qty.observed"] == 1
        lane_ok = obs["pipeline.lane.observed"] == 1
        for e, k, ln, has_lane, q in zip(
            obs["pipeline.edge"][live],
            obs["pipeline.k"][live],
            obs["pipeline.lane"][live],
            lane_ok[live],
            obs["pipeline.qty"][live],
        ):
            dest = self.lane_end[int(ln)] if has_lane else self.edge_head[int(e)]
            pos[(dest, int(k))] = pos.get((dest, int(k)), 0.0) + float(q)
        if self.lot_keys is not None:  # Small and Full: one row per (strait, good, lane, next edge)
            queued = np.asarray(obs["queue_lots.qty"]).sum(axis=1)
            for (_c, k, ln, _next), q in zip(self.lot_keys, queued):
                if q > 0.0 and ln is not None:
                    key = (self.lane_end[int(ln)], int(k))
                    pos[key] = pos.get(key, 0.0) + float(q)
        else:  # Tiny: one entry per lot
            live = obs["queue_lots.qty.observed"] == 1
            for k, ln, q in zip(obs["queue_lots.k"][live], obs["queue_lots.lane"][live], obs["queue_lots.qty"][live]):
                key = (self.lane_end[int(ln)], int(k))
                pos[key] = pos.get(key, 0.0) + float(q)
        return pos

    def _rates(self, obs):
        """Weekly use of each good at each node."""
        rate = dict(self.nominal_in)
        forecast = obs["demand_forecast.qty"]
        for key, row in self.demand_row.items():
            rate[key] = float(forecast[row, 0])
        g_bar = obs["graph_now.grid.G_bar"]
        for key, (gi, share) in self.grid_use.items():
            rate[key] = share * float(g_bar[gi])
        cap = obs["graph_now.fab.cap_eff"]
        for key, fi in self.fab_use.items():
            rate[key] = float(cap[fi])
        # a terminal feeds its grid the same week: it needs what the grid burns
        for (origin, k), out in self.by_origin.items():
            if (origin, k) in self.nominal_in or (origin, k) in self.demand_row:
                continue
            fed = sum(rate.get((sl["dest"], k), 0.0) for sl in out if sl["tau"] == 0)
            if fed > 0.0:
                rate[(origin, k)] = fed
        return rate

    def _warning(self, obs):
        """Each warning unit's score above ``warn_level``, rescaled to 0..1."""
        score = np.where(obs["warning.score.observed"] == 1, obs["warning.score"], 0.0)
        level = self.p["warn_level"]
        return np.clip((score - level) / max(1e-9, 1.0 - level), 0.0, 1.0)

    def _pending_bans(self, obs):
        """(edge, good) -> the earliest week an announced sanction takes effect."""
        live = obs["pending_prohibitions.edge.observed"] == 1
        bans: dict[tuple[int, int], int] = {}
        for e, k, w in zip(
            obs["pending_prohibitions.edge"][live],
            obs["pending_prohibitions.k"][live],
            obs["pending_prohibitions.effective_week"][live],
        ):
            key = (int(e), int(k))
            bans[key] = min(bans.get(key, 10**9), int(w))
        return bans

    @staticmethod
    def _cover_weeks(dest, k, position, rate):
        r = rate.get((dest, k), 0.0)
        return position.get((dest, k), 0.0) / r if r > 0.0 else np.inf


def _node_entry(inst, node_id):
    for n in inst["nodes"]:
        if n["id"] == node_id:
            return n
    return {}
