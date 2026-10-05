"""BC agent v12, the behaviour-cloning baseline: two small networks that copy a re-planning clairvoyant teacher.

Every week:
1. Flows. ``SlotFeatures`` turns the observation into one row of numbers per action slot (a route + a good; 47 on
   Tiny, 43 + the number of goods in general). The flow network, an MLP 3 x 128 shared by every slot, reads each
   row and answers the share of the route's nominal capacity to send, clipped to [0, MAX_SHARE];
   flows = share x nominal capacity x action_mask.
2. Strait releases. ``ReleaseFeatures`` turns it into one row per release group (strait, good, route out; 31 numbers
   on Tiny, 27 + goods). The release network (3 x 128) answers the share of the route out's nominal capacity to
   release; ``with_floor`` raises it, while the strait is open, to at least RELEASE_FLOOR of what waits plus what
   arrives this week; ``release_action`` puts it on the first open override slot with ``release_mode`` 1.

The teacher (training only, ``experiments/bc/replan_lib.py``) re-plans every week from the episode's real state with
the true future, following the game's rules for factories and power. The networks were trained on 2,000 Tiny
episodes, with "answering less than the teacher" counting 4x (flows) and 2x (releases), so the agent leans towards
shipping more when it can't tell what's coming. Details and results: experiments/bc/README.md ("BC baseline v12").

The weights fit Tiny only: on another network the feature counts differ and ``Agent`` refuses them; the code itself
reads every shape from the config and runs on Small and Full (retrain there). When weights.npz holds ``enc_*`` and
``dec_*`` arrays (the shared network of run 11), the flow network is that one instead.

Server rules: numpy only, weights loaded at module level, no randomness (nothing to seed). The feature objects keep
a little state within an episode (first-week levels, the straits' last weeks), so build one Agent per episode.
"""

from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WEIGHTS_PATH = HERE / "weights.npz"
MAX_SHARE = 2.0  # the network's output is clipped to [0, MAX_SHARE] x nominal capacity
RELEASE_FLOOR = 0.5  # the least share of (waiting + arriving this week) a pair releases while its strait is open
FLOOR_OPEN = 0.5  # "open": the strait's open fraction at least this
MODES = ("sea", "air", "pipeline", "grid")


def _load_weights():
    if not WEIGHTS_PATH.exists():
        return None
    with np.load(WEIGHTS_PATH) as data:
        return {key: data[key] for key in data.files}


WEIGHTS = _load_weights()  # at import time: the CPU budget does not meter it


class SlotFeatures:
    """Turns one observation into a (number of slots, number of features) table.

    ``__init__`` looks up once, from the config, everything that never changes in an episode (each slot's edges,
    straits, stock rows, nominal flow); ``__call__`` reads this week's observation.
    """

    def __init__(self, config):
        static, layout = config["static"], config["layout"]
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        self.T = int(config["T"])
        self.n_goods = len(static["commodities"]["id"])
        self.u0_edge = np.asarray(edges["u0"], dtype=float)
        self.c0_edge = np.maximum(np.asarray(edges["c0"], dtype=float), 1e-9)
        self.tau0_edge = np.maximum(np.asarray(edges["tau0"], dtype=float), 1.0)

        stock_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        choke_row = {int(c): i for i, c in enumerate(layout["chokepoints"])}
        warn_row = {int(i): j for j, (kind, i) in enumerate(layout["warning_units"]) if kind == "chokepoint"}
        plan = self._nominal_plan(static)
        supply_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["supply_slots"])}
        # the health of a factory, grid or packaging plant: (observation key, its row) per node
        health = {int(n): ("graph_now.fab.cap_eff", i) for i, n in enumerate(layout["fabs"])}
        health |= {int(n): ("graph_now.grid.G_bar", i) for i, n in enumerate(layout["grids"])}
        health |= {int(n): ("graph_now.osat.thr_eff", i) for i, n in enumerate(layout["osats"])}
        region = static["nodes"]["region"]

        self.slots = []  # one dict per action slot, everything precomputed
        for s, (e, k, lane) in enumerate(zip(slots["edge"], slots["k"], slots["lane"], strict=True)):
            route = list(lanes["edges"][lane]) if lane is not None else [e]
            straits = list(lanes["chokepoints"][lane]) if lane is not None else []
            final_head = int(edges["head"][route[-1]])
            self.slots.append(
                {
                    "e": int(e),
                    "k": int(k),
                    "lane": lane,
                    "route": np.asarray(route, dtype=int),
                    "later": np.asarray(route[1:], dtype=int),
                    "tail_row": stock_row.get((int(edges["tail"][e]), int(k))),
                    "head_row": stock_row.get((final_head, int(k))),
                    "choke_rows": [choke_row[c] for c in straits if c in choke_row],
                    "warn_rows": [warn_row[c] for c in straits if c in warn_row],
                    "strait_nodes": set(int(c) for c in straits),
                    "mode": [float(edges["mode"][e] == m) for m in MODES],
                    "plan": plan.get(
                        (edges["id"][e], static["commodities"]["id"][k], self._lane_id(static, lane)), 0.0
                    ),
                    "tail": int(edges["tail"][e]),
                    "final_head": final_head,
                    "supply_row": supply_row.get((int(edges["tail"][e]), int(k))),
                    "tail_health": health.get(int(edges["tail"][e])),
                    "head_health": health.get(final_head),
                    "regions": {region[int(edges["tail"][e])], region[final_head]},
                }
            )
        # the other routes of the same good from the same source, and into the same destination
        for d in self.slots:
            d["same_tail"] = [
                i for i, o in enumerate(self.slots) if o is not d and o["k"] == d["k"] and o["tail"] == d["tail"]
            ]
            d["same_head"] = [
                i
                for i, o in enumerate(self.slots)
                if o is not d and o["k"] == d["k"] and o["final_head"] == d["final_head"]
            ]
        self.n_features = 43 + self.n_goods  # the length of one row of __call__
        # the shared network's averaging matrices: row i averages the routes with i's source / destination / good
        keys = [
            [d["tail"] for d in self.slots],
            [d["final_head"] for d in self.slots],
            [d["k"] for d in self.slots],
        ]
        self.share_matrices = np.stack([np.array([[float(a == b) for b in key] for a in key]) for key in keys])
        self.share_matrices /= self.share_matrices.sum(axis=2, keepdims=True)
        self.health0 = {}  # each health value's first-week level, set on the first call
        self.history = []  # (strait open fraction, warning score) arrays of earlier weeks, latest last
        self.u0 = np.maximum(np.array([self.u0_edge[d["e"]] for d in self.slots]), 1e-9)  # nominal capacity per slot
        self.demand0 = None  # the first week's mean demand forecast, set on the first call

    @staticmethod
    def _lane_id(static, lane):
        return None if lane is None else static["lanes"]["id"][lane]

    @staticmethod
    def _nominal_plan(static):
        """(edge id, good id, lane id) -> weekly flow of the normal plan (the week-0 shipments)."""
        plan = {}
        for p in static["instance"]["initial_state"]["pipeline"]:
            if p["dispatch_week"] == 0:
                key = (p["edge"], p["k"], p.get("lane"))
                plan[key] = plan.get(key, 0.0) + float(p["qty"])
        return plan

    def __call__(self, obs):
        week = int(obs["week"][0])
        u = obs["graph_now.u"]
        u_seen = obs["graph_now.u.observed"].astype(bool)
        u_ratio = np.where(u_seen, u / np.maximum(self.u0_edge, 1e-9), 1.0)  # 1 where the capacity is not shown
        c_ratio = obs["graph_now.c"] / self.c0_edge
        tau_ratio = obs["graph_now.tau"] / self.tau0_edge
        tariff = obs["graph_now.tariff"]
        open_frac = obs["graph_now.open"]
        warning = obs["warning.score"]
        stock = obs["stock.qty"]

        # shipments in transit: (good, lane) for lane shipments, (good, edge) for the others
        live = obs["pipeline.qty.observed"].astype(bool)
        on_lane = live & obs["pipeline.lane.observed"].astype(bool)
        p_k, p_e, p_lane, p_qty = obs["pipeline.k"], obs["pipeline.edge"], obs["pipeline.lane"], obs["pipeline.qty"]

        # announced sanctions not yet in force, closures acting now with a known end
        pend = obs["pending_prohibitions.edge.observed"].astype(bool)
        pend_e = obs["pending_prohibitions.edge"][pend]
        pend_k = obs["pending_prohibitions.k"][pend]
        pend_w = obs["pending_prohibitions.effective_week"][pend]
        clos = obs["closure_end.chokepoint.observed"].astype(bool)
        clos_c = obs["closure_end.chokepoint"][clos]
        end_seen = obs["closure_end.end_week.observed"][clos].astype(bool)
        clos_end = np.where(end_seen, obs["closure_end.end_week"][clos], -1)  # -1: end not announced

        # demand: scaled by the first week's forecast, so the numbers stay near 1
        forecast = float(np.mean(obs["demand_forecast.qty"]))
        if self.demand0 is None:
            self.demand0 = max(forecast, 1e-9)
        global_part = [
            week / self.T,
            (self.T - week + 1) / self.T,
            forecast / self.demand0,
            float(np.mean(obs["demand_forecast.qty"][:, :2])) / self.demand0,  # the next two weeks
            float(np.mean(obs["demand_forecast.qty"][:, 4:])) / self.demand0,  # weeks 4 to 7 ahead
            float(np.sum(obs["backlog.qty"])) / self.demand0,
            float(np.sum(obs["last_week.sinks.lost"])) / self.demand0,
            float(np.max(warning)) if warning.size else 0.0,
            float(np.min(open_frac)) if open_frac.size else 1.0,
        ]
        self.global_part = global_part  # ReleaseFeatures reuses this week's globals

        requested = obs["last_week.clip.requested"]
        executed = obs["last_week.clip.executed"]
        mask = obs["action_mask"]
        rows = []
        # how much each route can carry this week, in units: capacity x mask x its straits' openness
        route_open = np.array(
            [min((open_frac[i] for i in d["choke_rows"]), default=1.0) for d in self.slots], dtype=float
        )
        carry = np.array([u_ratio[d["e"]] for d in self.slots]) * self.u0 * mask * route_open
        # announcements: threats and sanction notices name an edge, tariff notices a region
        msg = obs["messages.msg_id.observed"].astype(bool)
        m_chan, m_kind = obs["messages.channel"][msg], obs["messages.target_kind"][msg]
        m_target = obs["messages.target"][msg]
        m_k = np.where(obs["messages.k.observed"][msg].astype(bool), obs["messages.k"][msg], -1)
        # the health of factories, grids and packaging plants, against their first-week level
        for d in self.slots:
            for h in (d["tail_health"], d["head_health"]):
                if h is not None and h not in self.health0:
                    self.health0[h] = max(float(obs[h[0]][h[1]]), 1e-9)
        past = self.history[-2:]
        while len(past) < 2:
            past = [(open_frac, warning), *past]  # before week 1: as now
        for s, d in enumerate(self.slots):
            e, k, u0 = d["e"], d["k"], self.u0[s]
            if d["lane"] is not None:
                transit = p_qty[on_lane & (p_k == k) & (p_lane == d["lane"])].sum()
            else:
                transit = p_qty[live & ~on_lane & (p_k == k) & (p_e == e)].sum()
            hit = np.isin(pend_e, d["route"]) & (pend_k == k)
            weeks_to_sanction = float(np.min(pend_w[hit]) - week) if hit.any() else 0.0
            closing = [end for c, end in zip(clos_c, clos_end, strict=True) if int(c) in d["strait_nodes"]]
            weeks_to_reopen = max([float(end - week) for end in closing if end >= 0], default=0.0)
            good = [0.0] * self.n_goods
            good[k] = 1.0
            rows.append(
                [
                    # the route this week
                    u_ratio[e],
                    c_ratio[e],
                    tau_ratio[e],
                    float(np.max(tariff[d["route"], k])),
                    float(mask[s]),
                    min((open_frac[i] for i in d["choke_rows"]), default=1.0),
                    max((warning[i] for i in d["warn_rows"]), default=0.0),
                    float(np.min(u_ratio[d["later"]])) if d["later"].size else 1.0,
                    float(d["lane"] is not None),
                    *d["mode"],
                    # stocks and flows, in weeks of nominal capacity (log1p keeps big stocks readable)
                    np.log1p(stock[d["tail_row"]] / u0) if d["tail_row"] is not None else 0.0,
                    np.log1p(stock[d["head_row"]] / u0) if d["head_row"] is not None else 0.0,
                    np.log1p(transit / u0),
                    requested[s] / u0,
                    executed[s] / u0,
                    d["plan"] / u0,
                    # announcements
                    float(hit.any()),
                    weeks_to_sanction / self.T,
                    float(len(closing) > 0),
                    weeks_to_reopen / self.T,
                    *good,
                    *global_part,
                    # the other routes
                    np.log1p(carry[d["same_tail"]].sum() / u0),
                    np.log1p(carry[d["same_head"]].sum() / u0),
                    # what lies ahead at both ends
                    np.log1p(obs["graph_now.supply.avail"][d["supply_row"]] / u0)
                    if d["supply_row"] is not None
                    else 0.0,
                    float(obs[d["tail_health"][0]][d["tail_health"][1]]) / self.health0[d["tail_health"]]
                    if d["tail_health"] is not None
                    else 1.0,
                    float(obs[d["head_health"][0]][d["head_health"][1]]) / self.health0[d["head_health"]]
                    if d["head_health"] is not None
                    else 1.0,
                    # announcements about this route
                    np.log1p(np.sum((m_kind == 1) & (m_chan == 4) & np.isin(m_target, d["route"]))),  # threats
                    np.log1p(np.sum((m_kind == 1) & (m_chan == 3) & np.isin(m_target, d["route"]))),  # sanctions
                    np.log1p(
                        np.sum(
                            (m_kind == 3)
                            & (m_chan <= 2)
                            & np.isin(m_target, list(d["regions"]))
                            & ((m_k == -1) | (m_k == k))
                        )
                    ),  # tariff notices on its source's or destination's region
                    # the last two weeks of its straits
                    min((past[-1][0][i] for i in d["choke_rows"]), default=1.0),
                    min((past[-2][0][i] for i in d["choke_rows"]), default=1.0),
                    max((past[-1][1][i] for i in d["warn_rows"]), default=0.0),
                ]
            )
        self.history.append((open_frac.copy(), warning.copy()))
        return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)


class ReleaseFeatures:
    """Turns one observation into a (number of release groups, number of features) table, and back into an action.

    A release group is one (strait, good, out edge): the oracle's plan releases one quantity per group (its lanes
    summed), and it goes on the group's first override slot open this week. Call ``SlotFeatures`` first each week:
    the globals come from it.
    """

    def __init__(self, config):
        static, layout = config["static"], config["layout"]
        edges, ov = static["edges"], static["override_slots"]
        self.T = int(config["T"])
        self.n_goods = len(static["commodities"]["id"])
        self.pool = static["commodities"]["pool"]  # "tb" (tanker/bulk) or "ct" (container) per good
        self.u0_edge = np.maximum(np.asarray(edges["u0"], dtype=float), 1e-9)
        self.c0_edge = np.maximum(np.asarray(edges["c0"], dtype=float), 1e-9)
        stock_row = {(int(n), int(k)): i for i, (n, k) in enumerate(layout["stock_slots"])}
        choke_row = {int(c): i for i, c in enumerate(layout["chokepoints"])}
        warn_row = {int(i): j for j, (kind, i) in enumerate(layout["warning_units"]) if kind == "chokepoint"}
        pair_row = {(int(c), int(k)): i for i, (c, k) in enumerate(layout["release_pairs"])}
        self.n_pairs = len(layout["release_pairs"])
        self.n_slots = len(ov["chokepoint"])
        # the queue: Small and Full give one dense table (lot_keys x arrival week), Tiny a list of lots
        self.lot_keys = layout.get("lot_keys")
        # the lanes through each strait: (lane, the edge that enters the strait, the edge that leaves it)
        self.entries = []
        for lane, (lane_edges, straits) in enumerate(
            zip(static["lanes"]["edges"], static["lanes"]["chokepoints"], strict=True)
        ):
            for i, e_in in enumerate(lane_edges[:-1]):
                if int(edges["head"][e_in]) in {int(c) for c in straits}:
                    self.entries.append((lane, int(edges["head"][e_in]), int(e_in), int(lane_edges[i + 1])))

        self.groups = {}  # (strait, good, out edge) -> its facts, in order of first override slot
        for o, (c, k, e) in enumerate(zip(ov["chokepoint"], ov["k"], ov["out_edge"], strict=True)):
            key = (int(c), int(k), int(e))
            if key not in self.groups:
                self.groups[key] = {
                    "c": key[0],
                    "k": key[1],
                    "e": key[2],
                    "slots": [],
                    "pair": pair_row.get((key[0], key[1])),
                    "choke_row": choke_row.get(key[0]),
                    "warn_row": warn_row.get(key[0]),
                    "head_row": stock_row.get((int(edges["head"][key[2]]), key[1])),
                }
                if self.lot_keys is not None:
                    keys = [tuple(int(v) for v in lk) for lk in self.lot_keys]  # (strait, good, lane, next edge)
                    g = self.groups[key]
                    g["rows_out"] = [
                        i for i, lk in enumerate(keys) if lk[0] == key[0] and lk[1] == key[1] and lk[3] == key[2]
                    ]
                    g["rows_all"] = [i for i, lk in enumerate(keys) if lk[0] == key[0] and lk[1] == key[1]]
            self.groups[key]["slots"].append(o)
        self.groups = list(self.groups.values())
        self.u0 = np.array([self.u0_edge[g["e"]] for g in self.groups])  # nominal capacity of each out edge
        self.n_features = 27 + self.n_goods  # the length of one row of __call__ (9 globals)
        self.available = np.zeros(len(self.groups))  # this week: waiting at the pair + arriving this week (units)

    def _queue(self, obs, g, week):
        """(waiting to leave by g's out edge, waiting at g's strait for g's good, weeks the oldest has waited)."""
        if self.lot_keys is not None:
            q = obs["queue_lots.qty"] * obs["queue_lots.qty.observed"]
            out, all_ = q[g["rows_out"]], q[g["rows_all"]]
            weeks = np.nonzero(out.sum(0) > 0)[0]  # column w - 1: the cargo that reached the strait in week w
            age = float(week - (weeks[0] + 1)) if weeks.size else 0.0
            return float(out.sum()), float(all_.sum()), age
        live = obs["queue_lots.qty.observed"].astype(bool)
        here = live & (obs["queue_lots.chokepoint"] == g["c"]) & (obs["queue_lots.k"] == g["k"])
        out = here & (obs["queue_lots.next_edge"] == g["e"])
        qty = obs["queue_lots.qty"]
        age = float(week - obs["queue_lots.arrival_week"][out].min()) if out.any() else 0.0
        return float(qty[out].sum()), float(qty[here].sum()), age

    def _arriving(self, obs, g, week):
        """(arriving at g's strait for g's good this week, next week, this week by lanes that leave by g's edge)."""
        live = obs["pipeline.qty.observed"].astype(bool) & obs["pipeline.lane.observed"].astype(bool)
        live &= obs["pipeline.k"] == g["k"]
        p_lane, p_edge, p_week, p_qty = (
            obs["pipeline.lane"],
            obs["pipeline.edge"],
            obs["pipeline.arrival_week"],
            obs["pipeline.qty"],
        )
        now = nxt = now_out = 0.0
        for lane, c, e_in, e_out in self.entries:
            if c != g["c"]:
                continue
            here = live & (p_lane == lane) & (p_edge == e_in)
            q_now = float(p_qty[here & (p_week == week)].sum())
            now += q_now
            nxt += float(p_qty[here & (p_week == week + 1)].sum())
            now_out += q_now if e_out == g["e"] else 0.0
        return now, nxt, now_out

    def __call__(self, obs, global_part):
        week = int(obs["week"][0])
        u_seen = obs["graph_now.u.observed"].astype(bool)
        u_ratio = np.where(u_seen, obs["graph_now.u"] / self.u0_edge, 1.0)
        c_ratio = obs["graph_now.c"] / self.c0_edge
        open_frac, warning, stock = obs["graph_now.open"], obs["warning.score"], obs["stock.qty"]
        live = obs["pipeline.qty.observed"].astype(bool)
        clos = obs["closure_end.chokepoint.observed"].astype(bool)
        clos_c = obs["closure_end.chokepoint"][clos]
        end_seen = obs["closure_end.end_week.observed"][clos].astype(bool)
        clos_end = np.where(end_seen, obs["closure_end.end_week"][clos], -1)  # -1: end not announced
        ov_mask = obs["override_mask"]

        rows = []
        for i, g in enumerate(self.groups):
            c, k, e, u0, r = g["c"], g["k"], g["e"], self.u0[i], g["choke_row"]
            waiting_out, waiting_all, age = self._queue(obs, g, week)
            arriving, arriving_next, arriving_out = self._arriving(obs, g, week)
            self.available[i] = waiting_all + arriving
            transit = obs["pipeline.qty"][live & (obs["pipeline.edge"] == e) & (obs["pipeline.k"] == k)].sum()
            kappa = obs[f"graph_now.kappa.{self.pool[k]}"][r] if r is not None else 0.0
            closing = [end for cc, end in zip(clos_c, clos_end, strict=True) if int(cc) == c]
            weeks_to_reopen = max([float(end - week) for end in closing if end >= 0], default=0.0)
            good = [0.0] * self.n_goods
            good[k] = 1.0
            rows.append(
                [
                    # the queue at the strait, in weeks of the out route's nominal capacity
                    np.log1p(waiting_out / u0),
                    np.log1p(waiting_all / u0),
                    age / self.T,
                    # the cargo reaching the strait: most of it leaves in the week it arrives
                    np.log1p(arriving / u0),
                    np.log1p(arriving_next / u0),
                    np.log1p(arriving_out / u0),
                    # the strait and the out route this week
                    open_frac[r] if r is not None else 1.0,
                    np.log1p(kappa / u0),
                    u_ratio[e],
                    c_ratio[e],
                    float(obs["graph_now.tariff"][e, k]),
                    float(np.any(ov_mask[g["slots"]])),
                    warning[g["warn_row"]] if g["warn_row"] is not None else 0.0,
                    float(obs["graph_now.war_risk"][r]) / 2 if r is not None else 0.0,
                    float(len(closing) > 0),
                    weeks_to_reopen / self.T,
                    # where the cargo goes
                    np.log1p(stock[g["head_row"]] / u0) if g["head_row"] is not None else 0.0,
                    np.log1p(transit / u0),
                    *good,
                    *global_part,
                ]
            )
        return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)

    def with_floor(self, qty, obs, floor=RELEASE_FLOOR):
        """``qty`` raised, per (strait, good) pair whose strait is open, to at least ``floor`` x what it has available.

        The pair's groups share the extra in proportion to what the network gave them (or to their out routes'
        capacity when it gave nothing). Call after ``__call__`` of the same week (it reads ``available``).
        """
        qty = np.maximum(np.asarray(qty, dtype=float), 0.0).copy()
        open_frac = obs["graph_now.open"]
        pairs = {}
        for i, g in enumerate(self.groups):
            if (
                g["choke_row"] is not None
                and open_frac[g["choke_row"]] >= FLOOR_OPEN
                and obs["override_mask"][g["slots"]].any()
            ):
                pairs.setdefault(g["pair"], []).append(i)
        for members in pairs.values():
            need = floor * self.available[members[0]]  # the pair's waiting + arriving (the same for its groups)
            total = qty[members].sum()
            if total >= need or need <= 0:
                continue
            weight = qty[members] if total > 0 else self.u0[members]
            qty[members] += (need - total) * weight / weight.sum()
        return qty

    def release_action(self, qty, obs):
        """(override_qty, release_mode) that release ``qty[i]`` from group i this week.

        Each group's quantity goes on its first override slot open this week (``override_mask``); a (strait, good)
        pair with any open group takes ``release_mode`` 1 (our quantities, 0 included: 0 holds the cargo), the
        others keep the default release (0).
        """
        override_qty = np.zeros(self.n_slots)
        release_mode = np.zeros(self.n_pairs, dtype=np.int64)
        for g, q in zip(self.groups, qty, strict=True):
            open_slots = [o for o in g["slots"] if obs["override_mask"][o]]
            if open_slots and g["pair"] is not None:
                override_qty[open_slots[0]] = max(float(q), 0.0)
                release_mode[g["pair"]] = 1
        return override_qty, release_mode


def normalise(x, mean, std):
    """Scale each feature to about mean 0, spread 1; clipped, so a value never seen in training stays bounded."""
    return np.clip((x - mean) / std, -10.0, 10.0)


def _layers(weights, prefix):
    """The (W, b) pairs of one MLP in weights.npz, in order."""
    n = 1
    while f"{prefix}W{n + 1}" in weights:
        n += 1
    return [(weights[f"{prefix}W{i}"], weights[f"{prefix}b{i}"]) for i in range(1, n + 1)]


def _mlp(layers, h):
    """ReLU after every layer but the last."""
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def mlp_forward(weights, x, prefix=""):
    """A plain network: (rows, features) -> (rows,) share of nominal capacity, any depth. Plain numpy.

    ``prefix`` picks the network in weights.npz: "" the flows', "rel_" the releases'.
    """
    h = normalise(x, weights[prefix + "mean"], weights[prefix + "std"])
    return _mlp(_layers(weights, prefix), h)[:, 0]


def shared_forward(weights, x, share):
    """The shared flow network (run 11): (slots, features) -> (slots,), routes deciding together.

    ``share`` is ``SlotFeatures.share_matrices``: (3, slots, slots) means over same source, destination, good.
    """
    h = np.maximum(_mlp(_layers(weights, "enc_"), normalise(x, weights["mean"], weights["std"])), 0.0)
    return _mlp(_layers(weights, "dec_"), np.concatenate([h, *(m @ h for m in share)], axis=1))[:, 0]


class Agent:
    def __init__(self, config=None):
        if WEIGHTS is None:
            raise FileNotFoundError(f"{WEIGHTS_PATH} is missing: train it with experiments/bc/train_replan.py first")
        if "rel_mean" not in WEIGHTS:
            raise ValueError(f"{WEIGHTS_PATH} has no release network: retrain it with experiments/bc/train_replan.py")
        self.features = SlotFeatures(config)
        self.releases = ReleaseFeatures(config)
        for name, feats, prefix in (("flow", self.features, ""), ("release", self.releases, "rel_")):
            if feats.n_features != WEIGHTS[prefix + "mean"].shape[0]:
                raise ValueError(
                    f"weights.npz's {name} network expects {WEIGHTS[prefix + 'mean'].shape[0]} features, this "
                    f"network gives {feats.n_features}: the weights were trained on another network (task)"
                )

    def act(self, observation):
        x = self.features(observation)
        if "enc_W1" in WEIGHTS:
            share = shared_forward(WEIGHTS, x, self.features.share_matrices)
        else:
            share = mlp_forward(WEIGHTS, x)
        share = np.clip(share, 0.0, MAX_SHARE)
        flows = share * self.features.u0 * observation["action_mask"]
        release_x = self.releases(observation, self.features.global_part)
        release = np.clip(mlp_forward(WEIGHTS, release_x, "rel_"), 0.0, MAX_SHARE) * self.releases.u0
        release = self.releases.with_floor(release, observation)  # the safety net
        override_qty, release_mode = self.releases.release_action(release, observation)
        return {"flows": flows, "override_qty": override_qty, "release_mode": release_mode}
