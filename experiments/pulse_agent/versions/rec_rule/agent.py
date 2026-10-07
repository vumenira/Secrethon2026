"""``mpc_det`` with a recovery forecast: each week it plans ``h_weeks`` weeks and plays the first.

The benchmark's planner assumes the network stays as observed for the whole window (a closed strait stays closed). This
one lets each closed strait and each grid short of power end (``recovery.py``), by a rule (``mode``, ``quantile``,
``caution``) that can differ per kind and per event type (``rules``). As set here: a militarised closure fades as the
published duration law of its type says, given how long it has been seen (the expected share of each week it is still
on); a storm's closure and a grid's shortage are planned as over from this week on, the best of the rules measured on
Small (the earlier the planned recovery of these two, the lower the cost, down to none); the year-long closure a CN-TW
conflict puts on the Taiwan Strait is kept. The rest is
``agents/mpc_det_h20``: the planner is shockbench-flow's code (MIT) in ``sbflow/`` (``scripts/vendor_planner.py``),
solved with the HiGHS that SciPy bundles, and this file translates between the agent's arrays and the observation the
planner reads. It reads no warnings and no announcements. A ``params.json`` beside this file replaces ``PARAMS``. A week
the forecast fails is planned as ``mpc_det`` would; a week the planner fails sends the maximum.
"""

import json
import traceback

from pathlib import Path

import numpy as np
from recovery import Recovery
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.policies import lp_common
from sbflow.policies.mpc_det import MpcDet, MpcDetParams
from sbflow.policies.registry import PolicyContext


PARAMS = {
    "h_weeks": 20,  # the planning window in weeks; None: the network's longest lead time L (24 on small and full)
    # the impairments forecast to end, of closure, capacity, grid, supply, fab, osat (the others are kept)
    "kinds": ["closure", "grid"],
    "mode": "mean",  # "mean": the expected share of each week still impaired; "quantile": impaired until a quantile
    "quantile": 0.5,  # of the impairment's remaining life; 0: planned as over from this week on
    "caution": 1.0,  # the power on the probability an impairment is still on: below 1, recoveries come later
    # {a kind or an event type: {"mode", "quantile", "caution"}}: its own rule in place of the three above; the event
    # types of the two kinds are weather_closure, militarised_closure and energy_shock
    "rules": {
        "weather_closure": {"mode": "quantile", "quantile": 0.0},
        "grid": {"mode": "quantile", "quantile": 0.0},
    },
}
if (Path(__file__).parent / "params.json").is_file():
    PARAMS |= json.loads((Path(__file__).parent / "params.json").read_text())


class Planner(MpcDet):
    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self.recovery, self.failed = Recovery(self._inst, PARAMS), 0  # failed: weeks planned on persistence

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) if PARAMS["h_weeks"] is None else PARAMS["h_weeks"]

    def _window_arrays(self, inst, obs, H_t):
        try:
            self.recovery.update(obs, self._memory)
            return self.recovery.arrays(obs, self._memory, H_t)
        except Exception:  # noqa: BLE001 - any failure of the forecast: this week is planned on persistence
            traceback.print_exc()
            self.failed += 1
            return lp_common.persistence_arrays(inst, obs, self._memory, H_t)


def nullable(obs, key, cast=float):
    """A column as a list, None where it is not observed."""
    return [cast(v) if seen else None for v, seen in zip(obs[key].tolist(), obs[f"{key}.observed"].tolist())]


def rows(obs, block, fields):
    """The live entries of a padded list, as columns."""
    live = np.flatnonzero(obs[f"{block}.{fields[0]}.observed"])
    return {f: obs[f"{block}.{f}"][live].tolist() for f in fields}, live


class Agent:
    def __init__(self, config=None):
        static, layout = config["static"], config["layout"]
        self.static, self.seed = static, config["policy_seed"]
        self.planner, self.broken = None, False  # the planner starts on the first observation
        self.stock_slots, self.supply_slots = layout["stock_slots"], layout["supply_slots"]
        self.demands = layout["demands"]
        self.fabs, self.grids, self.osats = layout["fabs"], layout["grids"], layout["osats"]
        self.lot_keys = layout.get("lot_keys")  # small and full: the rows of the dense queue_lots block
        edges, lanes, slots = static["edges"], static["lanes"], static["action_slots"]
        if self.lot_keys is not None:  # the edge each lot entered its strait by: the lane's edge before its next edge
            self.lot_entry = []
            for _c, _k, lane, nxt in self.lot_keys:
                along = lanes["edges"][lane]
                self.lot_entry.append(along[along.index(nxt) - 1] if nxt in along[1:] else nxt)
            self.lot_transit = [edges["tau0"][e] for e in self.lot_entry]
        pairs = {tuple(pair): i for i, pair in enumerate(layout["release_pairs"])}
        override = static["override_slots"]
        self.pairs = pairs
        self.override_pair = [pairs[(c, k)] for c, k in zip(override["chokepoint"], override["k"])]
        # the fallback: each slot's nominal capacity and the straits its lane passes
        self.capacity = np.array([edges["u0"][e] for e in slots["edge"]], dtype=float)
        position = {node: i for i, node in enumerate(layout["chokepoints"])}
        passes = lanes["chokepoints"]
        self.through = [[position[c] for c in passes[lane]] if lane is not None else [] for lane in slots["lane"]]

    def act(self, observation):
        if not self.broken:
            try:
                obs = self.wire(observation)
                if self.planner is None:
                    planner = Planner(MpcDetParams(), PolicyContext())
                    planner.reset(self.static, obs, self.seed)
                    self.planner = planner
                return self.action(self.planner.act(obs))
            except Exception:  # noqa: BLE001 - any failure of the planner: this week sends the maximum
                traceback.print_exc()
                self.broken = self.planner is None  # a planner that never started is not tried again
        return self.send_max(observation)

    def wire(self, o):
        """The observation the planner reads (the scorer's own layout), rebuilt from the agent's arrays."""
        blackout = o["action_mask.observed"][0] == 0
        keep = np.flatnonzero(o["stock.qty.observed"])
        stock = {
            "node": [self.stock_slots[i][0] for i in keep],
            "k": [self.stock_slots[i][1] for i in keep],
            "qty": o["stock.qty"][keep].tolist(),
        }
        keep = np.flatnonzero(o["backlog.qty.observed"])
        backlog = {
            "node": [self.demands[i][0] for i in keep],
            "k": [self.demands[i][1] for i in keep],
            "qty": o["backlog.qty"][keep].tolist(),
        }
        pipeline, live = rows(o, "pipeline", ("qty", "edge", "k", "arrival_week"))
        pipeline["lane"] = nullable(o, "pipeline.lane", int)
        pipeline["lane"] = [pipeline["lane"][i] for i in live]
        wip, _live = rows(o, "wip", ("qty", "node", "k", "out_week"))
        pending, live = rows(o, "pending_prohibitions", ("edge", "k", "effective_week"))
        d, h = np.nonzero(o["demand_forecast.qty.observed"])
        forecast = {
            "node": [self.demands[i][0] for i in d],
            "k": [self.demands[i][1] for i in d],
            "h": h.tolist(),
            "qty": o["demand_forecast.qty"][d, h].tolist(),
        }
        return {
            "week": int(o["week"][0]),
            "stock": stock,
            "backlog": backlog,
            "pipeline": pipeline,
            "queue_lots": self.lots(o),
            "wip": wip,
            "graph_now": self.graph(o),
            "demand_forecast": forecast if d.size else None,
            "pending_prohibitions": None if blackout and not live.size else pending,
        }

    def lots(self, o):
        fields = (
            "qty",
            "lot_id",
            "chokepoint",
            "k",
            "lane",
            "next_edge",
            "arrival_week",
            "dispatch_week",
            "entry_edge",
        )
        if self.lot_keys is None:  # tiny: the lots as a list
            return rows(o, "queue_lots", fields)[0]
        key, week = np.nonzero(o["queue_lots.qty.observed"])  # row: a (strait, k, lane, next edge); column: week - 1
        order = np.lexsort((key, week))
        key, week = key[order].tolist(), (week[order] + 1).tolist()
        out = {f: [] for f in fields}
        for i, (r, w) in enumerate(zip(key, week)):
            c, k, lane, nxt = self.lot_keys[r]
            lot = (
                float(o["queue_lots.qty"][r, w - 1]),
                i,
                c,
                k,
                lane,
                nxt,
                w,
                w - self.lot_transit[r],
                self.lot_entry[r],
            )
            for f, v in zip(fields, lot):
                out[f].append(v)
        return out

    def graph(self, o):
        if o["override_mask.observed"][0] == 0:  # a blackout week: the planner keeps what it saw last
            return None
        keep = np.flatnonzero(o["graph_now.supply.avail.observed"])
        e, k = np.nonzero(o["graph_now.prohibited"])
        te, tk = np.nonzero(o["graph_now.tariff"])
        war = zip(o["graph_now.war_risk"].tolist(), o["graph_now.war_risk.observed"].tolist())
        return {
            "u": nullable(o, "graph_now.u"),
            "c": nullable(o, "graph_now.c"),
            "tau": nullable(o, "graph_now.tau", int),
            "open": nullable(o, "graph_now.open"),
            "kappa": {pool: nullable(o, f"graph_now.kappa.{pool}") for pool in ("tb", "ct")},
            "war_risk": [WAR_RISK_CLASSES[code] if seen else None for code, seen in war],
            "supply": {
                "node": [self.supply_slots[i][0] for i in keep],
                "k": [self.supply_slots[i][1] for i in keep],
                "avail": o["graph_now.supply.avail"][keep].tolist(),
            },
            "fab": {"node": self.fabs} | {f: nullable(o, f"graph_now.fab.{f}") for f in ("R", "alpha_bar", "cap_eff")},
            "grid": {"node": self.grids} | {f: nullable(o, f"graph_now.grid.{f}") for f in ("G_bar", "y_bar")},
            "osat": {"node": self.osats} | {f: nullable(o, f"graph_now.osat.{f}") for f in ("R", "thr_eff")},
            "prohibited": {"edge": e.tolist(), "k": k.tolist()},
            "tariff": {"edge": te.tolist(), "k": tk.tolist(), "rate": o["graph_now.tariff"][te, tk].tolist()},
        }

    def action(self, wire):
        """The planner's action as the agent's arrays."""
        flows = np.zeros(len(self.capacity))
        sent = wire.get("flows") or {"slot": [], "qty": []}
        flows[sent["slot"]] = sent["qty"]
        override, mode = np.zeros(len(self.override_pair)), np.zeros(len(self.pairs), dtype=np.int64)
        released = wire.get("overrides") or {"slot": [], "qty": []}
        for slot, qty in zip(released["slot"], released["qty"]):
            override[slot], mode[self.override_pair[slot]] = qty, 1
        held = wire.get("hold") or {"chokepoint": [], "k": []}
        for c, k in zip(held["chokepoint"], held["k"]):
            mode[self.pairs[(c, k)]] = 2
        return {"flows": flows, "override_qty": override, "release_mode": mode}

    def send_max(self, o):
        """The kit heuristic: every allowed route at capacity, a lane through a strait times its open fraction."""
        flows = self.capacity * o["action_mask"]
        open_now, seen = o["graph_now.open"], o["graph_now.open.observed"] == 1
        for s, straits in enumerate(self.through):
            for c in straits:
                if seen[c]:
                    flows[s] *= max(float(open_now[c]), 0.0)
        return {"flows": flows}
