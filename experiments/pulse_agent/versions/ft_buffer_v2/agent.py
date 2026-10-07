"""``agents/mpc_buffer`` tuned for Full, with one addition outside the planner: fuel pulsed at grids that are short of it.

This folder is the second version of ``agents/ft_buffer`` (the version uploaded on 2026-10-06, kept as it was). It
differs by one rule in ``pulse.py``, switched on by ``pulse_tail`` in ``params.json``: a fab stops counting toward a
pulse once a lot started now could not reach a sink before the episode ends. Without the key (or with None) it plays
as ``agents/ft_buffer`` does; ``reserve.py`` and ``sbflow/`` are the same files. On root 12345 the rule adds 0.001 on
Full (16 episodes) and 0.008 on Small (64 episodes).

Each week it plans ``h_weeks`` weeks and plays the first, as ``agents/mpc_buffer`` does (``mpc_det`` with a fuel reserve
at grids, a ramp on shed load and a window-end value, in ``reserve.py``). Two things are new here.

``pulse.py`` changes the week's flows from a terminal to its grid after the planner has chosen them. The simulator
serves a grid's base load before its fabs, and a grid's output barely exceeds the two together, so a fuel segment
short by a few per cent takes all the fabs' energy for as long as the shortage lasts (the planner's window does not
see this: from its second week it sheds base load to power a fab, which the simulator never does). The pulse holds the
short fuel at the terminal for some weeks and delivers a full week's burn in others: the same energy, but the fabs
run in the whole weeks. On root 12345 it lifts the score from 0.68 to 0.84 on Full (16 episodes) and from 0.76 to
0.83 on Small (64 episodes), with the settings of ``params.json``.

``reserve.py`` has more knobs than ``agents/mpc_buffer``'s: ``short_premium`` (a fuel shortfall priced by the chips it
costs under the pulse, so a scarce fuel goes to the grid that loses most), and ``end_value_fuel``, ``end_value_grids``,
``end_value_wip``, ``base_first_weeks`` and ``base_first_price``, which were measured and left at their defaults.

The benchmark's planner assumes the network stays as observed (a closed strait stays closed, an announced sanction
starts on its week) and demand follows the forecast, solves a linear program for the cheapest flows over the window,
and sends this week's. So it holds no fuel against events that have not started, it does not care when a shortage it
expects to last is taken, and it prices what is left at the window's end at the end-of-episode salvage price.
``reserve.py`` changes the window's LP on all three: a soft minimum on the fuel a grid holds, a price on shed base load
that falls over the window, and a share of the customs value for goods in stock, in transit or in process when the
window ends before the episode does. The rest is ``agents/mpc_det_h20``: the planner is shockbench-flow's code (MIT) in
``sbflow/`` (``scripts/vendor_planner.py``, unchanged), solved with the HiGHS that SciPy bundles, and this file
translates between the agent's arrays and the observation the planner reads. It reads no warnings and no
announcements.

``PARAMS`` holds the settings; with its defaults the agent plays as ``mpc_det_h20`` does. The ``params.json`` beside
this file replaces them with the settings measured best on Full (root 12345, 16 episodes). A week the reserve's code
fails is planned as ``mpc_det`` would (counted in ``Planner.failed``); a week the pulse's code fails sends the
planner's flows as they were (``Agent.pulse_failed``); a week the planner fails sends the maximum.
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pulse
import reserve
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.policies import lp_common
from sbflow.policies.base import StepTelemetry
from sbflow.policies.mpc_det import MpcDet, MpcDetParams
from sbflow.policies.registry import PolicyContext


PARAMS = {
    "h_weeks": 20,  # the planning window in weeks; None: the network's longest lead time L (24 on small and full)
    "demand_scale": 1.0,  # the forecast demand of every window week times this: above 1 plans for more than forecast
    "throughput_scale": 1.0,  # each strait's throughput per pool times this: below 1 plans for less passage than seen
    # the fuel a grid should hold, in weeks of its full-rate burn of that fuel (the fuel's share of the grid times its
    # nominal deliverable power), capped at the slot's storage; 0: no reserve
    "reserve_weeks": 0.0,
    "reserve_price": 0.001,  # the price of a unit-week of fuel missing below that level, as a share of the grid's VOLL
    "reserve_fuels": "all",  # "all": every fuel a grid burns; "rationed": only its rationed fuel (LNG)
    # True: the cover a grid's storage cannot hold is asked of the node feeding it over a zero-lead edge (its terminal)
    "reserve_feeders": False,
    # shed base load costs this share of VOLL more for every window week still to come (a ramp down to VOLL in the
    # window's last week), so a plan sheds later rather than earlier; with a reserve, keep it above reserve_price, or
    # the reserve pays for shedding early; 0: no ramp
    "shed_ramp": 0.0,
    # while the window ends before the episode: goods in stock, in transit or in process at the window's end are worth
    # this share of their customs value where that is above the salvage price; 0: the salvage price
    "end_value": 0.0,
    "end_value_fuel": None,  # the same share for the goods a grid burns (fuels) when it should differ; None: end_value
    # False: fuel held at grids and flows into grids keep the salvage price at the window's end (no end value there)
    "end_value_grids": True,
    # what a lot in process at the window's end is worth: "input" (its wafer) or "product" (its raw chip)
    "end_value_wip": "input",
    # the window weeks (from the first) in which shed base load at a grid with fabs carries the base-first price, so a
    # plan runs a fab only on energy above the base load, as the simulator does; 1: the planner's own (the first week)
    "base_first_weeks": 1,
    "base_first_price": "package",  # "package": the planner's first-week price; "tight": just above a fab's energy value
    # True: a fuel that a grid with fabs is short of is held at its terminal until the grid can run a whole week, and
    # delivered then (pulse.py), so the fabs run in some weeks instead of never
    "pulse": False,
    "pulse_hold_weeks": 6.0,  # a fuel whose tank would take longer than this to fill for one whole week is not held
    "pulse_fill": 0.9,  # fuel is held up to this share of the terminal's storage
    "pulse_margin": 1.0,  # a whole week delivers this many times the week's burn of a fuel the terminal runs short of
    # the rationed fuel (its grid stock must first be lifted to the rationing threshold, which the off weeks burn) is
    # delivered again when the terminal holds enough for this many whole weeks in a row, or is full
    "pulse_run": 1.0,
    # a grid not yet pulsed starts only when its fabs would get less than this share of their draw as things are
    # (a mild shortage leaves them most of it every week, which a pulse does not beat); 1: any shortfall starts one
    "pulse_partial": 0.5,
    "pulse_memory": 4.0,  # a grid counts as pulsed for this many weeks after the last week a fuel was held for it
    # a fuel is held only if this many whole weeks of the grid's fabs pay for the energy a full tank defers (lost
    # when the shortage ends with the tank full); 0: every fuel with a terminal may be held
    "pulse_payback": 0.0,
    # a number: a fab asks for no energy once a lot started now would reach no sink before the episode ends with this
    # many weeks to spare (its chips are then worth nothing, and a pulse would shed base load for them); None: off
    "pulse_tail": None,
    # a fuel segment's shortfall at a grid with fabs costs this share of the chips it loses under the pulse (the fabs'
    # weekly chips less VOLL times their energy, per unit of the fuel's weekly burn), so the plan sends a scarce fuel
    # to the grid that loses most; 0: every shortfall costs VOLL alike, as the planner has it
    "short_premium": 0.0,
    # debugging: True raises any failure (of the reserve's code or of the planner) out of act, so a local run counts
    # the week as an error instead of planning it without the reserve or sending the maximum
    "strict": False,
}
if (Path(__file__).parent / "params.json").is_file():
    PARAMS |= json.loads((Path(__file__).parent / "params.json").read_text())


class Planner(MpcDet):
    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self.failed = 0  # weeks planned as mpc_det would, after a failure of the reserve's code

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) if PARAMS["h_weeks"] is None else int(PARAMS["h_weeks"])

    def _window_arrays(self, inst, obs, H_t):
        """The persistence forecast with demand and the straits' throughput scaled, as the package's ``MpcDetVariant``.

        Backlog is not scaled: the planner folds it into week-1 demand later (``lp_common.window_marks``).
        """
        arrays = super()._window_arrays(inst, obs, H_t)
        scaled = {"demand": PARAMS["demand_scale"], "kappa": PARAMS["throughput_scale"]}
        scaled |= {"kappa_now": PARAMS["throughput_scale"]}
        if all(v == 1 for v in scaled.values()):
            return arrays
        out = dict(arrays)
        for key, factor in scaled.items():
            if factor != 1:
                out[key] = arrays[key] * factor
                out[key].flags.writeable = False
        return out

    def act(self, obs):
        """``MpcDet.act`` with the window's model built by ``reserve.rolled_lp``; a week that fails is planned plain."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        H_t = lp_common.window_length(self._H, t, inst.T)
        arrays = self._window_arrays(inst, obs, H_t)
        rules = self.params.planning_rules
        model = res = None
        try:
            model = reserve.rolled_lp(inst, obs, arrays, H_t, PARAMS, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
            if not res.ok and reserve.changes(model):
                raise RuntimeError(f"no solution of the window with the reserve: {res.trail}")
        except Exception:  # noqa: BLE001 - any failure of the reserve's code: this week is planned without it
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.failed += 1
            model = lp_common.rolled_lp(inst, obs, arrays, H_t, planning_rules=rules)
            res = self._session.solve(lp_common.to_highs_lp(model), lp_common.WindowShape.of(model))
        self._record(model, res)
        if res.ok:
            action = lp_common.week1_action(inst, model, res.x, obs, lp_common.prohibited_now(self._memory, t))
        else:
            action = self._fallback.act(obs)
        self.telemetry.append(
            StepTelemetry(t, time.perf_counter() - start, res.iterations, res.trail, fallback=not res.ok)
        )
        return action


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
        self.pulse, self.pulse_failed = None, 0
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
                    self.pulse = pulse.Pulse(planner._inst, PARAMS) if PARAMS["pulse"] else None
                    self.pulse_failed = 0  # weeks the pulse's code failed: the planner's flows were sent as they were
                return self.action(self.pulsed(obs, self.planner.act(obs)))
            except Exception:  # noqa: BLE001 - any failure of the planner: this week sends the maximum
                if PARAMS["strict"]:
                    raise
                traceback.print_exc()
                self.broken = self.planner is None  # a planner that never started is not tried again
        return self.send_max(observation)

    def pulsed(self, obs, action):
        """The planner's action with the fuel of short grids held or delivered by ``pulse.Pulse``; as it was on failure."""
        if self.pulse is None:
            return action
        try:
            memory = self.planner._memory
            banned = lp_common.prohibited_now(memory, int(obs["week"]))
            return self.pulse.adjust(obs, memory.values, banned, action)
        except Exception:  # noqa: BLE001 - any failure of the pulse's code: the planner's flows stand
            if PARAMS["strict"]:
                raise
            traceback.print_exc()
            self.pulse_failed += 1
            return action

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
