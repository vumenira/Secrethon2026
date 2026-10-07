"""An analysis probe, NOT a submission: ``mpc_det`` whose planning window a harness can fill with ground truth.

``experiments/approaches/foresight.py`` loads this folder, sets ``Planner.foresight`` to a function that returns the
window arrays of the week (built from the scenario's true marks, which no agent sees) and ``Planner.h_weeks`` to the
window length, and plays it to measure what each kind of knowledge about the future would be worth to the planner.
Without a harness it plays exactly as ``agents/mpc_det_h20`` (``foresight`` is None). Do not pack or upload it.

The planner is shockbench-flow's code (MIT) in ``sbflow/``, a copy of ``agents/mpc_det_h20/sbflow`` with one added
switch, ``BASE_FIRST_EVERY_WEEK`` in ``sbflow/oracle/lp.py`` (off: the package's model). The differences to
``agents/mpc_det_h20/agent.py`` are ``Planner`` (the hook, and the window's fab hits passed on to the LP, which
``mpc_det`` never has under persistence) and nothing else.
"""

import time
import traceback

import numpy as np
from sbflow.instance.schema import WAR_RISK_CLASSES
from sbflow.marks import FabHit
from sbflow.policies import lp_common as L
from sbflow.policies.base import StepTelemetry
from sbflow.policies.mpc_det import MpcDet, MpcDetParams
from sbflow.policies.registry import PolicyContext


H_WEEKS = 20  # the planning window in weeks; None: the network's longest lead time L (24 on small and full)


class Planner(MpcDet):
    foresight = None  # the harness's hook: (planner, obs, H_t, lp_common) -> (window arrays, [(fab, onset, severity)])
    h_weeks = H_WEEKS

    def reset(self, static, obs, policy_seed):
        super().reset(static, obs, policy_seed)
        self.failed, self.hits, self.stats = 0, (), {}  # failed: weeks the hook raised and persistence was planned

    def _horizon(self, inst, plan):
        return super()._horizon(inst, plan) if self.h_weeks is None else self.h_weeks

    def _window_arrays(self, inst, obs, H_t):
        self.hits = ()
        if self.foresight is None:
            return L.persistence_arrays(inst, obs, self._memory, H_t)
        try:
            arrays, hits = type(self).foresight(self, obs, H_t, L)
            self.hits = tuple(FabHit(fab=int(f), onset=float(on), severity=float(sev)) for f, on, sev in hits)
            return L.read_only(L.with_now(arrays))
        except Exception:  # noqa: BLE001 - counted and reported by the harness: a ceiling must not hide a failed week
            traceback.print_exc()
            self.failed += 1
            self.hits = ()
            return L.persistence_arrays(inst, obs, self._memory, H_t)

    def act(self, obs):
        """``MpcDet.act`` with the window's fab hits passed to the LP (they scrap the observed WIP)."""
        start = time.perf_counter()
        inst, t = self._inst, int(obs["week"])
        self._memory.update(inst, obs)
        H_t = L.window_length(self._H, t, inst.T)
        arrays = self._window_arrays(inst, obs, H_t)
        model = L.rolled_lp(inst, obs, arrays, H_t, fab_hits=self.hits, planning_rules=self.params.planning_rules)
        res = self._session.solve(L.to_highs_lp(model), L.WindowShape.of(model))
        self._record(model, res)
        if res.ok:
            action = L.week1_action(inst, model, res.x, obs, L.prohibited_now(self._memory, t))
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
        self.act_errors = 0  # weeks the planner raised and the maximum was sent (the harness reports them)
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
                self.act_errors += 1
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
