"""Three changes to the planner's window LP: a fuel reserve at grids, a ramp on shed load, and a window-end value.

The planner (``sbflow``, the benchmark's ``mpc_det``) plans on "what I see now stays as it is", so it holds no fuel
against events that have not started, it is indifferent to when a shortage it expects to last is taken, and it prices
whatever is left when its window ends at the end-of-episode salvage price. The changes are added to the benchmark's own
builder (``sbflow.oracle.lp._Builder``) just before it stacks the model, so ``sbflow/`` stays as vendored. They are
columns, rows and entries of the planning rules' price table (``meta["priority"]``), which enters the objective only:
the cost parts, the action mapping and the warm start (the same columns and rows in every window week) are untouched.

The reserve (``reserve_weeks`` > 0). For a fuel k that a grid g burns, in every window week t:

    I[g, k, t] + short[g, k, t] >= target[g, k, t],   short >= 0,   priced reserve_price * VOLL_g per unit-week

- ``target = min(cover * shares[k] * deliverable_g, storage[g, k])``: ``cover`` weeks of the grid's full-rate burn of
  the fuel (a fuel unit is one GWh; nominal values of the instance, never the week's impaired ones), with
  ``cover = min(reserve_weeks, weeks of the episode after t)``, so nothing is asked for after the last week.
- ``reserve_feeders``: the node that feeds the grid over a zero-lead edge (its terminal) holds the same fuel for it.
  With the switch on, the cover the grid's own storage cannot hold is asked of those nodes, each up to its storage, so
  ``reserve_weeks`` may exceed the grid's storage (about 4 weeks of burn).

The ramp (``shed_ramp`` > 0). Shed base load of window week h is priced ``shed_ramp * VOLL_g * (H - h)`` above VOLL at
every grid (H the window's weeks): shedding a week later is cheaper by ``shed_ramp * VOLL``, so of two plans that shed
the same the planner takes the one that sheds later. It is also the reserve's guard. A price on low stock rewards a
plan that sheds early and keeps the fuel when its forecast says the shortage lasts (a unit shed in week h instead of
week h' > h saves h' - h weeks of the reserve's price), at the same grid, which the simulator decides, or at another
grid by sending the fuel elsewhere, which the agent decides. With ``shed_ramp`` above ``reserve_price`` that never
pays, and the reserve is filled only with fuel no grid would burn in the window.

The end value (``end_value`` > 0), only while the window ends before the episode does. Goods at the window's end are
credited ``end_value * v_k`` (the commodity's customs value) where that is above the salvage price the builder gives
them: stock at nodes other than supply nodes and chokepoints, flows still in transit (times the strait's open share
when they head for one), lots and OSAT starts still in process (at their input's value, as the builder's salvage).
Queues at straits keep their salvage price, and so do the stocks the planning rules price (a fab's wafers, an OSAT's raw
chips), which the simulator moves on by itself.

Added in ``agents/ft_buffer`` (each off by default, so the model is ``agents/mpc_buffer``'s without them):

- the shortfall premium (``short_premium`` > 0): a price on the builder's fuel shortfall column at grids with fabs,
  ``Builder.add_short_premium``. Measured on Full with the pulse: +0.010.
- the end value's options: ``end_value_fuel`` (another share for fuels, not measured), ``end_value_grids`` False (no
  end value for fuel at grids or on its way to them: -0.18 on Full, the planner then runs the nuclear fuel down) and
  ``end_value_wip`` "product" (a lot in process valued at its chip).
- base load first in every window week (``base_first_weeks`` > 1), ``Builder.add_base_first``: -0.02 to -0.04 on
  Full. It stops the plan from counting on fabs that get no energy, but it also prices unavoidable shedding far
  above VOLL; ``pulse.py`` gets the fabs their energy instead.

Added in ``agents/ft_buffer_v4`` (each off by default):

- ``premium_small``: the premium of a fuel whose segment is smaller than the fabs' draw is per unit of that draw, not
  of its own burn (it costs the fabs a unit of energy per unit short; it cannot stop them).
- ``premium_limit`` (``Builder.add_free_short``): the premium is taken off the part of a fuel's shortfall that the
  grid's scarcest fuel makes useless, so a plentiful fuel stops outbidding the scarce one on an edge they share. The
  scarcest fuel is read off the window's start state (``Builder.fuel_cover``: stock at grid and terminal, lots
  waiting at a strait, shipments under way, over ``cover_weeks``). It adds one column and one row per grid and fuel.
- ``plan_duty`` (``Builder.add_duty``): from the window's second week the lots of a grid's fabs are bounded at the
  share of capacity that cover allows. Measured: -0.001 alone on both tasks and -0.001 to -0.003 next to the other
  two; left off.

Added in ``agents/mpc_pulse_rec_v10_kr`` (off by default):

- ``premium_whole`` (``Builder.add_whole``): ``premium_small`` prices a unit of a small fuel as a unit of the fabs'
  energy, which it is only in a week the grid runs whole; in any other week the fabs get nothing whatever the small
  fuel gives. Under the pulse the weeks that are not whole are the share of its burn that a fuel which can take all
  the fabs' energy is short by. So in every window week the small fuel's shortfall is free of the premium up to
  that share of its own burn, the share being the plan's own shortfall of the grid's other fuels in that week (one
  column and two rows per grid and small fuel; no forecast and no setting). Without it a small fuel that shares an
  edge with the scarce one took the edge first, every week, for energy the fabs never saw.

Added in ``agents/mpc_pulse_rec_v13`` (off by default here; ``agent.PARAMS`` has ``strait_reserve`` on):

- the reserve at straits (``strait_reserve``, ``strait_weeks``; ``Builder.strait_targets``). Tanker cargo waits at a
  strait for as long as the agent holds it there, and a strait's queue has no limit, where a terminal and its grid
  hold a few weeks of burn. The reserve above therefore ends where the storage does, and a strait that closes for a
  year is bridged for those few weeks; the clairvoyant plan bridges it with fuel it parked beyond the strait
  beforehand (30,000 GWh of gas and 34,000 of crude at one strait in one episode). The same soft minimum is put on
  a fuel's queue at the last strait before the terminals:

      Q[c, k, t] + short[c, k, t] >= target[c, k, t],   short >= 0,   priced as the grids' reserve

  for a fuel k at a strait c it reaches through another strait (what waits there is past whatever closes upstream),
  with ``target = cover * min(sum over the edges from c into a grid's terminal of min(u[e, t], the burn it feeds),
  the strait's throughput of the fuel's pool)``, ``cover = min(strait_weeks, weeks of the episode after t)``, and 0 in
  a week the strait's queue is charged at a war-risk rate. Like the grids' reserve it is filled only with fuel no
  grid would burn in the window (the ramp), and after the terminals (they also earn the end value).
- the reserve at supply nodes (``reserve_sources``; in ``Builder.reserve_targets``). A supply node keeps what is
  not shipped, up to its storage, at no cost, and what it cannot keep is lost. The plan saw no worth in that stock,
  so of two supplies of a grid whose stores are full it cut whichever saved freight, and a supply node that pipes
  fuel straight into the grid stood at one week of stock where it could have stood at four (in one episode the
  other route was then cut, and the grid ran dry fourteen weeks before the episode's end). With the switch the
  cover that neither the grid nor its terminal can hold is asked of the supply nodes with an edge straight into
  the grid, each up to its storage, at the same price. Measured and left off (it takes that loss back and loses
  as much elsewhere).

What was measured is at the two switches in ``agent.PARAMS``.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sbflow.marks import fixed_fp_errors
from sbflow.oracle.lp import _Builder
from sbflow.policies import lp_common


FUELS = ("all", "rationed")
BASE_FIRST_PRICES = ("package", "tight")
WIP_VALUES = ("input", "product")
TIGHT_MARGIN = 1.05  # the "tight" base-first price is this times the most a unit of energy is worth at the grid's fabs


@dataclass(frozen=True)
class Settings:
    """The knobs of ``agent.PARAMS`` that this module reads, checked, with the week the window starts in."""

    weeks: float  # reserve_weeks
    price: float  # reserve_price
    fuels: str  # reserve_fuels
    feeders: bool  # reserve_feeders
    ramp: float  # shed_ramp
    end_value: float
    week: int  # the episode week of the window's first week
    T: int  # the episode's weeks
    bf_weeks: int = 1  # base_first_weeks
    bf_price: str = "package"  # base_first_price
    ev_grids: bool = True  # end_value_grids
    ev_wip: str = "input"  # end_value_wip
    ev_fuel: float | None = None  # end_value_fuel
    short_premium: float = 0.0
    stock_early: float = 0.0
    premium_small: bool = False
    premium_limit: bool = False
    premium_whole: bool = False
    plan_duty: bool = False
    cover_weeks: float = 4.0
    pulse: bool = False  # whether pulse.Pulse plays (it decides what a short fuel leaves the fabs)
    payback: float = 0.0  # pulse_payback
    strait_weeks: float = 0.0  # strait_weeks where strait_reserve is on, else 0
    sources: bool = False  # reserve_sources

    @classmethod
    def of(cls, params: dict, week: int, T: int) -> "Settings":
        out = cls(
            weeks=float(params["reserve_weeks"]),
            price=float(params["reserve_price"]),
            fuels=params["reserve_fuels"],
            feeders=bool(params["reserve_feeders"]),
            ramp=float(params["shed_ramp"]),
            end_value=float(params["end_value"]),
            week=int(week),
            T=int(T),
            bf_weeks=int(params.get("base_first_weeks", 1)),
            bf_price=params.get("base_first_price", "package"),
            ev_grids=bool(params.get("end_value_grids", True)),
            ev_wip=params.get("end_value_wip", "input"),
            ev_fuel=None if params.get("end_value_fuel") is None else float(params["end_value_fuel"]),
            short_premium=float(params.get("short_premium", 0.0)),
            stock_early=float(params.get("stock_early", 0.0)),
            premium_small=bool(params.get("premium_small", False)),
            premium_limit=bool(params.get("premium_limit", False)),
            premium_whole=bool(params.get("premium_whole", False)),
            plan_duty=bool(params.get("plan_duty", False)),
            cover_weeks=float(params.get("cover_weeks", 4.0)),
            pulse=bool(params.get("pulse", False)),
            payback=float(params.get("pulse_payback", 0.0)),
            strait_weeks=float(params.get("strait_weeks", 0.0)) if params.get("strait_reserve", False) else 0.0,
            sources=bool(params.get("reserve_sources", False)),
        )
        numbers = (out.weeks, out.price, out.ramp, out.end_value, out.bf_weeks, out.ev_fuel or 0.0, out.short_premium, out.stock_early)
        numbers += (out.strait_weeks,)
        if not all(np.isfinite(v) and v >= 0 for v in numbers) or out.fuels not in FUELS:
            raise ValueError(f"reserve settings out of range: {out}")
        if out.bf_price not in BASE_FIRST_PRICES or out.ev_wip not in WIP_VALUES:
            raise ValueError(f"reserve settings out of range: {out}")
        return out

    @property
    def reserve(self) -> bool:
        return self.weeks > 0 and self.price > 0

    @property
    def strait(self) -> bool:
        return self.strait_weeks > 0 and self.price > 0


class Builder(_Builder):
    """The benchmark's builder with the reserve, the ramp and the end value added before the model is stacked."""

    def __init__(self, inst, marks, planning_rules: bool, settings: Settings) -> None:
        super().__init__(inst, marks, planning_rules=planning_rules)
        self.settings = settings

    def finish(self):
        st = self.settings
        end_value = st.end_value > 0 and st.week + self.T - 1 < st.T  # the window ends before the episode does
        base_first = st.bf_weeks > 1 and self.T > 1
        changed = st.reserve or st.ramp > 0 or end_value or base_first or st.short_premium > 0 or st.plan_duty
        changed = changed or st.strait
        if changed:
            if not self.rules:  # the builder keeps the price table only with the planning rules
                raise ValueError("the reserve, the ramp and the end value are prices of the planning rules' table")
            keys = list(self.cols.keys)  # the builder's own columns, before the reserve's
            where = {key: j for j, key in enumerate(keys)}
            ruled = frozenset(self.priority)  # the columns the planning rules price
            if end_value:
                self.add_end_value(keys, ruled)
            if base_first:
                self.add_base_first(where)
            if st.short_premium > 0:
                self.add_short_premium(where)
            if end_value and st.stock_early > 0:
                self.add_stock_early(where)
            if st.plan_duty and self.T > 1:
                self.add_duty(where)
            if st.ramp > 0:
                self.add_ramp(where)
            if st.reserve:
                self.add_reserve(where)
            if st.strait:
                self.add_strait_reserve(where)
        model = super().finish()
        model.meta["reserve"] = changed  # False: the model is the benchmark's own
        return model

    def add_price(self, j: int, price) -> None:
        """Add ``price`` (a scalar or one value per window week) to column template ``j``'s entry of the table."""
        self.priority[j] = np.asarray(self.priority.get(j, 0.0), dtype=float) + price

    # ----- the reserve -----------------------------------------------------------------------------------------------
    def reserve_targets(self) -> list[tuple[tuple, int, np.ndarray, float]]:
        """(name, stock slot, target per window week, price per unit-week) of every soft minimum."""
        inst, st = self.inst, self.settings
        S, E = inst.stock_slots, inst.edges
        left = st.T - (st.week + self.t - 1)  # weeks of the episode after each window week
        cover = np.minimum(st.weeks, np.maximum(left, 0)).astype(float)
        out = []
        asked: dict[int, list] = {}  # feeder slot -> [target, the highest price of the grids it feeds]
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            price = st.price * ga.voll
            for k in ga.fuels:
                if st.fuels == "rationed" and k != ga.rationed:
                    continue
                s = self.slot(g, k, "grid fuel")
                want = cover * ga.shares[k] * ga.deliverable
                target = np.minimum(want, S[s].storage)
                out.append((("reserve", go, k), s, target, price))
                rest = want - target
                if not st.feeders or not rest.any():
                    continue
                held = 0.0  # reserve_sources: what the feeder can hold of the rest
                for e in inst.in_edges[g]:  # the cover its storage cannot hold, at the node that feeds it at once
                    ed = E[e]
                    if k not in ed.K or ed.tau != 0 or ed.tail in self.chk or ed.tail in self.supply:
                        continue
                    fs = inst.slot_index.get((ed.tail, k))
                    if fs is None:
                        continue
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + rest
                    entry[1] = max(entry[1], price)
                    held = S[fs].storage
                    break  # one feeder per grid and fuel: the first in edge order
                over = np.maximum(rest - held, 0.0) if st.sources else np.zeros(self.T)
                for e in inst.in_edges[g]:  # reserve_sources: what is left, at the supply nodes that ship straight in
                    fs = inst.slot_index.get((E[e].tail, k))
                    if not over.any() or k not in E[e].K or E[e].tail not in self.supply or fs is None:
                        continue
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + over  # each on its own: a supply node's stock costs nothing to keep
                    entry[1] = max(entry[1], price)
        for fs, (target, price) in asked.items():
            out.append((("reserve_feeder", fs), fs, np.minimum(target, S[fs].storage), price))
        return out

    def add_reserve(self, where: dict[tuple, int]) -> None:
        """The soft minimum of every reserve slot (module docstring)."""
        for name, s, target, price in self.reserve_targets():
            j = self.cols.add(name)  # short: the stock missing below the target
            self.priority[j] = price
            r = self.ub.row(name, -target)  # -I - short <= -target
            self.ub.add(r, where[("I", s)], -1.0)
            self.ub.add(r, j, -1.0)

    # ----- the reserve at straits ------------------------------------------------------------------------------------
    def strait_targets(self) -> list[tuple[int, int, np.ndarray, float]]:
        """(strait, fuel, target per window week, price per unit-week) of every soft minimum on a queue.

        The straits are those a fuel reaches through another strait on some lane (the last straits before the
        terminals: fuel that waits there is past what closes upstream), and only tanker cargo, which the agent
        releases itself. The target is ``cover`` weeks of what the strait can pass on to the grids behind it: over
        its edges into a grid or a grid's terminal, the edge's capacity in the window (0 where the fuel is
        prohibited) up to the burn that node feeds, and no more than the strait's throughput. No target in a week the
        queue is charged at a war-risk rate. The price is the grids' reserve's, at the dearest grid behind the strait.
        """
        inst, st, marks = self.inst, self.settings, self.marks
        left = st.T - (st.week + self.t - 1)  # weeks of the episode after each window week
        cover = np.minimum(st.strait_weeks, np.maximum(left, 0)).astype(float)
        burn, price = defaultdict(float), defaultdict(float)  # (node, fuel) -> the burn it feeds, its reserve price
        for g in inst.grids:
            ga = inst.nodes[g].grid
            for k in ga.fuels:
                for node in [g] + self.feeders(g, k):
                    burn[(node, k)] += ga.shares[k] * ga.deliverable
                    price[(node, k)] = max(price[(node, k)], st.price * ga.voll)
        behind = {(c, k) for li, lane in enumerate(inst.lanes) for c in lane.chokepoints[1:] for k in inst.lane_K[li]}
        out = []
        for c, k, lane in self.queues:
            if lane is not None or (c, k) not in behind:
                continue
            ci = inst.chokepoint_ordinal[c]
            rate, worth = np.zeros(self.T), 0.0
            for e in inst.out_edges[c]:
                head = inst.edges[e].head
                if k in inst.edges[e].K and burn.get((head, k), 0.0) > 0:
                    rate += np.minimum(np.where(marks.prohibited[:, e, k], 0.0, marks.u[:, e]), burn[(head, k)])
                    worth = max(worth, price[(head, k)])
            calm = marks.h_queue[:, ci, k] <= inst.nodes[c].chokepoint.queue_holding[k][0] * (1.0 + 1e-9)
            target = np.where(calm, cover * np.minimum(rate, marks.kappa[:, ci, inst.commodity_pool[k]]), 0.0)
            if worth > 0 and target.any():
                out.append((c, k, target, worth))
        return out

    def add_strait_reserve(self, where: dict[tuple, int]) -> None:
        """The soft minimum on a fuel's queue at the last strait before the terminals (module docstring)."""
        for c, k, target, price in self.strait_targets():
            name = ("reserve_strait", c, k)
            j = self.cols.add(name)  # short: the queue missing below the target
            self.priority[j] = price
            r = self.ub.row(name, -target)  # -Q - short <= -target
            self.ub.add(r, where[("Q", c, k, None)], -1.0)
            self.ub.add(r, j, -1.0)

    # ----- the ramp --------------------------------------------------------------------------------------------------
    def add_ramp(self, where: dict[tuple, int]) -> None:
        """Shed base load costs ``shed_ramp * VOLL`` more for every window week still to come (module docstring)."""
        inst = self.inst
        later = (self.T - self.t).astype(float)  # the window weeks after each week
        for go, g in enumerate(inst.grids):
            self.add_price(where[("ysh", go)], self.settings.ramp * inst.nodes[g].grid.voll * later)

    # ----- what a fab's energy is worth ------------------------------------------------------------------------------
    def chip_worth(self, go: int) -> list[tuple[float, float, float]]:
        """(nominal lots per week, energy per lot, the most a chip is worth) of each fab of the grid that draws energy.

        A chip is worth the largest shortage penalty of its packaged good, or its end value if that is more.
        """
        inst, share = self.inst, self.settings.end_value
        packaged: dict[int, set[int]] = {}
        for o in inst.osats:
            for raw, pk in inst.nodes[o].osat.packages.items():
                packaged.setdefault(raw, set()).add(pk)
        out = []
        for fo in inst.grid_fabs[go]:
            fa = inst.nodes[inst.fabs[fo]].fab
            if fa.e <= 0:
                continue
            goods = packaged.get(fa.product, set()) | {fa.product}
            worth = max((d.pi * (self.T if d.backlog else 1) for d in inst.demands if d.k in goods), default=0.0)
            out.append((fa.cap0, fa.e, max(worth, max(share * inst.commodities[k].v for k in goods))))
        return out

    def add_short_premium(self, where: dict[tuple, int]) -> None:
        """Price a fuel segment's shortfall at a grid with fabs by the chips it costs when the fuel is pulsed.

        With ``pulse.Pulse`` a grid short of a share s of a fuel's weekly burn runs whole in about 1 - s of the weeks,
        so each unit short costs the grid's fabs 1 / burn of a week's chips, less the base load their energy would
        have served: ``(chips per week - VOLL * the fabs' energy per week) / burn`` per unit, at nominal rates. The
        planner's window counts the fabs as running on shed base load from its second week (it prices the shortfall
        at VOLL only), so it does not care which grid a scarce fuel goes to; ``short_premium`` times that price on the
        builder's shortfall column ("short", grid, fuel) says which grid loses most.
        """
        inst = self.inst
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = self.chip_worth(go)
            gain = sum(lots * worth for lots, _e, worth in fabs) - ga.voll * sum(lots * e for lots, e, _w in fabs)
            if ga.priority != "base_first" or gain <= 0:
                continue
            draw = sum(lots * e for lots, e, _w in fabs)  # the fabs' full draw per week
            small, others = [], []  # premium_whole: (fuel, its premium) of the small fuels; the fuels that are not
            for k in ga.fuels:
                j = where.get(("short", go, k))
                burn = ga.shares[k] * ga.deliverable
                if j is not None and burn > 0:
                    # premium_small: a segment smaller than the fabs' draw costs them a unit of energy per unit
                    # short (it cannot stop them), so a unit of it is worth the gain per unit of draw, not of burn
                    per = max(burn, draw) if self.settings.premium_small else burn
                    self.add_price(j, self.settings.short_premium * gain / per)
                    priced_small = self.settings.premium_small and burn < draw
                    if self.settings.premium_whole and priced_small:
                        small.append((k, self.settings.short_premium * gain / per))
                    elif self.settings.premium_limit:
                        self.add_free_short(go, k, j, self.settings.short_premium * gain / per, draw)
                    if not priced_small:
                        others.append(k)
            if small and others:
                self.add_whole(go, small, others, where)

    def add_whole(self, go: int, small: list[tuple[int, float]], others: list[int], where: dict[tuple, int]) -> None:
        """Take a small fuel's premium off the part of its shortfall that falls in weeks the grid is not whole.

        ``small``: the grid's fuels priced per unit of the fabs' draw (``premium_small``), each with that premium;
        ``others``: its fuels that can take all the fabs' energy. A unit of a small fuel is a unit of fab energy in
        a whole week and nothing to the fabs in any other. With the pulse a grid short of a share s of another
        fuel's burn runs whole in about 1 - s of the weeks (``add_short_premium``), so of a small fuel's burn the
        share s is needed in no whole week: its shortfall is free up to that. s is the plan's own, week by week:
        the other fuels' shortfall columns over their burn, summed (one of them short is the usual case; two short
        in different weeks leave the sum of both dark, in the same weeks the larger, and the sum is the one a
        linear program can use). The grid's power cancels from the ratio of two fuels' burns, so the row's
        coefficients are the ratios of their shares. A column ("whole", grid, fuel) earns the premium back, at most
        the small fuel's shortfall (a row) and at most that share of its burn (a row).
        """
        ga = self.inst.nodes[self.inst.grids[go]].grid
        for k, premium in small:
            j = self.cols.add(("whole", go, k))
            self.priority[j] = -premium
            r = self.ub.row(("whole", go, k))  # whole - short_k <= 0
            self.ub.add(r, j, 1.0)
            self.ub.add(r, where[("short", go, k)], -1.0)
            r = self.ub.row(("whole_dark", go, k))  # whole - sum_f (burn_k / burn_f) short_f <= 0
            self.ub.add(r, j, 1.0)
            for f in others:
                self.ub.add(r, where[("short", go, f)], -ga.shares[k] / ga.shares[f])

    # ----- what the fuel at hand and under way covers --------------------------------------------------------------
    def fuel_cover(self) -> dict[int, dict[int, float]]:
        """Grid ordinal -> fuel -> the share of the fuel's full burn covered over the next ``cover_weeks`` weeks.

        What counts: the stock at the grid and at the nodes that feed it over a zero-lead edge (its terminal), the
        lots waiting at a strait for their last edge to one of them, and the shipments under way to them, each edge's
        taken as a weekly rate (what is on the edge over its lead time). All of it is in the window's start state, so
        it is this week's observation and nothing else. 1 means the fuel is not short as far as can be seen.
        """
        if getattr(self, "_cover", None) is not None:
            return self._cover
        inst, E, weeks = self.inst, self.inst.edges, max(self.settings.cover_weeks, 1.0)
        held, rate = defaultdict(float), defaultdict(float)
        for node, k, q in inst.initial_state.stock:
            held[(node, k)] += q
        for sh in inst.initial_state.pipeline:
            rate[(E[sh.edge].head, sh.k)] += sh.qty / max(E[sh.edge].tau, 1)
        for lot in inst.initial_state.queue_lots:
            held[(E[lot.next_edge].head, lot.k)] += lot.qty
        out = {}
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            cover = {}
            for k in ga.fuels:
                nodes = [g] + self.feeders(g, k)
                burn = ga.shares[k] * float(self.marks.G_bar[0, go])
                have = sum(held[(v, k)] + weeks * rate[(v, k)] for v in nodes)
                cover[k] = 1.0 if burn <= 0 else min(1.0, have / (weeks * burn))
            out[go] = cover
        self._cover = out
        return out

    def feeders(self, g: int, k: int) -> list[int]:
        """The nodes that feed grid ``g`` fuel ``k`` over a zero-lead edge and hold a stock of it (its terminal)."""
        inst = self.inst
        return [
            inst.edges[e].tail
            for e in inst.in_edges[g]
            if k in inst.edges[e].K and inst.edges[e].tau == 0 and inst.edges[e].tail not in self.chk
            and inst.edges[e].tail not in self.supply and (inst.edges[e].tail, k) in inst.slot_index
        ]

    def add_free_short(self, go: int, k: int, j_short: int, premium: float, draw: float) -> None:
        """Take the premium off the part of fuel ``k``'s shortfall that the grid's scarcest fuel makes useless.

        The fabs of a grid run in about the share of weeks its scarcest fuel covers (``fuel_cover``, among the fuels
        whose segment is at least the fabs' draw). Another fuel delivered beyond that share powers no extra whole
        week, yet the premium priced every unit of it, and a unit of a small segment dearly, so on an edge both fuels
        share the plentiful fuel outbid the scarce one. A column ("free", grid, fuel), at most the fuel's shortfall
        (a row) and at most ``(1 - least cover)`` of its burn (a bound), earns the premium back.
        """
        inst = self.inst
        ga = inst.nodes[inst.grids[go]].grid
        cover = self.fuel_cover()[go]
        big = [f for f in ga.fuels if ga.shares[f] * ga.deliverable >= draw]
        least = min((cover[f] for f in big), default=1.0)
        spare = cover[k] > least + 1e-6  # not the scarcest fuel itself
        allowance = (1.0 - least) * ga.shares[k] * self.marks.G_bar[:, go] if spare else np.zeros(self.T)
        j = self.cols.add(("free", go, k), ub=np.maximum(allowance, 0.0))
        self.priority[j] = -premium
        r = self.ub.row(("free", go, k))  # free - short <= 0
        self.ub.add(r, j, 1.0)
        self.ub.add(r, j_short, -1.0)

    # ----- the duty the pulse gives a grid's fabs --------------------------------------------------------------------
    def duty(self, go: int) -> float:
        """The share of its capacity that the fabs of grid ``go`` can expect to run over the next weeks.

        From ``fuel_cover``: a fuel the pulse holds and delivers in whole weeks gives the fabs the share of weeks it
        covers; a fuel smaller than the fabs' draw takes a unit of their energy per unit short; any other short fuel
        (not held: no terminal, not worth a tank, or no pulse at all) leaves them what is left above the base load.
        """
        inst, st = self.inst, self.settings
        g = inst.grids[go]
        ga = inst.nodes[g].grid
        fabs = self.chip_worth(go)
        draw = sum(lots * e for lots, e, _w in fabs)
        if draw <= 0 or ga.priority != "base_first":
            return 1.0
        cover = self.fuel_cover()[go]
        packaged = defaultdict(set)
        for o in inst.osats:
            for raw, pk in inst.nodes[o].osat.packages.items():
                packaged[raw].add(pk)
        gain = 0.0  # as pulse.Pulse counts it: a whole week's chips at the shortage penalty less the base load
        for fo in inst.grid_fabs[go]:
            fa = inst.nodes[inst.fabs[fo]].fab
            if fa.e > 0:
                goods = packaged[fa.product] | {fa.product}
                gain += fa.cap0 * (max((d.pi for d in inst.demands if d.k in goods), default=0.0) - ga.voll * fa.e)
        room = max(float(self.marks.G_bar[0, go]) * sum(ga.shares.values()) - float(self.marks.y_bar[0, go]), 0.0)
        duty = 1.0
        for k in ga.fuels:
            c, burn = cover[k], ga.shares[k] * ga.deliverable
            if c >= 1.0 - 1e-9:
                continue
            if burn < draw:
                duty *= max(0.0, 1.0 - (1.0 - c) * burn / draw)
                continue
            threshold = inst.params.psi * ga.ibar[k] if k == ga.rationed else 0.0
            worth_it = st.payback <= 0 or (gain > 0 and (threshold + burn) * ga.voll <= st.payback * gain)
            if st.pulse and self.feeders(g, k) and worth_it:
                duty = min(duty, c)
            else:
                duty = min(duty, max(0.0, (room - (1.0 - c) * burn) / draw))
        return float(min(max(duty, 0.0), 1.0))

    def add_duty(self, where: dict[tuple, int]) -> None:
        """Bound the lots of a grid's fabs from the window's second week at ``duty`` times their capacity.

        The window runs a fab from its second week on at capacity (on shed base load), whatever the grid's fuel. The
        pulse runs it in the share of weeks the scarcest fuel covers. With the bound the plan counts on the chips that
        will be made, and plans wafers, the other fuels and the other fabs for them. It is a bound on the lot
        columns, not a price on shed load; the first week keeps the builder's own rule.
        """
        inst = self.inst
        later = self.t >= 2
        for go in range(len(inst.grids)):
            share = self.duty(go)
            if share >= 1.0 - 1e-6:
                continue
            for fo in inst.grid_fabs[go]:
                j = where[("p", fo)]
                ub = np.broadcast_to(np.asarray(self.cols.ub[j], dtype=float), (self.T,))
                self.cols.ub[j] = np.where(later, ub * share, ub)

    def add_stock_early(self, where: dict[tuple, int]) -> None:
        """Make chips kept at their fab a little better than free to hold, so a plan builds that stock early.

        With the end value a plan fills a fab's chip store for the credit at the window's end, and it does so with the
        window's last lots, to save holding cost. A fab whose outlets are cut (it can ship less than it can make) is
        therefore planned at its outlets' rate now and at capacity later, every week again, and never builds the stock
        that could leave when the outlets recover. The price is ``-(1 + stock_early)`` times the slot's holding cost
        per unit-week on the fab's product stock, only while the end value applies.
        """
        inst = self.inst
        for f in inst.fabs:
            s = self.slot(f, inst.nodes[f].fab.product, "fab product")
            j = where.get(("I", s))
            if j is not None:
                self.add_price(j, -(1.0 + self.settings.stock_early) * inst.stock_slots[s].holding)

    # ----- base load first in every window week ----------------------------------------------------------------------
    def tight_price(self, go: int) -> float:
        """The least price on shed base load, above VOLL, at which no plan sheds base load to power a fab of the grid.

        ``TIGHT_MARGIN`` times the most a unit of energy is worth at the grid's fabs (a lot takes e_f of it and yields
        at most one chip, ``chip_worth``), less VOLL; 0 when no fab of the grid draws energy or VOLL exceeds that.
        """
        best = max((worth / e for _lots, e, worth in self.chip_worth(go)), default=0.0)
        return max(TIGHT_MARGIN * best - self.inst.nodes[self.inst.grids[go]].grid.voll, 0.0)

    def add_base_first(self, where: dict[tuple, int]) -> None:
        """Price shed base load at grids with fabs in window weeks 2..``base_first_weeks`` as the builder does in week 1.

        The simulator serves base load before fabs at a ``base_first`` grid, every week. The builder's planning rules
        say so in the window's first week only (a price on shed base load, ``_Builder.base_first_price``), so from week
        2 on a plan may shed base load to power a fab, which never happens: the plan counts on chips from fabs that get
        no energy, sends them wafers, and does not care which grid a fuel shortage falls on. With the price in every
        week the plan runs a fab only on energy above the base load. ``base_first_price`` picks the price: "package",
        the builder's own (the instance's largest shortage penalty over the grid's smallest e_f), or "tight"
        (``tight_price``), which overprices unavoidable shedding the least. The first week keeps the builder's price.
        """
        inst, st = self.inst, self.settings
        on = ((self.t >= 2) & (self.t <= st.bf_weeks)).astype(float)
        for go, g in enumerate(inst.grids):
            if inst.nodes[g].grid.priority != "base_first":
                continue
            price = self.base_first_price(go) if st.bf_price == "package" else self.tight_price(go)
            if price > 0:
                self.add_price(where[("ysh", go)], price * on)

    # ----- the window's end ------------------------------------------------------------------------------------------
    def add_end_value(self, keys: list[tuple], ruled: frozenset) -> None:
        """Credit goods at the window's end ``end_value * v_k`` where that exceeds their salvage (module docstring).

        ``end_value_grids`` False leaves fuel held at grids and flows into grids out (they keep the salvage price):
        the grid's fuel is then the reserve's business. ``end_value_wip`` "product" values a lot in process at its
        product (the raw chip) instead of its input (the wafer).
        """
        inst, marks, share, t, T = self.inst, self.marks, self.settings.end_value, self.t, self.T
        K, S, E = inst.commodities, inst.stock_slots, inst.edges
        grids = frozenset(() if self.settings.ev_grids else inst.grids)
        fuels = frozenset(k for g in inst.grids for k in inst.nodes[g].grid.fuels)
        fuel_share = share if self.settings.ev_fuel is None else self.settings.ev_fuel  # of a good some grid burns
        last = (t == T).astype(float)
        for j, key in enumerate(keys):
            tag = key[0]
            if tag == "I":
                slot = S[key[1]]
                if slot.node in self.supply or j in ruled or slot.node in grids:
                    continue
                gain, when = (fuel_share if slot.k in fuels else share) * K[slot.k].v - slot.salvage, last
            elif tag == "x":
                e, k = key[1], key[2]
                ed = E[e]
                if ed.head in self.supply or ed.head in grids:
                    continue
                gain = (fuel_share if k in fuels else share) * K[k].v - self.nu(ed.head, k, "arrival")
                when = (t + ed.tau > T).astype(float)
                if ed.head in self.chk:  # still short of a strait: worth what the strait lets through
                    when = when * min(1.0, max(0.0, float(marks.o[T - 1, inst.chokepoint_ordinal[ed.head]])))
            elif tag == "p":
                fo = key[1]
                fa = inst.nodes[inst.fabs[fo]].fab
                nu = S[self.slot(inst.fabs[fo], fa.input, "fab input")].salvage
                worth = K[fa.product if self.settings.ev_wip == "product" else fa.input].v
                gain, when = share * worth - nu, (t + fa.tau > T) * (1.0 - marks.sigma_scr[:, fo])
            elif tag == "xi":
                oo, kp = key[1], key[2]
                o = inst.osats[oo]
                oa = inst.nodes[o].osat
                kr = next(raw for raw, packaged in oa.packages.items() if packaged == kp)
                gain = share * K[kr].v - S[self.slot(o, kr, "OSAT raw chip")].salvage
                when = (t + oa.tau > T).astype(float)
            else:
                continue
            if gain > 0:
                self.add_price(j, -gain * when)


@fixed_fp_errors
def build_lp(inst, marks, settings: Settings, *, planning_rules: bool):
    """``sbflow.oracle.lp.build_lp`` with ``Builder``: the same instance rung and the same checks (V3)."""
    inst = inst.at_digest(marks.instance_digest)
    if marks.instance_hash != inst.hash or marks.T != inst.T or marks.instance_digest != inst.content_digest:
        raise ValueError("marks were computed for another instance or horizon (V3)")
    return Builder(inst, marks, planning_rules, settings).build()


def changes(model) -> bool:
    """Whether ``model`` carries a reserve, a ramp or an end value (False: it is the benchmark's own model)."""
    return bool(model.meta.get("reserve"))


def rolled_lp(inst, obs: dict, arrays, H: int, params: dict, *, planning_rules: bool):
    """``lp_common.rolled_lp`` of week ``obs["week"]`` with the reserve, the ramp and the end value of ``params``."""
    settings = Settings.of(params, int(obs["week"]), inst.T)
    inst_r, backlog = lp_common.rolled_window(inst, obs, H)
    marks = lp_common.window_marks(inst_r, arrays, backlog)
    return build_lp(inst_r, marks, settings, planning_rules=planning_rules)
