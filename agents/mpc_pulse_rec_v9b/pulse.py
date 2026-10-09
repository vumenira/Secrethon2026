"""Hold a fuel a grid is short of at its terminal until the grid can run a whole week, then deliver it.

The simulator serves a grid's base load before its fabs (``base_first``), and a grid's nominal output barely exceeds its
base load plus what its fabs draw: a fuel segment short by a few per cent of the grid's output takes all the fabs'
energy, every week the shortage lasts. The clairvoyant plan (a linear program without that rule) sheds base load and
runs the fabs; an agent cannot. It can do the same thing over time: keep the short fuel at the terminal for some weeks
(the grid sheds that segment's whole output) and deliver a full week's burn in others (no shedding, the fabs run). The
energy burnt is the same; the base load shed rises by exactly the energy the fabs get, which is worth several times
VOLL in chips.

``Pulse.adjust`` changes the week's flows on the zero-lead edges into a grid with fabs (terminal to grid), after the
planner has chosen them, and, with ``pulse_wafers``, the wafer flows into the fabs of pulsed grids. Each week, for each
such grid:

- the fabs' draw this week is known (wafers on hand, capacity), and so is the most each fuel segment can give (the
  fuel at the grid and at its terminal, the rationing of the gas segment on last week's stock);
- if the grid can serve its base load and that draw, the week is ON: while the grid is being pulsed, each fuel the
  terminal cannot keep supplying is delivered as needed and no more (the planner would send it all); otherwise the
  planner's flows stand, raised to what the week needs;
- if not, the week is OFF. A grid not yet pulsed starts only if its fabs would get less than ``pulse_partial`` of
  their draw as things are (a mild shortage leaves them most of it every week, which a pulse does not beat). Then a
  fuel whose segment cannot run in full is held at the terminal; the rationed fuel is delivered once, to lift the
  grid's stock to the rationing threshold, when the terminal holds that charge and the whole weeks after it. Nothing is
  held when a short fuel cannot be held (it comes by pipe or from a source), fills its tank too slowly
  (``pulse_hold_weeks``), or would take too long to pay for the tank it strands when the shortage ends
  (``pulse_payback``); the terminal never fills past ``pulse_fill`` of its storage.

A fab counts toward that draw only while a lot started now can still reach a sink before the episode ends
(``pulse_tail``, ``Pulse.sale_lead``): later lots are worth nothing, and a pulse would shed base load for them. In the
last weeks of an episode the planner's flows therefore stand.

Two rules added in ``agents/ft_buffer_v3``, each off unless its knob is set:

- ``pulse_rundown``: near the last week a lot can still be sold, the rationed fuel is delivered for as many whole
  weeks as are left (not ``pulse_run``), and no fuel is held that cannot give a whole week by then.
- ``pulse_wafers`` (``Pulse.top_up_wafers``): the planner's window runs a fab from its second week on (on shed base
  load), so it sends an idle fab one week of that plan's lots and stops while those wafers sit there; a whole week then
  finds less than the fab can process. At a pulsed grid the fabs' wafers (at the fab, on their way, sent this week)
  are topped up to that many weeks of capacity over the direct edges from material nodes that are not in the fleet
  row, while the fab's chip stock is under ``pulse_wafers_room`` of its storage and its lots can still be sold.
  ``pulse_wafers_catchup`` (the target grows by the planner's own level per idle week) was tried on single episodes
  and left off.

One rule added in ``agents/ft_buffer_v4``, off unless its knob is set:

- ``pulse_small``: a fuel whose whole segment (its share of the grid's deliverable power) is smaller than the fabs'
  full draw cannot take all their energy: short of it, the fabs still get their draw less what the segment misses.
  Such a fuel never keeps a week from being ON and never makes the pulse stand down. The week is ON when every other
  fuel's segment can run in full, the fabs then get what the small segments leave, and in OFF weeks a small fuel that
  is short is held with the others, for the ON weeks where each unit of it is fab energy. Without the rule a small
  fuel that stopped arriving (and so could never be made whole) switched the pulse off for the whole grid.

Three rules added in ``agents/mpc_pulse_rec_v6``, each off unless its knob is set. They only lower flows the planner
sends, and they run after the rules above (``Pulse.adjust``):

- ``fuel_cancel`` (``Pulse.cancel_unburnable``): a dispatch of a fuel from a source straight into a grid is cut to
  what the grid can still burn. The grid's stock, what is on its way and the dispatch together never exceed the
  fuel's full-rate burn of the weeks left, so the part cut could not have been burnt whatever happens; it would have
  cost its freight, tariff and holding and earned the salvage price. The planner sends it for the window-end value
  (``reserve.py``). The rationed fuel is left out (its stock also sets next week's ration).
- ``ration_trim`` (``Pulse.trim_rationed``): the simulator loads every segment of a grid by one factor, served
  energy over available power (``Pulse.grid_week``). In a week the fabs draw less than the margin that factor is
  below 1: the segment that needs no fuel is throttled and the fuels burn in its place. The one throttle an agent has
  is the rationing rule: next week's rationed segment gives its share times this week's closing stock over the
  threshold. So in a week that sheds nothing, the rationed fuel delivered from the terminal is cut to the week's burn
  plus the stock that rations next week's segment to what next week needs: base load plus the fabs' draw from the
  wafers they will have, less what the other segments give (``ration_trim_margin`` of the fabs' full draw is left as
  room). The rest stays at the terminal for a week that would shed. Only while the grid and its terminal hold less
  than ``ration_trim_cover`` weeks of the fuel: with plenty, the saved fuel would never be needed. If next week's
  draw turns out higher than forecast, the fabs lose the difference for that week; the week is then treated as ON
  (the stock is lifted again), never as a shortage to pulse.
- ``wafers_fit`` (``Pulse.fit_wafers``): the planner's window runs an idle fab from its second week, so it keeps
  sending the wafers that plan would use, into a store that is full. In a week a grid sheds base load (its fabs get
  nothing) and is not being pulsed, the wafers sent toward each of its fabs are cut so that the store, what is on its
  way there and the week's dispatches fit the store. A pulsed grid is left alone: its fabs run in the whole weeks
  to come on the wafers gathered meanwhile, and cutting those cost more chips than the disposal it saved (-0.003 on
  Small when tried).

Three rules added in ``agents/mpc_pulse_rec_v7``, each off unless its knob is set:

- ``ration_trim_release`` (``Pulse.passing``): ``ration_trim`` forecasts next week's fab draw from the wafers the fabs
  will have, and it counted everything waiting at a strait that is not shut as arriving in time. A throttled strait
  passes on a small part of its queue, so the draw was forecast too high for as long as wafers queued there, and the
  rationed fuel was left to burn in the slack (all of the slack that weeks after a trim still had, apart from the
  episode's last weeks). With the knob, what the straits pass on this week is worked out by the simulator's own
  release rule (``sbflow.dynamics.chokepoint.release``) from the lots waiting, the shipments reaching a strait this
  week, the observed throughput by pool and the capacity of the next edges.
- ``fuel_trim`` (``Pulse.trim_fuels``): the same slack, taken off a fuel that is not rationed. A segment gives no more
  than the fuel on hand, and a zero-lead delivery arrives before the grid runs, so the terminal's delivery of the
  week sets the week's segment and no forecast is needed. In a week the grid can serve its base load and the fabs'
  whole draw, such a fuel is delivered so that it has on hand what its segment burns once the slack is taken off it:
  no more (the rest waits at the terminal, also what the segment could not burn anyway) and no less (the planner's
  window may leave a segment idle that the simulator loads). Only while grid and terminal hold less than
  ``fuel_trim_cover`` weeks of the fuel. It runs before ``ration_trim``, which reads the week's load factor.
- ``osat_fit`` (``Pulse.fit_osats``): in a week an OSAT's store of a packaged chip is full when it begins, the raw
  chips sent toward the OSAT are cut to what the planner's own flows take out of that store this week.

One rule added in ``agents/mpc_pulse_rec_v8``, off unless its knob is set (and left off in ``agent.PARAMS``: it
gains on Small and not on Full):

- ``open_hold`` (``Pulse.opening``, ``Pulse.plan_short``): the pulse starts only once a grid is short. Until then the
  grid burns a whole week's gas every week, also in the opening weeks, when its fabs often start few lots (the
  planner sends them few wafers while the lots in process at the start keep the chip stores filling). When the
  shortage comes, that gas is missing for whole weeks that would have run the fabs on the wafers gathered meanwhile.
  The planner sees the shortage coming: its window plan leaves the gas segment short weeks before the grid is. With
  the knob, the episode's first plan picks the grids whose rationed fuel's segment it leaves short by
  ``open_hold_short`` of its capacity or more over window weeks 4 to 12 (``OPEN_SPAN``); in weeks
  ``open_hold_first`` to ``open_hold_last`` of the episode that fuel stays at the terminal there, and the grid counts
  as pulsed: after the last held week the rules above find a grid without gas and a filled terminal, and deliver in
  whole weeks. Which grids: those whose rationed fuel the pulse can hold (a terminal, ``pulse_payback``) and, with
  ``open_hold_alone``, where the plan leaves no other fuel short that can take all the fabs' energy (one whose
  segment is no smaller than their draw): gas saved beside such a fuel buys no whole week. A hold ends early when
  the terminal reaches ``pulse_fill`` of its storage (the pulse's own rule then starts the whole weeks), and a hold
  that ended is not taken up again. Two options were measured and are off: ``open_hold_recheck`` (every week's plan
  decides for its own week, so a hold can also start later and ends when the plan stops expecting the shortage) and
  ``open_hold_draw`` (a week is held only if the fabs would draw less than that share of their full draw).

It reads the week's observation (stock, shipments, lots waiting at straits), the planner's memory of the network as
observed (grid power and base load, fab capacity factors, edge capacities and costs, strait throughput, tariffs,
prohibitions) and the instance; of the planner's window only, with ``open_hold``, the fuel shortfalls of its solution
(``Pulse.plan_short``; a caller that passes no plan to ``Pulse.adjust`` plays without that rule). It changes the
quantities of flows the planner already sends: on terminal-to-grid edges (``fuel_trim`` up or down to the week's
need), with ``pulse_wafers`` on material-to-fab edges (only upward), with the three rules of
``agents/mpc_pulse_rec_v6`` on source-to-grid, terminal-to-grid and wafer flows (only downward), and with ``osat_fit``
on raw-chip flows (only downward).
"""

import heapq
import math
from collections import defaultdict

from sbflow.dynamics import chokepoint
from sbflow.dynamics.state import Lot


TOL = 1e-9
# open_hold: the window weeks (from 1, both ends in) over which the plan's shortfall of a fuel is read. The first three
# show the stock at hand, which the hold itself runs down, and not what is coming; past the twelfth the forecast has
# ended most impairments
OPEN_SPAN = (4, 12)


class Pulse:
    def __init__(self, inst, params: dict) -> None:
        self.inst = inst
        self.hold_weeks = float(params["pulse_hold_weeks"])  # do not hold a fuel whose tank takes longer to fill
        self.fill = float(params["pulse_fill"])  # the share of the terminal's storage up to which fuel is held
        self.margin = float(params["pulse_margin"])  # the week's burn is covered this many times before a week is ON
        self.run = float(params["pulse_run"])  # the rationed fuel's tank is charged when it lasts this many whole weeks
        self.partial = float(params["pulse_partial"])  # a pulse starts when the fabs get less than this share as it is
        self.memory = float(params["pulse_memory"])  # a grid counts as pulsed this many weeks after its last held week
        payback = float(params["pulse_payback"])  # weeks of whole running that must pay for a stranded tank; 0: any
        psi = inst.params.psi
        chk, supply = set(inst.chokepoints), set(inst.supply_nodes)
        slots = defaultdict(list)
        for s, (e, k, _lane) in enumerate(inst.action_slots):
            slots[(e, k)].append(s)
        packaged = defaultdict(set)
        for o in inst.osats:
            for raw, pk in inst.nodes[o].osat.packages.items():
                packaged[raw].add(pk)
        self.grids = []
        small = bool(params.get("pulse_small", False))
        self.small: dict[int, set[int]] = {}  # grid ordinal -> its fuels smaller than the fabs' draw (pulse_small)
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
            if ga.priority != "base_first" or not fabs:
                continue
            gain = 0.0  # what a whole week is worth: the fabs' chips less the base load their energy would serve
            for fo in fabs:
                fa = inst.nodes[inst.fabs[fo]].fab
                goods = packaged[fa.product] | {fa.product}
                worth = max((d.pi for d in inst.demands if d.k in goods), default=0.0)
                gain += fa.cap0 * (worth - ga.voll * fa.e)
            fuels = {}
            for k in ga.fuels:
                feeders = []
                for e in inst.in_edges[g]:
                    ed = inst.edges[e]
                    tail = (ed.tail, k)
                    if k in ed.K and ed.tau == 0 and ed.tail not in chk and ed.tail not in supply:
                        if tail in inst.slot_index and slots[(e, k)]:
                            feeders.append((e, ed.tail, inst.stock_slots[inst.slot_index[tail]].storage, slots[(e, k)]))
                threshold = psi * ga.ibar[k] if k == ga.rationed else 0.0
                tank_cost = (threshold + ga.shares[k] * ga.deliverable) * ga.voll  # the energy a full tank defers
                worth_it = payback <= 0 or (gain > 0 and tank_cost <= payback * gain)
                fuels[k] = (ga.shares[k], threshold, feeders, worth_it)
            self.grids.append((go, g, fabs, fuels, sum(ga.shares.values())))
            # pulse_small: a fuel whose whole segment is smaller than the fabs' full draw cannot take all their energy,
            # so it never keeps a week from being ON and never makes the pulse stand down
            draw = sum(inst.nodes[inst.fabs[fo]].fab.cap0 * inst.nodes[inst.fabs[fo]].fab.e for fo in fabs)
            self.small[go] = {k for k in ga.fuels if ga.shares[k] * ga.deliverable < draw} if small else set()
        # pulse_tail: a fab asks for no energy once a lot started now would reach no sink before the episode ends, with
        # this many weeks to spare (the lots are worth nothing then, and a pulse sheds base load for them); None: off
        tail = params.get("pulse_tail")
        self.tail = None if tail is None else float(tail)
        self.lead = {} if tail is None else {fo: self.sale_lead(fo) for fo in range(len(inst.fabs))}
        # pulse_rundown: near the last week a lot of the grid's fabs can still be sold, the rationed fuel is delivered
        # for as many whole weeks as are left (not pulse_run), and no fuel is held that cannot give a whole week by then
        self.rundown = bool(params.get("pulse_rundown", False))
        self.last = {}  # grid ordinal -> the last week a lot started at one of its fabs can still be sold
        for go, _g, fabs, _fuels, _shares in self.grids:
            leads = [self.lead[fo] for fo in fabs if fo in self.lead and math.isfinite(self.lead[fo])]
            self.last[go] = inst.T - 1 if not leads else int(inst.T - min(leads) - (self.tail or 0.0))
        # pulse_wafers: at a pulsed grid the fabs' wafers (at the fab, on their way and sent this week) are topped up
        # to this many weeks of the fab's capacity over the direct edges from material nodes, while the fab's chip
        # stock is under pulse_wafers_room of its storage; 0: the planner's wafer flows stand
        self.wafers = float(params.get("pulse_wafers", 0.0))
        self.wafers_room = float(params.get("pulse_wafers_room", 0.5))
        # pulse_wafers_catchup: the target grows by what the planner itself keeps at hand for the fab (one week of
        # its plan) with every week the fab stays idle, up to pulse_wafers weeks of capacity, and nothing is added in
        # a week the fab ran: the fab makes up the lots the plan wanted, no more (more would only crowd its outlets)
        self.catchup = bool(params.get("pulse_wafers_catchup", False))
        self.spell: dict[int, int] = {}  # fab ordinal -> idle weeks in a row on a pulsed grid
        self.base: dict[int, float] = {}  # fab ordinal -> the planner's own wafers at hand when the idle spell began
        self.edge_slots = defaultdict(list)  # edge -> every action slot on it (they share its capacity)
        self.tail_slots = defaultdict(list)  # (tail node, commodity) -> every action slot that draws on that stock
        for s, (e, k, _lane) in enumerate(inst.action_slots):
            self.edge_slots[e].append(s)
            self.tail_slots[(inst.edges[e].tail, k)].append(s)
        fleet = {e for e, _lane, _dtau in inst.dup_items}  # edges of the fleet row: more flow there cuts other flows
        self.wafer_edges = {}  # fab ordinal -> [(nominal freight, edge, tail node, slots)] the direct ways in, cheapest first
        for fo, f in enumerate(inst.fabs):
            k = inst.nodes[f].fab.input
            ways = []
            for e in inst.in_edges[f]:
                ed = inst.edges[e]
                if k in ed.K and ed.tail not in chk and e not in fleet and slots[(e, k)] and (ed.tail, k) in inst.slot_index:
                    ways.append((ed.c0, e, ed.tail, slots[(e, k)]))
            self.wafer_edges[fo] = sorted(ways)
        self.rate: dict[tuple[int, int], float] = {}  # (grid ordinal, fuel) -> smoothed weekly arrivals at its terminal
        self.held: dict[int, int] = {}  # grid ordinal -> the last week a fuel was held for it
        # (week, grid ordinal, the rule, {good: quantity}) of the weeks it changed a flow ("osat": the OSAT's ordinal)
        self.log: list[tuple] = []
        # ----- added in agents/mpc_pulse_rec_v6 (module docstring), each off unless its knob is set -----
        # fuel_cancel: a fuel sent from a source straight into a grid is cut to what the grid can still burn
        self.cancel = bool(params.get("fuel_cancel", False))
        self.source_ways = []  # (grid ordinal, grid, fuel, its share, [(edge, slots)]) of the fuels not rationed
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            for k in ga.fuels if self.cancel else ():
                ways = [e for e in inst.in_edges[g] if k in inst.edges[e].K and inst.edges[e].tail in supply]
                ways = [(e, slots[(e, k)]) for e in ways if slots[(e, k)]]
                if ways and k != ga.rationed and (g, k) in inst.slot_index:
                    self.source_ways.append((go, g, k, ga.shares[k], ways))
        # ration_trim: in a week without shedding the rationed fuel delivered is cut to the week's burn plus the stock
        # that rations next week's segment to what next week needs, with ration_trim_margin of the fabs' full draw as
        # room, while grid and terminal hold less than ration_trim_cover weeks of the fuel's full burn
        self.trim = bool(params.get("ration_trim", False))
        self.trim_margin = float(params.get("ration_trim_margin", 0.0))
        self.trim_cover = float(params.get("ration_trim_cover", math.inf))
        self.trimmed: dict[int, int] = {}  # grid ordinal -> the last week its rationed fuel was left under threshold
        self.full_draw = {}  # grid ordinal -> the energy its fabs draw in a week at nominal capacity
        for go, _g, fabs, _fuels, _shares in self.grids:
            attrs = [inst.nodes[inst.fabs[fo]].fab for fo in fabs]
            self.full_draw[go] = sum(fa.cap0 * fa.e for fa in attrs)
        # wafers_fit: no wafers toward a fab of a grid that sheds base load beyond what the fab's store can take
        self.fit = bool(params.get("wafers_fit", False))
        # ----- added in agents/mpc_pulse_rec_v7 (module docstring), each off unless its knob is set -----
        # ration_trim_release: what a strait passes on in time for next week is worked out by the simulator's release
        # rule (ration_trim's forecast of next week's need took the whole queue of an open strait)
        self.trim_release = bool(params.get("ration_trim_release", False))
        # fuel_trim: in a week without shedding a fuel that is not rationed is delivered up to what its segment burns
        # once the week's slack is taken off it, while grid and terminal hold less than fuel_trim_cover weeks of its
        # full burn; fuel_trim_margin of the fabs' full draw is left as room
        self.fuel_trim = bool(params.get("fuel_trim", False))
        self.fuel_cover = float(params.get("fuel_trim_cover", math.inf))
        self.fuel_margin = float(params.get("fuel_trim_margin", 0.0))
        # osat_fit: an OSAT whose store of a packaged chip is full is sent no more of its raw chip than the store
        # ships out that week
        self.osat_fit = bool(params.get("osat_fit", False))
        self.stores = []  # (osat ordinal, osat, raw chip, packaged chip, the store's size, [(edge out, its slots)])
        for oi, o in enumerate(inst.osats if self.osat_fit else ()):
            for raw, pk in sorted(inst.nodes[o].osat.packages.items()):
                storage = inst.stock_slots[inst.slot_index[(o, pk)]].storage
                outlets = [(e, slots[(e, pk)]) for e in inst.out_edges[o] if slots[(e, pk)]]
                if storage:
                    self.stores.append((oi, o, raw, pk, storage, outlets))
        # ----- added in agents/mpc_pulse_rec_v8 (module docstring), off unless its knob is set -----
        # open_hold: in weeks open_hold_first..open_hold_last a grid's rationed fuel stays at the terminal where the
        # episode's first window plan leaves its segment short by open_hold_short of its capacity or more (OPEN_SPAN)
        self.open_hold = bool(params.get("open_hold", False))
        self.open_first = int(params.get("open_hold_first", 2))
        self.open_last = int(params.get("open_hold_last", 4))
        self.open_short = float(params.get("open_hold_short", 0.1))
        # open_hold_alone: not where the plan leaves another fuel short that can take all the fabs' energy
        self.open_alone = bool(params.get("open_hold_alone", True))
        # open_hold_recheck: every week's plan decides for its week; False: the first plan decides for all of them
        self.open_recheck = bool(params.get("open_hold_recheck", False))
        # open_hold_draw: only in a week the fabs would draw less than this share of their full draw; above 1: any week
        self.open_draw = float(params.get("open_hold_draw", 2.0))
        self.opened: set[int] | None = None  # the grids the first plan chose (open_hold_recheck False)
        self.open_at: dict[int, int] = {}  # grid ordinal -> the last week open_hold held its fuel

    def plan_short(self, plan) -> dict[tuple[int, int], float]:
        """What the week's window plan leaves each fuel segment short by: {(grid ordinal, fuel): share of capacity}.

        ``plan`` is the planner's window model and its solution. The builder's shortfall column of a fuel segment
        (("short", grid, fuel): what the segment gives below its share of the grid's load, for lack of the fuel)
        summed over the window weeks of ``OPEN_SPAN``, over the segment's capacity in those weeks (the bound of
        ("G", grid, fuel): its share of the grid's power as forecast). A window that ends before those weeks, or a
        model without the planning rules' shortfall columns, expects nothing.
        """
        model, x = plan
        tmpl, nc, weeks = model.meta["template"], int(model.meta["nc"]), int(model.T)
        if len(x) != weeks * nc or len(model.ub) != weeks * nc:
            raise ValueError(f"the plan is not a window of {weeks} weeks of {nc} columns")
        out = {}
        for go, _g, _fabs, fuels, _shares in self.grids:
            for k in fuels:
                short, cap = tmpl.get(("short", go, k)), tmpl.get(("G", go, k))
                if short is None or cap is None:
                    continue
                span = range(OPEN_SPAN[0] - 1, min(OPEN_SPAN[1], weeks))
                room = sum(float(model.ub[h * nc + cap]) for h in span)
                if room > 0:
                    out[(go, k)] = sum(float(x[h * nc + short]) for h in span) / room
        return out

    def opening(self, t: int, plan) -> set[int]:
        """The grids whose rationed fuel ``open_hold`` keeps at the terminal in week ``t`` (module docstring).

        A grid with fabs whose rationed fuel has a terminal and is worth a tank (``pulse_payback``), in a week of
        ``open_hold_first``..``open_hold_last``, when the plan (``plan_short``) leaves that fuel's segment short by
        ``open_hold_short`` or more and, with ``open_hold_alone``, no other fuel that can take all the fabs' energy
        (one that is not among the grid's small fuels, ``pulse_small``). With ``open_hold_recheck`` the week's own plan
        decides, and a week without a plan holds nothing; without it the first plan seen decides for every week.
        """
        if not self.open_hold or plan is None or (self.open_recheck and not self.open_first <= t <= self.open_last):
            return set()
        if not self.open_recheck and self.opened is not None:
            return self.opened if self.open_first <= t <= self.open_last else set()
        short = self.plan_short(plan)
        chosen = set()
        for go, g, _fabs, fuels, _shares in self.grids:
            r = self.inst.nodes[g].grid.rationed
            if r not in fuels or short.get((go, r), 0.0) < self.open_short:
                continue
            _share, _threshold, feeders, worth_it = fuels[r]
            others = [k for k in fuels if k != r and k not in self.small[go]]
            if not feeders or not worth_it:
                continue  # by pipe or from a source, or not worth a tank: the pulse does not hold it either
            if self.open_alone and any(short.get((go, k), 0.0) >= self.open_short for k in others):
                continue  # another fuel will stop the fabs: gas saved buys no whole week
            chosen.add(go)
        if not self.open_recheck:
            self.opened = chosen
            return chosen if self.open_first <= t <= self.open_last else set()
        return chosen

    @staticmethod
    def deliver(sent: dict, reach: list, new: float) -> None:
        """Set a fuel's delivery over ``reach`` ((slots, the most the edge carries) per edge) to ``new`` in all."""
        left = new
        for ss, q in reach:  # fill the edges in order; the lanes of an edge share its flow
            part = min(left, q)
            for i, s in enumerate(ss):
                sent[s] = part if i == 0 else 0.0
            left -= part

    def fit_osats(self, obs: dict, values: dict, sent: dict, stock: dict) -> bool:
        """Send an OSAT whose store of a packaged chip is full no more raw chips than the store ships out; True if any.

        An OSAT packages whatever raw chips it holds, up to its throughput, and the packaged chips above the store's
        size are disposed of. The planner was seen to keep sending raw chips to an OSAT whose outlets stayed cut for
        most of an episode, every one of them packaged and disposed of. In a week the store is full when it begins,
        the raw chips sent toward the OSAT (straight in or by a lane that ends there) are scaled down alike to what
        the planner's own flows take out of the store this week, each outlet within its observed capacity. What is
        not sent stays at the fab, for another OSAT or a later week.
        """
        t = int(obs["week"])
        changed = False
        for oi, o, raw, pk, storage, outlets in self.stores:
            if stock.get((o, pk), 0.0) < storage * (1 - TOL):
                continue  # the store has room: the planner's flows stand
            leaving = sum(min(sum(sent.get(s, 0.0) for s in ss), float(values["u"][e])) for e, ss in outlets)
            live = [s for s in self.toward(o, raw) if sent.get(s, 0.0) > 0]
            planned = sum(sent[s] for s in live)
            if planned <= leaving + TOL * storage:
                continue
            for s in live:
                sent[s] *= leaving / planned
            self.log.append((t, oi, "osat", {raw: planned - leaving}))
            changed = True
        return changed

    def sale_lead(self, fo: int) -> float:
        """The weeks from a lot's start at fab ``fo`` to the earliest week its chip can serve a demand (inf: never).

        The fab's lead time, the fastest way to an OSAT that packages its chip, the OSAT's lead time, and the fastest
        way on to a sink that asks for the packaged chip, at nominal lead times: goods in stock at a week's end leave
        the week after (a strait passes them on the week they arrive) and serve demand the week they arrive.
        """
        inst = self.inst
        chk = set(inst.chokepoints)

        def reach(source: int, k: int) -> dict[int, float]:
            best = {source: 0.0}
            heap = [(0.0, source)]
            while heap:
                d, u = heapq.heappop(heap)
                if d > best.get(u, math.inf):
                    continue
                for e in inst.out_edges[u]:
                    ed = inst.edges[e]
                    if k not in ed.K:
                        continue
                    nd = d + ed.tau + (0 if u in chk else 1)
                    if nd < best.get(ed.head, math.inf):
                        best[ed.head] = nd
                        heapq.heappush(heap, (nd, ed.head))
            return best

        f = inst.fabs[fo]
        fa = inst.nodes[f].fab
        to_osat = reach(f, fa.product)
        lead = math.inf
        for o in inst.osats:
            oa = inst.nodes[o].osat
            if fa.product not in oa.packages or o not in to_osat:
                continue
            pk = oa.packages[fa.product]
            to_sink = reach(o, pk)
            for d in inst.demands:
                if d.k == pk and d.dbar > 0 and d.node in to_sink:
                    lead = min(lead, fa.tau + to_osat[o] + oa.tau + to_sink[d.node])
        return lead

    def top_up_wafers(self, obs, values, prohibited, sent, stock, arriving, whole_now) -> bool:
        """Raise the week's wafer flows into the fabs of pulsed grids to ``pulse_wafers`` weeks of capacity; True if any.

        The planner's window runs a fab from its second week on (on shed base load), so it sends an idle fab only what
        one week of that plan processes and stops while those wafers sit there. A pulsed fab runs in whole weeks only,
        and then it can process a full week's capacity: it should find it. For each fab of a grid that held a fuel in
        the last ``pulse_memory`` weeks, whose lots can still be sold and whose chip stock leaves room, the wafers at
        the fab after this week's lots, on their way (shipments, lots waiting at a strait for their last edge) and sent
        this week are counted, and what is missing to the target is added on the direct edges from material nodes
        that are not in the fleet row, cheapest first, within each edge's capacity and its tail's stock.
        """
        inst, t = self.inst, int(obs["week"])
        coming: dict[tuple[int, int], float] = defaultdict(float)  # (node, commodity) -> on its way, after this week
        pp = obs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            if int(week) > t:
                coming[(inst.edges[int(e)].head, int(k))] += float(q)
        ql = obs.get("queue_lots") or {"k": [], "qty": [], "next_edge": []}
        for k, q, nxt in zip(ql["k"], ql["qty"], ql["next_edge"]):
            coming[(inst.edges[int(nxt)].head, int(k))] += float(q)
        drawn: dict[tuple[int, int], float] = {}  # (tail node, commodity) -> sent from that stock this week, so far
        any_change = False
        for go, _g, fabs, _fuels, _shares in self.grids:
            if t - self.held.get(go, -10**9) > self.memory:
                for fo in fabs:
                    self.spell.pop(fo, None)
                continue  # not a pulsed grid: the planner's belief that its fabs run every week holds
            moved = {}
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                k = fa.input
                R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                if R <= 0 or not self.wafer_edges[fo]:
                    continue
                cap = alpha * R * fa.cap0
                on_hand = stock.get((f, k), 0.0) + arriving.get((f, k), 0.0)
                ran = bool(whole_now.get(go))
                left = on_hand - (min(cap, on_hand) if ran else 0.0)  # after this week's lots
                planned = sum(sent.get(s, 0.0) for s in self.tail_slots_into(f, k))
                have = left + coming.get((f, k), 0.0) + planned
                storage = inst.stock_slots[inst.slot_index[(f, k)]].storage
                goal = min(self.wafers * cap, self.fill * storage if storage else math.inf)
                if self.catchup:  # the lots the planner wanted and power denied, not more
                    if ran:
                        self.spell[fo] = 0
                        continue  # the fab ran: the planner supplies what it runs next
                    weeks = self.spell.get(fo, 0) + 1
                    self.spell[fo] = weeks
                    if weeks == 1:
                        self.base[fo] = have  # what the planner keeps at hand: one week of its own plan for the fab
                    goal = min(goal, self.base.get(fo, 0.0) * (weeks + 1))
                if self.tail is not None and t + 1 + self.lead[fo] + self.tail > inst.T:
                    continue  # wafers sent now arrive too late for a lot that can be sold
                chips = inst.stock_slots[inst.slot_index[(f, fa.product)]]
                if chips.storage and stock.get((f, fa.product), 0.0) > self.wafers_room * chips.storage:
                    continue  # its chips are not leaving: more lots would fill the store
                need = goal - have
                added = 0.0
                for _cost, e, tail, ss in self.wafer_edges[fo]:
                    if need <= TOL * max(cap, 1.0):
                        break
                    live = [s for s in ss if s in sent]
                    if not live or bool(prohibited[e, k]):
                        continue
                    key = (tail, k)
                    if key not in drawn:
                        drawn[key] = sum(sent.get(s, 0.0) for s in self.tail_slots[key])
                    spare = min(float(values["u"][e]) - sum(sent.get(s, 0.0) for s in self.edge_slots[e]),
                                stock.get(key, 0.0) - drawn[key])
                    add = min(need, spare)
                    if add <= TOL * max(cap, 1.0):
                        continue
                    sent[live[0]] += add
                    drawn[key] += add
                    need -= add
                    added += add
                if added > 0:
                    moved[k] = moved.get(k, 0.0) + added
            if moved:
                any_change = True
                self.log.append((t, go, "waf", moved))
        return any_change

    def tail_slots_into(self, node: int, k: int) -> list[int]:
        """The action slots that send commodity ``k`` straight into ``node`` (cached)."""
        cache = self.__dict__.setdefault("_into", {})
        if (node, k) not in cache:
            cache[(node, k)] = [s for s, (e, kk, _lane) in enumerate(self.inst.action_slots) if kk == k and self.inst.edges[e].head == node]
        return cache[(node, k)]

    # ----- the rules of agents/mpc_pulse_rec_v6 ---------------------------------------------------------------------
    def slots_into(self, node: int, k: int, tau: int) -> list[int]:
        """The action slots that send commodity ``k`` straight into ``node`` over an edge of lead time ``tau``."""
        return [s for s in self.tail_slots_into(node, k) if self.inst.edges[self.inst.action_slots[s][0]].tau == tau]

    def grid_week(self, go: int, g: int, fabs, fuels, shares, values, stock, arriving, sent) -> dict:
        """Week t at one grid as the simulator will run it, from the observation and the flows about to be sent.

        The simulator's step 7 (``dynamics/sim.py``, ``production.allocate_energy``): a fuel's segment can give its
        share of the grid's power, for the rationed fuel times last week's closing stock over the rationing threshold
        (at most 1), and never more than the fuel on hand after the week's arrivals; the segment that needs no fuel
        gives its share. Base load is served first and the fabs get what is left, up to their draw (energy per lot
        times the lots that capacity and the wafers on hand allow). Every segment is then loaded by one factor, the
        energy served over the power available, and each fuel burns its available output times that factor.
        """
        inst = self.inst
        G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
        null = G * (shares - sum(fuel[0] for fuel in fuels.values()))  # the segment that needs no fuel
        seg = {}  # fuel -> (the most its segment can give, the fuel on hand, what it has available)
        for k, (share, threshold, _feeders, _worth_it) in fuels.items():
            here = stock.get((g, k), 0.0)
            have = here + arriving.get((g, k), 0.0) + sum(sent.get(s, 0.0) for s in self.slots_into(g, k, 0))
            limit = share * G * (min(1.0, here / threshold) if threshold > 0 else 1.0)
            seg[k] = (limit, have, min(limit, have))
        avail = null + sum(av for _limit, _have, av in seg.values())
        fab = {}  # fab ordinal -> (wafers on hand, the lots they and capacity allow, the energy those lots draw)
        for fo in fabs:
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
            wafers = stock.get((f, fa.input), 0.0) + arriving.get((f, fa.input), 0.0)
            lots = min(alpha * R * fa.cap0, wafers) if R > 0 else 0.0
            fab[fo] = (wafers, lots, fa.e * lots / R if R > 0 else 0.0)
        draw = sum(energy for _wafers, _lots, energy in fab.values())
        served = min(y, avail)
        rho = min(1.0, (avail - served) / draw) if draw > 0 else 0.0  # the share of their draw the fabs get
        load = (served + rho * draw) / avail if avail > 0 else 0.0
        return {"G": G, "y": y, "null": null, "seg": seg, "served": served, "fab": fab, "rho": rho, "load": load}

    def cancel_unburnable(self, obs: dict, values: dict, sent: dict, stock: dict) -> bool:
        """Cut the fuel sent from a source straight into a grid to what the grid can still burn; True if any was cut.

        A fuel's segment burns at most its share of the grid's power a week. Stock at the grid, shipments and lots
        under way to it and the week's other flows into it are burnt first, so of a dispatch only what fits under the
        full-rate burn of the weeks left (and of the weeks from its arrival) can ever be burnt. The rest would sit at
        the grid to the end: it is cut where that saves money, which is where the freight and tariff observed this
        week plus the holding cost from its arrival to the end are no less than the salvage price it would earn. At a
        source the fuel costs nothing to leave (its stock is refilled to a cap every week).
        """
        inst, t = self.inst, int(obs["week"])
        under_way: dict[tuple[int, int], float] = defaultdict(float)
        pp = obs["pipeline"]
        for e, k, q in zip(pp["edge"], pp["k"], pp["qty"]):
            under_way[(inst.edges[int(e)].head, int(k))] += float(q)
        ql = obs.get("queue_lots") or {"k": [], "qty": [], "next_edge": []}
        for k, q, nxt in zip(ql["k"], ql["qty"], ql["next_edge"]):
            under_way[(inst.edges[int(nxt)].head, int(k))] += float(q)
        moved = {}
        for go, g, k, share, ways in self.source_ways:
            if (g, k) not in stock:
                continue  # not observed this week: the planner's flows stand
            slot = inst.stock_slots[inst.slot_index[(g, k)]]
            burn = share * max(inst.nodes[g].grid.deliverable, float(values["grid_G"][go]))
            mine = {s for _e, ss in ways for s in ss}
            others = sum(sent.get(s, 0.0) for s in self.tail_slots_into(g, k) if s not in mine)
            room = burn * (inst.T - t + 1) - stock[(g, k)] - under_way[(g, k)] - others
            late: dict[int, float] = {}  # lead time -> what the weeks from an arrival that late can still burn
            for e, ss in ways:
                ed = inst.edges[e]
                weeks = max(inst.T - t - ed.tau, 0)  # the weeks a unit sent now would sit at the grid
                cost = float(values["c"][e]) + float(values["tariff"][e, k]) * inst.commodities[k].v
                saves = cost + slot.holding * weeks >= slot.salvage
                late.setdefault(ed.tau, burn * max(inst.T - t - ed.tau + 1, 0))
                for s in ss:
                    q = sent.get(s, 0.0)
                    keep = min(q, max(min(room, late[ed.tau]), 0.0)) if saves else q
                    room -= keep
                    late[ed.tau] -= keep
                    if q - keep > TOL * max(burn, 1.0):
                        sent[s] = keep
                        moved[(go, k)] = moved.get((go, k), 0.0) + q - keep
        for (go, k), q in moved.items():
            self.log.append((t, go, "cancel", {k: q}))
        return bool(moved)

    def passing(self, obs: dict, values: dict, prohibited, action: dict) -> dict:
        """What the straits pass on this week over an edge one week long: {(the edge's head, commodity): quantity}.

        The simulator's release (``sbflow.dynamics.chokepoint.release``: the action's overrides first, then arrival
        cohorts in order, each within the next edge's capacity and the throughput left in its pool) on the lots
        waiting at the straits and the shipments that reach one this week, with capacities, throughput and
        prohibitions as observed. It leaves out the fleet's slack, which can only lower a release.
        """
        inst, t = self.inst, int(obs["week"])
        lots = []
        ql = obs.get("queue_lots") or {"qty": []}
        fields = ("lot_id", "chokepoint", "k", "lane", "next_edge", "dispatch_week", "entry_edge", "arrival_week")
        for i, q in enumerate(ql["qty"]):
            ident, c, k, lane, onward, sent_in, entry, arrived = (int(ql[f][i]) for f in fields)
            lots.append(Lot(ident, c, k, float(q), lane, onward, sent_in, entry, arrived))
        pp = obs["pipeline"]
        for e, k, lane, q, week in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], pp["arrival_week"]):
            edge = inst.edges[int(e)]
            if int(week) == t and edge.head in inst.chokepoint_ordinal and lane is not None:  # it joins the book now
                onward = inst.lane_through[(int(lane), edge.head)][1]
                new = Lot(-1 - len(lots), edge.head, int(k), float(q), int(lane), onward, t - edge.tau, int(e), t)
                lots.append(new)
        asked = action.get("overrides") or {"slot": [], "qty": []}
        held = action.get("hold") or {"chokepoint": [], "k": []}
        overrides = {int(s): float(q) for s, q in zip(asked["slot"], asked["qty"])}
        holds = frozenset((int(c), int(k)) for c, k in zip(held["chokepoint"], held["k"]))
        capacity, throughput = [float(u) for u in values["u"]], [[float(x) for x in row] for row in values["kappa"]]
        out: dict[tuple[int, int], float] = defaultdict(float)
        for r in chokepoint.release(inst, lots, capacity, throughput, prohibited, overrides, holds):
            if inst.edges[r.edge].tau == 1:
                out[(inst.edges[r.edge].head, r.k)] += r.qty
        return out

    def trim_rationed(self, obs, values, prohibited, sent: dict, stock: dict, arriving, weeks, passing=None) -> bool:
        """Cut the rationed fuel delivered in a week without shedding to what this week and next need; True if any.

        ``weeks`` holds ``grid_week`` of every grid for the flows as they stand. Next week's rationed segment gives
        its share of the grid's power times this week's closing stock over the threshold, so the closing stock
        ``threshold * needed / share of power`` makes it give exactly ``needed``: next week's base load and fab draw
        less what the other segments will give, at this week's grid power and capacity factors. The fabs' draw is
        taken from the wafers they will have: those left after this week's lots, the shipments due next week, what is
        sent this week over an edge one week long and whatever a strait may pass on in time (all it holds if it is not
        shut; with ``passing``, the result of ``Pulse.passing``, what its release rule lets through); the other fuels
        from their stock after this week's burn, next week's arrivals and what the terminal will hold when this week
        ends (a node sends only what it held when the week began). The delivery that leaves this week's burn plus that
        stock on hand replaces a larger one (never one that would leave this week's own segment short, and never less
        than the terminal has to pass on to stay under ``pulse_fill``).
        """
        inst, t = self.inst, int(obs["week"])
        chk = set(inst.chokepoints)
        soon: dict[tuple[int, int], float] = defaultdict(float)  # (node, commodity) -> what may arrive next week
        pp = obs["pipeline"]
        for e, k, lane, q, week in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], pp["arrival_week"]):
            head = inst.edges[int(e)].head
            if int(week) == t + 1:
                soon[(head, int(k))] += float(q)
            elif passing is None and int(week) == t and head in chk and lane is not None:  # at a strait this week
                onward = inst.edges[inst.lane_through[(int(lane), head)][1]]  # it may pass it on
                if onward.tau == 1 and float(values["open"][inst.chokepoint_ordinal[head]]) > 0:
                    soon[(onward.head, int(k))] += float(q)
        ql = obs.get("queue_lots") or {"chokepoint": [], "k": [], "qty": [], "next_edge": []}
        for c, k, q, nxt in zip(ql["chokepoint"], ql["k"], ql["qty"], ql["next_edge"]) if passing is None else ():
            onward = inst.edges[int(nxt)]
            if onward.tau == 1 and float(values["open"][inst.chokepoint_ordinal[int(c)]]) > 0:
                soon[(onward.head, int(k))] += float(q)
        for key, q in (passing or {}).items():  # ration_trim_release: what the straits' release rule passes on
            soon[key] += q
        changed = False
        for go, g, fabs, fuels, _shares in self.grids:
            r = inst.nodes[g].grid.rationed
            if r not in fuels or (g, r) not in stock:
                continue
            wk = weeks[go]
            share, threshold, feeders, _worth_it = fuels[r]
            cap, here = share * wk["G"], stock[(g, r)]
            limit, have, av = wk["seg"][r]
            if threshold <= 0 or cap <= 0 or wk["null"] <= 0:
                continue  # no rationing rule to throttle with, or no segment that runs without fuel
            if wk["served"] < wk["y"] * (1 - TOL) or have < limit * (1 - TOL):
                continue  # the week sheds base load, or the fuel on hand already limits the segment
            tank = inflow = room = 0.0
            reach = []  # (slots, the most the edge can carry of the fuel this week), as in adjust
            for e, tail, storage, ss in feeders:
                live = [s for s in ss if s in sent]
                held = stock.get((tail, r), 0.0)
                tank += held
                inflow += arriving.get((tail, r), 0.0)
                room += storage
                if live and not bool(prohibited[e, r]):
                    reach.append((live, min(held, float(values["u"][e]))))
            if not reach or here + arriving.get((g, r), 0.0) + tank + inflow > self.trim_cover * cap:
                continue  # nothing to hold it back at, or plenty of the fuel: none saved would be needed
            need = wk["y"] + self.trim_margin * self.full_draw[go]  # next week's base load and fab draw
            for fo, (wafers, lots, _energy) in wk["fab"].items():
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                if R > 0:
                    coming = soon[(f, fa.input)] + sum(sent.get(s, 0.0) for s in self.slots_into(f, fa.input, 1))
                    need += fa.e * min(alpha * R * fa.cap0, wafers - wk["rho"] * lots + coming) / R
            others = wk["null"]  # what the other segments will give next week
            for k, (other_share, _threshold, ways, _w) in fuels.items():
                if k == r:
                    continue
                _limit, on_hand, gives = wk["seg"][k]
                more = soon[(g, k)] + sum(sent.get(s, 0.0) for s in self.slots_into(g, k, 1))
                for e, tail, _storage, ss in ways:
                    if any(s in sent for s in ss) and not bool(prohibited[e, k]):
                        left = stock.get((tail, k), 0.0) - sum(sent.get(s, 0.0) for s in self.tail_slots[(tail, k)])
                        more += min(max(left + arriving.get((tail, k), 0.0), 0.0), float(values["u"][e]))
                others += min(other_share * wk["G"], on_hand - gives * wk["load"] + more)
            needed = need - others  # what the rationed segment has to give next week
            if needed >= cap * (1 - TOL):
                continue  # all of it: nothing to trim
            close = threshold * max(needed, 0.0) / cap  # the closing stock that rations the segment to that
            new = max(close + av * wk["load"], limit) - here - arriving.get((g, r), 0.0)
            new = max(new, min(tank + inflow - self.fill * room, sum(q for _ss, q in reach)), 0.0)
            planned = sum(sent[s] for ss, _q in reach for s in ss)
            if new >= planned - TOL * max(cap, 1.0):
                continue  # the flows as they stand deliver no more than that
            left = new
            for ss, q in reach:  # fill the edges in order; the lanes of an edge share its flow
                part = min(left, q)
                for i, s in enumerate(ss):
                    sent[s] = part if i == 0 else 0.0
                left -= part
            self.trimmed[go] = t
            self.log.append((t, go, "trim", {r: new}))
            changed = True
        return changed

    def trim_fuels(self, obs: dict, values: dict, prohibited, sent: dict, stock: dict, arriving, weeks) -> set:
        """Deliver a fuel that is not rationed as the week needs it and no more; the grids whose flows it changed.

        A segment gives no more than the fuel on hand after the week's arrivals, and a delivery over a zero-lead edge
        arrives before the grid runs: what the terminal delivers this week limits this week's segment (no forecast is
        involved, unlike the rationed fuel, whose limit is last week's closing stock). ``weeks`` holds ``grid_week``
        of every grid for the flows as they stand. In a week the grid can serve its base load and the fabs' whole draw
        with these fuels delivered in full, the slack is what the power then available exceeds that need by, and the
        delivery of each such fuel is set so that the fuel on hand is its segment's output less the slack (less
        ``fuel_trim_margin`` of the fabs' full draw): the load factor is 1, the segment that needs no fuel runs in
        full, and the fuel not delivered waits at the terminal. So does a delivery beyond what the segment can burn
        this week, which keeps the grid's stock from blunting the cut in a later week; and a delivery the planner left
        under the week's need is raised to it (its window is free to leave a segment idle that the simulator loads).
        Never below what is already at the grid, never less than the terminal has to pass on to stay under
        ``pulse_fill``, and only while grid and terminal hold less than ``fuel_trim_cover`` weeks of the fuel's full
        burn (a plentiful fuel is left to burn: cutting it burns more of the others). A week that cannot be made
        whole is left as it is: there the pulse has decided what is held. It runs before ``trim_rationed``, which
        reads the week's load factor off the flows this rule leaves.
        """
        inst, t = self.inst, int(obs["week"])
        hit = set()
        for go, g, _fabs, fuels, _shares in self.grids:
            wk = weeks[go]
            r = inst.nodes[g].grid.rationed
            given = wk["null"]  # what the segments this rule leaves alone give this week
            # (fuel, its segment's cap, on hand without the terminal, the most it can give, floor, planned, reach)
            mine = []
            for k, (_share, _threshold, feeders, _worth_it) in fuels.items():
                limit, have, av = wk["seg"][k]
                tank = inflow = room = 0.0
                reach = []  # (slots, the most the edge can carry of the fuel this week beside its other flows)
                for e, tail, storage, ss in feeders:
                    live = [s for s in ss if s in sent]
                    held = stock.get((tail, k), 0.0)
                    tank += held
                    inflow += arriving.get((tail, k), 0.0)
                    room += storage
                    if live and not bool(prohibited[e, k]):
                        others = sum(sent.get(s, 0.0) for s in self.edge_slots[e] if s not in live)
                        reach.append((live, min(held, max(float(values["u"][e]) - others, 0.0))))
                plenty = stock.get((g, k), 0.0) + arriving.get((g, k), 0.0) + tank + inflow > self.fuel_cover * limit
                if k == r or (g, k) not in stock or not reach or limit <= 0 or plenty:
                    given += av  # rationed, not observed, nothing to hold it back at, or plenty of the fuel
                    continue
                planned = sum(sent[s] for ss, _q in reach for s in ss)
                feed = sum(q for _ss, q in reach)
                floor = min(max(tank + inflow - self.fill * room, 0.0), feed)  # what the terminal has to pass on
                mine.append((k, limit, have - planned, min(limit, have - planned + feed), floor, planned, reach))
            want = wk["y"] + sum(energy for _wafers, _lots, energy in wk["fab"].values())
            slack = given + sum(most for _k, _limit, _base, most, _floor, _planned, _reach in mine) - want
            if not mine or slack < -TOL * wk["G"]:
                continue  # no fuel to set, or the week cannot serve base load and the fabs' draw: the flows stand
            slack -= self.fuel_margin * self.full_draw[go]
            moved = {}
            for k, limit, base, most, floor, planned, reach in mine:
                new = max(most - min(max(slack, 0.0), most) - base, floor, 0.0)
                slack -= most - min(limit, base + new)  # what this fuel's segment gives up
                if abs(new - planned) <= TOL * max(limit, 1.0):
                    continue
                left = new
                for ss, q in reach:  # fill the edges in order; the lanes of an edge share its flow
                    part = min(left, q)
                    for i, s in enumerate(ss):
                        sent[s] = part if i == 0 else 0.0
                    left -= part
                moved[k] = new
            if moved:
                self.log.append((t, go, "fuel", moved))
                hit.add(go)
        return hit

    def toward(self, node: int, k: int) -> list[int]:
        """The action slots whose goods of commodity ``k`` end at ``node``: straight in, or by a lane ending there."""
        cache = self.__dict__.setdefault("_toward", {})
        if (node, k) not in cache:
            inst = self.inst
            cache[(node, k)] = [
                s for s, (e, kk, lane) in enumerate(inst.action_slots)
                if kk == k and inst.edges[e if lane is None else inst.lanes[lane].edges[-1]].head == node
            ]
        return cache[(node, k)]

    def fit_wafers(self, obs: dict, sent: dict, stock: dict, arriving, weeks: dict) -> bool:
        """Cut the wafers sent toward the fabs of a grid that sheds base load to what their stores take; True if any.

        Such a fab starts no lot this week, and the planner's window still runs it from its second week, so the plan
        sends what that week would use: with the store full, all of it is disposed of on arrival. Counted against the
        store: the wafers at the fab, the shipments under way and the lots waiting at a strait that end there, and the
        week's dispatches toward it, which are scaled down alike by what does not fit. Only at a grid that is not
        being pulsed (no fuel held for it in the last ``pulse_memory`` weeks): there the fabs stay idle until the
        shortage ends, while a pulsed grid's fabs need the wafers gathered in its OFF weeks.
        """
        inst, t = self.inst, int(obs["week"])
        bound: dict[tuple[int, int], float] = defaultdict(float)  # (node, commodity) -> under way there after this week
        pp = obs["pipeline"]
        for e, k, lane, q, week in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], pp["arrival_week"]):
            end = inst.edges[int(e) if lane is None else inst.lanes[int(lane)].edges[-1]].head
            if int(week) > t or end != inst.edges[int(e)].head:
                bound[(end, int(k))] += float(q)
        ql = obs.get("queue_lots") or {"k": [], "qty": [], "lane": [], "next_edge": []}
        for k, q, lane, nxt in zip(ql["k"], ql["qty"], ql["lane"], ql["next_edge"]):
            end = inst.edges[int(nxt) if lane is None else inst.lanes[int(lane)].edges[-1]].head
            bound[(end, int(k))] += float(q)
        changed = False
        for go, _g, fabs, _fuels, _shares in self.grids:
            if weeks[go]["served"] >= weeks[go]["y"] * (1 - TOL):
                continue  # base load is served: the fabs get what is left, and the planner's wafers stand
            if t - self.held.get(go, -10**9) <= self.memory:
                continue  # a pulsed grid: its fabs run in the whole weeks to come, on the wafers gathered meanwhile
            moved = {}
            for fo in fabs:
                f = inst.fabs[fo]
                k = inst.nodes[f].fab.input
                storage = inst.stock_slots[inst.slot_index[(f, k)]].storage
                if (f, k) not in stock or not storage:
                    continue
                live = [s for s in self.toward(f, k) if sent.get(s, 0.0) > 0]
                planned = sum(sent[s] for s in live)
                excess = stock[(f, k)] + arriving.get((f, k), 0.0) + bound[(f, k)] + planned - storage
                if planned <= 0 or excess <= TOL * storage:
                    continue
                keep = max(1.0 - excess / planned, 0.0)
                for s in live:
                    sent[s] *= keep
                moved[k] = moved.get(k, 0.0) + planned * (1.0 - keep)
            if moved:
                changed = True
                self.log.append((t, go, "fit", moved))
        return changed

    def adjust(self, obs: dict, values: dict, prohibited, action: dict, plan=None) -> dict:
        """The planner's wire action of the week with the flows into pulsed grids changed (module docstring).

        ``plan``: the planner's window model of the week and its solution, read by ``open_hold`` alone; None (a
        caller that has none, or a week the planner fell back): the week is played without that rule.
        """
        inst, t = self.inst, int(obs["week"])
        last = t >= inst.T  # the last week: nothing to hold fuel for
        if not action.get("flows") or (last and not self.cancel):
            return action
        opening = set() if last else self.opening(t, plan)  # open_hold: the grids whose rationed fuel waits this week
        stock = {(int(n), int(k)): float(q) for n, k, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
        arriving: dict[tuple[int, int], float] = defaultdict(float)
        pp = obs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            if int(week) == t:
                arriving[(inst.edges[int(e)].head, int(k))] += float(q)
        sent = dict(zip(action["flows"]["slot"], action["flows"]["qty"]))
        changed = False
        whole_now = {}  # grid ordinal -> whether the grid runs a whole week (its fabs get their draw)
        for go, g, fabs, fuels, shares in () if last else self.grids:
            G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
            want = 0.0  # the energy the grid's fabs ask for this week
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                if self.tail is not None and t + self.lead[fo] + self.tail > inst.T:
                    continue  # a lot started now reaches no sink before the episode ends
                if R > 0:
                    wafers = stock.get((f, fa.input), 0.0) + arriving.get((f, fa.input), 0.0)
                    want += fa.e * min(alpha * R * fa.cap0, wafers) / R
            target = min(want, G * shares - y)
            state, unheld = {}, set()  # unheld: fuels not worth a tank (pulse_payback)
            able = G * (shares - sum(fuel[0] for fuel in fuels.values()))  # the unmodelled segment
            for k, (share, threshold, feeders, worth_it) in fuels.items():
                here = stock.get((g, k), 0.0)
                ration = min(1.0, here / threshold) if threshold > 0 else 1.0
                on_hand = here + arriving.get((g, k), 0.0)
                tank = inflow = room = 0.0
                reach = []  # (slots, the most the edge can carry of the fuel this week)
                for e, tail, storage, ss in feeders:
                    live = [s for s in ss if s in sent]
                    held = stock.get((tail, k), 0.0)
                    tank += held
                    inflow += arriving.get((tail, k), 0.0)
                    room += storage
                    if live and not bool(prohibited[e, k]):
                        reach.append((live, min(held, float(values["u"][e]))))
                feed = sum(q for _ss, q in reach)
                key = (go, k)
                self.rate[key] = inflow if key not in self.rate else 0.7 * self.rate[key] + 0.3 * inflow
                state[k] = (share * G, threshold, ration, on_hand, tank, inflow, room, reach, feed)
                able += min(share * G * ration, on_hand + feed)
                if not worth_it:
                    unheld.add(k)
            small = self.small[go]
            lost = 0.0  # what the small fuels' segments miss this week whatever is delivered: the fabs lose that much
            for k in small:
                burn, _thr, ration, on_hand, _tank, _inflow, _room, _reach, feed = state[k]
                lost += max(burn - min(burn * ration, on_hand + feed), 0.0)
            target = min(target, G * shares - y - lost)  # the most the fabs can get this week
            if go in opening and self.open_at.get(go, t - 1) == t - 1:
                # open_hold: the plan expects the rationed fuel to run short, so it waits at the terminal for the
                # whole weeks to come (whatever the fabs ask for this week), and the grid counts as pulsed. Only
                # while the terminal has room (a full one starts the whole weeks, as in the pulse's own rule), and
                # in weeks in a row: a hold that ended is not taken up again (the grid's stock would be burnt twice)
                r = inst.nodes[g].grid.rationed
                burn, _thr, ration, on_hand, tank, inflow, room, reach, feed = state[r]
                if reach and tank + inflow < self.fill * room and want < self.open_draw * self.full_draw[go]:
                    gives = able - min(burn * ration, on_hand + feed) + min(burn * ration, on_hand)
                    whole_now[go] = target > TOL * max(y, 1.0) and gives >= y + target - TOL * G
                    self.held[go] = self.open_at[go] = t
                    if sum(sent[s] for ss, _q in reach for s in ss) > TOL * max(burn, 1.0):
                        self.deliver(sent, reach, 0.0)
                        changed = True
                    self.log.append((t, go, "open", {r: 0.0}))
                    continue  # the other fuels' flows stand, as in a week the pulse holds a fuel
            if target <= TOL * max(y, 1.0):
                continue  # no fab to power this week: the planner's flows stand
            whole = able >= y + target - TOL * G
            if not whole and self.trimmed.get(go) == t - 1:
                # ration_trim left the rationed fuel's stock under the threshold last week and this week needs more
                # than was forecast: only that segment is short, by what was misjudged. The week is treated as ON
                # (its flows lift the stock again), never as a shortage to pulse
                burn, _thr, ration, on_hand, _tank, _inflow, _room, _reach, feed = state[inst.nodes[g].grid.rationed]
                whole = able + min(burn, on_hand + feed) - min(burn * ration, on_hand + feed) >= y + target - TOL * G
                if whole:
                    self.log.append((t, go, "miss", {}))
            whole_now[go] = whole
            pulsed = t - self.held.get(go, -10**9) <= self.memory
            ahead = max(self.last[go] - t, 0) if self.rundown else math.inf  # sellable weeks after this one
            short = set()  # the fuels whose segment cannot run in full this week
            if not whole:
                if not pulsed and able - y >= self.partial * target:
                    continue  # the fabs get most of their draw every week as it is
                fixable = True
                waited = False  # a fuel that can take all the fabs' energy is short: the week waits for it
                for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    if min(burn * ration, on_hand + feed) >= burn * (1 - TOL):
                        continue
                    if k in small:
                        if reach and k not in unheld:
                            short.add(k)  # kept for the next ON week, where each unit of it is fab energy
                        continue
                    short.add(k)
                    waited = True
                    lack = burn + threshold - tank - on_hand  # what the tank still has to gather for a whole week
                    wait = min(self.hold_weeks, ahead)  # the weeks the tank may take to fill
                    if not reach or k in unheld or lack > wait * max(inflow, self.rate[(go, k)]):
                        fixable = False  # by pipe or from a source, not worth a tank, or a tank that fills too slowly
                if not fixable or not waited:
                    continue  # the grid cannot be made whole by waiting: the planner's flows stand
                self.held[go] = t
            moved = {}
            for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                if not reach:
                    continue
                planned = sum(sent[s] for ss, _q in reach for s in ss)
                if whole:
                    need = min(max(burn * self.margin + threshold - on_hand, 0.0), feed)
                    lasting = tank + inflow - need >= burn  # the terminal covers next week too
                    new = need if pulsed and not lasting and k not in unheld else max(planned, need)
                else:
                    if k not in short:
                        continue  # this segment can run in full: the planner's flow stands
                    new = 0.0
                    if threshold > 0 and ration < 1:  # lift the grid's stock to the rationing threshold in one week
                        charge = max(threshold - on_hand + burn * ration, 0.0)
                        rate = min(max(inflow, self.rate[(go, k)]), burn)
                        run = min(self.run, max(inst.T - t, 1))  # the whole weeks the tank must then last
                        if self.rundown:
                            run = min(run, max(ahead, 1))  # no more than can still be sold after the charge week
                        full = tank + inflow >= self.fill * room  # the tank cannot hold more: start whatever it gives
                        if charge <= feed and tank - charge + inflow >= burn:
                            if full or tank - charge + inflow >= burn + (run - 1) * (burn - rate):
                                new = charge
                    new = max(new, min(tank + inflow - self.fill * room, feed))  # never past the terminal's storage
                    new = max(new, 0.0)
                if abs(new - planned) <= TOL * max(burn, 1.0):
                    continue
                left = new
                for ss, q in reach:  # fill the edges in order; the lanes of an edge share its flow
                    part = min(left, q)
                    for i, s in enumerate(ss):
                        sent[s] = part if i == 0 else 0.0
                    left -= part
                moved[k] = new
            if moved:
                changed = True
                self.log.append((t, go, "on" if whole else "off", moved))
        top_up = not last and self.wafers > 0
        if top_up and self.top_up_wafers(obs, values, prohibited, sent, stock, arriving, whole_now):
            changed = True
        if not last and (self.fit or self.trim or self.fuel_trim):  # the cuts, on the flows as they stand
            weeks = {
                go: self.grid_week(go, g, fabs, fuels, shares, values, stock, arriving, sent)
                for go, g, fabs, fuels, shares in self.grids
            }
            if self.fit and self.fit_wafers(obs, sent, stock, arriving, weeks):
                changed = True
            hit = self.trim_fuels(obs, values, prohibited, sent, stock, arriving, weeks) if self.fuel_trim else ()
            for go, g, fabs, fuels, shares in self.grids:
                if go in hit:  # the week's segments changed: the rationed fuel's trim reads the new load factor
                    weeks[go] = self.grid_week(go, g, fabs, fuels, shares, values, stock, arriving, sent)
                    changed = True
            if self.trim:
                passing = self.passing(obs, values, prohibited, action) if self.trim_release else None
                if self.trim_rationed(obs, values, prohibited, sent, stock, arriving, weeks, passing):
                    changed = True
        if not last and self.osat_fit and self.fit_osats(obs, values, sent, stock):
            changed = True
        if self.cancel and self.cancel_unburnable(obs, values, sent, stock):
            changed = True
        if not changed:
            return action
        out = dict(action)
        out["flows"] = {"slot": list(action["flows"]["slot"]), "qty": [sent[s] for s in action["flows"]["slot"]]}
        return out
