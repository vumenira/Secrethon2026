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

Added in ``agents/ft_buffer_v5``: ``Pulse.forecast``, the rule above played forward for the planner's window (the
share of the fabs' full draw each grid will have in each of the next weeks). It changes nothing in ``adjust`` and is
called only when ``plan_calendar`` is set. On ten Small and three Full episodes of root 12345 it calls 98 % of
grid-weeks right one week ahead, 92 and 89 % four weeks ahead, 86 and 75 % eight weeks ahead (the window's own belief,
every week ON, is right in 27 to 31 % on Small and 54 % on Full). The planner told that forecast scored worse
(``reserve.py``): a fab planned idle gets no wafers, and ``adjust`` counts a fab's draw by the wafers it holds, so
the pulse stops holding fuel for it.

It reads the week's observation (stock, shipments, lots waiting at straits), the planner's memory of the network as
observed (grid power and base load, fab capacity factors, edge capacities, prohibitions) and the instance; never the
planner's window, its forecast or its solution. It changes the quantities of flows the planner already sends: on
terminal-to-grid edges, and with ``pulse_wafers`` on material-to-fab edges (only upward).
"""

import heapq
import math
from collections import defaultdict


TOL = 1e-9


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
        self.log: list[tuple] = []  # (week, grid ordinal, "on" | "off", {fuel: sent}) of the weeks it changed a flow

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

    def forecast(self, obs: dict, values: dict, prohibited, H: int) -> dict[int, list[float]]:
        """Grid ordinal -> the share of its fabs' full draw the grid will have above base load in the next ``H`` weeks.

        This week first. It is ``adjust``'s own rule played forward on what this week's observation shows, one grid
        at a time: the fuel at the grid and at its terminal, the shipments under way to them by their arrival week,
        and after the last week whose arrivals are all under way (the shortest lead time into the terminal) the
        smoothed weekly arrivals ``adjust`` keeps (``rate``). The grid's power, its base load and the fabs'
        capacity are taken as observed for every week. In a week the planner's own flow would stand, the forecast
        delivers what the grid needs. A fuel without a terminal is held at this week's level. Nothing in the pulse's
        memory is changed.
        """
        inst, t = self.inst, int(obs["week"])
        E = inst.edges
        stock = {(int(n), int(k)): float(q) for n, k, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
        under_way: dict[tuple[int, int, int], float] = defaultdict(float)  # (node, commodity, arrival week) -> quantity
        pp = obs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            under_way[(E[int(e)].head, int(k), int(week))] += float(q)
        out = {}
        for go, g, fabs, fuels, shares in self.grids:
            G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
            full = 0.0  # the fabs' draw at capacity, whatever is left of the episode
            asks = []  # (the last week the fab counts toward a pulse, its draw at capacity, its draw this week)
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                full += fa.e * alpha * fa.cap0
                if R > 0:
                    until = inst.T if self.tail is None else inst.T - self.lead[fo] - self.tail
                    wafers = stock.get((f, fa.input), 0.0) + under_way.get((f, fa.input, t), 0.0)
                    asks.append((until, fa.e * alpha * fa.cap0, fa.e * min(alpha * R * fa.cap0, wafers) / R))
            sim = {}
            for k, (share, threshold, feeders, worth_it) in fuels.items():
                tails = sorted({tail for _e, tail, _storage, _ss in feeders})
                reach = [e for e, _tail, _storage, _ss in feeders if not bool(prohibited[e, k])]
                leads = [E[e].tau for tail in tails for e in inst.in_edges[tail] if k in E[e].K]
                inflow = sum(under_way.get((tail, k, t), 0.0) for tail in tails)
                key = (go, k)
                sim[k] = {
                    "burn": share * G,
                    "threshold": threshold,
                    "here": stock.get((g, k), 0.0),
                    "tank": sum(stock.get((tail, k), 0.0) for tail in tails),
                    "room": sum(inst.stock_slots[inst.slot_index[(tail, k)]].storage for tail in tails),
                    "store": inst.stock_slots[inst.slot_index[(g, k)]].storage,
                    "cap": sum(float(values["u"][e]) for e in reach),
                    "tails": tails,
                    "known": min(leads, default=0),  # the weeks from now whose arrivals are all under way
                    "rate": inflow if key not in self.rate else 0.7 * self.rate[key] + 0.3 * inflow,
                    "held": worth_it,
                }
            small = self.small[go]
            last_held = self.held.get(go, -10**9)
            series = []
            for w in range(int(H)):
                week = t + w
                want = sum(now if w == 0 else cap for until, cap, now in asks if week <= until)
                target = min(want, G * shares - y)
                able = G * (shares - sum(fuel[0] for fuel in fuels.values()))
                state = {}
                for k, s in sim.items():
                    thr = s["threshold"]
                    ration = min(1.0, s["here"] / thr) if thr > 0 else 1.0
                    on_hand = s["here"] + under_way.get((g, k, week), 0.0)
                    if not s["tails"]:  # no terminal: what the segment can give this week, every week
                        if w == 0:
                            s["level"] = min(s["burn"] * ration, on_hand)
                        state[k] = (s["burn"], 0.0, 1.0, s["level"], 0.0, 0.0, 0.0, False, 0.0)
                        able += s["level"]
                        continue
                    inflow = sum(under_way.get((tail, k, week), 0.0) for tail in s["tails"])
                    if w >= max(s["known"], 1):
                        inflow = max(inflow, s["rate"])
                    feed = min(s["tank"], s["cap"])
                    state[k] = (s["burn"], thr, ration, on_hand, s["tank"], inflow, s["room"], bool(s["tails"]) and s["cap"] > 0, feed)
                    able += min(s["burn"] * ration, on_hand + feed)
                lost = 0.0
                for k in small:
                    burn, _thr, ration, on_hand, _tank, _inflow, _room, _reach, feed = state[k]
                    lost += max(burn - min(burn * ration, on_hand + feed), 0.0)
                target = min(target, G * shares - y - lost)
                mode, short = "stand", set()
                if target > TOL * max(y, 1.0):
                    whole = able >= y + target - TOL * G
                    pulsed = week - last_held <= self.memory
                    ahead = max(self.last[go] - week, 0) if self.rundown else math.inf
                    mode = "on" if whole else "stand"
                    if not whole and (pulsed or able - y < self.partial * target):
                        fixable, waited = True, False
                        for k, (burn, thr, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                            if min(burn * ration, on_hand + feed) >= burn * (1 - TOL):
                                continue
                            if k in small:
                                if reach and sim[k]["held"]:
                                    short.add(k)
                                continue
                            short.add(k)
                            waited = True
                            lack = burn + thr - tank - on_hand
                            wait = min(self.hold_weeks, ahead)
                            if not reach or not sim[k]["held"] or lack > wait * max(inflow, sim[k]["rate"]):
                                fixable = False
                        if fixable and waited:
                            mode, last_held = "off", week
                        else:
                            short = set()
                g_av = G * (shares - sum(fuel[0] for fuel in fuels.values()))
                ready = {}
                for k, (burn, thr, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    new = 0.0
                    if reach:
                        new = min(max(burn * self.margin + thr - on_hand, 0.0), feed)  # an ON week, or the planner's flow
                        if mode == "off" and k in short:
                            new = 0.0
                            if thr > 0 and ration < 1:
                                charge = max(thr - on_hand + burn * ration, 0.0)
                                rate = min(max(inflow, sim[k]["rate"]), burn)
                                run = min(self.run, max(inst.T - week, 1))
                                if self.rundown:
                                    run = min(run, max(ahead, 1))
                                full_tank = tank + inflow >= self.fill * room
                                if charge <= feed and tank - charge + inflow >= burn:
                                    if full_tank or tank - charge + inflow >= burn + (run - 1) * (burn - rate):
                                        new = charge
                            new = max(new, min(tank + inflow - self.fill * room, feed), 0.0)
                    ready[k] = (new, min(burn * ration, on_hand + new))
                    g_av += ready[k][1]
                energy = min(want, max(g_av - y, 0.0))
                load = (energy + min(y, g_av)) / g_av if g_av > 0 else 0.0
                for k, (burn, thr, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    new, available = ready[k]
                    s = sim[k]
                    if s["tails"]:
                        s["here"] = min(max(on_hand + new - available * load, 0.0), s["store"] if s["store"] else math.inf)
                        s["tank"] = min(max(tank + inflow - new, 0.0), room if room else math.inf)
                series.append(min(1.0, max(0.0, (g_av - y) / full)) if full > 0 else 1.0)
            out[go] = series
        return out

    def adjust(self, obs: dict, values: dict, prohibited, action: dict) -> dict:
        """The planner's wire action of the week with the flows into pulsed grids changed (module docstring)."""
        inst, t = self.inst, int(obs["week"])
        if t >= inst.T or not action.get("flows"):
            return action  # the last week: nothing to hold fuel for
        stock = {(int(n), int(k)): float(q) for n, k, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
        arriving: dict[tuple[int, int], float] = defaultdict(float)
        pp = obs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            if int(week) == t:
                arriving[(inst.edges[int(e)].head, int(k))] += float(q)
        sent = dict(zip(action["flows"]["slot"], action["flows"]["qty"]))
        changed = False
        whole_now = {}  # grid ordinal -> whether the grid runs a whole week (its fabs get their draw)
        for go, g, fabs, fuels, shares in self.grids:
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
            if target <= TOL * max(y, 1.0):
                continue  # no fab to power this week: the planner's flows stand
            whole = able >= y + target - TOL * G
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
        if self.wafers > 0 and self.top_up_wafers(obs, values, prohibited, sent, stock, arriving, whole_now):
            changed = True
        if not changed:
            return action
        out = dict(action)
        out["flows"] = {"slot": list(action["flows"]["slot"]), "qty": [sent[s] for s in action["flows"]["slot"]]}
        return out
