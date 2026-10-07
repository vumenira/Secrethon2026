"""Hold a fuel a grid is short of at its terminal until the grid can run a whole week, then deliver it.

The simulator serves a grid's base load before its fabs (``base_first``), and a grid's nominal output barely exceeds its
base load plus what its fabs draw: a fuel segment short by a few per cent of the grid's output takes all the fabs'
energy, every week the shortage lasts. The clairvoyant plan (a linear program without that rule) sheds base load and
runs the fabs; an agent cannot. It can do the same thing over time: keep the short fuel at the terminal for some weeks
(the grid sheds that segment's whole output) and deliver a full week's burn in others (no shedding, the fabs run). The
energy burnt is the same; the base load shed rises by exactly the energy the fabs get, which is worth several times
VOLL in chips.

``Pulse.adjust`` changes the week's flows on the zero-lead edges into a grid with fabs (terminal to grid), after the
planner has chosen them, and nothing else. Each week, for each such grid:

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

It reads the observation and the planner's memory of the network only.
"""

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
        self.rate: dict[tuple[int, int], float] = {}  # (grid ordinal, fuel) -> smoothed weekly arrivals at its terminal
        self.held: dict[int, int] = {}  # grid ordinal -> the last week a fuel was held for it
        self.log: list[tuple] = []  # (week, grid ordinal, "on" | "off", {fuel: sent}) of the weeks it changed a flow

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
        for go, g, fabs, fuels, shares in self.grids:
            G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
            want = 0.0  # the energy the grid's fabs ask for this week
            for fo in fabs:
                f = inst.fabs[fo]
                fa = inst.nodes[f].fab
                R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
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
            if target <= TOL * max(y, 1.0):
                continue  # no fab to power this week: the planner's flows stand
            whole = able >= y + target - TOL * G
            pulsed = t - self.held.get(go, -10**9) <= self.memory
            short = set()  # the fuels whose segment cannot run in full this week
            if not whole:
                if not pulsed and able - y >= self.partial * target:
                    continue  # the fabs get most of their draw every week as it is
                fixable = True
                for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    if min(burn * ration, on_hand + feed) >= burn * (1 - TOL):
                        continue
                    short.add(k)
                    lack = burn + threshold - tank - on_hand  # what the tank still has to gather for a whole week
                    if not reach or k in unheld or lack > self.hold_weeks * max(inflow, self.rate[(go, k)]):
                        fixable = False  # by pipe or from a source, not worth a tank, or a tank that fills too slowly
                if not fixable or not short:
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
        if not changed:
            return action
        out = dict(action)
        out["flows"] = {"slot": list(action["flows"]["slot"]), "qty": [sent[s] for s in action["flows"]["slot"]]}
        return out
