"""A buffer of packaged chips at lost-sales sinks, topped up after the planner has chosen the week's flows.

A lost-sales sink serves the week's demand from what it holds, and what it cannot serve is lost at the shortage
penalty. The planner's window plans on the demand forecast, so it sends a sink the forecast less its stock and the
sink ends every planned week empty: each week demand comes in above the forecast, the difference is lost, also while
packaged chips wait one edge away. The window has no reason to move them. It credits a chip alike wherever it is
when the window ends, and a chip not yet shipped has paid no freight.

A price in the window does not fix this (tried first, as a soft minimum on a sink's stock): a linear program cannot
say "a sink sells what it holds while demand is unmet", so the plan met the minimum by not selling in its first
week, which is not the agent's to decide, and sent nothing more.

The rule, for each lost-sales sink and good, after the planner (and the pulse, which touches other flows):

- **What the sink is short of a buffer by.** The stock it should still hold at the end of next week if demand comes in
  as forecast is ``share`` of next week's forecast, where ``share = exp(sink_cover * sd) - 1`` and ``sd`` is the
  standard deviation of the log of next week's demand around its forecast, from the demand's own law in the instance
  (an AR(1) of coefficient phi and innovation sigma, of which this week's and next week's innovations are still to
  come: ``sigma * sqrt(1 + phi^2)``). What it will hold is counted from the observation: its stock and this week's
  arrivals less this week's forecast (never below nothing), next week's arrivals under way, and this week's flows on
  the one-week edges into it.
- **What may fill it.** Only stock the plan itself leaves unused: the least of the planned end-of-week stock at the
  edge's tail over the window's first ``sink_idle`` weeks. A chip the plan ships or sells in those weeks is not
  touched, so no sale the plan counts on is given up for a buffer.
- **How.** The flow on a one-week edge from such a tail to the sink is raised, cheapest edge first, within the edge's
  capacity left by every flow on it and the tail's stock left by every flow out of it (the simulator scales all the
  flows of an edge or a stock that is over-asked, so nothing may be). Edges of the fleet's row are left alone.

It needs the week's plan (``Planner.plan``): a week the planner fell back, or played an older plan, tops up nothing.
"""

import math
from collections import defaultdict


TOL = 1e-9


class SinkBuffer:
    def __init__(self, inst, params):
        self.inst = inst
        self.cover = float(params["sink_cover"])  # the buffer, in standard deviations of next week's demand
        self.idle = int(params["sink_idle"])  # the plan's weeks a chip must stay unused at its node to be moved
        if not (math.isfinite(self.cover) and self.cover >= 0 and self.idle >= 1):
            raise ValueError(f"sink_cover {self.cover!r} and sink_idle {self.idle!r} are not a buffer's settings")
        chk = set(inst.chokepoints)
        fleet = {e for e, _lane, _dtau in inst.dup_items}  # edges of the fleet row: more flow there cuts other flows
        slots = defaultdict(list)  # (edge, commodity) -> its action slots
        self.edge_slots = defaultdict(list)  # edge -> every action slot on it (they share its capacity)
        self.tail_slots = defaultdict(list)  # (tail node, commodity) -> every action slot that draws on that stock
        for s, (e, k, _lane) in enumerate(inst.action_slots):
            slots[(e, k)].append(s)
            self.edge_slots[e].append(s)
            self.tail_slots[(inst.edges[e].tail, k)].append(s)
        # (node, good, its stock slot, the buffer as a share of next week's forecast, every slot that arrives next
        # week, [(nominal freight, edge, tail, slots)] the edges a top-up may use, cheapest first)
        self.sinks = []
        for d in inst.demands:
            if d.backlog or d.pi <= 0 or (d.node, d.k) not in inst.slot_index:
                continue
            due, ways = [], []
            for e in inst.in_edges[d.node]:
                ed = inst.edges[e]
                if d.k not in ed.K or ed.tau != 1 or ed.tail in chk:
                    continue
                due.extend(slots[(e, d.k)])
                if e not in fleet and slots[(e, d.k)] and (ed.tail, d.k) in inst.slot_index:
                    ways.append((ed.c0, e, ed.tail, slots[(e, d.k)]))
            share = math.exp(self.cover * d.sigma * math.sqrt(1.0 + d.phi**2)) - 1.0
            if ways and share > 0:
                self.sinks.append((d.node, d.k, inst.slot_index[(d.node, d.k)], share, due, sorted(ways)))
        self.log = []  # (week, sink node, good, quantity added) of the weeks it raised a flow

    def unused(self, plan, node: int, k: int) -> float:
        """What the plan leaves at ``node`` of good ``k`` through its first ``sink_idle`` weeks: the least end stock."""
        model, x = plan
        tmpl, nc, weeks = model.meta["template"], int(model.meta["nc"]), int(model.T)
        j = tmpl.get(("I", self.inst.slot_index[(node, k)]))
        if j is None or len(x) != weeks * nc:
            return 0.0
        return max(min(float(x[h * nc + j]) for h in range(min(self.idle, weeks))), 0.0)

    def adjust(self, obs, values, prohibited, action, plan):
        """The wire ``action`` with the flows into sinks short of their buffer raised (module docstring).

        ``obs``: the wire observation; ``values``: the planner's memory of the network as observed; ``prohibited``
        (E, K): the prohibitions in force this week; ``plan``: the week's window model and its solution, or None.
        """
        inst, t = self.inst, int(obs["week"])
        forecast = obs.get("demand_forecast")
        if plan is None or forecast is None or not action.get("flows") or t >= inst.T or not self.sinks:
            return action
        ahead = {(int(n), int(k), int(h)): float(q) for n, k, h, q in zip(forecast["node"], forecast["k"], forecast["h"], forecast["qty"])}
        stock = {(int(n), int(k)): float(q) for n, k, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])}
        now, soon = defaultdict(float), defaultdict(float)  # (node, good) -> arriving this week, next week
        pp = obs["pipeline"]
        for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
            if int(week) == t:
                now[(inst.edges[int(e)].head, int(k))] += float(q)
            elif int(week) == t + 1:
                soon[(inst.edges[int(e)].head, int(k))] += float(q)
        sent = dict(zip(action["flows"]["slot"], action["flows"]["qty"]))
        free: dict[tuple[int, int], float] = {}  # (tail, good) -> what a top-up may still take of its stock
        changed = False
        for node, k, s, share, due, ways in self.sinks:
            this_week, next_week = ahead.get((node, k, 0)), ahead.get((node, k, 1))
            if this_week is None or next_week is None:
                continue
            left = max(stock.get((node, k), 0.0) + now.get((node, k), 0.0) - this_week, 0.0)  # at this week's end
            there = left + soon.get((node, k), 0.0) + sum(sent.get(x, 0.0) for x in due)  # next week, before sales
            storage = inst.stock_slots[s].storage
            need = min((1.0 + share) * next_week - there, (storage - there) if storage else math.inf)
            added = 0.0
            for _cost, e, tail, ss in ways:
                if need <= TOL * max(next_week, 1.0):
                    break
                live = [x for x in ss if x in sent]
                if not live or bool(prohibited[e, k]):
                    continue
                key = (tail, k)
                if key not in free:
                    out = sum(sent.get(x, 0.0) for x in self.tail_slots[key])
                    free[key] = min(self.unused(plan, tail, k), stock.get(key, 0.0) - out)
                spare = float(values["u"][e]) - sum(sent.get(x, 0.0) for x in self.edge_slots[e])
                add = min(need, spare, free[key])
                if add <= TOL * max(next_week, 1.0):
                    continue
                sent[live[0]] += add
                free[key] -= add
                need -= add
                added += add
            if added > 0:
                changed = True
                self.log.append((t, node, k, added))
        if not changed:
            return action
        out = dict(action)
        out["flows"] = {"slot": list(action["flows"]["slot"]), "qty": [sent[x] for x in action["flows"]["slot"]]}
        return out
