"""Patch reserve.py of a v4 working copy with rules 2 and 3 (run from the folder that holds reserve.py)."""

import sys

p = sys.argv[1] if len(sys.argv) > 1 else "reserve.py"
s = open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"


def rep(a, b, count=1):
    global s
    a, b = a.replace("\n", nl), b.replace("\n", nl)
    assert s.count(a) == count, (s.count(a), a[:70])
    s = s.replace(a, b)


rep("from dataclasses import dataclass\n", "from collections import defaultdict\nfrom dataclasses import dataclass\n")
rep(
    "    premium_small: bool = False\n",
    "    premium_small: bool = False\n    premium_limit: bool = False\n    plan_duty: bool = False\n    cover_weeks: float = 4.0\n"
    "    pulse: bool = False  # whether pulse.Pulse plays (it decides what a short fuel leaves the fabs)\n    payback: float = 0.0  # pulse_payback\n",
)
rep(
    '            premium_small=bool(params.get("premium_small", False)),\n',
    '            premium_small=bool(params.get("premium_small", False)),\n'
    '            premium_limit=bool(params.get("premium_limit", False)),\n'
    '            plan_duty=bool(params.get("plan_duty", False)),\n'
    '            cover_weeks=float(params.get("cover_weeks", 4.0)),\n'
    '            pulse=bool(params.get("pulse", False)),\n'
    '            payback=float(params.get("pulse_payback", 0.0)),\n',
)
rep(
    "        changed = st.reserve or st.ramp > 0 or end_value or base_first or st.short_premium > 0\n",
    "        changed = st.reserve or st.ramp > 0 or end_value or base_first or st.short_premium > 0 or st.plan_duty\n",
)
rep(
    "            if end_value and st.stock_early > 0:\n                self.add_stock_early(where)\n",
    "            if end_value and st.stock_early > 0:\n                self.add_stock_early(where)\n            if st.plan_duty and self.T > 1:\n                self.add_duty(where)\n",
)
rep(
    """                    per = max(burn, draw) if self.settings.premium_small else burn
                    self.add_price(j, self.settings.short_premium * gain / per)
""",
    '''                    per = max(burn, draw) if self.settings.premium_small else burn
                    self.add_price(j, self.settings.short_premium * gain / per)
                    if self.settings.premium_limit:
                        self.add_free_short(go, k, j, self.settings.short_premium * gain / per, draw)

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
''',
)
open(p, "w", encoding="utf-8", newline="").write(s)
print("reserve patched:", p)
