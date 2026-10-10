"""Apply the strait reserve to agents/mpc_pulse_rec_v13w/reserve.py (run once; it asserts every anchor)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "agents" / "mpc_pulse_rec_v13w" / "reserve.py"
s = p.read_bytes().decode("utf-8")
orig = s
crlf = chr(13) + chr(10) in s
s = s.replace(chr(13) + chr(10), chr(10))
assert "strait_weeks" not in s, "already patched"


def swap(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


# 1. the module docstring
swap(
    '''  column and two rows per grid and small fuel; no forecast and no setting). Without it a small fuel that shares an
  edge with the scarce one took the edge first, every week, for energy the fabs never saw.
"""
''',
    '''  column and two rows per grid and small fuel; no forecast and no setting). Without it a small fuel that shares an
  edge with the scarce one took the edge first, every week, for energy the fabs never saw.

Added in ``agents/mpc_pulse_rec_v13w`` (off by default):

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
"""
''',
)

# 2. the settings
swap(
    '''    payback: float = 0.0  # pulse_payback
''',
    '''    payback: float = 0.0  # pulse_payback
    strait_weeks: float = 0.0  # strait_weeks where strait_reserve is on, else 0
''',
)
swap(
    '''            payback=float(params.get("pulse_payback", 0.0)),
        )
''',
    '''            payback=float(params.get("pulse_payback", 0.0)),
            strait_weeks=float(params.get("strait_weeks", 0.0)) if params.get("strait_reserve", False) else 0.0,
        )
''',
)
swap(
    '''        if not all(np.isfinite(v) and v >= 0 for v in numbers) or out.fuels not in FUELS:
''',
    '''        numbers += (out.strait_weeks,)
        if not all(np.isfinite(v) and v >= 0 for v in numbers) or out.fuels not in FUELS:
''',
)
swap(
    '''    @property
    def reserve(self) -> bool:
        return self.weeks > 0 and self.price > 0
''',
    '''    @property
    def reserve(self) -> bool:
        return self.weeks > 0 and self.price > 0

    @property
    def strait(self) -> bool:
        return self.strait_weeks > 0 and self.price > 0
''',
)

# 3. finish
swap(
    '''        changed = st.reserve or st.ramp > 0 or end_value or base_first or st.short_premium > 0 or st.plan_duty
''',
    '''        changed = st.reserve or st.ramp > 0 or end_value or base_first or st.short_premium > 0 or st.plan_duty
        changed = changed or st.strait
''',
)
swap(
    '''            if st.reserve:
                self.add_reserve(where)
        model = super().finish()
''',
    '''            if st.reserve:
                self.add_reserve(where)
            if st.strait:
                self.add_strait_reserve(where)
        model = super().finish()
''',
)

# 4. the methods, before the ramp
swap(
    '''    # ----- the ramp --------------------------------------------------------------------------------------------------
''',
    '''    # ----- the reserve at straits ------------------------------------------------------------------------------------
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
''',
)
p.write_bytes((s.replace(chr(10), chr(13) + chr(10)) if crlf else s).encode("utf-8"))
print("reserve.py:", len(orig), "->", len(s), "characters")
