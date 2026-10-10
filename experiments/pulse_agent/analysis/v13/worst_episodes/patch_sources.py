"""Apply the reserve at supply nodes to agents/mpc_pulse_rec_v13w/reserve.py (run once; it asserts every anchor)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
p = ROOT / "agents" / "mpc_pulse_rec_v13w" / "reserve.py"
s = p.read_bytes().decode("utf-8")
orig = s
crlf = chr(13) + chr(10) in s
s = s.replace(chr(13) + chr(10), chr(10))
assert "reserve_sources" not in s, "already patched"


def swap(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


# 1. the module docstring
swap(
    '''  a week the strait's queue is charged at a war-risk rate. Like the grids' reserve it is filled only with fuel no
  grid would burn in the window (the ramp), and after the terminals (they also earn the end value).
"""
''',
    '''  a week the strait's queue is charged at a war-risk rate. Like the grids' reserve it is filled only with fuel no
  grid would burn in the window (the ramp), and after the terminals (they also earn the end value).
- the reserve at supply nodes (``reserve_sources``; in ``Builder.reserve_targets``). A supply node keeps what is
  not shipped, up to its storage, at no cost, and what it cannot keep is lost. The plan saw no worth in that stock,
  so of two supplies of a grid whose stores are full it cut whichever saved freight, and a supply node that pipes
  fuel straight into the grid stood at one week of stock where it could have stood at four (in one episode the
  other route was then cut, and the grid ran dry fourteen weeks before the episode's end). With the switch the
  cover that neither the grid nor its terminal can hold is asked of the supply nodes with an edge straight into
  the grid, each up to its storage, at the same price.
"""
''',
)

# 2. the settings
swap(
    '''    strait_weeks: float = 0.0  # strait_weeks where strait_reserve is on, else 0
''',
    '''    strait_weeks: float = 0.0  # strait_weeks where strait_reserve is on, else 0
    sources: bool = False  # reserve_sources
''',
)
swap(
    '''            strait_weeks=float(params.get("strait_weeks", 0.0)) if params.get("strait_reserve", False) else 0.0,
''',
    '''            strait_weeks=float(params.get("strait_weeks", 0.0)) if params.get("strait_reserve", False) else 0.0,
            sources=bool(params.get("reserve_sources", False)),
''',
)

# 3. the targets
swap(
    '''                for e in inst.in_edges[g]:  # the cover its storage cannot hold, at the node that feeds it at once
                    ed = E[e]
                    if k not in ed.K or ed.tau != 0 or ed.tail in self.chk or ed.tail in self.supply:
                        continue
                    fs = inst.slot_index.get((ed.tail, k))
                    if fs is None:
                        continue
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + rest
                    entry[1] = max(entry[1], price)
                    break  # one feeder per grid and fuel: the first in edge order
''',
    '''                held = 0.0  # reserve_sources: what the feeder can hold of the rest
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
                    take = np.minimum(over, S[fs].storage)
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + take
                    entry[1] = max(entry[1], price)
                    over = over - take
''',
)
p.write_bytes((s.replace(chr(10), chr(13) + chr(10)) if crlf else s).encode("utf-8"))
print("reserve.py:", len(orig), "->", len(s), "characters")
