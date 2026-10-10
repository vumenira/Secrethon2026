from pathlib import Path

p = Path(__file__).resolve().parents[2] / "agents" / "mpc_pulse_rec_v13w" / "reserve.py"
s = p.read_bytes().decode("utf-8")
crlf = chr(13) + chr(10) in s
s = s.replace(chr(13) + chr(10), chr(10))
old = '''  cover that neither the grid nor its terminal can hold is asked of the supply nodes with an edge straight into
  the grid, each up to its storage, at the same price.
"""
'''
new = '''  cover that neither the grid nor its terminal can hold is asked of the supply nodes with an edge straight into
  the grid, each up to its storage, at the same price. Measured and left off (it takes that loss back and loses
  as much elsewhere).

What was measured is at the two switches in ``agent.PARAMS``.
"""
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_bytes((s.replace(chr(10), chr(13) + chr(10)) if crlf else s).encode("utf-8"))
print("done")
