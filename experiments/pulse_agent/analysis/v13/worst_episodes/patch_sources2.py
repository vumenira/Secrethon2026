"""reserve_sources: every supply node is asked up to its storage on its own (the first no longer uses up the rest)."""
from pathlib import Path

p = Path(__file__).resolve().parents[2] / "agents" / "mpc_pulse_rec_v13w" / "reserve.py"
s = p.read_bytes().decode("utf-8")
crlf = chr(13) + chr(10) in s
s = s.replace(chr(13) + chr(10), chr(10))
old = '''                    take = np.minimum(over, S[fs].storage)
                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + take
                    entry[1] = max(entry[1], price)
                    over = over - take
'''
new = '''                    entry = asked.setdefault(fs, [np.zeros(self.T), 0.0])
                    entry[0] = entry[0] + over  # each on its own: a supply node's stock costs nothing to keep
                    entry[1] = max(entry[1], price)
'''
assert s.count(old) == 1
s = s.replace(old, new)
p.write_bytes((s.replace(chr(10), chr(13) + chr(10)) if crlf else s).encode("utf-8"))
print("done")
