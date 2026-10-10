"""Write the files this folder's edits started from (for a three-way merge): outputs/s13/base/agent.py, pulse.py."""
from pathlib import Path

root = Path("agents/mpc_pulse_rec_v13s")
out = Path("outputs/s13/base")
s = (root / "agent.py").read_bytes().decode("utf-8")
a = s.index("Two rules are added in this folder, ``agents/mpc_pulse_rec_v13s``")
b = s.index("The rest is ``agents/mpc_det_h20``")
s = s[:a] + s[b:]
a = s.index("    # ----- added in agents/mpc_pulse_rec_v13s (pulse.py; they need \"pulse\" on) -----")
b = s.index("    # ----- added in agents/mpc_pulse_rec_v13 (this file): the week's CPU -----")
s = s[:a] + s[b:]
(out / "agent.py").write_bytes(s.encode("utf-8"))
# the copy taken at 17:45 for the census play: the same file with the first form of the mild_start entry
c = Path("outputs/recovery_tune/agents/s13_census/agent.py").read_bytes().decode("utf-8")
a = c.index("    # ----- added in agents/mpc_pulse_rec_v13s")
b = c.index("    # ----- added in agents/mpc_pulse_rec_v13 (this file): the week's CPU -----")
c = c[:a] + c[b:]
print("agent.py base equals the census copy less its entry:", c == s, len(s.splitlines()), "lines")
(out / "pulse.py").write_bytes(Path("agents/mpc_pulse_rec_v12/pulse.py").read_bytes())
