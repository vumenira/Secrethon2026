"""Put this folder's PARAMS block on the current agents/mpc_pulse_rec_v13/agent.py (which changed after the copy)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
mine = (ROOT / "agents" / "mpc_pulse_rec_v13w" / "agent.py").read_bytes().decode("utf-8")
base = (ROOT / "agents" / "mpc_pulse_rec_v13" / "agent.py").read_bytes().decode("utf-8")
assert chr(13) not in mine and chr(13) not in base
start = mine.index("    # ----- added in agents/mpc_pulse_rec_v13w (reserve.py) -----\n")
end = mine.index("    # ----- the pulse (pulse.py) -----\n")
block = mine[start:end]
anchor = '    "cover_weeks": 4.0,\n    # ----- the pulse (pulse.py) -----\n'
assert base.count(anchor) == 1 and "strait_reserve" not in base
out = base.replace(anchor, '    "cover_weeks": 4.0,\n' + block + "    # ----- the pulse (pulse.py) -----\n")
(Path(__file__).parent / "agent_before_rebase.py.txt").write_bytes(mine.encode("utf-8"))
(ROOT / "agents" / "mpc_pulse_rec_v13w" / "agent.py").write_bytes(out.encode("utf-8"))
print("block of", block.count("\n"), "lines put on the current v13 agent.py;", len(base), "->", len(out), "characters")
