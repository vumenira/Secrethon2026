"""The parameter checks of agents/mpc_pulse_rec_v3 at load, and its defaults against agents/mpc_pulse_rec_v2's."""

import importlib
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
WORK = Path(__file__).parent / "smoke_agents" / "check_v3"
MODULES = ("agent", "recovery", "reserve", "pulse")


def load(folder, params):
    if WORK.exists():
        shutil.rmtree(WORK)
    shutil.copytree(ROOT / "agents" / folder, WORK, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params is not None:
        (WORK / "params.json").write_text(json.dumps(params))
    for name in MODULES:
        sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if p != str(WORK)]
    sys.path.insert(0, str(WORK))
    importlib.invalidate_caches()
    try:
        return dict(importlib.import_module("agent").PARAMS)
    finally:
        sys.path.remove(str(WORK))
        for name in list(sys.modules):
            if name in MODULES or name.split(".")[0] == "sbflow":
                sys.modules.pop(name)


old, new = load("mpc_pulse_rec_v2", None), load("mpc_pulse_rec_v3", None)
print("PARAMS only in v3:", {k: new[k] for k in new if k not in old})
print("PARAMS only in v2:", {k: old[k] for k in old if k not in new})
print("PARAMS that differ:", {k: (old[k], new[k]) for k in old if k in new and old[k] != new[k]})
print("order of the old entries kept:", [k for k in new if k in old] == list(old))
v3 = json.loads((ROOT / "agents" / "ft_buffer_v3" / "params.json").read_text())
print("ft_buffer_v3's params.json against the defaults:", {k: (v, new[k]) for k, v in v3.items() if new[k] != v} or "all equal")
good = [v3, {"pulse_tail": 0, "pulse_rundown": False, "pulse_wafers": 0}, {"pulse_tail": None}, {"pulse_wafers": 2}]
good += [{"pulse_wafers_room": 0.25, "pulse_wafers_catchup": True}, {"stock_early": 0.5}, {"kinds": [], "strict": True}]
bad = [{"pulse_rundown": 1}, {"pulse_rundown": "true"}, {"pulse_wafers": True}, {"pulse_wafers": -1}, {"pulse_wafers": None}]
bad += [{"pulse_wafers_room": "0.5"}, {"pulse_wafers_catchup": 0}, {"stock_early": -0.1}, {"stock_early": None}]
bad += [{"pulse_wafer": 1.0}, {"pulse_tail": "2"}, {"demand_scale": 1.0}]
for params in good:
    load("mpc_pulse_rec_v3", params)
    print("accepted:", json.dumps(params)[:100])
for params in bad:
    try:
        load("mpc_pulse_rec_v3", params)
    except (ValueError, TypeError) as exc:
        print("refused: ", json.dumps(params), "->", type(exc).__name__, str(exc)[:90])
    else:
        raise SystemExit(f"NOT refused: {params}")
