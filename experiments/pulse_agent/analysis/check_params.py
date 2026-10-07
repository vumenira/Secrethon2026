"""The parameter checks of agents/rec_merged, and the rule each event type gets from the defaults."""
import importlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
work = Path(__file__).parent / "check_agent"
if work.exists():
    shutil.rmtree(work)
shutil.copytree(ROOT / "agents" / "rec_merged", work, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
sys.path.insert(0, str(work))


def load(params):
    for name in ("agent", "recovery"):
        sys.modules.pop(name, None)
    if params is None:
        (work / "params.json").unlink(missing_ok=True)
    else:
        (work / "params.json").write_text(json.dumps(params))
    importlib.invalidate_caches()
    return importlib.import_module("agent")


agent = load(None)
print("defaults:", json.dumps(agent.PARAMS))
import recovery

class Inst:  # the rule table needs no instance
    pass

rec = recovery.Recovery.__new__(recovery.Recovery)
rec.params, rec.rules = agent.PARAMS, {}
for kind, events in recovery.EVENTS.items():
    for event in events:
        print(f"  {kind:9s} {event:20s} {rec.rule(kind, event)}")

good = [
    {"kinds": ["closure", "grid"]},
    {"kinds": ["capacity"]},
    {"h_weeks": 24},
    {"h_weeks": None, "kinds": []},
    {"rules": {}},
    {"rules": {"capacity": {"first": "open"}, "port_slowdown": {"mode": "keep", "first": None}}},
    {"mode": "quantile", "quantile": 0, "first": "mean", "caution": 2},
]
bad = [
    {"by_event": {"port_stoppage": {"first": "open"}}},
    {"kinds": ["closures"]},
    {"rules": {"weather_closures": {"mode": "quantile"}}},
    {"rules": {"grid": {"quantiles": 0.0}}},
    {"rules": {"grid": {"mode": "median"}}},
    {"rules": {"port_stoppage": {"first": "closed"}}},
    {"rules": {"grid": {"quantile": "half"}}},
    {"mode": "kept"},
    {"H_weeks": 24},
]
for params in good:
    load(params)
    print("accepted:", json.dumps(params))
for params in bad:
    try:
        load(params)
    except ValueError as exc:
        print("refused: ", json.dumps(params), "->", exc)
    else:
        raise SystemExit(f"NOT refused: {params}")
(work / "params.json").unlink(missing_ok=True)
