"""Shadow test of agents/rec_merged: every week of a play, the window arrays of the merged forecast against the two
copies it was merged from, bit for bit, on the same observation and memory (root 12345 only).

uv run python shadow.py --episodes=0,1,2            (one process)
uv run python shadow.py --count=64 --n_jobs=2       (all 64)
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
TUNE = ROOT / "outputs" / "recovery_tune"
RULES_Q0 = {"weather_closure": {"mode": "quantile", "quantile": 0.0}, "grid": {"mode": "quantile", "quantile": 0.0}}
# the PARAMS the cached variants ran with: their agent.py defaults with their params.json over them
RULE_PARAMS = {
    "h_weeks": 20,
    "kinds": ["closure", "grid"],
    "mode": "mean",
    "quantile": 0.5,
    "caution": 1.0,
    "rules": RULES_Q0,
}
CAP_PARAMS = {
    "h_weeks": 20,
    "kinds": ["capacity"],
    "mode": "mean",
    "quantile": 0.5,
    "first": None,
    "caution": 1.0,
    "by_event": {"port_stoppage": {"first": "open"}, "port_slowdown": {"mode": "keep"}},
}
REFS = {
    "rule_now": (ROOT / "agents" / "rec_rule" / "recovery.py", RULE_PARAMS),
    "rule_snap": (TUNE / "agents" / "t1b_v3_q0" / "recovery.py", RULE_PARAMS),
    "cap_now": (ROOT / "agents" / "rec_capacity" / "recovery.py", CAP_PARAMS),
    "cap_snap": (TUNE / "agents" / "t1a_v3_final" / "recovery.py", CAP_PARAMS),
}
PAIRS = [("m_cg", "rule_now"), ("m_cg", "rule_snap"), ("m_cap", "cap_now"), ("m_cap", "cap_snap")]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # not put in sys.modules: the agent's own ``recovery`` stays the merged one
    return module


def differ(a, b):
    """The arrays of two windows that are not equal bit for bit: {name: largest absolute difference}."""
    if a is None or b is None:
        return {"<failed>": float("nan")}
    out = {}
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out[name] = float("inf")
        elif not np.array_equal(np.asarray(a[name]), np.asarray(b[name])):
            x, y = np.asarray(a[name], dtype=float), np.asarray(b[name], dtype=float)
            out[name] = float(np.max(np.abs(x - y))) if x.shape == y.shape else float("inf")
    return out


def shadow(agent, spec, episodes):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        with without_secret_like():
            factory = load_agent_class(agent, "submission_shadow")
            module, merged = sys.modules["submission_shadow"], sys.modules["recovery"]
            refs = {name: (load(f"ref_{name}", path), params) for name, (path, params) in REFS.items()}
            lp_common = module.lp_common
            stats = {
                "weeks": 0,
                "diff": {f"{a}|{b}": [] for a, b in PAIRS},
                "union": [],
                "errors": [],
                "moved": {"m_cg": 0, "m_cap": 0, "all": 0},
                "events": {},
                "dyads": None,
            }
            Planner = module.Planner
            reset0, window0 = Planner.reset, Planner._window_arrays

            def reset(self, static, obs, policy_seed, reset0=reset0, stats=stats, refs=refs, merged=merged):
                reset0(self, static, obs, policy_seed)
                P = module.PARAMS
                self.shadows = {
                    "m_cg": merged.Recovery(self._inst, P | {"kinds": ["closure", "grid"]}),
                    "m_cap": merged.Recovery(self._inst, P | {"kinds": ["capacity"]}),
                }
                for name, (mod, params) in refs.items():
                    self.shadows[name] = mod.Recovery(self._inst, dict(params))
                stats["dyads"] = {
                    name: {str(k): len(v) for k, v in rec.dyads.items()}
                    for name, rec in self.shadows.items()
                    if hasattr(rec, "dyads")
                }

            def window(self, inst, obs, H_t, window0=window0, stats=stats, lp_common=lp_common):
                t = int(obs["week"])
                got = {}
                for name, rec in self.shadows.items():
                    try:
                        rec.update(obs, self._memory)
                        got[name] = rec.arrays(obs, self._memory, H_t)
                    except Exception as exc:  # noqa: BLE001
                        stats["errors"].append((name, t, repr(exc)))
                        got[name] = None
                failed = self.failed
                played = window0(self, inst, obs, H_t)
                if self.failed != failed:
                    stats["errors"].append(("played", t, "forecast failed"))
                stats["weeks"] += 1
                for a, b in PAIRS:
                    d = differ(got[a], got[b])
                    if d:
                        stats["diff"][f"{a}|{b}"].append((t, d))
                plain = lp_common.persistence_arrays(inst, obs, self._memory, H_t)
                for name in ("m_cg", "m_cap"):
                    stats["moved"][name] += bool(differ(got[name], plain))
                stats["moved"]["all"] += bool(differ(played, plain))
                # the three kinds together: straits and grids as the closure and grid forecast, edges as the capacity one
                if got["m_cg"] is not None and got["m_cap"] is not None:
                    bad = {}
                    for name in played:
                        source = got["m_cap"] if name in ("u", "u_now") else got["m_cg"]
                        if not np.array_equal(np.asarray(played[name]), np.asarray(source[name])):
                            bad[name] = True
                    if bad:
                        stats["union"].append((t, sorted(bad)))
                for kind, _element, event, *_ in self.recovery.elements:
                    if kind in ("closure", "grid"):
                        for layer in self.recovery.layers[(kind, _element)]:
                            key = f"{kind}:{layer[1]}"
                            stats["events"][key] = stats["events"].get(key, 0) + 1
                    else:
                        key = f"{kind}:{event}"
                        stats["events"][key] = stats["events"].get(key, 0) + 1
                return played

            Planner.reset, Planner._window_arrays = reset, window
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            traj = rollout(inst, shim, omega, regime, _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        rows.append(
            {
                "episode": n,
                "J_policy_cents": traj.J_cents,
                "fallback_weeks": sum(took_fallback(r) for r in traj.records),
                "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                "forecast_failed": getattr(planner, "failed", None),
                "broken": getattr(shim.agent, "broken", None),
                "seconds": round(seconds, 3),
                **stats,
            }
        )
        unload_agent()
    return rows


def main(episodes=None, count=3, n_jobs=1, agent=str(ROOT / "agents" / "rec_merged"), out=None):
    from joblib import Parallel, delayed
    from shockbench_flow_agent import EpisodeSet

    es = EpisodeSet.build("small", 64, entropy=12345, n_jobs=1, verbose=False)
    if episodes is None:
        ns = list(es.episodes)[:count]
    else:
        ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else str(episodes).split(","))]
    chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
    parts = Parallel(n_jobs=n_jobs)(delayed(shadow)(agent, es._spec, [int(n) for n in c]) for c in chunks)
    rows = [r for part in parts for r in part]
    rows.sort(key=lambda r: r["episode"])
    if out:
        Path(out).write_text(json.dumps(rows, indent=1, default=str))
    total = {key: sum(len(r["diff"][key]) for r in rows) for key in rows[0]["diff"]}
    events = {}
    for r in rows:
        for key, value in r["events"].items():
            events[key] = events.get(key, 0) + value
    print(f"episodes {len(rows)}, weeks {sum(r['weeks'] for r in rows)}")
    print("weeks whose window differs, per pair:", total)
    print("weeks the three kinds together are not the union of the two:", sum(len(r["union"]) for r in rows))
    print("errors:", sum(len(r["errors"]) for r in rows), [r["errors"][:2] for r in rows if r["errors"]][:3])
    print("forecast failures of the played agent:", sum(r["forecast_failed"] or 0 for r in rows))
    print("fallback weeks:", sum(r["fallback_weeks"] for r in rows), "broken:", sum(bool(r["broken"]) for r in rows))
    print("agent errors:", [r["first_error"] for r in rows if r["first_error"]][:3])
    print("weeks the forecast moves the window:", {k: sum(r["moved"][k] for r in rows) for k in rows[0]["moved"]})
    print("impairment-weeks by kind and event:", dict(sorted(events.items())))
    print("dyad edges:", rows[0]["dyads"])
    for key, n in total.items():
        if n:
            first = next((r["episode"], r["diff"][key][0]) for r in rows if r["diff"][key])
            print(f"first difference of {key}: episode {first[0]}, week {first[1][0]}: {first[1][1]}")
    print("costs:", [(r["episode"], r["J_policy_cents"]) for r in rows[:4]])


if __name__ == "__main__":
    fire.Fire(main)
