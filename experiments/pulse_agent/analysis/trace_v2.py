"""Replay agents/ft_buffer_v2 with a logging copy of its pulse and solve the clairvoyant plan; one dump per episode.

The agent folder is loaded as it is (never edited): after the load, ``pulse.Pulse`` is replaced in memory by a subclass
whose ``adjust`` is the same code with a record of every grid's decision. The episode's cost is checked against the
cached row ``ft_v2`` / ``ft_s_v2`` (to the cent) when the settings are the folder's.

uv run python trace_v2.py --task=full --episodes=16
uv run python trace_v2.py --task=small --episodes=64 --only=3,7,12
"""

import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402

AGENT = ROOT / "agents" / "ft_buffer_v2"
TOL = 1e-9


def tracing(base):
    """``base`` (the folder's ``pulse.Pulse``) with ``adjust`` recording each grid's decision; the logic is copied."""

    class TracingPulse(base):
        def __init__(self, inst, params):
            super().__init__(inst, params)
            self.trace = []

        def adjust(self, obs, values, prohibited, action):
            inst, t = self.inst, int(obs["week"])
            if t >= inst.T or not action.get("flows"):
                return action
            stock = {
                (int(n), int(k)): float(q)
                for n, k, q in zip(obs["stock"]["node"], obs["stock"]["k"], obs["stock"]["qty"])
            }
            arriving = defaultdict(float)
            pp = obs["pipeline"]
            for e, k, q, week in zip(pp["edge"], pp["k"], pp["qty"], pp["arrival_week"]):
                if int(week) == t:
                    arriving[(inst.edges[int(e)].head, int(k))] += float(q)
            sent = dict(zip(action["flows"]["slot"], action["flows"]["qty"]))
            changed = False
            for go, g, fabs, fuels, shares in self.grids:
                G, y = float(values["grid_G"][go]), float(values["grid_y"][go])
                want = 0.0
                tailed = 0
                for fo in fabs:
                    f = inst.fabs[fo]
                    fa = inst.nodes[f].fab
                    R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
                    if self.tail is not None and t + self.lead[fo] + self.tail > inst.T:
                        tailed += 1
                        continue
                    if R > 0:
                        wafers = stock.get((f, fa.input), 0.0) + arriving.get((f, fa.input), 0.0)
                        want += fa.e * min(alpha * R * fa.cap0, wafers) / R
                target = min(want, G * shares - y)
                state, unheld = {}, set()
                able = G * (shares - sum(fuel[0] for fuel in fuels.values()))
                for k, (share, threshold, feeders, worth_it) in fuels.items():
                    here = stock.get((g, k), 0.0)
                    ration = min(1.0, here / threshold) if threshold > 0 else 1.0
                    on_hand = here + arriving.get((g, k), 0.0)
                    tank = inflow = room = 0.0
                    reach = []
                    for e, tail, storage, ss in feeders:
                        live = [s for s in ss if s in sent]
                        held = stock.get((tail, k), 0.0)
                        tank += held
                        inflow += arriving.get((tail, k), 0.0)
                        room += storage
                        if live and not bool(prohibited[e, k]):
                            reach.append((live, min(held, float(values["u"][e]))))
                    feed = sum(q for _ss, q in reach)
                    key = (go, k)
                    self.rate[key] = inflow if key not in self.rate else 0.7 * self.rate[key] + 0.3 * inflow
                    state[k] = (share * G, threshold, ration, on_hand, tank, inflow, room, reach, feed)
                    able += min(share * G * ration, on_hand + feed)
                    if not worth_it:
                        unheld.add(k)
                rec = {
                    "t": t,
                    "go": go,
                    "G": G,
                    "y": y,
                    "want": want,
                    "target": target,
                    "able": able,
                    "tailed": tailed,
                    "fuels": {
                        k: {
                            "burn": s[0],
                            "threshold": s[1],
                            "ration": s[2],
                            "on_hand": s[3],
                            "tank": s[4],
                            "inflow": s[5],
                            "room": s[6],
                            "reach": bool(s[7]),
                            "feed": s[8],
                            "rate": self.rate[(go, k)],
                            "full": min(s[0] * s[2], s[3] + s[8]) >= s[0] * (1 - TOL),
                        }
                        for k, s in state.items()
                    },
                    "why": None,
                    "short": {},
                    "moved": {},
                }
                self.trace.append(rec)
                if target <= TOL * max(y, 1.0):
                    rec["why"] = "no_target"
                    continue
                whole = able >= y + target - TOL * G
                pulsed = t - self.held.get(go, -(10**9)) <= self.memory
                rec["pulsed"] = pulsed
                short = set()
                if not whole:
                    if not pulsed and able - y >= self.partial * target:
                        rec["why"] = "partial_ok"
                        continue
                    fixable = True
                    for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                        if min(burn * ration, on_hand + feed) >= burn * (1 - TOL):
                            continue
                        short.add(k)
                        lack = burn + threshold - tank - on_hand
                        cause = "holdable"
                        if not reach or k in unheld or lack > self.hold_weeks * max(inflow, self.rate[(go, k)]):
                            fixable = False
                            cause = "no_terminal" if not reach else ("payback" if k in unheld else "slow")
                        rec["short"][k] = cause
                    if not fixable or not short:
                        rec["why"] = "unfixable" if short else "no_short"
                        continue
                    self.held[go] = t
                rec["why"] = "on" if whole else "off"
                moved = {}
                for k, (burn, threshold, ration, on_hand, tank, inflow, room, reach, feed) in state.items():
                    if not reach:
                        continue
                    planned = sum(sent[s] for ss, _q in reach for s in ss)
                    if whole:
                        need = min(max(burn * self.margin + threshold - on_hand, 0.0), feed)
                        lasting = tank + inflow - need >= burn
                        new = need if pulsed and not lasting and k not in unheld else max(planned, need)
                    else:
                        if k not in short:
                            continue
                        new = 0.0
                        if threshold > 0 and ration < 1:
                            charge = max(threshold - on_hand + burn * ration, 0.0)
                            rate = min(max(inflow, self.rate[(go, k)]), burn)
                            run = min(self.run, max(inst.T - t, 1))
                            full = tank + inflow >= self.fill * room
                            if charge <= feed and tank - charge + inflow >= burn:
                                if full or tank - charge + inflow >= burn + (run - 1) * (burn - rate):
                                    new = charge
                        new = max(new, min(tank + inflow - self.fill * room, feed))
                        new = max(new, 0.0)
                    rec["moved"][k] = (planned, new)
                    if abs(new - planned) <= TOL * max(burn, 1.0):
                        continue
                    left = new
                    for ss, q in reach:
                        part = min(left, q)
                        for i, s in enumerate(ss):
                            sent[s] = part if i == 0 else 0.0
                        left -= part
                    moved[k] = new
                if moved:
                    changed = True
                    self.log.append((t, go, "on" if whole else "off", moved))
            if not changed:
                return action
            out = dict(action)
            out["flows"] = {"slot": list(action["flows"]["slot"]), "qty": [sent[s] for s in action["flows"]["slot"]]}
            return out

    return TracingPulse


def one(spec, n, out_dir, override, plan):
    from shockbench_flow.dynamics.env import Env, rollout
    from shockbench_flow.dynamics.sim import initial_stock
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.base import reset_policy
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
    seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
    start = time.perf_counter()
    with without_secret_like():
        factory = load_agent_class(str(AGENT), "submission_trace")
        mod = sys.modules[factory.__module__]
        mod.PARAMS.update({"strict": True, **(override or {})})
        pm = sys.modules["pulse"]
        pm.Pulse = tracing(pm.Pulse)
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    pulse = agent.pulse
    pulse_dump = None
    if pulse is not None:
        pulse_dump = {
            "trace": pulse.trace,
            "log": pulse.log,
            "lead": dict(pulse.lead),
            "grids": [
                (go, g, list(fabs), {k: (v[0], v[1], [(e, tail, storage) for e, tail, storage, _ss in v[2]], v[3]) for k, v in fuels.items()}, shares)
                for go, g, fabs, fuels, shares in pulse.grids
            ],
        }
    unload_agent()
    recs = traj.records
    played = {
        "J": traj.J_cents / 100,
        "salvage": traj.salvage,
        "errors": errors,
        "seconds": time.perf_counter() - start,
        "stock": np.array([r.stock for r in recs]),
        "disposal": np.array([r.disposal for r in recs]),
        "lift": np.array([r.lift for r in recs]),
        "lots": np.array([r.lots_started for r in recs]),
        "scrapped": np.array([r.scrapped for r in recs]),
        "energy": np.array([r.energy for r in recs]),
        "served_load": np.array([r.served_load for r in recs]),
        "shed": np.array([r.shed for r in recs]),
        "demand": np.array([r.demand for r in recs]),
        "served": np.array([r.served for r in recs]),
        "lost": np.array([r.lost for r in recs]),
        "backlog": np.array([r.backlog for r in recs]),
        "segment": [dict(r.segment) for r in recs],
        "packaged": [dict(r.packaged) for r in recs],
        "x": [dict(r.x) for r in recs],
        "queue": [dict(r.queue) for r in recs],
        "costs": [r.costs.as_dict() for r in recs],
    }
    inst_r = inst.at_digest(marks.instance_digest)
    dump = {
        "episode": n,
        "task": task,
        "played": played,
        "pulse": pulse_dump,
        "i0": np.asarray(initial_stock(inst_r), dtype=float),
        "marks": {
            name: np.asarray(getattr(marks, name))
            for name in ("y_bar", "G_bar", "G_bar_now", "R", "R_now", "alpha_bar", "R_osat", "o", "supply", "u", "demand")
        },
    }
    if plan:  # the clairvoyant plan under the package's planning rules: the whole-episode LP at week 1
        truth = F.truth_of(inst, omega, marks)
        got = {}
        with without_secret_like():
            factory = load_agent_class(str(F.PROBE), F.MODULE)
            mod = sys.modules[factory.__module__]
            lp = sys.modules["sbflow.oracle.lp"]
            L = mod.L
            P = mod.Planner
            P.h_weeks, P.foresight = 10**6, F.Window(truth, {"fields": "all", "demand": True})

            def act(self, obs):
                inst_v, t = self._inst, int(obs["week"])
                self._memory.update(inst_v, obs)
                H = L.window_length(self._H, t, inst_v.T)
                arrays = self._window_arrays(inst_v, obs, H)
                assert self.failed == 0
                lp.BASE_FIRST_EVERY_WEEK = False
                t0 = time.perf_counter()
                model = L.rolled_lp(inst_v, obs, arrays, H, fab_hits=self.hits, planning_rules=True)
                res = lp.solve_oracle(model)
                weekly, salvage = lp.lp_costs(model, res.x)
                got.update(
                    J=res.J_usd,
                    status=res.status,
                    seconds=time.perf_counter() - t0,
                    columns=list(model.columns),
                    x=np.asarray(res.x).reshape(H, model.meta["nc"]),
                    costs=[w.as_dict() for w in weekly],
                    salvage=float(salvage),
                )
                return {"week": t, "flows": {"slot": [], "qty": []}, "overrides": None, "hold": None}

            P.act = act
            shim = _metered_shim(factory, None)
            env = Env(fallback=fallback)
            obs, info = env.reset(inst, regime, omega, seed, marks=marks, policy_name="submission")
            reset_policy(shim, info["static"], obs, seed, info.get("omega"))
            shim.act(obs)
            plan_errors = list(shim.errors)
        unload_agent()
        got["errors"] = plan_errors
        dump["plan"] = got
    out = Path(out_dir) / f"{n}.pkl"
    out.write_bytes(pickle.dumps(dump))
    return n, played["J"], errors, round(played["seconds"]), (dump.get("plan") or {}).get("J")


def main(task="full", entropy=12345, episodes=16, only=None, n_jobs=2, tag="v2", override=None, plan=True, check=None):
    from joblib import Parallel, delayed

    es = F.tune.episode_set(task, entropy, episodes, 1)
    refs = {r["episode"]: r for r in es.references}
    ns = list(es.episodes)
    if only is not None:
        only = [only] if isinstance(only, int) else [int(v) for v in only]
        ns = [n for n in ns if n in only]
    out_dir = HERE / "trace" / f"{task}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    override = dict(override) if override else None
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(es._spec, int(n), str(out_dir), override, plan) for n in ns)
    cached = None
    if check is None:
        check = {"full": "ft_v2", "small": "ft_s_v2"}[task] if not override else ""
    if check:
        rows = json.loads((F.tune.run_dir(task, entropy, episodes) / f"{check}.json").read_text())["rows"]
        cached = {r["episode"]: r["J_policy_cents"] / 100 for r in rows}
    same = 0
    for n, J, errors, seconds, plan_J in results:
        r = refs[n]
        gap = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        line = f"episode {n} level {r['stratum']}: {(r['J_naive_cents'] / 100 - J) / gap:.4f} ({seconds} s)"
        if plan_J is not None:
            line += f", plan {(r['J_naive_cents'] / 100 - plan_J) / gap:.4f}"
        if cached is not None:
            ok = abs(cached[n] - J) < 0.005
            same += ok
            line += f", {'same as' if ok else 'DIFFERS from'} {check}"
        if errors:
            line += f" ERRORS {errors[:1]}"
        print(line, flush=True)
    if cached is not None:
        print(f"{same} of {len(results)} episodes cost the same as {check} to the cent")
    print(f"{time.perf_counter() - start:.0f} s; dumps in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
