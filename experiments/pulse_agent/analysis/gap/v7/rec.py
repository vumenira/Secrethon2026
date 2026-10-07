"""Play an agent folder with its records kept and the gate of the gas trim noted for every grid and week.

The folder is loaded as the scorer loads it and never edited; entries of PARAMS are replaced in memory (``strict`` on).
``Pulse.trim_rationed`` is wrapped: before it runs, ``why`` walks the same gates on a copy of the flows and notes where
the grid left the rule. Each episode's cost is compared with a cached row to the cent. One dump per episode in
``v7/traj/<tag>_<task>_<entropy>_<n>.pkl``.

uv run python rec.py --task=small --n_jobs=2
"""

import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v6"))
import common as C  # noqa: E402
import probe as P  # noqa: E402

TOL = 1e-9


def why(self, obs, values, prohibited, sent, stock, arriving, weeks, passing=None):
    """{grid ordinal: (label, numbers)}: where ``Pulse.trim_rationed`` of agents/mpc_pulse_rec_v6 leaves each grid.

    ``passing`` (agents/mpc_pulse_rec_v7, ``ration_trim_release``): what the straits pass on, in place of their queues.
    """
    inst, t = self.inst, int(obs["week"])
    chk = set(inst.chokepoints)
    soon = defaultdict(float)
    strait = defaultdict(float)  # the part of ``soon`` that a strait has yet to pass on (the whole queue is counted)
    pp = obs["pipeline"]
    for e, k, lane, q, week in zip(pp["edge"], pp["k"], pp["lane"], pp["qty"], pp["arrival_week"]):
        head = inst.edges[int(e)].head
        if int(week) == t + 1:
            soon[(head, int(k))] += float(q)
        elif passing is None and int(week) == t and head in chk and lane is not None:
            onward = inst.edges[inst.lane_through[(int(lane), head)][1]]
            if onward.tau == 1 and float(values["open"][inst.chokepoint_ordinal[head]]) > 0:
                soon[(onward.head, int(k))] += float(q)
                strait[(onward.head, int(k))] += float(q)
    ql = obs.get("queue_lots") or {"chokepoint": [], "k": [], "qty": [], "next_edge": []}
    for c, k, q, nxt in zip(ql["chokepoint"], ql["k"], ql["qty"], ql["next_edge"]) if passing is None else ():
        onward = inst.edges[int(nxt)]
        if onward.tau == 1 and float(values["open"][inst.chokepoint_ordinal[int(c)]]) > 0:
            soon[(onward.head, int(k))] += float(q)
            strait[(onward.head, int(k))] += float(q)
    for key, q in (passing or {}).items():
        soon[key] += q
        strait[key] += q
    out = {}
    for go, g, fabs, fuels, _shares in self.grids:
        r = inst.nodes[g].grid.rationed
        wk = weeks[go]
        info = {
            "G": wk["G"], "y": wk["y"], "null": wk["null"], "served": wk["served"], "rho": wk["rho"], "load": wk["load"],
            "pulsed": t - self.held.get(go, -10**9) <= self.memory, "off": self.held.get(go) == t,
            "seg": {k: v for k, v in wk["seg"].items()},
            "draw": sum(energy for _w, _l, energy in wk["fab"].values()),
        }
        if r not in fuels or (g, r) not in stock:
            out[go] = ("no_rationed", info)
            continue
        share, threshold, feeders, _worth_it = fuels[r]
        cap, here = share * wk["G"], stock[(g, r)]
        limit, have, av = wk["seg"][r]
        info.update(cap=cap, here=here, limit=limit, have=have, av=av, threshold=threshold, feeders=len(feeders))
        if threshold <= 0 or cap <= 0 or wk["null"] <= 0:
            out[go] = ("no_rule", info)
            continue
        if wk["served"] < wk["y"] * (1 - TOL):
            out[go] = ("sheds", info)
            continue
        if have < limit * (1 - TOL):
            out[go] = ("on_hand_limits", info)
            continue
        tank = inflow = room = 0.0
        reach = []
        for e, tail, storage, ss in feeders:
            live = [s for s in ss if s in sent]
            held = stock.get((tail, r), 0.0)
            tank += held
            inflow += arriving.get((tail, r), 0.0)
            room += storage
            if live and not bool(prohibited[e, r]):
                reach.append((live, min(held, float(values["u"][e]))))
        info.update(tank=tank, inflow=inflow, room=room, reach=len(reach))
        if not reach:
            out[go] = ("no_terminal" if not feeders else "no_reach", info)
            continue
        if here + arriving.get((g, r), 0.0) + tank + inflow > self.trim_cover * cap:
            out[go] = ("cover", info)
            continue
        need = wk["y"] + self.trim_margin * self.full_draw[go]
        info["fabs"] = {}
        for fo, (wafers, lots, _energy) in wk["fab"].items():
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            R, alpha = float(values["fab_R"][fo]), float(values["fab_alpha"][fo])
            if R > 0:
                coming = soon[(f, fa.input)] + sum(sent.get(s, 0.0) for s in self.slots_into(f, fa.input, 1))
                need += fa.e * min(alpha * R * fa.cap0, wafers - wk["rho"] * lots + coming) / R
                info["fabs"][fo] = {
                    "wafers": wafers, "lots": lots, "soon": soon[(f, fa.input)], "strait": strait[(f, fa.input)],
                    "cap": alpha * R * fa.cap0, "R": R, "alpha": alpha, "e": fa.e,
                    "sent1": {s: sent.get(s, 0.0) for s in self.slots_into(f, fa.input, 1) if sent.get(s, 0.0) > 0},
                }
        others = wk["null"]
        for k, (other_share, _threshold, ways, _w) in fuels.items():
            if k == r:
                continue
            _limit, on_hand, gives = wk["seg"][k]
            more = soon[(g, k)] + sum(sent.get(s, 0.0) for s in self.slots_into(g, k, 1))
            for e, tail, _storage, ss in ways:
                if any(s in sent for s in ss) and not bool(prohibited[e, k]):
                    left = stock.get((tail, k), 0.0) - sum(sent.get(s, 0.0) for s in self.tail_slots[(tail, k)])
                    more += min(max(left + arriving.get((tail, k), 0.0), 0.0), float(values["u"][e]))
            others += min(other_share * wk["G"], on_hand - gives * wk["load"] + more)
        needed = need - others
        info.update(need=need, others=others, needed=needed)
        if needed >= cap * (1 - TOL):
            out[go] = ("needs_all", info)
            continue
        close = threshold * max(needed, 0.0) / cap
        own = max(close + av * wk["load"], limit) - here - arriving.get((g, r), 0.0)
        fill = min(tank + inflow - self.fill * room, sum(q for _ss, q in reach))
        new = max(own, fill, 0.0)
        planned = sum(sent[s] for ss, _q in reach for s in ss)
        info.update(close=close, own=own, fill=fill, new=new, planned=planned)
        if new >= planned - TOL * max(cap, 1.0):
            # the flows as they stand deliver no more: which bound holds the delivery up, or nothing is delivered
            if planned <= TOL * max(cap, 1.0):
                label = "none_sent"
            elif fill >= max(own, 0.0) and fill > 0:
                label = "fill"
            else:
                label = "as_low"
            out[go] = (label, info)
            continue
        out[go] = ("trim_fill" if fill > max(own, 0.0) else "trim", info)
    return out


def one(task, n, folder, params, entropy, tag):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    start = time.perf_counter()
    notes, before = {}, {}
    with without_secret_like():
        factory = load_agent_class(str(folder), "submission_rec")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params)
        pulse = sys.modules["pulse"]
        if hasattr(pulse.Pulse, "trim_rationed"):
            orig = pulse.Pulse.trim_rationed

            def trim_rationed(self, obs, values, prohibited, sent, stock, arriving, weeks, *args, **kwargs):
                try:
                    passing = args[0] if args else kwargs.get("passing")
                    notes[int(obs["week"])] = why(self, obs, values, prohibited, dict(sent), stock, arriving, weeks, passing)
                except Exception as exc:  # noqa: BLE001 - the note is a by-product; the play goes on
                    notes[int(obs["week"])] = {"error": repr(exc)}
                return orig(self, obs, values, prohibited, sent, stock, arriving, weeks, *args, **kwargs)

            pulse.Pulse.trim_rationed = trim_rationed
        orig_pulsed = module.Agent.pulsed

        def pulsed(self, obs, action):
            flows = action.get("flows") or {"slot": [], "qty": []}
            before[int(obs["week"])] = dict(zip([int(s) for s in flows["slot"]], [float(q) for q in flows["qty"]]))
            return orig_pulsed(self, obs, action)

        module.Agent.pulsed = pulsed
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    log = list(agent.pulse.log) if agent.pulse is not None else []
    failed = (getattr(agent.planner, "forecast_failed", None), getattr(agent.planner, "reserve_failed", None), agent.pulse_failed)
    unload_agent()
    dump = {
        "episode": n, "task": task, "entropy": entropy, "J": int(traj.J_cents), "salvage": float(traj.salvage),
        "records": traj.records, "log": log, "why": notes, "before": before, "errors": errors, "failed": failed,
    }
    path = HERE / "traj" / f"{tag}_{task}_{entropy}_{n}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(dump))
    return n, int(traj.J_cents), errors[:1], failed, round(time.perf_counter() - start, 1)


def main(task="small", folder="mpc_pulse_rec_v6", params="{}", base="u_v6", tag="v6", entropy=12345, episodes=None, only=None, n_jobs=2, redo=False):
    import json

    from joblib import Parallel, delayed

    params = P.parse(params)
    params.setdefault("strict", True)
    root = Path(folder) if Path(folder).is_absolute() else C.ROOT / "agents" / folder
    episodes = episodes or C.TOTAL[task]
    out_dir = C.OUT / f"{task}-e{entropy}-n{episodes}"
    refs = {r["episode"]: r for r in json.loads((out_dir / "references.json").read_text())}
    cached = {r["episode"]: r["J_policy_cents"] for r in json.loads((out_dir / f"{base}.json").read_text())["rows"]} if base else {}
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or not (HERE / "traj" / f"{tag}_{task}_{entropy}_{n}.pkl").exists()]
    start = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(one)(task, n, root, params, entropy, tag) for n in todo) if todo else []
    same = 0
    for n, J, errors, failed, seconds in results:
        ok = cached.get(n) == J
        same += ok
        r = refs[n]
        score = (r["J_naive_cents"] - J) / (r["J_naive_cents"] - r["J_oracle_cents"])
        diff = J - cached[n] if n in cached else None
        print(f"episode {n} level {r['stratum']}: {score:.4f} {'same as' if ok else 'DIFFERS from'} {base} (by {diff} cents); failures {failed}{' ERRORS ' + str(errors) if errors else ''} ({seconds} s)", flush=True)
    print(f"{same} of {len(results)} episodes cost the same as {base} to the cent; {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire(main)
