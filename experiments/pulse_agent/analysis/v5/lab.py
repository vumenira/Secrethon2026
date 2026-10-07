"""Shared pieces of the v5 experiments: play one episode of an agent folder with its settings changed and hooks set.

Nothing here is an agent. ``play`` loads a folder as the scorer does, changes ``PARAMS`` of the loaded module, lets a
caller patch the loaded modules (``patch(modules)``), rolls the episode out and returns the trajectory, the agent and
what the hooks kept. ``calendar_of`` reads from a finished trajectory, for every grid with fabs and every week, the
share of the fabs' full draw that the grid had above its base load (the simulator's own rule, recomputed from the
records).
"""

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "recovery_tune"

V4 = json.loads((ROOT / "agents" / "ft_buffer_v4" / "params.json").read_text())


def folder(name):
    for base in (ROOT / "agents", OUT / "agents", HERE / "agents"):
        if (base / name / "agent.py").is_file():
            return str(base / name)
    raise ValueError(name)


def world(task, n, entropy=12345):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))


def references(task, entropy=12345):
    run = OUT / f"{task}-e{entropy}-n{64 if task == 'small' else 16}"
    return json.loads((run / "references.json").read_text())


def cached(task, name, entropy=12345):
    run = OUT / f"{task}-e{entropy}-n{64 if task == 'small' else 16}"
    return [r["J_policy_cents"] for r in json.loads((run / f"{name}.json").read_text())["rows"]]


def play(task, n, agent, params=None, patch=None, keep_plans=False, world_=None, entropy=12345, tag="lab"):
    """One episode. Returns {"traj", "agent", "plans" (week -> (H, nc) solution, template), "errors", "inst", "marks"}."""
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = world_ if world_ is not None else world(task, n, entropy)
    plans, template = {}, {}
    with without_secret_like():
        factory = load_agent_class(folder(agent), f"submission_{tag}")
        module = sys.modules[factory.__module__]
        module.PARAMS.update({**(params or {}), "strict": True})
        if keep_plans:
            box = [0]
            orig_act = module.Planner.act

            def act(self, obs):
                box[0] = int(obs["week"])
                return orig_act(self, obs)

            def record(self, model, res):
                if res.ok:
                    template.update(model.meta["template"])
                    plans[box[0]] = np.asarray(res.x, dtype=float).reshape(model.T, model.meta["nc"]).copy()

            module.Planner.act = act
            module.Planner._record = record
        if patch is not None:
            patch({"agent": module, "reserve": sys.modules["reserve"], "pulse": sys.modules["pulse"]})
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    agent_obj = shim.agent
    errors = list(shim.errors)
    unload_agent()
    return {"traj": traj, "agent": agent_obj, "plans": plans, "template": template, "errors": errors, "inst": inst, "marks": marks}


def fab_grids(inst):
    """Grid ordinals with a fab that draws energy, served base load first."""
    out = []
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        if ga.priority == "base_first" and any(inst.nodes[inst.fabs[fo]].fab.e > 0 for fo in inst.grid_fabs[go]):
            out.append(go)
    return out


def calendar_of(inst, marks, traj):
    """(T, G) the share of the fabs' full draw (at this week's capacity) the grid had above its base load; nan: no fab.

    Recomputed from the records as the simulator computes it: the available output of every segment (its share of the
    grid's power times the rationing factor on last week's stock, at most the fuel on hand after arrivals), less the
    base load. Also returns (T, G) the room in energy and (T, G) the full draw.
    """
    T, G = inst.T, len(inst.grids)
    share, room, draw = np.full((T, G), np.nan), np.zeros((T, G)), np.zeros((T, G))
    psi = inst.params.psi
    rec = traj.records
    from shockbench_flow.dynamics.sim import initial_stock

    i0 = np.asarray(initial_stock(inst), dtype=float)
    for ti in range(T):
        r = rec[ti]
        prev = i0 if ti == 0 else rec[ti - 1].stock
        for go, g in enumerate(inst.grids):
            ga = inst.nodes[g].grid
            Gb, yb = float(marks.G_bar[ti, go]), float(marks.y_bar[ti, go])
            g_av = ga.shares.get(None, 0.0) * Gb
            for k in ga.fuels:
                s = inst.slot_index[(g, k)]
                on_hand = float(r.stock[s] + r.disposal[s] + r.segment.get((go, k), 0.0))
                ration = 1.0
                if k == ga.rationed:
                    thr = psi * ga.ibar[k]
                    ration = 1.0 if prev[s] >= thr else prev[s] / thr
                g_av += min(ga.shares[k] * Gb * ration, on_hand)
            full = 0.0
            for fo in inst.grid_fabs[go]:
                fa = inst.nodes[inst.fabs[fo]].fab
                if fa.e > 0:
                    full += fa.e * float(marks.alpha_bar[ti, fo]) * fa.cap0  # e p / R with p = alpha R cap0
            room[ti, go], draw[ti, go] = g_av - yb, full
            if full > 0:
                share[ti, go] = min(1.0, max(0.0, (g_av - yb) / full))
            elif any(inst.nodes[inst.fabs[fo]].fab.e > 0 for fo in inst.grid_fabs[go]):
                share[ti, go] = 1.0 if g_av - yb >= 0 else 0.0
    return share, room, draw


def parts_of(traj):
    """The episode's cost by component (bn USD) and its salvage credit."""
    keys = list(traj.records[0].costs.as_dict())
    out = {k: sum(getattr(r.costs, k) for r in traj.records) / 1e9 for k in keys}
    out["salvage"] = -float(traj.salvage) / 1e9
    return out
