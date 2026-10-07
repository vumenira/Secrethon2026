"""CPU seconds per week of buf_knobs settings on Small, root 12345: time.process_time around reset and act.

The settings are played one after another on each episode (interleaved), in this one process, so the machine's load is
about the same for all of them. Week 1 includes Agent(config) and the planner's reset, as the server counts it.

uv run python t3b_weekcpu.py --episodes=4 --settings=<file.json: {name: params}> --out=<file.json>
"""

import json
import shutil
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent


def folder(name, params):
    out = HERE / "weekcpu_agents" / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "agents" / "buf_knobs", out, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if params:
        (out / "params.json").write_text(json.dumps(params))
    return out


def main(settings, episodes=4, first=0, out="t3b_weekcpu.json"):
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent import EpisodeSet
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    specs = json.loads(Path(settings).read_text())
    es = EpisodeSet.build("small", 64, entropy=12345, n_jobs=1, verbose=False)
    task, entropy, regime, fq, cache = es._spec
    roots = {name: folder(name, params) for name, params in specs.items()}
    rows = []
    for n in range(first, first + episodes):
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        for name, root in roots.items():
            with without_secret_like():
                factory = load_agent_class(str(root), f"submission_{name}")
                shim = _metered_shim(factory, None)
                cpu, wall = [], []
                inner_reset, inner_act = shim.reset, shim.act
                carry = [0.0, 0.0]

                def reset(static, obs, seed, inner_reset=inner_reset, carry=carry):
                    c, w = time.process_time(), time.perf_counter()
                    inner_reset(static, obs, seed)
                    carry[0], carry[1] = time.process_time() - c, time.perf_counter() - w

                def act(obs, inner_act=inner_act, carry=carry, cpu=cpu, wall=wall):
                    c, w = time.process_time(), time.perf_counter()
                    action = inner_act(obs)
                    cpu.append(carry[0] + time.process_time() - c)
                    wall.append(carry[1] + time.perf_counter() - w)
                    carry[0] = carry[1] = 0.0
                    return action

                shim.reset, shim.act = reset, act
                seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
                traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
            planner = getattr(shim.agent, "planner", None)
            telemetry = getattr(planner, "telemetry", [])
            rows.append(
                {
                    "name": name,
                    "episode": n,
                    "J_policy_cents": traj.J_cents,
                    "fallback_weeks": sum(took_fallback(r) for r in traj.records),
                    "errors": len(shim.errors),
                    "cpu": [round(v, 4) for v in cpu],
                    "wall": [round(v, 4) for v in wall],
                    "iterations": [int(s.iterations) for s in telemetry],
                    "lp_fallback": sum(bool(s.fallback) for s in telemetry),
                    "trail": sorted({str(s.trail) for s in telemetry}),
                }
            )
            unload_agent()
            c = np.array(cpu)
            print(
                f"episode {n} {name:8s} J {traj.J_cents} cpu total {c.sum():6.2f} s, slowest week {c.max():.3f} s"
                f" (week {int(c.argmax()) + 1}), weeks 2-20 mean {c[1:20].mean():.3f} s",
                flush=True,
            )
        Path(out).write_text(json.dumps({"settings": specs, "rows": rows}))
    names = list(specs)
    base = names[0]
    stat = {}
    for name in names:
        c = np.array([r["cpu"] for r in rows if r["name"] == name])
        it = np.array([r["iterations"] for r in rows if r["name"] == name])
        stat[name] = (c.sum(axis=1).mean(), c.max(), c[:, 1:20].mean(), np.sort(c.ravel())[-5:].mean(), it.sum(axis=1).mean())
    print(f"\n{'setting':8s} {'cpu/episode':>12s} {'slowest week':>13s} {'weeks 2-20':>11s} {'top-5 weeks':>12s} {'iterations':>11s}")
    for name in names:
        s, b = stat[name], stat[base]
        print(
            f"{name:8s} {s[0]:7.2f} x{s[0] / b[0]:4.2f} {s[1]:8.3f} x{s[1] / b[1]:4.2f} {s[2]:6.3f} x{s[2] / b[2]:4.2f}"
            f" {s[3]:7.3f} x{s[3] / b[3]:4.2f} {s[4]:8.0f} x{s[4] / b[4]:4.2f}"
        )


if __name__ == "__main__":
    fire.Fire(main)
