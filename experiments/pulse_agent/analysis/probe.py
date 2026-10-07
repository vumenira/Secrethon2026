"""Single-episode probes: several scratch agents on a few episodes, each against a cached variant of the harness.

uv run python <this> --agents=v4_off,v4_r1 --episodes=17,12 --task=small --base=ft_s_v3

Prints, per episode and agent, the cost difference to the base as a share of that episode's naive-to-clairvoyant gap
(positive: the agent is better), and the sum over the probed episodes.
"""

import json
from pathlib import Path

import fire


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "recovery_tune"


def folder(name):
    for base in (HERE / "agents", ROOT / "agents", OUT / "agents"):
        if (base / name / "agent.py").is_file():
            return str(base / name)
    raise ValueError(name)


def main(agents, episodes, task="small", base="ft_s_v3", entropy=12345):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    names = list(agents) if isinstance(agents, (list, tuple)) else [a for a in str(agents).split(",") if a]
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    run = OUT / f"{task}-e{entropy}-n{64 if task == 'small' else 16}"
    refs = json.loads((run / "references.json").read_text())
    cached = json.loads((run / f"{base}.json").read_text())["rows"]
    cache = str(default_cache_dir())
    total = {name: 0.0 for name in names}
    money = {name: 0.0 for name in names}
    print(f"{task}, against {base}: difference as a share of the episode's gap (bn USD)")
    for n in ns:
        inst, omega, marks, fallback = _world(task, entropy, n, REPLICATIONS, cache)
        gap = refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]
        jb = cached[n]["J_policy_cents"]
        line = f"  episode {n:3d} (level {refs[n]['stratum']}, base {(refs[n]['J_naive_cents'] - jb) / gap:.3f})"
        for name in names:
            with without_secret_like():
                factory = load_agent_class(folder(name), f"submission_{name}")
                shim = _metered_shim(factory, None)
                traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
            errors = len(shim.errors)
            unload_agent()
            d = (jb - traj.J_cents) / gap
            total[name] += d
            money[name] += (jb - traj.J_cents) / 1e11
            line += f" | {name} {d:+.4f} ({(jb - traj.J_cents) / 1e11:+.1f}){' ERRORS ' + str(errors) if errors else ''}"
        print(line, flush=True)
    print("  sum over the probed episodes: " + " | ".join(f"{name} {total[name]:+.4f} ({money[name]:+.1f} bn)" for name in names))


if __name__ == "__main__":
    fire.Fire(main)
