"""Single-episode probes of settings of ``agents/ft_buffer_v5`` against a cached variant of the harness.

uv run python <this> --episodes=60,25 --variants="e,e_l,e_l_s" --task=small

A variant name is a "+"-joined list of the keys of ``KEYS`` below (each a change to v4's params); "v4" is v4 itself.
Prints, per episode and variant, the cost difference to the cached base (bn USD and share of the episode's gap;
positive: the variant is better) and the sum over the probed episodes.
"""

import sys
from pathlib import Path

import fire


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lab  # noqa: E402


KEYS = {
    "v4": {},
    "e": {"plan_calendar": "energy"},
    "lots": {"plan_calendar": "lots"},
    "l": {"calendar_load": True},
    "s": {"calendar_wafers": "short"},
    "i": {"calendar_wafers": "idle"},
    "w2": {"calendar_weeks": 2},
    "w3": {"calendar_weeks": 3},
    "w4": {"calendar_weeks": 4},
    "w6": {"calendar_weeks": 6},
    "w8": {"calendar_weeks": 8},
    "w12": {"calendar_weeks": 12},
    "nowaf": {"pulse_wafers": 0.0},
    "waf2": {"pulse_wafers": 2.0},
    "tight": {"calendar_shed": "tight"},
}


def params_of(name):
    out = dict(lab.V4)
    for key in name.split("+"):
        out |= KEYS[key]
    return out


def main(episodes, variants="e+l+s", task="small", base=None, agent="ft_buffer_v5"):
    ns = [int(n) for n in (episodes if isinstance(episodes, (list, tuple)) else [episodes])]
    names = list(variants) if isinstance(variants, (list, tuple)) else [v for v in str(variants).split(",") if v]
    base = base or ("ft_s_v4" if task == "small" else "ft_v4")
    refs, cached = lab.references(task), lab.cached(task, base)
    total = {name: 0.0 for name in names}
    gaps = 0.0
    print(f"{task}, against {base}: bn USD (share of the episode's gap); positive: better")
    for n in ns:
        w = lab.world(task, n)
        gap = (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 1e11
        gaps += gap
        line = f"  episode {n:3d} (level {refs[n]['stratum']}, gap {gap:7.1f})"
        for name in names:
            res = lab.play(task, n, agent, params_of(name), world_=w, tag="pf")
            d = (cached[n] - res["traj"].J_cents) / 1e11
            total[name] += d
            failed = getattr(res["agent"].planner, "failed", 0) + getattr(res["agent"].planner, "calendar_failed", 0) + getattr(res["agent"], "pulse_failed", 0)
            line += f" | {name} {d:+7.2f} ({d / gap:+.4f}){' ERR' if res['errors'] or failed else ''}"
        print(line, flush=True)
    print("  sum: " + " | ".join(f"{name} {total[name]:+.1f} bn ({total[name] / gaps:+.4f})" for name in names))


if __name__ == "__main__":
    fire.Fire(main)
