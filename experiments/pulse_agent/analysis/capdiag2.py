"""Check agents/rec_capacity's reading of edges (``Edges.read``) against the scenario's events, week by week.

Saves the per-episode arrays (instant share, week-average share, the strikes' factor, the other cuts' factor) so that
later questions need no world rebuilt.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(REPO / "agents" / "rec_capacity"))
OUT = Path(__file__).parent / "caparrays.pkl"


def worlds(episodes=64, entropy=12345, task="small"):
    if OUT.exists():
        return pickle.loads(OUT.read_bytes())
    from shockbench_flow import marks as M
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    cache = str(default_cache_dir())
    out = []
    for n in range(episodes):
        inst, omega, marks, _fb = _world(task, entropy, n, REPLICATIONS, cache)
        params = M.mark_params_from_json(str(omega.arrays["meta_mark_params"]))
        events = M.read_events(inst, omega.arrays)
        t = M._weeks(inst.T)
        u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
        finite = np.isfinite(u0)

        def factor(evs, key="u_now"):
            g = M._graph_marks(inst, tuple(evs), params, t)
            f = np.ones_like(g[key])
            f[:, finite] = g[key][:, finite] / u0[finite]
            return f

        strikes = [q for q in events if q.type == M.PORT_STRIKE]
        others = [q for q in events if q.type in (M.REGIONAL_CONFLICT, M.SANCTION)]
        share = np.ones_like(marks.u_now)
        share[:, finite] = np.asarray(marks.u_now)[:, finite] / u0[finite]
        avg = np.ones_like(marks.u)
        avg[:, finite] = np.asarray(marks.u)[:, finite] / u0[finite]
        out.append(
            dict(
                episode=n,
                finite=finite,
                u0=u0,
                share=share,
                avg=avg,
                strike=factor(strikes),
                other=factor(others),
                strike_avg=factor(strikes, "u"),
                other_avg=factor(others, "u"),
                strikes=[(q.onset, q.duration, q.severity, q.target) for q in strikes],
                tail_region=[inst.nodes[e.tail].region for e in inst.edges],
                head_region=[inst.nodes[e.head].region for e in inst.edges],
                mode=[e.mode for e in inst.edges],
            )
        )
        print(n, flush=True)
    OUT.write_bytes(pickle.dumps(out))
    return out


def main():
    from recovery import Edges
    from sbflow.disruption.params import PoissonParams
    from sbflow.marks import MarkParams

    data = worlds()
    confusion = Counter()
    level_ok = Counter()
    examples = []
    for w in data:
        edges = Edges(PoissonParams(), MarkParams())
        T, E = w["share"].shape
        for e in np.flatnonzero(w["finite"]):
            for t in range(T):
                s = w["share"][t, e]
                event, level = edges.read(int(e), float(s))
                fs, fo = w["strike"][t, e], w["other"][t, e]
                if fs >= 1.0:
                    truth = "none" if fo >= 1.0 else "long"
                elif abs(fs - 0.07) < 1e-9:
                    truth = "stoppage"
                elif 0.4 < fs < 0.8:
                    truth = "slowdown"
                else:
                    truth = "strikes"
                if s >= 1.0:
                    continue
                confusion[(truth, event or "kept")] += 1
                if event is not None:
                    ok = abs(level - fo) < 1e-6
                    level_ok[(truth, event, "level right" if ok else "level wrong")] += 1
                    if not ok and len(examples) < 12:
                        examples.append((w["episode"], int(e), t + 1, round(float(s), 4), event, round(level, 4), round(float(fo), 4), round(float(fs), 4)))
    print("cells by (truth, the agent's reading):")
    for k, v in sorted(confusion.items(), key=lambda kv: -kv[1]):
        print(f"   {k[0]:10s} {k[1]:14s} {v:7d}")
    print("the level a strike returns to:")
    for k, v in sorted(level_ok.items(), key=lambda kv: -kv[1]):
        print("  ", k, v)
    print("examples of a wrong level (episode, edge, week, share, event, level read, true level, strike factor):")
    for x in examples:
        print("  ", x)


if __name__ == "__main__":
    main()
