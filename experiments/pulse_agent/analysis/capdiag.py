"""Ground truth of the edge-capacity impairments the agent sees: who caused them, how the agent classifies them.

Throwaway analysis: root 12345, Small, 64 episodes. Writes a pickle of per-cell rows for later questions.
"""

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


REPO = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
sys.path.insert(0, str(REPO / "agents" / "mpc_recovery"))
OUT = Path(__file__).parent / "capcells.pkl"


def main(episodes=64, entropy=12345, task="small"):
    from recovery import strike
    from sbflow.disruption.params import PoissonParams
    from shockbench_flow import marks as M
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.omega import codes
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    poisson = PoissonParams()
    cache = str(default_cache_dir())
    cells = Counter()  # (class, causes) -> (week, edge) cells
    spells = Counter()
    rows = []
    per_episode = []
    for n in range(episodes):
        inst, omega, marks, _fb = _world(task, entropy, n, REPLICATIONS, cache)
        params = M.mark_params_from_json(str(omega.arrays["meta_mark_params"]))
        events = M.read_events(inst, omega.arrays)
        t = M._weeks(inst.T)
        u0 = np.array([np.inf if e.u0 is None else e.u0 for e in inst.edges])
        finite = np.isfinite(u0)
        groups = {
            "strike": tuple(q for q in events if q.type == M.PORT_STRIKE),
            "war": tuple(q for q in events if q.type == M.REGIONAL_CONFLICT),
            "sanction": tuple(q for q in events if q.type == M.SANCTION),
        }
        fac = {}
        for name, evs in groups.items():
            g = M._graph_marks(inst, evs, params, t)
            f = np.ones_like(g["u_now"])
            f[:, finite] = g["u_now"][:, finite] / u0[finite]
            fac[name] = f
        share = np.ones_like(marks.u_now)
        share[:, finite] = np.asarray(marks.u_now)[:, finite] / u0[finite]
        avg = np.ones_like(marks.u)
        avg[:, finite] = np.asarray(marks.u)[:, finite] / u0[finite]
        T, E = share.shape
        ep = Counter()
        strikes = [
            (q.onset, q.duration, q.severity, codes.TARGET_KINDS[q.target_kind], q.target) for q in groups["strike"]
        ]
        for e in np.flatnonzero(finite):
            first, seen = None, False
            for w in range(1, T + 1):  # the week whose observation shows the instant w - 1
                s = share[w - 1, e]
                if s >= 1.0:
                    first = None
                    continue
                if first is None:
                    first, seen = w, w > 1
                    new = True
                else:
                    new = False
                cls = strike(s, poisson) or "kept"
                causes = "+".join(k for k in ("strike", "war", "sanction") if fac[k][w - 1, e] < 1.0)
                cells[(cls, causes)] += 1
                ep[(cls, causes)] += 1
                if new:
                    spells[(cls, causes)] += 1
                # the true remaining life: weeks from now until the instant value is back to nominal (inf: never)
                back = np.flatnonzero(share[w - 1 :, e] >= 1.0)
                remaining = int(back[0]) if back.size else np.inf
                # until the value changes at all
                change = np.flatnonzero(np.abs(share[w - 1 :, e] - s) > 1e-9)
                until_change = int(change[0]) if change.size else np.inf
                nxt = float(share[w - 1 + change[0], e]) if change.size else np.nan
                rows.append(
                    dict(
                        episode=n,
                        week=w,
                        edge=int(e),
                        share=float(s),
                        cls=cls,
                        causes=causes,
                        first=first,
                        seen=seen,
                        remaining=remaining,
                        until_change=until_change,
                        next_share=nxt,
                        left=T - w + 1,  # window weeks left in the episode, this one included
                        avg=avg[w - 1 : w - 1 + 20, e].tolist(),
                        now=share[w - 1 : w - 1 + 21, e].tolist(),
                        u0=float(u0[e]),
                    )
                )
        per_episode.append(dict(episode=n, cells=dict(ep), strikes=strikes, edges=int(finite.sum()), T=T))
        print(n, "strikes", len(strikes), dict(ep), flush=True)
    print("\ncells (week, edge) by (agent's class, true causes):")
    for k, v in sorted(cells.items(), key=lambda kv: -kv[1]):
        print(f"  {k[0]:14s} {k[1]:24s} {v:7d} cells  {spells[k]:5d} spells")
    OUT.write_bytes(pickle.dumps(dict(rows=rows, per_episode=per_episode)))


if __name__ == "__main__":
    main()
