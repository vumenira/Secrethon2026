"""Where ends_needed let a run go on ("goes"), and the grid's weeks after it in that play and in the v12 play."""
import fire, numpy as np
from ana import instance, episodes_of, MEMORY

def main(episode, task="small", entropy=6006, episodes=128, name="r13rec_ends", base="r13rec_off", after=12, before=1, rule="goes", grid=""):
    inst, P = instance(task)
    N, K = inst.nodes, inst.commodities
    a = next(iter(episodes_of(task, entropy, episodes, name, str(episode))))[1]
    b = next(iter(episodes_of(task, entropy, episodes, base, str(episode))))[1]
    ta, tb = {tr["t"]: tr for tr in a["trace"]}, {tr["t"]: tr for tr in b["trace"]}
    print(f"episode {episode}: J {a['row']['J_policy_cents'] / 1e11:.1f} ({name}) against {b['row']['J_policy_cents'] / 1e11:.1f} ({base}): saves {(b['row']['J_policy_cents'] - a['row']['J_policy_cents']) / 1e11:+.1f} bn")
    events = [(t, go, moved) for t, tr in sorted(ta.items()) for (_t, go, r, moved) in tr["log"] if r == rule]
    print(f"{rule} entries:", [(t, N[inst.grids[go]].id, {K[k].id: round(v) for k, v in moved.items()}) for t, go, moved in events])
    shown = set()
    for t0, go, moved in events:
        if grid and N[inst.grids[go]].id != grid:
            continue
        if any(go == g2 and t0 - t1 < after for t1, g2 in shown):
            continue
        shown.add((t0, go))
        g = inst.grids[go]
        ga = N[g].grid
        r = ga.rationed
        fabs = [fo for fo in inst.grid_fabs[go] if N[inst.fabs[fo]].fab.e > 0]
        print(f"--- {N[g].id} from week {t0} (last sellable {P.last[go]}); per play: log | fab energy/full draw | shed/burn | lng here tank inflow sent seg (in weeks of burn) | week cost bn")
        for t in range(max(1, t0 - before), min(t0 + after, inst.T) + 1):
            line = f"{t:4d} "
            for ep, tr in ((a, ta.get(t)), (b, tb.get(t))):
                rec = ep["records"][t - 1]
                if tr is None:
                    line += " | (no trace)"
                    continue
                gr = tr["grids"][go]
                burn = ga.shares[r] * gr["G"] if gr["G"] > 0 else 1.0
                fu = gr["fuels"][r]
                logs = ",".join(rule_ for (_t, g2, rule_, _m) in tr["log"] if g2 == go)
                en = float(sum(rec.energy[fo] for fo in fabs))
                line += f" | {logs:12s} {en / P.full_draw[go]:4.2f} {rec.shed[go] / burn:4.2f}  {fu['here'] / burn:4.2f} {fu['tank'] / burn:4.2f} {fu['inflow'] / burn:4.2f} {fu['sent'] / burn:4.2f} {rec.segment.get((go, r), 0.0) / burn:4.2f}  {rec.cost_cents / 1e11:6.2f}"
            print(line)

if __name__ == "__main__":
    fire.Fire(main)
