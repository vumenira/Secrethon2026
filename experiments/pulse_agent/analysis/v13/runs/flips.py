"""Offline: at each run schedule_ends ended in a recorded v12 play, what the second form of ends_needed would decide.

Approximate (the grid's other segments are taken as full): needed = burn - (full draw - forecast draw).
"""
import fire, numpy as np
from collections import defaultdict
from ana import instance, episodes_of, MEMORY

def main(task="full", entropy=12345, episodes=16, name="r13rec_off", only="", show=60):
    inst, P = instance(task)
    N = inst.nodes
    tot = defaultdict(float)
    shown = 0
    for n, ep in episodes_of(task, entropy, episodes, name, only):
        rec, trace = ep["records"], {tr["t"]: tr for tr in ep["trace"]}
        first = {}
        for t in sorted(trace):
            tr = trace[t]
            for (_t, go, rule, moved) in tr["log"]:
                if rule != "lapse" or t - tr["held"].get(go, -10**9) <= MEMORY:
                    continue
                g = inst.grids[go]
                ga = N[g].grid
                r = ga.rationed
                gr = tr["grids"][go]
                fu = gr["fuels"][r]
                burn = ga.shares[r] * gr["G"]
                thr = inst.params.psi * ga.ibar[r]
                pl = tr["plan"]
                asked = most = target = 0.0
                for fo in inst.grid_fabs[go]:
                    fa = N[inst.fabs[fo]].fab
                    fb = gr["fabs"][fo]
                    if fa.e <= 0 or fb["R"] <= 0:
                        continue
                    if P.tail is not None and t + P.lead[fo] + P.tail > inst.T:
                        continue
                    cap = fb["alpha"] * fb["R"] * fa.cap0
                    now_lots = min(cap, fb["wafers"] + fb["warr"])
                    H = pl["H"] if pl is not None else 0
                    later = cap if pl is None or H <= 2 else min(cap, float(np.mean(np.maximum(pl["p"][fo, 2:min(6, H)], 0.0))))
                    asked += fa.e * max(now_lots, later) / fb["R"]
                    target += fa.e * now_lots / fb["R"]
                    most += fa.e * cap / fb["R"]
                needed = burn - (most - asked)
                comes = min(max(fu["inflow"], tr["rate"].get((go, r), 0.0)), burn)
                lacking = max(min(needed, burn) - comes, 0.0)
                weeks = max(P.last[go] - t, 0)
                worth = P.gain[go] / P.full_draw[go] + ga.voll
                on_hand = fu["here"] + fu["arr"]
                ration = min(1.0, fu["here"] / thr)
                now = min(burn * ration, on_hand + fu["tank"])
                lost = max(target - weeks * max(most - asked, 0.0), 0.0)
                waste = max(now - (inst.T - t) * lacking, 0.0)
                lhs, rhs = weeks * min(lacking, asked) * worth / 1e9, (lost * worth + waste * ga.voll) / 1e9
                mild = lhs < rhs
                tot["events"] += 1
                tot["would go on"] += mild
                key = (n, go)
                if mild and key not in first:
                    first[key] = t
                    tot["episodes-grids with a first flip"] += 1
                if mild and shown < show:
                    shown += 1
                    print(f"ep {n:3d} wk {t:3d} {N[g].id:8s} comes {comes / burn:4.2f} needed {needed / burn:4.2f} lacking {lacking:6.1f} GWh weeks {weeks:2d} draw now/forecast/full {target:4.0f}/{asked:4.0f}/{most:4.0f} buys {lhs:6.1f} bn < costs {rhs:6.1f} bn (lots {lost * worth / 1e9:5.1f}, fuel {waste * ga.voll / 1e9:5.1f})")
    for k, v in tot.items():
        print(f"  {k}: {v:.0f}")

if __name__ == "__main__":
    fire.Fire(main)
