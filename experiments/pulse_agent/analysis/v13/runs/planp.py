"""At given (episode, week, grid) of a recorded play: the plan's lots per fab of the grid over the window, and the real lots after."""
import fire, numpy as np
from ana import instance, episodes_of

def main(events, task="small", entropy=6006, episodes=128, name="r13rec_off"):
    inst, P = instance(task)
    N = inst.nodes
    evs = [tuple(e.split(":")) for e in (events if isinstance(events, str) else ",".join(map(str, events))).split(",")]
    eps = sorted({int(e[0]) for e in evs})
    data = {n: ep for n, ep in episodes_of(task, entropy, episodes, name, ",".join(map(str, eps)))}
    for n, t, grid in evs:
        n, t = int(n), int(t)
        ep = data[n]
        go = inst.grid_ordinal[inst.node_index[grid]]
        tr = {x["t"]: x for x in ep["trace"]}[t]
        pl = tr["plan"]
        for fo in inst.grid_fabs[go]:
            fa = N[inst.fabs[fo]].fab
            fb = tr["grids"][go]["fabs"][fo]
            cap = max(fb["alpha"] * fb["R"] * fa.cap0, 1.0)
            real = " ".join(f"{ep['records'][u - 1].lots_started[fo] / cap:4.2f}" for u in range(t, min(t + 8, inst.T) + 1))
            print(f"ep {n} wk {t} {N[inst.fabs[fo]].id:18s} wafers {(fb['wafers'] + fb['warr']) / cap:4.2f} plan p/cap wk1-10: {' '.join(f'{v / cap:4.2f}' for v in pl['p'][fo, :10])} | real lots/cap from this week: {real}")

if __name__ == "__main__":
    fire.Fire(main)
