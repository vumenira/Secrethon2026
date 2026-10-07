"""Per grid: the weeks its fabs ran, ours against the realizable plan, and the most whole weeks its fuel allows.

A whole week burns a full week of every fuel segment, so with the fuel that reached a grid in our play (starting stock
plus arrivals) no more than min over gas and crude of (that fuel / a full week's burn) weeks can be whole. The table
sets our fab energy (in whole weeks of the fabs' draw) beside that ceiling, beside the plan under our calendar and
beside the realizable plan, with each plan's own fuel ceiling.

uv run python duty.py collect --task=small
uv run python duty.py table --task=small
"""

import json
import pickle

import fire
import numpy as np

import common as C
import diff as Dm
import view as V


def collect(task="small"):
    refs = C.references(task)
    out = {}
    for n in sorted(refs):
        p = Dm.OUT / f"{task}_{n}.pkl"
        if not p.exists():
            continue
        plan = pickle.loads(p.read_bytes())
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        v = V.View(cal)
        T = cal.T
        xs = {"ours": d["zpi"].reshape(T, cal.nc), "cal": plan["x_cal"], "plan": plan["x"]}
        rows = {}
        for gi, (go, fabs, _S) in enumerate(cal.grids):
            g = cal.inst.grids[go]
            ga = cal.inst.nodes[g].grid
            draw = sum(cal.inst.nodes[cal.inst.fabs[fo]].fab.e * cal.inst.nodes[cal.inst.fabs[fo]].fab.cap0 for fo in fabs)
            r = {"draw": draw, "emax": float(cal.emax[gi].sum()) / draw}
            for label, x in xs.items():
                s = Dm.grid_series(v, cal, x, gi)
                e = {"E": float(s["E"].sum()) / draw, "E8": float(s["E"][:8].sum()) / draw, "shed": float(s["shed"].sum()), "fuel": {}}
                e["on"] = int(((s["shed"] <= 1e-7 * ga.base_load) & (s["E"] > 1e-6 * draw)).sum())
                e["on_low"] = int(((s["shed"] <= 1e-7 * ga.base_load) & (s["E"] > 1e-6 * draw) & (s["E"] < 0.5 * cal.emax[gi])).sum())
                e["noshed_idle"] = int(((s["shed"] <= 1e-7 * ga.base_load) & (s["E"] <= 1e-6 * draw)).sum())
                for k, f in s["fuel"].items():
                    if k == "nucfuel":
                        continue
                    burn = ga.shares[v.kid[k]] * ga.deliverable
                    # what reached the grid or its terminal over the episode = burnt + left at the end (disposal aside);
                    # "start" holds 0 and "arrived" that total, so the ceiling below is their sum
                    left = float(f["Ig"][-1] + f["It"][-1])
                    e["fuel"][k] = {"start": 0.0, "arrived": (float(f["G"].sum()) + left) / burn, "burnt": float(f["G"].sum()) / burn, "left": left / burn}
                rows_label = e
                r[label] = rows_label
            rows[v.node[g]] = r
        out[n] = rows
        print(n, flush=True)
    (C.HERE / f"duty_{task}.json").write_text(json.dumps(out, indent=1))


def table(task="small"):
    data = {int(k): v for k, v in json.loads((C.HERE / f"duty_{task}.json").read_text()).items()}
    ns = sorted(data)
    grids = list(data[ns[0]])
    print(f"{task}, {len(ns)} episodes; per grid, means per episode. Fab energy in whole weeks of the fabs' full draw; fuel in weeks of a segment's full burn")
    for g in grids:
        R = [data[n][g] for n in ns]
        print(f"  {g}: the fabs could draw in {np.mean([r['emax'] for r in R]):.1f} weeks' worth (events on the fabs and the grid counted)")
        for label, name in (("ours", "our play"), ("cal", "best plan, our calendar"), ("plan", "realizable plan")):
            E = np.array([r[label]["E"] for r in R])
            E8 = np.array([r[label]["E8"] for r in R])
            on = np.array([r[label]["on"] for r in R])
            low = np.array([r[label]["on_low"] for r in R])
            idle = np.array([r[label]["noshed_idle"] for r in R])
            line = f"    {name:24s} fab energy {E.mean():5.1f} (weeks 1-8: {E8.mean():4.1f}); weeks without shed that powered fabs {on.mean():5.1f} (of them under half the draw {low.mean():4.1f}); without shed and fabs idle {idle.mean():4.1f};"
            ceil = []
            for r in R:
                c = min((f["start"] + f["arrived"]) for f in r[label]["fuel"].values())
                ceil.append(min(c, 52.0))
            line += f" fuel ceiling {np.mean(ceil):5.1f}"
            for k in R[0][label]["fuel"]:
                line += f"; {k} start {np.mean([r[label]['fuel'][k]['start'] for r in R]):4.1f} arrived {np.mean([r[label]['fuel'][k]['arrived'] for r in R]):5.1f} burnt {np.mean([r[label]['fuel'][k]['burnt'] for r in R]):5.1f} left {np.mean([r[label]['fuel'][k]['left'] for r in R]):4.1f}"
            print(line)
        # how far our fab energy is from our own fuel ceiling, by episode
        gap = np.array([min(min((f["start"] + f["arrived"]) for f in r["ours"]["fuel"].values()), r["emax"]) - r["ours"]["E"] for r in R])
        gp = np.array([r["plan"]["E"] - r["ours"]["E"] for r in R])
        print(f"    our fab energy below the ceiling of the fuel that reached the grid in our play: mean {gap.mean():.1f} weeks (median {np.median(gap):.1f}); the realizable plan above ours: mean {gp.mean():.1f} (median {np.median(gp):.1f})")


if __name__ == "__main__":
    fire.Fire({"collect": collect, "table": table})
