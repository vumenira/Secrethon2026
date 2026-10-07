"""What the plans of the exact shed-week model (tight.py, calendar free) do in the opening and after, beside our play.

uv run python tightlook.py --task=small
"""

import pickle

import fire
import numpy as np

import common as C
import diff as Dm
import view as V


def main(task="small"):
    refs = C.references(task)
    acc = {}
    used = []
    for n in sorted(refs):
        p = C.HERE / "tight" / f"{task}_{n}.pkl"
        if not p.exists():
            continue
        plan = pickle.loads(p.read_bytes())
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        v = V.View(cal)
        T = cal.T
        xs = {"ours": d["zpi"].reshape(T, cal.nc), "plan": plan["x"]}
        zs = {"ours": plan["z_ours"], "plan": plan["z"]}
        used.append(n)
        for gi, (go, fabs, _S) in enumerate(cal.grids):
            g = cal.inst.grids[go]
            ga = cal.inst.nodes[g].grid
            name = v.node[g]
            draw = sum(cal.inst.nodes[cal.inst.fabs[fo]].fab.e * cal.inst.nodes[cal.inst.fabs[fo]].fab.cap0 for fo in fabs)
            ybar = np.asarray(cal.marks.y_bar)[:, go]
            for label, x in xs.items():
                s = Dm.grid_series(v, cal, x, gi)
                a = acc.setdefault((name, label), {k: 0.0 for k in ("noshed5", "noshed_late", "E8", "E_late", "gas4", "gas_del4", "gas_term4", "shed5", "shed_all", "crude4")})
                noshed = s["shed"] <= 1e-7 * ybar
                a["noshed5"] += noshed[:5].sum()
                a["noshed_late"] += noshed[5:].sum()
                a["E8"] += s["E"][:8].sum() / draw
                a["E_late"] += s["E"][8:].sum() / draw
                a["shed5"] += s["shed"][:5].sum()
                a["shed_all"] += s["shed"].sum()
                f = s["fuel"]["lng"]
                burn = ga.shares[v.kid["lng"]] * ga.deliverable
                a["gas4"] += f["G"][:4].sum() / burn
                a["gas_del4"] += (f["del"][:4].sum() / burn) if "del" in f else 0.0
                a["gas_term4"] += f["It"][3] / burn
                if "crude" in s["fuel"]:
                    a["crude4"] += s["fuel"]["crude"]["G"][:4].sum() / (ga.shares[v.kid["crude"]] * ga.deliverable)
    m = len(used)
    print(f"{task}: {m} episodes with a plan of the exact model (calendar free): {used}")
    print("  per grid, means per episode, ours | plan: weeks without shed in weeks 1-5 and in weeks 6-52; fab energy (whole weeks of the draw) in weeks 1-8 and 9-52; gas burnt in weeks 1-4, delivered from the terminal in weeks 1-4, left at the terminal after week 4 (weeks of full burn); crude burnt in weeks 1-4; base load shed in weeks 1-5 and in all (GWh)")
    for name in sorted({k[0] for k in acc}):
        o, p = acc[(name, "ours")], acc[(name, "plan")]
        f = lambda k, w=5, dp=2: f"{o[k] / m:{w}.{dp}f} | {p[k] / m:{w}.{dp}f}"  # noqa: E731
        print(f"  {name}: no shed wk 1-5 {f('noshed5')}; wk 6-52 {f('noshed_late', 5, 1)}; fab energy wk 1-8 {f('E8')}; wk 9-52 {f('E_late', 5, 1)}; gas burnt wk 1-4 {f('gas4')}; delivered {f('gas_del4')}; at the terminal after wk 4 {f('gas_term4')}; crude burnt wk 1-4 {f('crude4')}; shed wk 1-5 {f('shed5', 7, 0)}; all {f('shed_all', 8, 0)}")


if __name__ == "__main__":
    fire.Fire(main)
