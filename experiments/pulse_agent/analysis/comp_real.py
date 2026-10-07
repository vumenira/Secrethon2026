"""Our agent's cost above the realizable clairvoyant, by cost component and by where the chips are missing (traced episodes)."""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_v2 as A  # noqa: E402
import milp_real as M  # noqa: E402


def main(task="small", entropy=12345, tag="v2"):
    from shockbench_flow.hosting.tasks import task_generator
    from shockbench_flow.oracle.lp import build_lp, lp_costs

    inst, _ = task_generator(task)
    st = A.static(inst)
    total = {"small": 64, "full": 16}[task]
    refs = {r["episode"]: r for r in json.loads((A.ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{total}" / "references.json").read_text())}
    comp = defaultdict(float)
    E = defaultdict(lambda: defaultdict(float))
    gap = ours_J = real_J = 0.0
    lots = defaultdict(float)
    n_ep = 0
    for path in sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem)):
        n = int(path.stem)
        mp = M.OUT / f"{task}_{n}.pkl"
        if not mp.exists():
            continue
        d = pickle.loads(path.read_bytes())
        m = pickle.loads(mp.read_bytes())
        _inst, _omega, marks, _fb = M.world(task, entropy, n)
        model = build_lp(inst, marks)
        weekly, salvage = lp_costs(model, m["x"].ravel())
        p = d["played"]
        n_ep += 1
        gap += (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 100
        ours_J += p["J"]
        real_J += m["J"]
        for k in p["costs"][0]:
            comp[k] += sum(c[k] for c in p["costs"]) - sum(getattr(w, k) for w in weekly)
        comp["terminal credit"] += -(p["salvage"] - salvage)
        colm = {key: j for j, key in enumerate(m["columns"])}
        for go in st:
            if not st[go]["fabs"]:
                continue
            name = st[go]["id"]
            E[name]["ours"] += sum(p["energy"][:, fo].sum() for fo in st[go]["fabs"])
            E[name]["real"] += sum(m["x"][:, colm[("E", fo)]].sum() for fo in st[go]["fabs"])
            for fo in st[go]["fabs"]:
                fid = inst.nodes[inst.fabs[fo]].id
                lots[(name, fid, "ours")] += p["lots"][:, fo].sum()
                lots[(name, fid, "real")] += m["x"][:, colm[("p", fo)]].sum()
    print(f"{task}: {n_ep} traced episodes; ours above the realizable incumbent by {(ours_J - real_J) / 1e9:,.0f} bn = {(ours_J - real_J) / gap:.3f} of their gap")
    for k, x in sorted(comp.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:16s} {x / 1e9:9,.0f} bn  {x / gap:+.4f}")
    print("   fab energy by grid, ours and realizable (GWh), and lots by fab (ours / realizable):")
    for name, v in E.items():
        fabs = "; ".join(f"{fid} {lots[(g, fid, 'ours')] / lots[(g, fid, 'real')] if lots[(g, fid, 'real')] else float('nan'):.2f}" for (g, fid, who) in lots if g == name and who == "ours")
        print(f"   {name:9s} {v['ours']:9.0f} {v['real']:9.0f}  ours/realizable {v['ours'] / v['real'] if v['real'] else float('nan'):.2f} | lots: {fabs}")


if __name__ == "__main__":
    fire.Fire(main)
