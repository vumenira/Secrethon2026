"""One grid of one traced episode, week by week: what the fabs asked and got, the plan, the fuel, the pulse's state."""

import pickle
import sys
from pathlib import Path

import fire
import numpy as np


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import analyze_v2 as A  # noqa: E402


def main(task="small", episode=0, grid="grid_kr", tag="v2", fuel=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    d = pickle.loads((HERE / "trace" / f"{task}_{tag}" / f"{episode}.pkl").read_bytes())
    go = next(g for g in st if st[g]["id"] == grid)
    gw = A.grid_weeks(inst, st, d, go)
    plan = d["plan"]
    col = {key: j for j, key in enumerate(plan["columns"])}
    Ep = sum(plan["x"][:, col[("E", fo)]] for fo in st[go]["fabs"])
    pp = sum(plan["x"][:, col[("p", fo)]] for fo in st[go]["fabs"])
    yshp = plan["x"][:, col[("ysh", go)]]
    why = {r["t"]: r for r in d["pulse"]["trace"] if r["go"] == go}
    names = {k: inst.commodities[k].id for k in gw["fuels"]}
    print(f"{grid} episode {episode}: fuels {[(names[k], round(st[go]['fuels'][k]['share'], 3), 'terminal' if fu['terminal'] else 'no terminal') for k, fu in gw['fuels'].items()]}")
    ks = [k for k in gw["fuels"] if fuel is None or names[k] == fuel]
    head = f"{'wk':>3s} {'Ecap':>5s} {'want':>5s} {'got':>5s} {'plan':>5s} {'lots':>7s} {'plan':>7s} {'shed':>6s} {'plan':>6s} {'class':9s} {'pulse':10s}"
    for k in ks:
        head += f" | {names[k][:4]:>4s}: {'arr':>6s} {'deliv':>6s} {'burn':>6s} {'need':>6s} {'grid':>6s} {'term':>6s} {'rat':>4s}"
    print(head)
    for t in range(gw["T"]):
        r = why.get(t + 1, {})
        note = (r.get("why") or "") + ("".join(f" {names[k][:3]}:{c[:4]}" for k, c in r.get("short", {}).items()) if r.get("why") == "unfixable" else "")
        line = (
            f"{t + 1:3d} {gw['Ecap'][t]:5.0f} {gw['Ewant'][t]:5.0f} {gw['E'][t]:5.0f} {Ep[t]:5.0f} {gw['lots'][t]:7.0f} {pp[t]:7.0f}"
            f" {gw['shed'][t]:6.0f} {yshp[t]:6.0f} {gw['cls'][t]:9s} {note:10s}"
        )
        for k in ks:
            fu = gw["fuels"][k]
            line += (
                f" | {'':5s} {fu['a_term'][t] + fu['a_direct'][t]:6.0f} {fu['deliver'][t]:6.0f} {fu['burn'][t]:6.0f} {fu['whole_burn'][t]:6.0f}"
                f" {fu['Ig'][t + 1]:6.0f} {fu['It'][t + 1]:6.0f} {fu['ration'][t]:4.2f}"
            )
        print(line)
    print(f"totals: Ecap {gw['Ecap'].sum():.0f} got {gw['E'].sum():.0f} plan {Ep.sum():.0f}; lots {gw['lots'].sum():.0f} plan {pp.sum():.0f}; shed {gw['shed'].sum():.0f} plan {yshp.sum():.0f}")


if __name__ == "__main__":
    fire.Fire(main)
