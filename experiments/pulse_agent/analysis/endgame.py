"""Two follow-ups on the traced episodes: the held fuel left when lots stop being sellable, and the other costs.

1. Per grid: the bound "accepted fuels held, gas charge" again, but leaving in tank and grid what our agent left there
   at the last sellable week (reserve and stranded fuel). The difference is what that fuel could still have powered.
2. The played cost above the clairvoyant plan by cost component.
"""

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
import report_v2 as R  # noqa: E402


def main(task="full", tag="v2", entropy=12345, episodes=None):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    episodes = episodes or {"full": 16, "small": 64}[task]
    refs = {r["episode"]: r for r in json.loads((A.ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}" / "references.json").read_text())}
    paths = sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem))
    tot = defaultdict(lambda: defaultdict(float))
    comp = defaultdict(float)
    gap = 0.0
    name_k = {k: inst.commodities[k].id for k in range(len(inst.commodities))}
    left_tab = defaultdict(float)
    for path in paths:
        d = pickle.loads(path.read_bytes())
        n = int(path.stem)
        gap += (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 100
        for k in d["played"]["costs"][0]:
            comp[k] += sum(c[k] for c in d["played"]["costs"]) - sum(c[k] for c in d["plan"]["costs"])
        v_fab, lots_o, lots_p, _rows, _kinds = R.money(inst, d)
        accepted = {row[0]: {k for k, val in row[3].items() if val[2] and val[3]} for row in d["pulse"]["grids"]}
        for go in st:
            if not st[go]["fabs"]:
                continue
            gw = A.grid_weeks(inst, st, d, go)
            name = st[go]["id"]
            acc = accepted.get(go, set())
            target = float(gw["Eplan"].sum())
            live = gw["Ecap"] > A.EPS
            if not live.any():
                continue
            last = int(np.flatnonzero(live)[-1])
            b3 = A.bound(gw, acc, charge=True, cap=target)
            # the same with our agent's leftover kept: take it out of what is usable from the last sellable week on
            g2 = dict(gw)
            g2["fuels"] = {}
            for k, fu in gw["fuels"].items():
                f2 = dict(fu)
                if k in acc:
                    keep = fu["Ig"][last + 1] + fu["It"][last + 1]
                    left_tab[(name, name_k[k])] += keep
                    a = fu["a_direct"].copy()
                    a[last] -= keep  # may go negative: the cumulative sum is what the bound reads
                    f2["a_direct"] = a
                g2["fuels"][k] = f2
            b3k = min(A.bound(g2, acc, charge=True, cap=target), b3)
            got = min(float(gw["E"].sum()), target)
            usd = sum(max(lots_p[fo] - lots_o[fo], 0.0) * v_fab[fo] for fo in st[go]["fabs"])
            v = usd / (target - got) if target - got > A.EPS else 0.0
            row = tot[name]
            row["b3"] += b3
            row["b3k"] += b3k
            row["got"] += got
            row["usd_left"] += (b3 - b3k) * v
            row["usd_rest"] += (b3k - got) * v
            # the weeks the fabs had all the energy they asked for: had wafers been at hand, they draw their capacity
            full = float(np.where(gw["cls"] == "whole", gw["Ecap"], gw["E"]).sum())
            got_w = max(min(full, b3k), got)
            row["waf"] += got_w - got
            row["usd_waf"] += (got_w - got) * v
            row["on_cap"] += float(gw["Ecap"][gw["cls"] == "whole"].sum())
            row["on_got"] += float(gw["E"][gw["cls"] == "whole"].sum())
            row["on_plan"] += float(gw["Eplan"][gw["cls"] == "whole"].sum())
            row["plan"] += target
    print(f"{task}: {len(paths)} episodes; gap {gap / 1e9:,.0f} bn")
    print("1. Held fuel left in tank and grid at the last sellable week (accepted fuels; reserve and stranded fuel), GWh of fab energy:")
    print(f"   {'grid':9s} {'accepted+charge':>15s} {'leftover kept':>13s} {'got':>9s} | {'leftover':>9s} {'bn':>7s} {'of gap':>7s} | {'the rest':>9s} {'bn':>7s} {'of gap':>7s}")
    P = defaultdict(float)
    for name, r in tot.items():
        print(
            f"   {name:9s} {r['b3']:15.0f} {r['b3k']:13.0f} {r['got']:9.0f} | {r['b3'] - r['b3k']:9.0f} {r['usd_left'] / 1e9:7.1f} {r['usd_left'] / gap:7.4f} |"
            f" {r['b3k'] - r['got']:9.0f} {r['usd_rest'] / 1e9:7.1f} {r['usd_rest'] / gap:7.4f}"
        )
        for c, x in r.items():
            P[c] += x
    print(
        f"   {'pooled':9s} {P['b3']:15.0f} {P['b3k']:13.0f} {P['got']:9.0f} | {P['b3'] - P['b3k']:9.0f} {P['usd_left'] / 1e9:7.1f} {P['usd_left'] / gap:7.4f} |"
        f" {P['b3k'] - P['got']:9.0f} {P['usd_rest'] / 1e9:7.1f} {P['usd_rest'] / gap:7.4f}"
    )
    print("   Of 'the rest': fab energy the whole weeks would have drawn at capacity (wafers at hand), within the bound:")
    for name, r in list(tot.items()) + [("pooled", P)]:
        print(
            f"   {name:9s} whole weeks: capacity {r['on_cap']:9.0f} got {r['on_got']:9.0f} ({r['on_got'] / r['on_cap'] if r['on_cap'] else float('nan'):.2f}); wafers at hand would add {r['waf']:8.0f} GWh"
            f" = {r['usd_waf'] / 1e9:7.1f} bn = {r['usd_waf'] / gap:.4f} of the gap; then left of the rest {(r['usd_rest'] - r['usd_waf']) / gap:.4f}"
        )
    print("   fuel left, GWh of fuel: " + ", ".join(f"{g} {f} {x:,.0f}" for (g, f), x in sorted(left_tab.items()) if x > 1))
    print("2. Played cost above the plan by component, bn and share of the gap:")
    for k, x in sorted(comp.items(), key=lambda kv: -abs(kv[1])):
        print(f"   {k:14s} {x / 1e9:9,.0f} bn  {x / gap:+.4f}")


if __name__ == "__main__":
    fire.Fire(main)
