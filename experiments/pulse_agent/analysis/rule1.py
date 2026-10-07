"""The size of one pulse rule: a short fuel that cannot be fixed should not veto pulsing the fuels that can.

From the traced episodes: the grid-weeks in which the pulse stood down ("unfixable") although at least one short fuel
was holdable. In such a week, with the holdable fuels made whole and the others left as they were, the fabs would get
their draw less the unfixable segments' shortfall; a pulse on the holdable fuel gives that in a share of the weeks
equal to the fuel's arrival rate over its burn. The estimate is that energy minus what the fabs got.
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


def main(task="full", tag="v2"):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _ = task_generator(task)
    st = A.static(inst)
    name_k = {k: c.id for k, c in enumerate(inst.commodities)}
    report = json.loads((HERE / f"report_{task}.json").read_text())
    gap = report["gap"]
    rate = {g: (v["usd_planB0"] / (v["plan"] - v["B0"]) if v["plan"] - v["B0"] > 1 else 0.0) for g, v in report["tot"].items()}
    tot = defaultdict(lambda: defaultdict(float))
    paths = sorted((HERE / "trace" / f"{task}_{tag}").glob("*.pkl"), key=lambda q: int(q.stem))
    for path in paths:
        d = pickle.loads(path.read_bytes())
        p = d["played"]
        shares = {row[0]: row[4] for row in d["pulse"]["grids"]}
        lead = d["pulse"]["lead"]
        T = p["energy"].shape[0]
        for r in d["pulse"]["trace"]:
            if r["why"] != "unfixable":
                continue
            go, t = r["go"], r["t"]
            hold = [k for k, c in r["short"].items() if c == "holdable"]
            unfix = [k for k, c in r["short"].items() if c != "holdable"]
            fu = r["fuels"]
            missing = sum(fu[k]["burn"] - min(fu[k]["burn"] * fu[k]["ration"], fu[k]["on_hand"] + fu[k]["feed"]) for k in unfix)
            reach = r["target"] - missing  # the fabs' energy with the holdable fuels whole
            got = sum(p["energy"][t - 1, fo] for fo in st[go]["fabs"] if t + lead.get(fo, 0.0) <= T)
            key = (st[go]["id"], "+".join(sorted(name_k[k] for k in unfix)), "+".join(sorted(name_k[k] for k in hold)) or "-")
            row = tot[key]
            row["weeks"] += 1
            row["target"] += r["target"]
            row["got"] += got
            if reach > 0 and hold:
                duty = min(min(1.0, fu[k]["rate"] / fu[k]["burn"]) if fu[k]["burn"] > 0 else 1.0 for k in hold)
                row["could_weeks"] += 1
                row["pulse"] += max(reach * duty - got, 0.0)
                row["reach"] += reach
            elif reach > 0:
                row["already"] += 1
    print(f"{task}: {len(paths)} episodes. Grid-weeks the pulse stood down; 'reach' = fab energy with the holdable fuels whole and the rest as it was")
    print(f"{'grid':9s} {'cannot be fixed':18s} {'holdable':10s} {'weeks':>6s} {'fabs asked':>10s} {'got':>8s} | {'weeks a pulse could act':>23s} {'reach':>8s} {'pulse adds':>10s} {'bn':>7s} {'of gap':>7s}")
    total = 0.0
    for key, r in sorted(tot.items(), key=lambda kv: -kv[1]["pulse"]):
        usd = r["pulse"] * rate.get(key[0], 0.0)
        total += usd
        print(f"{key[0]:9s} {key[1]:18s} {key[2]:10s} {r['weeks']:6.0f} {r['target']:10.0f} {r['got']:8.0f} | {r['could_weeks']:23.0f} {r['reach']:8.0f} {r['pulse']:10.0f} {usd / 1e9:7.1f} {usd / gap:7.4f}")
    print(f"total: {total / 1e9:.0f} bn = {total / gap:.4f} of the gap (money at each grid's rate from the bound table)")


if __name__ == "__main__":
    fire.Fire(main)
