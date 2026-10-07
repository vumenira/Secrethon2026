"""Why the trim's forecast of next week's fab draw was too high: per fab, the wafers forecast against those it had.

uv run python fabdraw.py --task=small --n=3 --grid=grid_tw --first=20 --last=26
"""

import pickle
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402


def main(task="small", n=3, grid="grid_tw", first=1, last=None, tag="v6", entropy=12345):
    d = pickle.loads((HERE / "traj" / f"{tag}_{task}_{entropy}_{n}.pkl").read_bytes())
    inst, _omega, marks, _fb = C.world(task, n, entropy)
    inst = inst.at_digest(marks.instance_digest)
    node = [nd.id for nd in inst.nodes]
    go = next(i for i, g in enumerate(inst.grids) if node[g] == grid)
    recs, notes = d["records"], d["why"]
    for t in range(int(first), int(last or inst.T - 1) + 1):
        note = notes.get(t, {}).get(go)
        if not note or "fabs" not in note[1]:
            print(f"wk {t}: {note[0] if note else 'no note'}")
            continue
        i = note[1]
        nxt = recs[t]  # week t + 1
        print(f"wk {t} {note[0]}: forecast need {i.get('need', 0):.0f} (y {i['y']:.0f}) others {i.get('others', 0):.0f} needed {i.get('needed', 0):.0f}; next week realised fab energy {sum(float(nxt.energy[fo]) for fo in i['fabs']):.0f} shed {float(nxt.shed[go]):.0f}")
        for fo, fb in i["fabs"].items():
            f = inst.fabs[fo]
            fa = inst.nodes[f].fab
            s = inst.slot_index[(f, fa.input)]
            had = float(nxt.stock[s]) + float(nxt.lots_started[fo])  # wafers on hand next week before its lots
            fore = fb["wafers"] - i["rho"] * fb["lots"] + fb["soon"] + sum(fb["sent1"].values())
            line = f"   {node[f]:18s} wafers now {fb['wafers']:9.0f} lots now {fb['lots']:8.0f} soon {fb['soon']:8.0f} sent over 1-week edges {sum(fb['sent1'].values()):8.0f} -> forecast {fore:9.0f}, had {had:9.0f}; cap {fb['cap']:8.0f} (next week's marks: R {float(marks.R[t, fo]):.2f} alpha {float(marks.alpha_bar[t, fo]):.2f}, seen R {fb['R']:.2f} alpha {fb['alpha']:.2f}); lots started next week {float(nxt.lots_started[fo]):8.0f}"
            print(line)
            for slot, q in fb["sent1"].items():
                e, k, lane = inst.action_slots[slot]
                ed = inst.edges[e]
                done = float(recs[t - 1].executed.get(slot, 0.0))
                tail_stock = float(recs[t - 2].stock[inst.slot_index[(ed.tail, k)]]) if t >= 2 else float("nan")
                print(f"      slot {slot} {node[ed.tail]}->{node[ed.head]} lane {lane}: requested {q:9.0f} executed {done:9.0f}; tail stock when the week began {tail_stock:9.0f}; u {float(marks.u[t - 1, e]):9.0f}")


if __name__ == "__main__":
    fire.Fire(main)
