"""The fab calendars of one episode side by side: ours, the looser model's plan, the exact model's plan.

uv run python zshow.py --task=small --n=41 --weeks=30
"""

import json
import pickle

import fire
import numpy as np

import common as C


def main(task="small", n=41, weeks=30):
    grids = ["grid_tw", "grid_kr", "grid_jp", "grid_eu"]
    rows = {}
    p = C.HERE / "diff" / f"{task}_{n}.json"
    if p.exists():
        r = json.loads(p.read_text())
        rows["ours"], rows["loose"] = np.array(r["z_ours"]), np.array(r["z_final"])
    p = C.HERE / "tight" / f"{task}_{n}.pkl"
    if p.exists():
        t = pickle.loads(p.read_bytes())
        rows["exact"] = np.array(t["z"])
        rows.setdefault("ours", np.array(t["z_ours"]))
    for gi, g in enumerate(grids):
        for label, z in rows.items():
            print(f"  {g} {label:5s} " + "".join("#" if q > 0.5 else "." for q in z[gi, :weeks]) + f"   ({int(z[gi].sum())} weeks without shed)")


if __name__ == "__main__":
    fire.Fire(main)
