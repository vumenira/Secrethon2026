"""The fab calendar by week, ours against the realizable plan's (after putting back what does not pay): the share of
episodes in which each grid sheds no base load, and the number of such weeks per episode.

uv run python calmean.py --task=small --weeks=20
"""

import json

import fire
import numpy as np

import common as C


def main(task="small", weeks=20):
    refs = C.references(task)
    Zo, Zp, ns = [], [], []
    for n in sorted(refs):
        p = C.HERE / "diff" / f"{task}_{n}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        if "z_final" not in r:
            continue
        Zo.append(np.array(r["z_ours"]))
        Zp.append(np.array(r["z_final"]))
        ns.append(n)
    Zo, Zp = np.array(Zo), np.array(Zp)
    grids = ["grid_tw", "grid_kr", "grid_jp", "grid_eu"]
    print(f"{task}, {len(ns)} episodes: share of episodes in which the grid sheds no base load, weeks 1..{weeks}: ours | realizable plan")
    for gi, g in enumerate(grids[: Zo.shape[1]]):
        print(f"  {g} ours " + " ".join(f"{x:3.0%}" for x in Zo[:, gi, :weeks].mean(axis=0)))
        print(f"  {g} plan " + " ".join(f"{x:3.0%}" for x in Zp[:, gi, :weeks].mean(axis=0)))
        print(f"      weeks without shed per episode: ours {Zo[:, gi].sum(axis=1).mean():.1f}, plan {Zp[:, gi].sum(axis=1).mean():.1f}; in weeks 1-5: ours {Zo[:, gi, :5].sum(axis=1).mean():.2f}, plan {Zp[:, gi, :5].sum(axis=1).mean():.2f}; in weeks 6-52: ours {Zo[:, gi, 5:].sum(axis=1).mean():.1f}, plan {Zp[:, gi, 5:].sum(axis=1).mean():.1f}")


if __name__ == "__main__":
    fire.Fire(main)
