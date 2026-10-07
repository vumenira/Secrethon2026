"""Events that are on at week 1 and end inside the episode, and events that start early: per episode and in all.

uv run python ends.py --task=small --show=4,42,0,46
"""

from collections import Counter

import fire
import numpy as np

import common as C
import worlds as W


def main(task="small", show=None):
    from shockbench_flow.omega import codes

    refs = C.references(task)
    show = [] if show is None else [int(v) for v in (show if isinstance(show, (list, tuple)) else [show])]
    T = {"small": 52, "full": 104}[task]
    n_on, n_end, n_end13, n_new, n_new13 = [], [], [], [], []
    types_end, types_on, types_new = Counter(), Counter(), Counter()
    for n in sorted(refs):
        inst, omega, _marks, _fb = C.world(task, n)
        evs = W.events(inst, omega)
        on = [q for q in evs if q.onset <= 0 < q.end]
        ending = [q for q in on if q.end < T]
        new = [q for q in evs if 0 < q.onset < T]
        n_on.append(len(on))
        n_end.append(len(ending))
        n_end13.append(sum(q.end < 13 for q in ending))
        n_new.append(len(new))
        n_new13.append(sum(q.onset < 13 for q in new))
        for q in on:
            types_on[codes.EVENT_TYPES[q.type]] += 1
        for q in ending:
            types_end[codes.EVENT_TYPES[q.type]] += 1
        for q in new:
            types_new[codes.EVENT_TYPES[q.type]] += 1
        if n in show:
            print(f"episode {n}: on at week 1: {len(on)}; of them ending inside the episode: " + "; ".join(W.describe(inst, q) for q in sorted(ending, key=lambda q: q.end)))
            print(f"    starting in weeks 1-12: " + "; ".join(W.describe(inst, q) for q in new if q.onset < 12))
    print(f"{task}, {len(refs)} episodes: events on at week 1 per episode {np.mean(n_on):.1f}; of them ending inside the episode {np.mean(n_end):.1f} (before week 13: {np.mean(n_end13):.1f}); events starting inside the episode {np.mean(n_new):.1f} (before week 13: {np.mean(n_new13):.1f})")
    print(f"  on at week 1 by type: {dict(types_on)}")
    print(f"  ending inside the episode by type: {dict(types_end)}")
    print(f"  starting inside the episode by type: {dict(types_new)}")


if __name__ == "__main__":
    fire.Fire(main)
