"""The largest clusters whose extra fab energy comes mainly from one source (diffrep.split), to pick foresight checks.

uv run python topcl.py --task=small --match="shed more" --top=6
"""

import json

import fire

import common as C
from diffrep import split
from report import weights


def main(task="small", match="shed more", top=6, min_start=2):
    refs = C.references(task)
    w = weights(task, sorted(refs))
    rows = []
    for n in sorted(refs):
        p = C.HERE / "diff" / f"{task}_{n}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        for c in r.get("clusters", []):
            shares, dE = split(c, r.get("z_ours"))
            s = sum(v for k, v in shares.items() if match in k)
            if c["a"] >= min_start and dE > 0:
                rows.append((w[n] * c.get("step", 0.0) * s, n, c["grid"], c["a"], c["b"], s, c["on"], c["off"]))
    rows.sort(reverse=True)
    for val, n, g, a, b, s, on, off in rows[:top]:
        print(f"  episode {n} {g} weeks {a}-{b}: {val:.4f} pooled from '{match}' ({s:.0%} of the cluster); it runs {on}, sheds {off}")


if __name__ == "__main__":
    fire.Fire(main)
