"""Per episode: what each variant saves over the base, beside the capacity cells the agent classes as strikes."""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RUN = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune\small-e12345-n64")


def costs(name):
    rows = json.loads((RUN / f"{name}.json").read_text())["rows"]
    rows = sorted(rows, key=lambda r: r["episode"])
    return np.array([r["J_policy_cents"] for r in rows], dtype=float), [r["episode"] for r in rows]


def main(*names):
    names = list(names) or ["mean_capacity", "truth_capacity"]
    data = pickle.loads((Path(__file__).parent / "capcells.pkl").read_bytes())
    refs = json.loads((RUN / "references.json").read_text())
    base, eps = costs("mpc_det_h20")
    gains = {n: (base - costs(n)[0]) / 1e11 for n in names}
    gap = {r["episode"]: (r["J_naive_cents"] - r["J_oracle_cents"]) / 1e11 for r in refs}
    level = {r["episode"]: r["stratum"] for r in refs}
    per = {p["episode"]: p for p in data["per_episode"]}
    table = []
    for i, n in enumerate(eps):
        cells = Counter()
        for (cls, causes), v in per[n]["cells"].items():
            if cls == "kept":
                cells["kept"] += v
            elif causes == "strike":
                cells["strike_ok"] += v
            else:
                cells["strike_wrong"] += v
        table.append((n, level[n], gap[n], cells, [gains[k][i] for k in names], len(per[n]["strikes"])))
    print("episode level gap   strikes  cells: classed strike & is one / classed strike & is not / kept    saved bn: " + " ".join(names))
    order = sorted(table, key=lambda r: r[4][0])
    for n, lv, g, cells, gs, ns in order:
        print(f"{n:4d} {lv} {g:8.1f} {ns:3d}   {cells['strike_ok']:6d} {cells['strike_wrong']:6d} {cells['kept']:6d}    " + " ".join(f"{x:+9.3f}" for x in gs))
    a = np.array([[r[3]["strike_ok"], r[3]["strike_wrong"], r[3]["kept"]] for r in table], dtype=float)
    for j, k in enumerate(names):
        g = np.array([r[4][j] for r in table])
        rel = g / np.array([r[2] for r in table])
        print(f"\n{k}: total saved {g.sum():+.2f} bn; in episodes with no cell classed strike: {g[(a[:, 0] + a[:, 1]) == 0].sum():+.3f}"
              f" ({int(((a[:, 0] + a[:, 1]) == 0).sum())} episodes)")
        for label, col in (("classed strike, is one", 0), ("classed strike, is not", 1)):
            m = a[:, col] > 0
            print(f"   episodes with cells '{label}': {int(m.sum())}, saved {g[m].sum():+.2f}; without: {int((~m).sum())}, saved {g[~m].sum():+.2f}")
        only_ok = (a[:, 0] > 0) & (a[:, 1] == 0)
        only_wrong = (a[:, 0] == 0) & (a[:, 1] > 0)
        both = (a[:, 0] > 0) & (a[:, 1] > 0)
        for label, m in (("only true strikes", only_ok), ("only false strikes", only_wrong), ("both", both)):
            print(f"   {label}: {int(m.sum())} episodes, saved {g[m].sum():+.2f} bn, sum of saved/gap {rel[m].sum():+.4f}, better/worse {int((g[m] > 1e-6).sum())}/{int((g[m] < -1e-6).sum())}")


if __name__ == "__main__":
    main(*sys.argv[1:])
