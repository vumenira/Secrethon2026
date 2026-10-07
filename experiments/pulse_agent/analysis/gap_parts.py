"""Where the cost of a played variant sits against the clairvoyant plan: cost components, lots started, base load shed.

uv run python gap_parts.py --task=full --episodes=16 --names=fs_ev_h24,fs_pq_k0,fs_pq_all
"""

import json
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent


def main(task="full", entropy=12345, episodes=16, names="fs_ev_h24,fs_pq_k0,fs_pq_all"):
    out = ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}"
    refs = json.loads((out / "references.json").read_text())
    bound = json.loads((HERE / f"bound2_{task}_{episodes}.json").read_text())
    names = names.split(",") if isinstance(names, str) else list(names)
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float) / 100
    oracle = np.array([r["J_oracle_cents"] for r in refs], dtype=float) / 100
    gap = naive - oracle
    comps = ("shortage", "shed", "holding", "queue_holding", "freight", "war_risk", "tariff", "disposal")
    plans = {}
    for label in ("free", "bf"):
        d = [bound["details"][str(n)][label] for n in bound["episodes"]]
        plans[label] = {
            "J": np.array(bound[label]),
            "parts": {c: np.array([x["parts"][c] for x in d]) for c in comps},
            "lots": np.array([sum(x["lots"]) for x in d]),
            "shed": np.array([sum(x["shed"]) for x in d]),
        }
    played = {}
    for name in names:
        rows = json.loads((out / f"{name}.json").read_text())["rows"]
        played[name] = {
            "J": np.array([r["J_policy_cents"] for r in rows], dtype=float) / 100,
            "parts": {c: np.array([r["parts"][c] for r in rows]) for c in comps},
            "lots": np.array([sum(r["lots"]) for r in rows]),
            "shed": np.array([sum(r["shed"]) for r in rows]),
        }
    free = plans["free"]
    print(f"{task}: sums over the {len(refs)} episodes, bn USD; the naive-to-clairvoyant gap is {gap.sum() / 1e9:,.0f} bn")
    print(f"{'':22s} {'J':>9s} {'shortage':>9s} {'shed':>9s} {'other':>8s} {'lots, share of plan':>20s} {'shed energy, x plan':>20s}")
    for label, v in [("clairvoyant plan", free), ("plan, rule every week", plans["bf"])] + list(played.items()):
        other = sum(v["parts"][c].sum() for c in comps if c not in ("shortage", "shed"))
        print(
            f"{label:22s} {v['J'].sum() / 1e9:9,.0f} {v['parts']['shortage'].sum() / 1e9:9,.0f} {v['parts']['shed'].sum() / 1e9:9,.0f}"
            f" {other / 1e9:8,.0f} {v['lots'].sum() / free['lots'].sum():20.3f} {v['shed'].sum() / free['shed'].sum():20.3f}"
        )
    print("the cost above the clairvoyant plan, by component (share of the naive-to-clairvoyant gap, all episodes alike):")
    for name, v in played.items():
        extra = {c: (v["parts"][c] - free["parts"][c]).sum() / gap.sum() for c in comps}
        other = sum(extra[c] for c in comps if c not in ("shortage", "shed"))
        total = (v["J"] - free["J"]).sum() / gap.sum()
        print(f"  {name:20s} total {total:+.3f}: shortage {extra['shortage']:+.3f}, shed {extra['shed']:+.3f}, other {other:+.3f}")
    print("per episode: share of the gap closed; plan with the rule every week against the played variants")
    for i, r in enumerate(refs):
        line = f"  episode {r['episode']:2d} level {r['stratum']}: rule every week {(naive[i] - plans['bf']['J'][i]) / gap[i]:.3f}"
        for name, v in played.items():
            line += f"  {name} {(naive[i] - v['J'][i]) / gap[i]:.3f}"
        print(line)
    for name, v in played.items():
        above = int(((naive - v["J"]) / gap > (naive - plans["bf"]["J"]) / gap).sum())
        print(f"{name}: scores above the every-week-rule plan value in {above} of {len(refs)} episodes")


if __name__ == "__main__":
    fire.Fire(main)
