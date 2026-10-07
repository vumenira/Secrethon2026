"""The pinned ceiling by how far the calendar still moved: is the gain there where the pin held?

uv run python <this> --name=ft_s_v5_hc_pin_p1 --task=small
"""

import json
import sys
from pathlib import Path

import fire
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab  # noqa: E402


def main(name="ft_s_v5_hc_pin_p1", task="small", base=None):
    base = base or ("ft_s_v4" if task == "small" else "ft_v4")
    run = lab.OUT / f"{task}-e12345-n{64 if task == 'small' else 16}"
    rows = json.loads((run / f"{name}.json").read_text())["rows"]
    refs, cached = lab.references(task), lab.cached(task, base)
    gap = np.array([(r["J_naive_cents"] - r["J_oracle_cents"]) / 1e11 for r in refs])
    gain = np.array([(cached[r["episode"]] - r["J_policy_cents"]) / 1e11 for r in rows])
    moved = np.array([r["calendar_moved"] for r in rows])
    dshare = np.array([r["mean_share"][1] - r["mean_share"][0] for r in rows])
    print(f"{task} {name} against {base}: total gain {gain.sum():+.1f} bn of {gap.sum():.0f} ({gain.sum() / gap.sum():+.4f}); mean share of fab power {np.mean([r['mean_share'][0] for r in rows]):.4f} -> {np.mean([r['mean_share'][1] for r in rows]):.4f}")
    for label, mask in (
        ("the calendar held (mean share within 0.002)", np.abs(dshare) <= 0.002),
        ("the grids lost power (mean share down by more than 0.002)", dshare < -0.002),
        ("the grids gained power (up by more than 0.002)", dshare > 0.002),
    ):
        if mask.any():
            print(f"  {label}: {int(mask.sum())} episodes, gain {gain[mask].sum():+.1f} bn ({gain[mask].sum() / gap[mask].sum():+.4f} of their gaps), better {int((gain[mask] > 0).sum())} worse {int((gain[mask] < 0).sum())}, mean moved {moved[mask].mean():.4f}")
    if len(gain) > 3:
        c = np.corrcoef(dshare, gain / gap)[0, 1]
        slope = np.polyfit(dshare, gain / gap, 1)
        print(f"  correlation of the gain (share of gap) with the change of mean share: {c:+.2f}; fit: gain = {slope[1]:+.4f} {slope[0]:+.2f} x change")


if __name__ == "__main__":
    fire.Fire(main)
