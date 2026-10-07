"""Per episode: variant a against variant b from the cached costs, as a share of the episode's naive-to-clairvoyant gap.

uv run python <this> --a=ft_q_e20_fd8 --b=f_e10_w4 --task=full --episodes=16
"""

import json
from pathlib import Path

import fire
import numpy as np


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")


def main(a="ft_q_e20_fd8", b="f_e10_w4", task="full", entropy=12345, episodes=16, show=8):
    run = OUT / f"{task}-e{entropy}-n{episodes}"
    refs = json.loads((run / "references.json").read_text())
    ra = json.loads((run / f"{a}.json").read_text())["rows"]
    rb = json.loads((run / f"{b}.json").read_text())["rows"]
    gap = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float)
    ja = np.array([r["J_policy_cents"] for r in ra], dtype=float)
    jb = np.array([r["J_policy_cents"] for r in rb], dtype=float)
    naive = np.array([r["J_naive_cents"] for r in refs], dtype=float)
    d = (jb - ja) / gap  # the episode's own score difference, a minus b
    sa, sb = (naive - ja) / gap, (naive - jb) / gap
    strata = [r["stratum"] for r in refs]
    worse = np.flatnonzero(d < -1e-6)
    print(f"{task}: {a} against {b}: better in {int((d > 1e-6).sum())}, worse in {worse.size} of {len(d)} episodes")
    print(f"  per-episode score of {a}: min {sa.min():.3f}, lower quartile {np.quantile(sa, 0.25):.3f}, median {np.median(sa):.3f}; of {b}: min {sb.min():.3f}, median {np.median(sb):.3f}")
    print(f"  per-episode difference: mean {d.mean():+.4f}, median {np.median(d):+.4f}, worst {d.min():+.4f}, best {d.max():+.4f}")
    print(f"  the losses, summed in bn USD: {((ja - jb)[worse]).sum() / 1e11:.1f}; the gains: {((jb - ja)[d > 1e-6]).sum() / 1e11:.1f}")
    for i in sorted(worse, key=lambda i: d[i])[:show]:
        print(f"    episode {refs[i]['episode']:3d} level {strata[i]}: {sb[i]:.3f} -> {sa[i]:.3f} ({d[i]:+.4f}, {(ja[i] - jb[i]) / 1e11:+.1f} bn)")
    for key in ("fallback_weeks", "invalid_entries", "cpu_weeks"):
        print(f"  {key}: {a} {sum(r.get(key) or 0 for r in ra)}, {b} {sum(r.get(key) or 0 for r in rb)}")
    print(f"  first errors: {[r['first_error'] for r in ra if r.get('first_error')][:3]}")
    by = {s: float(d[[i for i, x in enumerate(strata) if x == s]].mean()) for s in sorted(set(strata))}
    print("  mean per-episode difference by harm level:", {s: round(v, 4) for s, v in by.items()})


if __name__ == "__main__":
    fire.Fire(main)
