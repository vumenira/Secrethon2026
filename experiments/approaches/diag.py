"""Where two variants differ: the episodes with the largest cost gap and the events of each (onsets in weeks).

uv run python experiments/approaches/diag.py --a=rec_mean --b=mpc_det_h20 --top=8
"""

import json
from collections import Counter
from pathlib import Path

import fire
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "outputs" / "recovery_tune"


def main(a, b="mpc_det_h20", task="small", entropy=12345, episodes=64, top=8, events=True):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.omega import codes
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    run = OUT / f"{task}-e{entropy}-n{episodes}"
    refs = json.loads((run / "references.json").read_text())
    ja = [r["J_policy_cents"] for r in json.loads((run / f"{a}.json").read_text())["rows"]]
    jb = [r["J_policy_cents"] for r in json.loads((run / f"{b}.json").read_text())["rows"]]
    gain = (np.array(jb, dtype=float) - np.array(ja, dtype=float)) / 1e11  # bn USD that a saves over b
    gap = np.array([r["J_naive_cents"] - r["J_oracle_cents"] for r in refs], dtype=float) / 1e11
    print(
        f"{a} minus {b}: total saved {gain.sum():+.1f} bn USD over {len(refs)} episodes;"
        f" naive-to-clairvoyant gap {gap.sum():.1f}"
    )
    for s in (1, 2, 3, 4):
        idx = [i for i, r in enumerate(refs) if r["stratum"] == s]
        print(f"  level {s}: {len(idx)} episodes, saved {gain[idx].sum():+.1f}, gap {gap[idx].sum():.1f}")
    order = np.argsort(-np.abs(gain))[:top]
    cache = str(default_cache_dir())
    for i in order:
        r = refs[i]
        print(f"episode {r['episode']} level {r['stratum']}: {a} saves {gain[i]:+.2f} bn (gap {gap[i]:.1f})")
        if not events:
            continue
        inst, omega, marks, _fallback = _world(task, entropy, r["episode"], REPLICATIONS, cache)
        keys = ("ev_type", "ev_onset", "ev_duration", "ev_severity", "ev_target", "ev_target_kind")
        ev = {k: np.asarray(omega.arrays[k]) for k in keys}
        live = np.flatnonzero((ev["ev_onset"] + ev["ev_duration"] > 0) & (ev["ev_onset"] < inst.T))
        count = Counter(codes.EVENT_TYPES[t] for t in ev["ev_type"][live])
        print("   events in the episode:", dict(count))
        for j in live:
            name = codes.EVENT_TYPES[ev["ev_type"][j]]
            if name in ("tariff", "sanction", "piracy"):
                continue
            print(
                f"     {name:20s} onset {ev['ev_onset'][j]:7.1f} lasts {ev['ev_duration'][j]:7.1f} wk"
                f" sev {ev['ev_severity'][j]:.2f}"
                f" target {codes.TARGET_KINDS[ev['ev_target_kind'][j]]} {ev['ev_target'][j]}"
            )


if __name__ == "__main__":
    fire.Fire(main)
