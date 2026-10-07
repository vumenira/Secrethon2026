"""Pooled scores of the realizable clairvoyant (milp_real.py) beside our agent, by harm level, for the episodes solved."""

import json
import sys
from pathlib import Path

import fire
import numpy as np


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "experiments" / "approaches"))
import foresight as F  # noqa: E402


def main(task="small", entropy=12345, ours="ft_s_v2", also="z_all", tag=""):
    total = {"small": 64, "full": 16}[task]
    es = F.tune.episode_set(task, entropy, total, 1)
    refs = list(es.references)
    out = F.tune.run_dir(task, entropy, total)
    res = json.loads((HERE / f"milp_{task}{tag}.json").read_text())
    played = {r["episode"]: r["J_policy_cents"] for r in json.loads((out / f"{ours}.json").read_text())["rows"]}
    other = {}
    for name in [n for n in also.split(",") if n and (out / f"{n}.json").exists()]:
        other[name] = {r["episode"]: r["J_policy_cents"] for r in json.loads((out / f"{name}.json").read_text())["rows"]}
    have = [r for r in refs if str(r["episode"]) in res and res[str(r["episode"])].get("J_usd") is not None]
    print(f"{task}: {len(have)} of {len(refs)} episodes solved; by harm level {[sum(r['stratum'] == s for r in have) for s in (1, 2, 3, 4)]}")
    status = {"optimal": 0, "time limit": 0}
    rows = []
    for r in have:
        m = res[str(r["episode"])]
        status["optimal" if m["status"] == 0 else "time limit"] += 1
        naive, oracle = r["J_naive_cents"], r["J_oracle_cents"]
        J_inc = min(int(round(m["J_usd"] * 100)), played[r["episode"]])  # our own trajectory is feasible in the model
        J_bnd = min(max(int(round(m["bound"] * 100)), oracle), J_inc)
        rows.append((r, played[r["episode"]], J_inc, J_bnd, int(round(m["J_usd"] * 100)) > played[r["episode"]]))
    print(f"solve status: {status}; incumbents worse than our own play (replaced by ours): {sum(x[4] for x in rows)}")

    def pooled(values):
        """The board's pooling on the solved episodes: levels weighted as the scorer weights them, where present."""
        from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

        by = {}
        for s in (1, 2, 3, 4):
            idx = [i for i, x in enumerate(rows) if x[0]["stratum"] == s]
            if idx:
                num = np.mean([rows[i][0]["J_naive_cents"] - values[i] for i in idx])
                den = np.mean([rows[i][0]["J_naive_cents"] - rows[i][0]["J_oracle_cents"] for i in idx])
                by[s] = (num, den)
        w = {s: STRATUM_WEIGHTS[s - 1] for s in by}
        num = sum(w[s] * by[s][0] for s in by)
        den = sum(w[s] * by[s][1] for s in by)
        return num / den, {s: by[s][0] / by[s][1] for s in by}

    lines = [("our agent (" + ours + ")", [x[1] for x in rows])]
    for name, vals in other.items():
        lines.append((name, [vals[x[0]["episode"]] for x in rows]))
    lines += [("realizable clairvoyant, incumbent", [x[2] for x in rows]), ("realizable clairvoyant, proven bound", [x[3] for x in rows])]
    for label, vals in lines:
        p, by = pooled(vals)
        print(f"   {label:40s} {p:.4f}   by level " + " ".join(f"{s}: {v:.3f}" for s, v in by.items()))
    gaps = [(x[2] - x[3]) / (x[0]["J_naive_cents"] - x[0]["J_oracle_cents"]) for x in rows]
    print(f"   incumbent to bound, per episode: median {np.median(gaps):.3f}, largest {max(gaps):.3f}; episodes within 0.01: {sum(g <= 0.01 for g in gaps)}")
    top = sorted(rows, key=lambda x: (x[1] - x[2]) / (x[0]["J_naive_cents"] - x[0]["J_oracle_cents"]), reverse=True)[:10]
    sc = lambda x, J: (x[0]["J_naive_cents"] - J) / (x[0]["J_naive_cents"] - x[0]["J_oracle_cents"])  # noqa: E731
    print("   largest differences: " + "; ".join(f"ep {x[0]['episode']} {sc(x, x[1]):.2f}->{sc(x, x[2]):.2f} (bound {sc(x, x[3]):.2f})" for x in top))


if __name__ == "__main__":
    fire.Fire(main)
