"""Compare cached harness rows of variants with the base's: exact cost match, errors, seconds (scratch)."""

import json
import sys
from pathlib import Path

import numpy as np


OUT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026\outputs\recovery_tune")


def main(n, base, base_n, *names):
    ref = json.loads((OUT / f"small-e12345-n{base_n}" / f"{base}.json").read_text())["rows"]
    refs = json.loads((OUT / f"small-e12345-n{n}" / "references.json").read_text())
    jb = {r["episode"]: r["J_policy_cents"] for r in ref}
    sb = {r["episode"]: r["seconds"] for r in ref}
    for name in names:
        doc = json.loads((OUT / f"small-e12345-n{n}" / f"{name}.json").read_text())
        rows = doc["rows"]
        same = sum(r["J_policy_cents"] == jb[r["episode"]] for r in rows)
        d = np.array([(jb[r["episode"]] - r["J_policy_cents"]) / 1e11 for r in rows])
        gap = np.array([(x["J_naive_cents"] - x["J_oracle_cents"]) / 1e11 for x in refs])
        strata = np.array([x["stratum"] for x in refs])
        sec = np.array([r["seconds"] for r in rows])
        print(
            f"{name}: params {doc['spec'].get('params')}\n"
            f"   episodes {len(rows)}, equal to {base} to the cent {same}, fallback weeks {sum(r['fallback_weeks'] for r in rows)},"
            f" cpu weeks {sum(r.get('cpu_weeks', 0) for r in rows)}, invalid {sum(r.get('invalid_entries', 0) for r in rows)},"
            f" errors {[r['first_error'] for r in rows if r.get('first_error')][:3]}\n"
            f"   seconds/episode mean {sec.mean():.2f} median {np.median(sec):.2f} max {sec.max():.1f}"
            f" (base file: mean {np.mean([sb[r['episode']] for r in rows]):.2f});"
            f" saved bn USD total {d.sum():+.1f}, by level "
            + " ".join(f"{s}: {d[strata[: len(d)] == s].sum():+.1f}" for s in (1, 2, 3, 4))
            + f"; largest {np.sort(d)[-3:][::-1].round(1).tolist()} smallest {np.sort(d)[:3].round(1).tolist()}"
        )


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), *sys.argv[4:])
