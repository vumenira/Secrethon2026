"""Print census rows of one episode and grid (no play)."""
import pickle
import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent


def main(root=12345, ep=14, go=2, codes="idle", name="s13_census", first=1, last=104, limit=40):
    d = pickle.load(open(HERE / name / f"full-e{root}" / f"ep{ep}.pkl", "rb"))
    want = codes.split(",") if isinstance(codes, str) else list(codes)
    n = 0
    for t, g, code, row in d["census"]:
        if g != go or not (first <= t <= last) or (want != ["all"] and code not in want):
            continue
        n += 1
        if n > limit:
            break
        fu = {k: tuple(round(f[x], 2 if x == "ration" else None) for x in ("burn", "ration", "on_hand", "tank", "inflow", "feed")) for k, f in row["fuels"].items()}
        fabs = {fo: (round(f["wafers"]), round(f["cap"]), f["R"], f["sold"], round(float(d["lots"][t - 1][fo]))) for fo, f in row["fabs"].items()}
        seg = {k: round(v) for (g2, k), v in d["segment"][t - 1].items() if g2 == go}
        extra = {k: row[k] for k in ("short", "also", "piping", "leaning", "blocked", "moved", "pulsed", "starting", "shocked") if row.get(k)}
        print(t, code, "m", round(row["power"] - row["y"], 1), "want", round(row["want"], 1), "target", round(row["target"], 1),
              "able-y", round(row["able"] - row["y"], 1), "shed", round(float(d["shed"][t - 1][go])), "fuels", fu, "seg", seg, "fabs", fabs, extra)


if __name__ == "__main__":
    fire.Fire(main)
