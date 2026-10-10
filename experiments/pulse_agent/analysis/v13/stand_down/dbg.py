import pickle
from collections import Counter
d = pickle.load(open("outputs/s13/s13_census/full-e12345/ep14.pkl", "rb"))
c = Counter()
for t, go, code, row in d["census"]:
    if go != 2 or t > d["last"][2]:
        continue
    ehat = sum(min(f["cap"], f["wafers"]) for f in row["fabs"].values() if f["R"] > 0 and f["sold"])
    powered = row["power"] - row["y"] > 1e-9 * max(row["y"], 1.0)
    c[(code, ehat > 0, bool(powered))] += 1
print(c, d["last"])
