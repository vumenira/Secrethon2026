import json, sys, os
from collections import defaultdict
root = "outputs/recovery_tune"
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}
sets = [("full-e12345-n16", "v12_all"), ("full-e2026-n16", "v12_all"), ("full-e777-n32", "v12_all")]
rows = []
for d, name in sets:
    p = os.path.join(root, d, name + ".json")
    if not os.path.exists(p):
        print("missing", p, [f for f in os.listdir(os.path.join(root, d)) if "v12" in f]); continue
    refs = {r["episode"]: r for r in json.load(open(os.path.join(root, d, "references.json")))}
    v11p = os.path.join(root, d, "v11_final.json")
    v11 = {r["episode"]: r for r in json.load(open(v11p))["rows"]} if os.path.exists(v11p) else {}
    for r in json.load(open(p))["rows"]:
        ref = refs[r["episode"]]
        if ref.get("excluded"): continue
        n, c, j = ref["J_naive_cents"]/1e11, ref["J_clairvoyant_cents"]/1e11, r["J_policy_cents"]/1e11
        rows.append(dict(set=d, ep=r["episode"], s=ref["stratum"], naive=n, clair=c, pol=j, att=n-c, short=j-c,
                         rss=(n-j)/(n-c), harm=ref["harm_usd"]/1e9,
                         v11=(v11[r["episode"]]["J_policy_cents"]/1e11 if r["episode"] in v11 else None)))
def pooled(rs):
    by = defaultdict(list)
    for r in rs: by[r["s"]].append(r)
    num = sum(W[s]*sum(x["naive"]-x["pol"] for x in v)/len(v) for s, v in by.items())
    den = sum(W[s]*sum(x["att"] for x in v)/len(v) for s, v in by.items())
    return num/den, den, by
sc, den, by = pooled(rows)
print(f"episodes {len(rows)}  pooled RSS {sc:.4f}  weighted mean attainable {den:.0f} bn (denominator)")
for s in sorted(by):
    v = by[s]
    print(f" level {s}: n={len(v)} mean att {sum(x['att'] for x in v)/len(v):.0f} bn, mean shortfall {sum(x['short'] for x in v)/len(v):.0f} bn, level RSS {1-sum(x['short'] for x in v)/sum(x['att'] for x in v):.4f}, weight {W[s]}")
# each episode's contribution to pooled shortfall
for r in rows:
    r["contrib"] = W[r["s"]]*r["short"]/len(by[r["s"]])/den
tot = sum(r["contrib"] for r in rows)
print(f"total shortfall (1-RSS) {tot:.4f}")
rows.sort(key=lambda r: -r["contrib"])
cum = 0
print("top episodes by contribution to the pooled shortfall:")
for i, r in enumerate(rows[:20]):
    cum += r["contrib"]
    print(f" {i+1:2d} {r['set']:16s} ep{r['ep']:2d} L{r['s']} rss {r['rss']:.3f} short {r['short']:6.0f} bn att {r['att']:6.0f} bn contrib {r['contrib']:.4f} cum {cum:.4f} ({cum/tot:.0%})")
import statistics
print("median per-episode rss", statistics.median(r["rss"] for r in rows))
for q in (0.1, 0.25, 0.5):
    k = int(len(rows)*q); print(f" worst {q:.0%} of episodes ({k}) hold {sum(r['contrib'] for r in rows[:k])/tot:.0%} of shortfall")
# what 1 bn per episode is worth
print(f"1 bn USD saved in every episode = {sum(W[s] for s in by)/den:.5f} RSS;  0.001 RSS = {den/1000/sum(W[s] for s in by):.1f} bn per episode")
