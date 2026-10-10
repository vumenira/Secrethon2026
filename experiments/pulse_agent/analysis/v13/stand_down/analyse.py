"""The census table: why the fabs of a grid got less than their capacity, week by week (no play; reads the pickles).

uv run python outputs/s13/analyse.py --name=s13_census --sets="full:12345:16,full:2026:16,full:777:32"
"""
import json
import pickle
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
W = {1: 0.50, 2: 0.30, 3: 0.15, 4: 0.05}
GRID = ["tw", "kr", "jp", "cn", "us", "eu", "sea", "in"]
FUEL = ["lng", "crude", "nuc"]


def load(name, sets):
    """Every episode's document with its weight in the pooled score (weight of its level / episodes of the level)."""
    docs = []
    for item in sets.split(","):
        task, entropy, episodes = item.split(":")
        folder = ROOT / "outputs" / "recovery_tune" / f"{task}-e{entropy}-n{episodes}"
        refs = json.loads((folder / "references.json").read_text())
        for ref in refs:
            p = HERE / name / f"{task}-e{entropy}" / f"ep{ref['episode']}.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:
                doc = pickle.load(f)
            doc["ref"], doc["set"] = ref, f"{entropy}"
            docs.append(doc)
    levels = defaultdict(int)
    for d in docs:
        levels[d["ref"]["stratum"]] += 1
    for d in docs:
        d["w"] = W[d["ref"]["stratum"]] / levels[d["ref"]["stratum"]]  # sums to 1 over the episodes
    att = sum(d["w"] * (d["ref"]["J_naive_cents"] - d["ref"]["J_oracle_cents"]) for d in docs) / 1e11
    return docs, att


def missing(f):
    """What a fuel's segment misses of its full week as the pulse read it (the terminal delivering all it can)."""
    return f["burn"] - min(f["burn"] * f["ration"], f["on_hand"] + f["feed"])


def main(name="s13_census", sets="full:12345:16", task="full", top=12, detail=True):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    docs, att = load(name, sets)
    print(f"{len(docs)} episodes; attainable (weighted mean) {att:.0f} bn per episode; 0.001 of score = {att / 1000:.2f} bn")
    packaged = defaultdict(set)
    for o in inst.osats:
        for raw, pk in inst.nodes[o].osat.packages.items():
            packaged[raw].add(pk)
    fabval, fab_e = {}, {}  # fab ordinal -> USD a lot is worth beyond the base load its energy would serve; its energy
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        for fo in inst.grid_fabs[go]:
            fa = inst.nodes[inst.fabs[fo]].fab
            goods = packaged[fa.product] | {fa.product}
            worth = max((d.pi for d in inst.demands if d.k in goods), default=0.0)
            fabval[fo] = worth - ga.voll * fa.e
            fab_e[fo] = fa.e
    feeder_slots = defaultdict(list)  # (grid ordinal, fuel) -> the action slots on its zero-lead edges into the grid
    for s, (e, k, _lane) in enumerate(inst.action_slots):
        ed = inst.edges[e]
        if ed.tau == 0 and ed.head in inst.grids:
            feeder_slots[(inst.grids.index(ed.head), k)].append(s)
    # reason -> [grid-weeks, weighted bn power-lost, weighted bn wafer-lost, plain bn power-lost]
    table = defaultdict(lambda: np.zeros(4))
    by_grid = defaultdict(lambda: np.zeros(4))
    mild = defaultdict(lambda: np.zeros(3))  # (grid, fuel) -> weeks, weighted bn lost, what a steady pulse recovers
    dead = defaultdict(lambda: np.zeros(3))  # (grid, blocked fuels) -> weeks, bn lost, bn if the others were pulsed
    h5 = defaultdict(float)
    h3 = defaultdict(lambda: np.zeros(3))
    leak = defaultdict(lambda: np.zeros(3))  # (grid, fuel, reason) -> weeks, weeks of burn, weighted bn as whole weeks
    under = defaultdict(lambda: np.zeros(3))  # mild weeks: (grid, fuel) -> weeks, delivered, deliverable (weeks of burn)
    per_ep = []
    runs = defaultdict(list)  # (set, episode, grid, reason) -> lengths of its spells
    for d in docs:
        w = d["w"]
        rows = {(t, go): (code, row) for t, go, code, row in d["census"]}
        logs = defaultdict(list)
        for t, go, rule, _moved in d["log"]:
            logs[(t, go)].append(rule)
        ep_tot = defaultdict(float)
        prev = {}
        scarce_at = {}  # (grid, fuel) -> the last week the pulse counted the fuel short, held or blocked
        for (t, go), (code, row) in sorted(rows.items()):
            tail = t > d["last"].get(go, 10**9)
            lost_p = lost_w = 0.0
            ehat, egot, vals = [], [], []
            for fo, fab in row["fabs"].items():
                started = float(d["lots"][t - 1][fo])
                want = min(fab["cap"], fab["wafers"]) if fab["R"] > 0 else 0.0
                if not fab["sold"]:
                    continue
                lost_p += max(want - started, 0.0) * fabval[fo]
                lost_w += max(fab["cap"] - want, 0.0) * fabval[fo]
                ehat.append(want * fab_e[fo] / fab["R"] if fab["R"] > 0 else 0.0)
                egot.append(float(d["energy"][t - 1][fo]))
                vals.append(fabval[fo] / fab_e[fo] * (fab["R"] if fab["R"] > 0 else 0.0))  # USD per unit of energy
            lost_p, lost_w = lost_p / 1e9, lost_w / 1e9
            sub = code
            if tail:
                sub = "tail"
            elif code == "idle":
                shock = row["power"] - row["y"] - row["lost"] <= 1e-9 * max(row["y"], 1.0)
                sub = "idle:shock" if shock else "idle:nowafers"
                if shock and "shock" in logs[(t, go)]:
                    sub = "idle:shock_held"
            elif code in ("on", "lapse"):
                sub = code + (":pulsed" if row.get("pulsed") else "")
            elif code == "off":
                sub = "off:" + (("pipe" if row["piping"] else "") + ("lean" if row["leaning"] else "") or "tank")
            elif code == "dead":
                sub = "dead:" + "+".join(sorted({v[0] for v in row["blocked"].values()}))
            table[sub] += (1, w * lost_p, w * lost_w, lost_p)
            by_grid[(GRID[go], sub)] += (1, w * lost_p, w * lost_w, lost_p)
            ep_tot[sub] += lost_p
            m = row["power"] - row["y"]
            # the leak: a scarce fuel burnt in a week the fabs did not get their draw (held, it would burn in a whole
            # week instead: every week of burn leaked is a whole week not made, worth the grid's gain)
            marked = set()
            for name_ in ("short", "also", "blocked", "piping"):
                marked |= set(row.get(name_, ()))
            if sub == "mild":
                marked |= {k for k, f in row["fuels"].items() if missing(f) > 1e-9 * f["burn"] and not f["small"]}
            for k in marked:
                scarce_at[(go, k)] = t
            got_all = sum(egot) >= sum(ehat) * (1 - 1e-6) and sum(ehat) > 0
            if not tail and not got_all:
                for k, f in row["fuels"].items():
                    if t - scarce_at.get((go, k), -99) <= 4 and f["burn"] > 0 and not f["small"]:
                        burnt = float(d["segment"][t - 1].get((go, k), 0.0))
                        if burnt > 0:
                            share = burnt / f["burn"]
                            leak[(GRID[go], FUEL[k], sub)] += (1, share, w * share * d["gain"][go] / 1e9)
            if sub == "mild":
                for k, f in row["fuels"].items():
                    if missing(f) > 1e-9 * f["burn"] and not f["small"] and f["reach"]:
                        sent = sum(d["executed"][t - 1].get(s, 0.0) for s in feeder_slots[(go, k)])
                        under[(GRID[go], FUEL[k])] += (1, sent / f["burn"], f["feed"] / f["burn"])
            if sub == "mild":  # what a steady pulse would recover: the share of burn missing, (burn - m) gain / draw
                for k, f in row["fuels"].items():
                    miss = missing(f)
                    if miss > 1e-9 * f["burn"] and not f["small"]:
                        rec = miss / f["burn"] * max(f["burn"] - m, 0.0) * d["gain"][go] / d["full_draw"][go] / 1e9
                        mild[(GRID[go], FUEL[k])] += (1, w * lost_p, w * rec)
            if sub.startswith("dead"):
                missb = sum(missing(row["fuels"][k]) for k in row["blocked"])
                others = [k for k in row["short"] if k not in row["blocked"] and not row["fuels"][k]["small"]]
                pot = max(row["target"] - missb, 0.0) / row["target"] if row["target"] > 0 else 0.0
                tag = ",".join(f"{FUEL[k]}:{v[0]}" for k, v in sorted(row["blocked"].items()))
                key = (GRID[go], tag, "others short" if others else "alone")
                dead[key] += (1, w * lost_p, w * lost_p * pot if others else 0.0)
            # H5: energy split pro rata in a partly powered week
            if len(ehat) > 1 and sum(egot) > 1e-9 and sum(egot) < sum(ehat) * (1 - 1e-6):
                left, best = sum(egot), 0.0
                for v, eh in sorted(zip(vals, ehat), reverse=True):
                    take = min(left, eh)
                    best += take * v
                    left -= take
                h5[GRID[go]] += w * (best - sum(v * e for v, e in zip(vals, egot))) / 1e9
            # H3: the terminal's deliveries of a piped fuel in weeks that are not whole
            if code not in ("on", "lapse") and not tail:
                for k, f in row["fuels"].items():
                    if f["pipes"] and f["terminal"] and f["burn"] > 0:
                        sent = sum(d["executed"][t - 1].get(s, 0.0) for s in feeder_slots[(go, k)])
                        if sent > 0:
                            h3[(GRID[go], sub)] += (1, w * sent / f["burn"] * d["gain"][go] / 1e9, sent / f["burn"])
            last = prev.get(go)
            if last and last[0] == sub and last[1] == t - 1:
                runs[(d["set"], d["episode"], go, sub)][-1] += 1
            else:
                runs[(d["set"], d["episode"], go, sub)].append(1)
            prev[go] = (sub, t)
        per_ep.append((d["set"], d["episode"], d["ref"]["stratum"], dict(ep_tot)))
    n = len(docs)
    print("\nreason                 grid-weeks  per-ep  power-lost bn/ep (weighted)  as score  wafer-lost bn/ep (w)  power-lost bn/ep (plain)")
    for key, v in sorted(table.items(), key=lambda kv: -kv[1][1]):
        print(f"{key:22s} {int(v[0]):8d} {v[0] / n:7.1f} {v[1]:14.1f} {v[1] / att:16.4f} {v[2]:14.1f} {v[3] / n:20.1f}")
    tot = sum(v[1] for v in table.values())
    weeks = int(sum(v[0] for v in table.values()))
    print(f"{'all':22s} {weeks:8d} {'':7s} {tot:14.1f} {tot / att:16.4f} {sum(v[2] for v in table.values()):14.1f}")
    if not detail:
        return
    print("\nby grid and reason (weighted bn per episode of power-lost; grid-weeks), the largest:")
    for key, v in sorted(by_grid.items(), key=lambda kv: -kv[1][1])[: 3 * top]:
        print(f"  {key[0]:4s} {key[1]:22s} {int(v[0]):6d} wk  {v[1]:8.1f} bn  {v[1] / att:.4f}   wafer-lost {v[2]:8.1f}")
    print("\nmild weeks by grid and short fuel: weeks, weighted bn lost per episode, what a steady pulse recovers")
    for key, v in sorted(mild.items(), key=lambda kv: -kv[1][2]):
        print(f"  {key[0]:4s} {key[1]:6s} {int(v[0]):6d} wk  lost {v[1]:8.2f}  pulse recovers {v[2]:8.2f} bn = {v[2] / att:.4f}")
    print("\ndead weeks by grid and blocked fuels: weeks, weighted bn lost, bn if the other short fuels were pulsed")
    for key, v in sorted(dead.items(), key=lambda kv: -kv[1][1])[: 2 * top]:
        print(f"  {key[0]:4s} {key[1]:34s} {key[2]:12s} {int(v[0]):6d} wk  lost {v[1]:8.2f}  reachable {v[2]:8.2f} = {v[2] / att:.4f}")
    print("\nleak: a scarce fuel burnt in a week the fabs did not get their draw (weeks, weeks of burn, weighted bn, score)")
    for key, v in sorted(leak.items(), key=lambda kv: -kv[1][2])[: 3 * top]:
        print(f"  {key[0]:4s} {key[1]:6s} {key[2]:20s} {int(v[0]):6d} wk {v[1]:8.1f} weeks of burn {v[2]:8.2f} bn = {v[2] / att:.4f}")
    tot_leak = sum(v[2] for v in leak.values())
    print(f"  all {tot_leak:8.2f} bn = {tot_leak / att:.4f}")
    by_reason = defaultdict(float)
    for key, v in leak.items():
        by_reason[key[2]] += v[2]
    print("  by reason:", {k: round(v, 1) for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1])})
    print("\nmild weeks: what the terminal delivered against what it could (weeks of burn)")
    for key, v in sorted(under.items()):
        print(f"  {key[0]:4s} {key[1]:6s} {int(v[0]):5d} wk  delivered {v[1]:7.2f}  deliverable {v[2]:7.2f}")
    print("\nH5 (energy split pro rata in a partly powered week): weighted bn per episode with the high-value fab first")
    for key, v in sorted(h5.items(), key=lambda kv: -kv[1]):
        print(f"  {key:4s} {v:8.2f} bn = {v / att:.4f}")
    print("\nH3 (a piped fuel's terminal deliveries in weeks not whole): weeks, weighted bn as whole weeks, weeks of burn")
    for key, v in sorted(h3.items(), key=lambda kv: -kv[1][1]):
        print(f"  {key[0]:4s} {key[1]:22s} {int(v[0]):6d} wk  {v[1]:8.2f} bn = {v[1] / att:.4f}   {v[2]:.1f} weeks of burn")
    print("\nspells of mild and dead weeks (count, weeks, median, max; weeks in spells of 4+ and 8+):")
    agg = defaultdict(list)
    for (_s, _e, go, sub), lens in runs.items():
        if sub == "mild" or sub.startswith("dead"):
            agg[(GRID[go], sub)].extend(lens)
    for key, lens in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
        a = np.array(lens)
        print(
            f"  {key[0]:4s} {key[1]:22s} spells {len(a):4d} weeks {a.sum():5d} median {np.median(a):4.0f} max {a.max():4d}"
            f" in 4+ {a[a >= 4].sum():5d} in 8+ {a[a >= 8].sum():5d}"
        )
    print("\nepisodes with the most mild / dead loss (set, episode, level, plain bn):")
    for kind in ("mild", "dead"):
        def total(r, kind=kind):
            return sum(v for k, v in r[3].items() if k.startswith(kind))
        ranked = sorted(per_ep, key=lambda r: -total(r))[:top]
        print(" ", kind, [(s, e, lvl, round(total((s, e, lvl, t)))) for s, e, lvl, t in ranked])


if __name__ == "__main__":
    fire.Fire(main)
