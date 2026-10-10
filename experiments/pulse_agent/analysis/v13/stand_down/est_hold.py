"""What holding a scarce fuel through the weeks the fabs cannot run could be worth, within the terminal's storage.

uv run python outputs/s13/est_hold.py --name=s13_census --sets="full:12345:16,full:2026:16,full:777:32"

No play; a ledger over the census pickles. Per episode, grid and fuel with a terminal: in a week the fabs cannot
run whatever is delivered (no wafers, the grid's own power under its base load, another fuel that cannot be held)
what the fuel's segment burnt could have waited at the terminal, as far as the terminal had room (pulse_fill of its
storage less what it held). In a later week the pulse held that fuel for a whole week (an "off" week that waited
for it), a week's burn out of that bank makes the week whole. Counted: the weeks made whole, at the grid's gain per
whole week. ``seen``: only where the fuel was reaching its terminal more slowly than the grid burns it when the
week came (the pulse's own test, the larger of the week's arrivals and their smoothed rate), or was short or held
in the last four weeks: what a rule could know. It ignores that the weeks after would themselves change.
"""
from collections import defaultdict

import fire
import numpy as np

from analyse import FUEL, GRID, load


def main(name="s13_census", sets="full:12345:16", task="full", top=14, fill=0.9):
    from shockbench_flow.hosting.tasks import task_generator

    inst, _params = task_generator(task)
    docs, att = load(name, sets)
    out = defaultdict(lambda: np.zeros(4))  # (kind of week, grid, fuel) -> weeks held, weeks of burn banked, whole weeks, bn
    cases = []
    for seen in (False, True):
        for d in docs:
            census = defaultdict(dict)
            for t, go, code, row in d["census"]:
                census[go][t] = (code, row)
            for go, weeks in census.items():
                last = d["last"][go]
                fuels = next(iter(weeks.values()))[1]["fuels"]
                for k, f0 in fuels.items():
                    if f0["small"] or not f0["terminal"]:
                        continue
                    bank, lately = 0.0, -99
                    made = defaultdict(float)
                    banked = defaultdict(float)
                    source = []  # the kinds of week the bank came from, oldest first: (kind, weeks of burn)
                    for t, (code, row) in sorted(weeks.items()):
                        if t > last:
                            break
                        f = row["fuels"][k]
                        burn = f["burn"]
                        if burn <= 0:
                            continue
                        powered = row["power"] - row["y"] > 1e-9 * max(row["y"], 1.0)
                        if k in row.get("short", ()) or k in row.get("also", ()):
                            lately = t
                        kind = None
                        if code == "idle":
                            kind = "nowafers" if powered else "shock"
                        elif code == "dead" and k not in row["blocked"]:
                            kind = "dead"
                        if kind is not None and (kind != "shock" or "shock" not in [r for tt, g2, r, _m in d["log"] if tt == t and g2 == go]):
                            scarce = max(f["inflow"], f["rate"] or 0.0) < burn * (1 - 1e-6) or t - lately <= 4
                            if scarce or not seen:
                                room = fill * f["room"] - f["tank"] - f["inflow"] - bank
                                gave = float(d["segment"][t - 1].get((go, k), 0.0))
                                keep = max(min(gave, room), 0.0)
                                if keep > 0:
                                    bank += keep
                                    banked[kind] += keep / burn
                                    source.append([kind, keep / burn])
                        elif code == "off" and k in row.get("short", ()) and bank >= burn * (1 - 1e-9):
                            if any(k2 != k and not row["fuels"][k2]["small"] for k2 in row["short"]):
                                continue  # the week waits for another fuel too: this fuel alone does not make it whole
                            bank -= burn
                            need = 1.0
                            while need > 1e-9 and source:
                                part = min(need, source[0][1])
                                made[source[0][0]] += part
                                source[0][1] -= part
                                need -= part
                                if source[0][1] <= 1e-9:
                                    source.pop(0)
                    for kind in banked:
                        bn = made[kind] * d["gain"][go] / 1e9
                        out[(seen, kind, GRID[go], FUEL[k])] += (1, banked[kind], made[kind], d["w"] * bn)
                        if bn > 0:
                            cases.append((round(d["w"] * bn, 2), round(bn, 1), seen, kind, GRID[go], FUEL[k], d["set"], d["episode"], d["ref"]["stratum"]))
    print(f"{len(docs)} episodes; 0.001 of score = {att / 1000:.2f} bn (weighted)")
    for seen in (False, True):
        print("\n" + ("where the fuel was seen to be scarce that week" if seen else "with hindsight (any such week)")
              + ": kind of week, grid, fuel: cases, weeks of burn banked, weeks made whole, weighted bn, score")
        tot = 0.0
        by_kind = defaultdict(float)
        for key, v in sorted(out.items(), key=lambda kv: -kv[1][3]):
            if key[0] == seen and v[3] > 0:
                print(f"  {key[1]:9s} {key[2]:4s} {key[3]:6s} {int(v[0]):4d} {v[1]:7.1f} {v[2]:7.1f} {v[3]:8.2f} bn = {v[3] / att:.4f}")
                tot += v[3]
                by_kind[key[1]] += v[3]
        print(f"  all {tot:8.2f} bn = {tot / att:.4f}   by kind: " + ", ".join(f"{k} {v:.2f} ({v / att:.4f})" for k, v in by_kind.items()))
    print("\nthe largest cases seen (weighted bn, plain bn, kind, grid, fuel, set, episode, level):")
    for c in sorted((c for c in cases if c[2]), reverse=True)[:top]:
        print("  ", c[0], c[1], *c[3:])


if __name__ == "__main__":
    fire.Fire(main)
