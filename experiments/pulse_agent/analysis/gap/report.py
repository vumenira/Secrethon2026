"""Pooled tables of the calendar LP's variants (solve.py): played, calendar-optimised, by decision group and by time.

uv run python report.py --task=small
"""

import json

import fire
import numpy as np

import common as C
import solve as S


def load(task, sets="all"):
    refs = C.references(task)
    res = {}
    for n in sorted(refs):
        p = C.HERE / "res" / f"{task}_{n}_{sets}.json"
        if p.exists():
            res[n] = json.loads(p.read_text())
    return refs, res


def main(task="small", sets="all", top=12):
    refs, res = load(task, sets)
    ns = sorted(res)
    cached = C.cached(task)
    print(f"{task}: {len(ns)} of {len(refs)} episodes solved; by harm level {[sum(refs[n]['stratum'] == s for n in ns) for s in (1, 2, 3, 4)]}")
    names = list(res[ns[0]]["J"])
    # validation
    same = sum(int(round(res[n]["J_played"] * 100)) == cached[n] for n in ns)
    worst = max(res[n]["residual"] for n in ns)
    both = sum(res[n]["both"] for n in ns)
    dev_model = max(abs(res[n]["J_model"] - res[n]["J_played"]) / abs(res[n]["J_played"]) for n in ns)
    dev_all = max(abs(res[n]["J"]["all"] - res[n]["J_played"]) / abs(res[n]["J_played"]) for n in ns if res[n]["J"]["all"] is not None)
    none_all = [n for n in ns if res[n]["J"]["all"] is None]
    print(f"validation: replayed cost equals the cached row to the cent in {same} of {len(ns)}; worst row residual of the model at our vector {worst:.1e};")
    print(f"  grid-weeks that shed base load and powered fabs {both}; the model's cost of our vector differs from ours by at most {dev_model:.1e} (relative);")
    print(f"  the LP with every column fixed differs from ours by at most {dev_all:.1e} (relative); episodes where it found no point: {none_all}")
    bad = {}
    for n in ns:
        for name, s in res[n]["status"].items():
            if s != "Optimal":
                bad.setdefault(name, []).append((n, s))
    if bad:
        print("  solves not optimal at the first try:")
        for name, items in bad.items():
            print(f"    {name}: {items[:6]}{' ...' if len(items) > 6 else ''}")

    above = {}  # variant -> episodes whose optimum came out dearer than our own play (which is feasible in every variant)

    def J_of(name):
        out = {}
        for n in ns:
            J = res[n]["J"].get(name)
            if J is None:
                return None
            if name != "all" and J > res[n]["J_played"] * (1 + 1e-9 * np.sign(res[n]["J_played"])):
                above.setdefault(name, []).append((n, (J - res[n]["J_played"]) / abs(res[n]["J_played"])))
            out[n] = J * 100  # as solved: never replaced by our play
        return out

    played = {n: res[n]["J_played"] * 100 for n in ns}
    sc = {}
    by = {}
    for name in ["played"] + names:
        J = played if name == "played" else J_of(name)
        if J is None:
            continue
        sc[name], by[name] = C.pooled(refs, J)
    lev = lambda name: " ".join(f"{s}: {v:.4f}" for s, v in by[name].items())  # noqa: E731
    missing = [name for name in names if name not in sc]
    print(f"  variants with an episode unsolved (left out of every table): {missing}")
    print(f"  variants whose optimum is dearer than our play in some episode (by more than 1e-9 of the cost): { {k: (len(v), max(x[1] for x in v)) for k, v in above.items()} }")
    # every variant is a restriction of the calendar LP, and a prefix fixed longer is a restriction of a shorter one
    below = {}
    for name in names:
        if name in ("all", "cal") or name not in sc:
            continue
        worst_rel = max((res[n]["J"]["cal"] - res[n]["J"][name]) / abs(res[n]["J"]["cal"]) for n in ns)
        if worst_rel > 1e-7:
            below[name] = worst_rel
    print(f"  variants cheaper than the calendar optimum in some episode by more than 1e-7 of the cost: {below}")
    order = [f"pre_{t}" for t in S.PREFIX[task] if f"pre_{t}" in sc]
    bad_order = 0
    worst_order = 0.0
    for a, b in zip(order, order[1:]):
        for n in ns:
            rel = (res[n]["J"][a] - res[n]["J"][b]) / abs(res[n]["J"][a])
            if rel > 1e-7:
                bad_order += 1
            worst_order = max(worst_order, rel)
    print(f"  prefix solves out of order (a longer fixed prefix cheaper by more than 1e-7 of the cost): {bad_order}; the largest such gap {worst_order:.1e}")
    check = {}
    for n in ns:
        p = C.HERE / "res" / f"{task}_{n}_check.json"
        if p.exists():
            c = json.loads(p.read_text())
            if c["J"].get("all_ns") is not None:
                check[n] = (c["J"]["all_ns"] - c["J_played"]) / abs(c["J_played"])
    if check:
        print(f"  every decision and rule fixed, stocks free: solved in {len(check)} of {len(ns)} episodes; cost minus ours between {min(check.values()):+.1e} and {max(check.values()):+.1e} of the cost")
    print(f"\nplayed {sc['played']:.4f} (by level {lev('played')})")
    if "all" in sc:
        print(f"every column fixed {sc['all']:.4f}")
    print(f"calendar optimised {sc['cal']:.4f} (by level {lev('cal')})   difference {sc['cal'] - sc['played']:+.4f}")
    gap = sc["cal"] - sc["played"]
    if "fix_x" in sc:
        print("\nBy decision group (score units; share of the difference in brackets)")
        print(f"  every flow as played, the simulator's rules free (fix_x): {sc['fix_x']:.4f}  gain over played {sc['fix_x'] - sc['played']:+.4f} ({(sc['fix_x'] - sc['played']) / gap:.0%})")
        print(f"  every flow free, the rules as played (fix_rules):          {sc['fix_rules']:.4f}  gain over played {sc['fix_rules'] - sc['played']:+.4f} ({(sc['fix_rules'] - sc['played']) / gap:.0%})")
        print(f"  {'group':12s} {'loss when fixed alone':>22s} {'gain when freed alone':>22s} {'gain freed with rules':>22s}")
        tot_fix = tot_free = tot_freex = 0.0
        for g in S.GROUPS:
            loss = sc["cal"] - sc[f"fix_{g}"]
            gain = sc[f"free_{g}"] - sc["played"]
            gx = sc[f"freex_{g}"] - sc["fix_x"] if f"freex_{g}" in sc else float("nan")
            tot_fix += loss
            tot_free += gain
            tot_freex += 0.0 if np.isnan(gx) else gx
            print(f"  {g:12s} {loss:14.4f} ({loss / gap:4.0%}) {gain:14.4f} ({gain / gap:4.0%}) {gx:14.4f} ({gx / gap:4.0%})")
        print(f"  {'sum':12s} {tot_fix:14.4f} ({tot_fix / gap:4.0%}) {tot_free:14.4f} ({tot_free / gap:4.0%}) {tot_freex:14.4f} ({tot_freex / gap:4.0%})")
        print("  other cuts of the flows (loss when fixed alone; gain when freed with the rules, over fix_x):")
        for g in ("fuel+", "mat+", "chip+", "lng+", "crude+", "nucfuel+", "fuel_up", "fuel_grid", "strait_fuel", "strait_rest"):
            if f"fix_{g}" in sc:
                loss, gx = sc["cal"] - sc[f"fix_{g}"], sc[f"freex_{g}"] - sc["fix_x"]
                print(f"  {g:12s} {loss:14.4f} ({loss / gap:4.0%}) {'':22s} {gx:14.4f} ({gx / gap:4.0%})")
        print("  pieces of the rules, each fixed on top of every flow (loss against fix_x):")
        for g in ("lots", "grid", "xi", "serve", "disp"):
            if f"fix_x_{g}" in sc:
                loss = sc["fix_x"] - sc[f"fix_x_{g}"]
                print(f"  {g:12s} {loss:14.4f} ({loss / gap:4.0%})")
    if any(k.startswith("pre_") for k in sc):
        T = {"small": 52, "full": 104}[task]
        print("\nBy time: weeks 1..t as played, the rest optimised (pre_t: every decision and rule; prex_t: the flows only)")
        prev, prevx, last = sc["cal"], sc["cal"], 0
        print(f"  {'weeks':>9s} {'score, all fixed to t':>22s} {'loss of these weeks':>20s} {'per week':>9s} | {'flows only':>10s} {'loss':>8s}")
        for t in list(S.PREFIX[task]) + [T]:
            cur = sc["played"] if t == T else sc[f"pre_{t}"]
            curx = sc["fix_x"] if t == T and "fix_x" in sc else sc.get(f"prex_{t}", float("nan"))
            print(f"  {last + 1:4d}-{t:<4d} {cur:22.4f} {prev - cur:13.4f} ({(prev - cur) / gap:4.0%}) {(prev - cur) / (t - last):9.5f} | {curx:10.4f} {prevx - curx:8.4f}")
            prev, prevx, last = cur, curx, t
        print("  only the weeks of a window free (gain over played; winx: its flows and the rules of every week, gain over fix_x):")
        tot = totx = 0.0
        for a, b in S.WINDOWS[task]:
            g1 = sc[f"win_{a}_{b}"] - sc["played"]
            g2 = sc[f"winx_{a}_{b}"] - sc.get("fix_x", sc["played"])
            tot += g1
            totx += g2
            print(f"  {a + 1:4d}-{b:<4d} {g1:10.4f} ({g1 / gap:4.0%})   {g2:10.4f} ({g2 / gap:4.0%})")
        print(f"  {'sum':9s} {tot:10.4f} ({tot / gap:4.0%})   {totx:10.4f} ({totx / gap:4.0%})")
        print("  one good's flows (strait releases included) of weeks 1..t as played, the rest free: loss of each stretch of weeks")
        for g in ("fuel+", "mat+", "chip+"):
            if f"fix_{g}" not in sc:
                continue
            prev, last, line = sc["cal"], 0, f"  {g:8s}"
            for _a, b in list(S.WINDOWS[task][:-1]) + [(0, T)]:
                cur = sc[f"fix_{g}"] if b == T else sc.get(f"preg_{g}_{b}", float("nan"))
                line += f"  {last + 1}-{b}: {prev - cur:.4f}"
                prev, last = cur, b
            print(line + f"   all weeks: {sc['cal'] - sc[f'fix_{g}']:.4f}")
    # per episode
    print(f"\nLargest differences, played to calendar optimised (share of the pooled difference each carries):")
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    counts = {s: sum(refs[n]["stratum"] == s for n in ns) for s in (1, 2, 3, 4)}
    den = sum(STRATUM_WEIGHTS[s - 1] * np.mean([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns if refs[n]["stratum"] == s]) for s in counts if counts[s])
    rows = []
    for n in ns:
        r = refs[n]
        D = r["J_naive_cents"] - r["J_oracle_cents"]
        w = STRATUM_WEIGHTS[r["stratum"] - 1] / counts[r["stratum"]]
        d = (res[n]["J_played"] - res[n]["J"]["cal"]) * 100
        rows.append((w * d / den, n, r["stratum"], (r["J_naive_cents"] - res[n]["J_played"] * 100) / D, (r["J_naive_cents"] - res[n]["J"]["cal"] * 100) / D, d / 1e11))
    rows.sort(reverse=True)
    cum = 0.0
    for share, n, s, a, b, bn in rows[:top]:
        cum += share
        print(f"  episode {n:2d} level {s}: played {a:.4f} calendar {b:.4f}  saving {bn:7.2f} bn  pooled {share:.4f} ({share / gap:4.0%}, cumulative {cum / gap:4.0%})")


def core(task="full", sets="core"):
    """The shorter list of variants (solve.py, sets="core"): the same tables as far as they go."""
    refs, res = load(task, sets)
    ns = sorted(res)
    cached = C.cached(task)
    print(f"{task}: {len(ns)} of {len(refs)} episodes solved; by harm level {[sum(refs[n]['stratum'] == s for n in ns) for s in (1, 2, 3, 4)]}")
    if not ns:
        return
    names = list(res[ns[0]]["J"])
    same = sum(int(round(res[n]["J_played"] * 100)) == cached[n] for n in ns)
    print(f"validation: replayed cost equals the cached row to the cent in {same} of {len(ns)}; worst row residual of the model at our vector {max(res[n]['residual'] for n in ns):.1e};")
    print(f"  grid-weeks that shed base load and powered fabs {sum(res[n]['both'] for n in ns)}; the model's cost of our vector differs from ours by at most {max(abs(res[n]['J_model'] - res[n]['J_played']) / abs(res[n]['J_played']) for n in ns):.1e} (relative)")
    chk = [(res[n]["J"]["all_ns"] - res[n]["J_played"]) / abs(res[n]["J_played"]) for n in ns if res[n]["J"].get("all_ns") is not None]
    print(f"  every decision and rule fixed, stocks free: solved in {len(chk)} of {len(ns)} episodes; cost minus ours between {min(chk):+.1e} and {max(chk):+.1e} of the cost" if chk else "  every decision and rule fixed: no episode solved")
    first = {}
    for n in ns:
        for name, s in res[n]["status"].items():
            if s != "Optimal":
                first.setdefault(s, []).append((n, name))
    print(f"  solves not optimal at the first try, by what followed: { {k: len(v) for k, v in first.items()} }")
    unsolved = sorted({name for n in ns for name in names if res[n]["J"].get(name) is None})
    print(f"  variants with an episode unsolved (left out): {unsolved}")
    played = {n: res[n]["J_played"] * 100 for n in ns}
    sc, by = {}, {}
    sc["played"], by["played"] = C.pooled(refs, played)
    worse, cheaper = {}, {}
    for name in names:
        if name in unsolved:
            continue
        sc[name], by[name] = C.pooled(refs, {n: res[n]["J"][name] * 100 for n in ns})
        w = max((res[n]["J"][name] - res[n]["J_played"]) / abs(res[n]["J_played"]) for n in ns)
        c = max((res[n]["J"]["cal"] - res[n]["J"][name]) / abs(res[n]["J"]["cal"]) for n in ns) if "cal" not in unsolved else float("nan")
        if w > 1e-7:
            worse[name] = w
        if c > 1e-7:
            cheaper[name] = c
    print(f"  variants dearer than our play in some episode by more than 1e-7 of the cost: {worse}; cheaper than the calendar optimum: {cheaper}")
    lev = lambda name: " ".join(f"{s}: {v:.4f}" for s, v in by[name].items())  # noqa: E731
    print(f"\nplayed {sc['played']:.4f} (by level {lev('played')})")
    if "cal" not in sc:
        return
    gap = sc["cal"] - sc["played"]
    print(f"calendar optimised {sc['cal']:.4f} (by level {lev('cal')})   difference {gap:+.4f}")
    g = lambda name: sc.get(name, float("nan"))  # noqa: E731
    print(f"every flow as played, the simulator's rules free: {g('fix_x'):.4f}  gain over played {g('fix_x') - sc['played']:+.4f} ({(g('fix_x') - sc['played']) / gap:.0%})")
    print(f"  of which the grids' burn and shed (lost when they too are fixed): {g('fix_x') - g('fix_x_grid'):.4f}; lot starts: {g('fix_x') - g('fix_x_lots'):.4f}")
    print(f"lot starts as played, the rest free: loss {sc['cal'] - g('fix_lots'):.4f} ({(sc['cal'] - g('fix_lots')) / gap:.0%})")
    print(f"  {'flows':10s} {'loss when fixed alone':>22s} {'gain when freed with the rules (over fix_x)':>44s}")
    for name in ("fuel", "mat", "chip", "strait", "fuel+", "mat+", "chip+", "lng+", "crude+", "nucfuel+"):
        if f"fix_{name}" in sc:
            loss, gain = sc["cal"] - sc[f"fix_{name}"], g(f"freex_{name}") - g("fix_x")
            print(f"  {name:10s} {loss:14.4f} ({loss / gap:4.0%}) {gain:30.4f} ({gain / gap:4.0%})")
    T = {"small": 52, "full": 104}[task]
    prev, prevx, last = sc["cal"], sc["cal"], 0
    print("by time: weeks 1..t as played, the rest optimised (every decision and rule | the flows only)")
    for t in list(S.PREFIX_CORE[task]) + [T]:
        cur = sc["played"] if t == T else g(f"pre_{t}")
        curx = g("fix_x") if t == T else g(f"prex_{t}")
        print(f"  {last + 1:4d}-{t:<4d} score {cur:.4f}  loss {prev - cur:.4f} ({(prev - cur) / gap:4.0%})  per week {(prev - cur) / (t - last):.5f} | flows only: score {curx:.4f}  loss {prevx - curx:.4f}")
        prev, prevx, last = cur, curx, t


if __name__ == "__main__":
    fire.Fire({"main": main, "core": core})
