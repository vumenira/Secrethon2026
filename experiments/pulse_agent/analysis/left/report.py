"""Pooled tables of the calendar LP's variants for agents/mpc_pulse_rec_v6 (solve.py).

uv run python report.py --task=small
"""

import json

import fire
import numpy as np

import common as C
import solve as S


def load(task):
    refs = C.references(task)
    res = {}
    for n in sorted(refs):
        p = C.HERE / "res" / f"{task}_{n}.json"
        if p.exists():
            res[n] = json.loads(p.read_text())
    return refs, res


def weights(task, ns):
    """Per episode, the factor that turns USD into pooled score units on the episodes ``ns``."""
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    refs = C.references(task)
    counts = {s: sum(refs[n]["stratum"] == s for n in ns) for s in (1, 2, 3, 4)}
    den = sum(STRATUM_WEIGHTS[s - 1] * np.mean([refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"] for n in ns if refs[n]["stratum"] == s]) for s in counts if counts[s]) / 100
    return {n: STRATUM_WEIGHTS[refs[n]["stratum"] - 1] / counts[refs[n]["stratum"]] / den for n in ns}


def main(task="small", top=14):
    refs, res = load(task)
    ns = sorted(res)
    cached = C.cached(task)
    print(f"{task}: {len(ns)} of {len(refs)} episodes solved; by harm level {[sum(refs[n]['stratum'] == s for n in ns) for s in (1, 2, 3, 4)]}")
    names = list(res[ns[0]]["J"])
    same = sum(int(round(res[n]["J_played"] * 100)) == cached[n] for n in ns)
    print(f"validation: replayed cost equals the cached row {C.CACHED[task]} to the cent in {same} of {len(ns)}; worst row residual of the model at our vector {max(res[n]['residual'] for n in ns):.1e};")
    print(f"  grid-weeks that shed base load and powered fabs {sum(res[n]['both'] for n in ns)}; the model's cost of our vector differs from ours by at most {max(abs(res[n]['J_model'] - res[n]['J_played']) / abs(res[n]['J_played']) for n in ns):.1e} (relative)")
    chk = [(res[n]["J"]["all_ns"] - res[n]["J_played"]) / abs(res[n]["J_played"]) for n in ns if res[n]["J"].get("all_ns") is not None]
    print(f"  every decision and rule fixed, stocks free: solved in {len(chk)} of {len(ns)} episodes; cost minus ours between {min(chk):+.1e} and {max(chk):+.1e} of the cost")
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
        c = max((res[n]["J"]["cal"] - res[n]["J"][name]) / abs(res[n]["J"]["cal"]) for n in ns)
        if w > 1e-7:
            worse[name] = w
        if c > 1e-7:
            cheaper[name] = c
    print(f"  variants dearer than our play in some episode by more than 1e-7 of the cost: {worse}; cheaper than the calendar optimum: {cheaper}")
    lev = lambda name: " ".join(f"{s}: {v:.4f}" for s, v in by[name].items())  # noqa: E731
    g = lambda name: sc.get(name, float("nan"))  # noqa: E731
    gap = sc["cal"] - sc["played"]
    print(f"\nplayed {sc['played']:.4f} (by level {lev('played')})")
    print(f"calendar optimised {sc['cal']:.4f} (by level {lev('cal')})   difference {gap:+.4f}")
    print(f"  difference by level: {' '.join(f'{s}: {by['cal'][s] - by['played'][s]:+.4f}' for s in by['cal'])}")
    print(f"every flow as played, the simulator's rules free (fix_x): {g('fix_x'):.4f}  gain over played {g('fix_x') - sc['played']:+.4f} ({(g('fix_x') - sc['played']) / gap:.0%})")
    print(f"  of which the grids' burn and shed (lost when they too are fixed): {g('fix_x') - g('fix_x_grid'):.4f}; lot starts: {g('fix_x') - g('fix_x_lots'):.4f}")
    print(f"lot starts as played, the rest free: loss {sc['cal'] - g('fix_lots'):.4f}")
    for name in ("fuel+", "mat+", "chip+"):
        print(f"  {name} flows as played, the rest free: loss {sc['cal'] - g(f'fix_{name}'):.4f} ({(sc['cal'] - g(f'fix_{name}')) / gap:.0%})")
    T = {"small": 52, "full": 104}[task]
    prev, prevx, last = sc["cal"], sc["cal"], 0
    print("\nBy time: weeks 1..t as played, the rest optimised (every decision and rule | the flows only)")
    for t in list(S.PREFIX[task]) + [T]:
        cur = sc["played"] if t == T else g(f"pre_{t}")
        curx = g("fix_x") if t == T else g(f"prex_{t}")
        print(f"  {last + 1:4d}-{t:<4d} score {cur:.4f}  loss {prev - cur:.4f} ({(prev - cur) / gap:4.0%})  per week {(prev - cur) / (t - last):.5f} | flows only: score {curx:.4f}  loss {prevx - curx:.4f}")
        prev, prevx, last = cur, curx, t
    print("\nThe opening: one piece of weeks 1..t as played, everything else free (loss against the calendar optimum)")
    for t in S.OPEN:
        print(f"  weeks 1-{t}: all {sc['cal'] - g(f'pre_{t}'):.4f}; flows {sc['cal'] - g(f'prex_{t}'):.4f} (fuel {sc['cal'] - g(f'o{t}_fuel'):.4f}, wafers {sc['cal'] - g(f'o{t}_mat'):.4f}, chips {sc['cal'] - g(f'o{t}_chip'):.4f});"
              f" rules {sc['cal'] - g(f'o{t}_rules'):.4f} (lot starts {sc['cal'] - g(f'o{t}_lots'):.4f}, grids {sc['cal'] - g(f'o{t}_grid'):.4f});"
              f" flows and lots {sc['cal'] - g(f'o{t}_x_lots'):.4f}; flows and grids {sc['cal'] - g(f'o{t}_x_grid'):.4f}")
        print(f"      only weeks 1-{t} free: gain over played {g(f'win_0_{t}') - sc['played']:.4f}; only their flows (and every week's rules) free: gain over fix_x {g(f'winx_0_{t}') - g('fix_x'):.4f}")
    w = weights(task, ns)
    print(f"\nLargest differences, played to calendar optimised (pooled units), with the opening's part (weeks 1-4 as played against free):")
    rows = []
    for n in ns:
        r = refs[n]
        D = (r["J_naive_cents"] - r["J_oracle_cents"]) / 100
        d = res[n]["J_played"] - res[n]["J"]["cal"]
        o4 = res[n]["J"]["pre_4"] - res[n]["J"]["cal"]
        rows.append((w[n] * d, n, r["stratum"], (r["J_naive_cents"] / 100 - res[n]["J_played"]) / D, (r["J_naive_cents"] / 100 - res[n]["J"]["cal"]) / D, w[n] * o4, o4 / D))
    rows.sort(reverse=True)
    cum = 0.0
    for share, n, s, a, b, o4, o4e in rows[:top]:
        cum += share
        print(f"  episode {n:2d} level {s}: played {a:.4f} calendar {b:.4f}  pooled {share:.4f} ({share / gap:4.0%}, cumulative {cum / gap:4.0%}); opening {o4:.5f} pooled ({o4e:.4f} of the episode's gap)")
    o4 = np.array([r[6] for r in rows])
    print(f"  the opening's loss per episode, as a share of the episode's own gap: mean {o4.mean():.4f}, median {np.median(o4):.4f}, quartiles {np.percentile(o4, 25):.4f} to {np.percentile(o4, 75):.4f}, min {o4.min():.4f}, max {o4.max():.4f}")
    (C.HERE / f"report_{task}.json").write_text(json.dumps({"sc": sc, "by": {k: {str(s): v for s, v in d.items()} for k, d in by.items()}}, indent=1))


if __name__ == "__main__":
    fire.Fire(main)
