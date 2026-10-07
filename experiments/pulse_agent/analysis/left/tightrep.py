"""Pooled tables of the exact shed-week model (tight.py) beside the looser calendar model.

uv run python tightrep.py --task=small
uv run python tightrep.py order --task=small     # the episodes by what the looser plan gains over our calendar
"""

import json
import pickle

import fire
import numpy as np

import common as C
from report import weights


def order(task="small", top=24):
    refs = C.references(task)
    w = weights(task, sorted(refs))
    rows = []
    for n in sorted(refs):
        p = C.HERE / "diff" / f"{task}_{n}.json"
        if p.exists():
            r = json.loads(p.read_text())
            rows.append((w[n] * (r["J_cal"] - r["J_final"]), n))
    rows.sort(reverse=True)
    print(",".join(str(n) for _v, n in rows[:top]))
    print(f"these carry {sum(v for v, _n in rows[:top]):.4f} of {sum(v for v, _n in rows):.4f} pooled")
    print("the rest: " + ",".join(str(n) for n in sorted(refs) if n not in [m for _v, m in rows[:top]]))


def main(task="small"):
    refs = C.references(task)
    w = weights(task, sorted(refs))
    res, loose = {}, {}
    for n in sorted(refs):
        p = C.HERE / "tight" / f"{task}_{n}.json"
        if p.exists():
            res[n] = json.loads(p.read_text())
        q = C.HERE / "diff" / f"{task}_{n}.json"
        base = json.loads((C.HERE / "res" / f"{task}_{n}.json").read_text())
        loose[n] = {"J_cal": base["J"]["cal"], "J_played": base["J_played"], "J_plan": base["J"]["cal"]}
        if q.exists():
            loose[n]["J_plan"] = min(json.loads(q.read_text())["J_final"], base["J"]["cal"])
    ns = sorted(res)
    print(f"{task}: the exact rule for shed weeks, {len(ns)} episodes solved; our play obeys its rows to {max(res[n]['residual_play'] for n in ns):.1e}")
    st = {}
    for n in ns:
        for k in ("cal", "plan_calendar", "free"):
            if k in res[n]:
                st.setdefault(k, {}).setdefault(res[n][k]["status"], []).append(n)
    for k, d in st.items():
        print(f"  {k}: " + "; ".join(f"{s}: {len(v)}" for s, v in d.items()))
    # our calendar under the exact rule, all solved episodes
    have = [n for n in ns if res[n]["cal"]["J"] is not None]
    d_cal = sum(w[n] * (res[n]["cal"]["J"] - loose[n]["J_cal"]) for n in have)
    d_bnd = sum(w[n] * (res[n]["cal"]["bound"] - loose[n]["J_cal"]) for n in have)
    g_loose = sum(w[n] * (loose[n]["J_played"] - loose[n]["J_cal"]) for n in have)
    print(f"  our calendar, {len(have)} episodes: the looser model's optimum is {g_loose:.4f} pooled above our play; under the exact rule it is {d_cal:.4f} lower (found), at least {max(d_bnd, 0.0):.4f} lower by the proven bounds")
    if len(have) == len(refs):
        a = C.pooled(refs, {n: res[n]["cal"]["J"] * 100 for n in have})
        print(f"    pooled, all episodes: our calendar under the exact rule {a[0]:.4f} (by level {' '.join(f'{s}: {v:.4f}' for s, v in a[1].items())})")
    fr = [n for n in ns if "free" in res[n] and res[n]["free"]["J"] is not None]
    if fr:
        g_t = sum(w[n] * (res[n]["cal"]["J"] - res[n]["free"]["J"]) for n in fr)
        g_tb = sum(w[n] * (res[n]["cal"]["J"] - res[n]["free"]["bound"]) for n in fr)
        g_l = sum(w[n] * (loose[n]["J_cal"] - loose[n]["J_plan"]) for n in fr)
        g_all = sum(w[n] * (loose[n]["J_cal"] - loose[n]["J_plan"]) for n in loose)
        print(f"  calendar free, {len(fr)} episodes: the looser model's plan is {g_l:.4f} pooled above its calendar optimum on them (of {g_all:.4f} on all episodes); under the exact rule the plan found is {g_t:.4f} above (proven bound {g_tb:.4f})")
        inf = [n for n in fr if res[n].get("plan_calendar", {}).get("status") == "Infeasible"]
        ok = [n for n in fr if res[n].get("plan_calendar", {}).get("J") is not None]
        print(f"    the looser plan's own calendar under the exact rule: impossible in {len(inf)} of {len([n for n in fr if 'plan_calendar' in res[n]])} episodes; a plan found in {len(ok)}")
        for n in sorted(fr, key=lambda n: -w[n] * (loose[n]["J_cal"] - loose[n]["J_plan"])):
            D = res[n]["D"]
            naive = refs[n]["J_naive_cents"] / 100
            sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
            print(f"    episode {n:2d}: played {sc(loose[n]['J_played']):.4f}; our calendar {sc(loose[n]['J_cal']):.4f} | exact {sc(res[n]['cal']['J']):.4f}; calendar free {sc(loose[n]['J_plan']):.4f} | exact {sc(res[n]['free']['J']):.4f} (bound {sc(res[n]['free']['bound']):.4f}, {res[n]['free']['status']})")


if __name__ == "__main__":
    fire.Fire({"main": main, "order": order})
