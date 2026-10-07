"""The opening weeks, ours against an optimised plan, episode by episode.

uv run python opening.py collect --task=small --name=cal      # weekly series of every episode, ours and the plan's
uv run python opening.py table --task=small --name=cal        # by item and week: ours, the plan, its spread
uv run python opening.py events --task=small                  # the opening's loss against what happens early
"""

import json
import pickle

import fire
import numpy as np

import common as C
import view as V
import worlds as W


def sol_of(task, n, name):
    d = pickle.loads((C.HERE / "res" / f"{task}_{n}_sol.pkl").read_bytes())
    return d["sols"][name], d["z"]


def collect(task="small", name="cal", weeks=16, src=None):
    """``src``: a folder of per-episode pickles {"x": (T, nc)} to read the plan from instead of res/*_sol.pkl."""
    refs = C.references(task)
    out = {}
    for n in sorted(refs):
        cal = C.Cal(task, n)
        d = C.load_traj(task, n)
        a = d["zpi"].reshape(cal.T, cal.nc)
        if src is None:
            b, z = sol_of(task, n, name)
        else:
            p = C.HERE / src / f"{task}_{n}.pkl"
            if not p.exists():
                continue
            dd = pickle.loads(p.read_bytes())
            b, z = dd["x"], dd.get("z")
        v = V.View(cal)
        sa, sb = V.series(v, a), V.series(v, b)
        zo, _both = cal.calendar(d["zpi"])
        out[n] = {"ours": {k: x[:weeks].copy() for k, x in sa.items()}, "opt": {k: x[:weeks].copy() for k, x in sb.items()}, "z": zo[:, :weeks].copy(), "grids": [v.node[inst_g] for inst_g in [cal.inst.grids[go] for go, _f, _s in cal.grids]]}
        print(n, flush=True)
    (C.HERE / f"opening_{task}_{name}.pkl").write_bytes(pickle.dumps(out))


def table(task="small", name="cal", weeks=8, only=None, items=None):
    data = pickle.loads((C.HERE / f"opening_{task}_{name}.pkl").read_bytes())
    ns = sorted(data) if only is None else [int(v) for v in (only if isinstance(only, (list, tuple)) else [only])]
    keys = list(data[ns[0]]["ours"])
    if items:
        items = items if isinstance(items, (list, tuple)) else [items]
        keys = [k for k in keys if any(str(i) in k for i in items)]
    print(f"{task}, {len(ns)} episodes, plan {name}: per item and week 1..{weeks}: ours mean | plan mean | plan sd across episodes | ours sd")
    for k in keys:
        A = np.array([data[n]["ours"][k][:weeks] for n in ns])
        B = np.array([data[n]["opt"][k][:weeks] for n in ns])
        if np.abs(A).max() + np.abs(B).max() < 1e-9:
            continue
        big = max(np.abs(A).max(), np.abs(B).max()) > 100
        f = (lambda x: f"{x:9.0f}") if big else (lambda x: f"{x:5.2f}")
        print(f"  {k}")
        print("      ours    " + " ".join(f(x) for x in A.mean(axis=0)))
        print("      plan    " + " ".join(f(x) for x in B.mean(axis=0)))
        print("      plan sd " + " ".join(f(x) for x in B.std(axis=0)))
        print("      ours sd " + " ".join(f(x) for x in A.std(axis=0)))


def calendar(task="small", weeks=12):
    """Our own fab calendar in the opening: per grid and week, the share of episodes that shed no base load."""
    data = pickle.loads((C.HERE / f"opening_{task}_cal.pkl").read_bytes())
    ns = sorted(data)
    Z = np.array([data[n]["z"][:, :weeks] for n in ns])
    print(f"{task}: share of the {len(ns)} episodes in which the grid shed no base load (our play), weeks 1..{weeks}")
    for gi, g in enumerate(data[ns[0]]["grids"]):
        print(f"  {g:8s} " + " ".join(f"{x:4.2f}" for x in Z[:, gi].mean(axis=0)))


def early(task, n, upto=12):
    """The events of an episode that act in weeks 1..upto: (text, onset, end, type name)."""
    from shockbench_flow.omega import codes

    inst, omega, _marks, _fb = C.world(task, n)
    out = []
    for q in W.events(inst, omega):
        if q.onset < upto and q.end > 0:
            out.append((W.describe(inst, q), q.onset, q.end, codes.EVENT_TYPES[q.type]))
    return out


def events(task="small", upto=12):
    """Per episode: the opening's loss (weeks 1-4 as played against free, USD) and the events of weeks 1..upto."""
    refs = C.references(task)
    rows = []
    for n in sorted(refs):
        res = json.loads((C.HERE / "res" / f"{task}_{n}.json").read_text())
        J = res["J"]
        ev = early(task, n, upto)
        physical = [e for e in ev if e[3] not in ("tariff", "piracy")]
        rows.append({"n": n, "level": refs[n]["stratum"], "o4": (J["pre_4"] - J["cal"]) / 1e9, "o4x": (J["prex_4"] - J["cal"]) / 1e9, "o8": (J["pre_8"] - J["cal"]) / 1e9, "o8x": (J["prex_8"] - J["cal"]) / 1e9,
                     "gap": (refs[n]["J_naive_cents"] - refs[n]["J_oracle_cents"]) / 1e11, "events": ev, "physical": physical,
                     "carried": [e for e in physical if e[1] <= 0], "new": [e for e in physical if e[1] > 0]})
    (C.HERE / f"opening_events_{task}.json").write_text(json.dumps(rows, indent=1))
    print(f"{task}: the opening's loss per episode (bn USD; weeks 1-4 as played against free, all | flows only; the same for weeks 1-8) and the events acting in weeks 1..{upto}")
    for r in sorted(rows, key=lambda r: -r["o4"]):
        print(f"  episode {r['n']:2d} level {r['level']}: {r['o4']:7.2f} | {r['o4x']:7.2f}   {r['o8']:7.2f} | {r['o8x']:7.2f}   gap {r['gap']:7.1f} bn; carried in {len(r['carried'])}, new {len(r['new'])}: " + "; ".join(e[0] for e in r["physical"][:6]))
    for key in ("o4", "o4x", "o8", "o8x"):
        x = np.array([r[key] for r in rows])
        calm = np.array([r[key] for r in rows if not r["physical"]])
        busy = np.array([r[key] for r in rows if r["physical"]])
        new = np.array([r[key] for r in rows if r["new"]])
        nonew = np.array([r[key] for r in rows if not r["new"]])
        print(f"  {key}: all {x.mean():.2f} (median {np.median(x):.2f}, sd {x.std():.2f}); no physical event in weeks 1..{upto} ({len(calm)}): {calm.mean() if len(calm) else float('nan'):.2f} (median {np.median(calm) if len(calm) else float('nan'):.2f});"
              f" some ({len(busy)}): {busy.mean() if len(busy) else float('nan'):.2f} (median {np.median(busy) if len(busy) else float('nan'):.2f}); a new one starts ({len(new)}): {new.mean() if len(new) else float('nan'):.2f}; none starts ({len(nonew)}): {nonew.mean() if len(nonew) else float('nan'):.2f}")


if __name__ == "__main__":
    fire.Fire({"collect": collect, "table": table, "events": events, "calendar": calendar})
