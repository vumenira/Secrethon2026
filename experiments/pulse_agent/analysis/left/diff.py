"""Question 2: where the realizable plan's fab calendar differs from ours, what each difference is worth and what
makes it possible.

Per episode: the realizable plan's calendar (left/milp, else the earlier incumbent) is first reduced to the
differences that pay: a grid-week where the plan neither sheds nor powers a fab takes our value; then every grid's
differences, every cluster of them (difference weeks at one grid no more than 3 weeks apart) and every run are put
back to our calendar where that costs nothing. What is left is valued cluster by cluster (the loss when it alone is
put back; the gain when it alone is added to our calendar) and its extra fab energy is traced to its source.

uv run python diff.py run --task=small --n_jobs=2
uv run python diff.py report --task=small
"""

import json
import pickle
import time

import fire
import numpy as np

import common as C
import view as V


OUT = C.HERE / "diff"
GAP = 3  # difference weeks at one grid no more than this many weeks apart form one cluster
TOL = 2e-5  # a difference pays when putting it back costs more than this share of the episode's gap


def plan_of(task, n, cal):
    p = C.HERE / "milp" / f"{task}_{n}.pkl"
    if p.exists():
        m = pickle.loads(p.read_bytes())
        return np.round(np.asarray(m["z"], dtype=float)), "new"
    p = C.HERE.parent / "milp" / f"{task}_{n}.pkl"
    if p.exists():
        m = pickle.loads(p.read_bytes())
        z = np.round(np.asarray(m["z"], dtype=float))
        if z.shape == (len(cal.grids), cal.T):
            return z, "old"
    return None, None


def clusters(diff):
    """[(gi, [weeks])] of a (G, T) bool array: per grid, difference weeks no more than GAP apart."""
    out = []
    for gi in range(diff.shape[0]):
        ws = np.flatnonzero(diff[gi])
        cur = []
        for w in ws:
            if cur and w - cur[-1] > GAP:
                out.append((gi, cur))
                cur = []
            cur.append(int(w))
        if cur:
            out.append((gi, cur))
    return out


def runs(diff, sign):
    """[(gi, [weeks])]: per grid, runs of consecutive difference weeks of one sign."""
    out = []
    for gi in range(diff.shape[0]):
        cur = []
        for w in range(diff.shape[1]):
            if diff[gi, w] and cur and (w - cur[-1] > 1 or sign[gi, w] != sign[gi, cur[-1]]):
                out.append((gi, cur))
                cur = []
            if diff[gi, w]:
                cur.append(w)
        if cur:
            out.append((gi, cur))
    return out


def grid_series(v, cal, a, gi):
    """Per-week arrays of one fab grid from a (T, nc) column array."""
    inst = cal.inst
    go, fabs, _S = cal.grids[gi]
    g = inst.grids[go]
    ga = inst.nodes[g].grid
    out = {"E": sum(v.col(a, ("E", fo)) for fo in fabs), "shed": v.col(a, ("ysh", go)), "null": v.col(a, ("G", go, None)), "fuel": {}}
    for k in ga.fuels:
        tl, e = v.terminal(go, k)
        f = {"G": v.col(a, ("G", go, k)), "Ig": v.stock(a, g, k)}
        arr = np.zeros(cal.T)
        if tl is not None:
            f["It"] = v.stock(a, tl, k)
            f["del"] = v.flow(a, e, k)
            arr += v.into(a, tl, k, arrive=True)
            f["sent"] = v.into(a, tl, k)
        else:
            f["It"] = np.zeros(cal.T)
            f["sent"] = np.zeros(cal.T)
        # what reaches the grid over edges with a lead (a source straight into the grid), by arrival week
        for e2 in inst.in_edges[g]:
            ed = inst.edges[e2]
            if k in ed.K and ed.tau > 0 and (e2, k) in v.flows:
                fl = v.flow(a, e2, k)
                sh = np.zeros(cal.T)
                sh[ed.tau :] = fl[: cal.T - ed.tau]
                arr += sh
                f["sent"] = f["sent"] + fl
        f["arr"] = arr
        out["fuel"][v.K[k]] = f
    return out


def one(task, n):
    start = time.perf_counter()
    refs = C.references(task)[n]
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    d = C.load_traj(task, n)
    cal = C.Cal(task, n)
    T, G = cal.T, len(cal.grids)
    z0, _both = cal.calendar(d["zpi"])
    zm, src = plan_of(task, n, cal)
    out = {"episode": n, "D": D, "J_played": d["J_cents"] / 100, "src": src}
    if zm is None:
        return out
    v = V.View(cal)
    solves = [0]

    def J_of(z):
        solves[0] += 1
        J, _st, _s, x = cal.lp(*cal.bounds(z=z))
        return (np.inf if J is None else J), x

    J_cal, x_cal = J_of(z0)
    J_m, x_m = J_of(zm)
    out["J_cal"], out["J_plan"] = J_cal, J_m
    if not np.isfinite(J_m) or J_m >= J_cal - TOL * D:
        out["clusters"] = []
        out["J_final"] = min(J_cal, J_m)
        out["seconds"] = round(time.perf_counter() - start, 1)
        return out
    # a grid-week where the plan neither sheds nor powers a fab takes our value
    xm = x_m[: cal.nn].reshape(T, cal.nc)
    z = zm.copy()
    for gi, (go, fabs, _S) in enumerate(cal.grids):
        E = sum(v.col(xm, ("E", fo)) for fo in fabs)
        shed = v.col(xm, ("ysh", go))
        ybar = np.asarray(cal.marks.y_bar)[:, go]
        idle = (E <= 1e-6 * np.maximum(cal.emax[gi], 1.0)) & (shed <= 1e-7 * ybar)
        z[gi, idle] = z0[gi, idle]
    J, x = J_of(z)
    if J > J_m + TOL * D:  # should not happen: keep the plan's own calendar then
        z, J, x = zm.copy(), J_m, x_m
    out["weeks_differ_raw"], out["weeks_differ_physical"] = int((zm != z0).sum()), int((z != z0).sum())
    tol = TOL * D
    # put back what does not pay: whole grids, clusters, runs; repeated until nothing more goes
    for _pass in range(3):
        changed = False
        diff = z != z0
        groups = [(gi, list(np.flatnonzero(diff[gi]))) for gi in range(G) if diff[gi].any()] + clusters(diff) + runs(diff, z)
        for gi, ws in groups:
            ws = [w for w in ws if z[gi, w] != z0[gi, w]]
            if not ws:
                continue
            zt = z.copy()
            zt[gi, ws] = z0[gi, ws]
            Jt, xt = J_of(zt)
            if Jt <= J + tol:
                z, J, x, changed = zt, min(J, Jt), xt, True
        if not changed:
            break
    out["J_final"] = J
    out["weeks_differ_final"] = int((z != z0).sum())
    xs = x[: cal.nn].reshape(T, cal.nc)
    xo = d["zpi"].reshape(T, cal.nc)
    diff = z != z0
    cl = []
    S_o = [grid_series(v, cal, xo, gi) for gi in range(G)]
    S_m = [grid_series(v, cal, xs, gi) for gi in range(G)]
    S_c = [grid_series(v, cal, x_cal[: cal.nn].reshape(T, cal.nc), gi) for gi in range(G)]
    for gi, ws in clusters(diff):
        zt = z.copy()
        zt[gi, ws] = z0[gi, ws]
        L = J_of(zt)[0] - J
        zt = z0.copy()
        zt[gi, ws] = z[gi, ws]
        A = J_cal - J_of(zt)[0]
        a, b = ws[0], ws[-1]
        span = slice(a, b + 1)
        on = [w for w in ws if z[gi, w] == 1]  # the plan sheds nothing where we shed
        off = [w for w in ws if z[gi, w] == 0]  # the plan sheds (or idles its fabs) where we shed nothing
        go = cal.grids[gi][0]
        rec = {"grid": v.node[cal.inst.grids[go]], "gi": gi, "a": a + 1, "b": b + 1, "on": [w + 1 for w in on], "off": [w + 1 for w in off], "loss_back": L, "gain_alone": A}
        for label, S in (("ours", S_o), ("plan", S_m), ("cal", S_c)):
            s = S[gi]
            rec[label] = {"E": float(s["E"][span].sum()), "shed": float(s["shed"][span].sum()), "null": float(s["null"][span].sum()), "E_on": float(s["E"][on].sum()) if on else 0.0, "E_off": float(s["E"][off].sum()) if off else 0.0}
        rec["emax_on"] = float(cal.emax[gi][on].sum()) if on else 0.0
        rec["partial_on"] = float(sum(S_m[gi]["E"][w] for w in on if S_m[gi]["E"][w] < 0.9 * cal.emax[gi][w]))
        rec["fuel"] = {}
        for k, fm in S_m[gi]["fuel"].items():
            fo = S_o[gi]["fuel"][k]
            fr = {}
            for label, f in (("ours", fo), ("plan", fm), ("cal", S_c[gi]["fuel"][k])):
                fr[label] = {
                    "G": float(f["G"][span].sum()),
                    "G_before": float(f["G"][:a].sum()),
                    "G_open": float(f["G"][: min(a, 5)].sum()),  # of that, in weeks 1..5 (the starting stock's weeks)
                    "arr_to_b": float(f["arr"][: b + 1].sum()),
                    "arr_before": float(f["arr"][:a].sum()),
                    "stock_before": float((f["Ig"][a - 1] + f["It"][a - 1]) if a > 0 else np.nan),
                    "stock_end": float(f["Ig"][b] + f["It"][b]),
                }
            # the week from which the plan has sent more of this fuel toward the grid than we have (10 % of the final surplus)
            ds = np.cumsum(fm["sent"] - fo["sent"])[: b + 1]
            fr["sent_more_to_b"] = float(ds[-1]) if len(ds) else 0.0
            first = np.flatnonzero(ds > 0.1 * max(ds[-1], 1e-9)) if len(ds) and ds[-1] > 0 else []
            fr["sent_from"] = int(first[0]) + 1 if len(first) else None
            rec["fuel"][k] = fr
        # the same fuel at every fab grid's terminal together: what arrived by the cluster's end, plan minus ours
        rec["arr_all"] = {}
        for k in rec["fuel"]:
            tot = 0.0
            for gj in range(G):
                if k in S_m[gj]["fuel"]:
                    tot += float(S_m[gj]["fuel"][k]["arr"][: b + 1].sum() - S_o[gj]["fuel"][k]["arr"][: b + 1].sum())
            rec["arr_all"][k] = tot
        # our own calendar around the week the plan began to send more
        rec["ours_off_before"] = {w + 1: int(z0[gi, max(w - 3, 0) : w + 1].min() == 0) for w in range(max(a - 8, 0), a + 1)}
        cl.append(rec)
    # the clusters added to our calendar one after another in the order they begin: each one's step (they sum to the
    # whole difference); a step that cannot be solved (a later cluster needs it) is carried to the next
    order = sorted(range(len(cl)), key=lambda i: (cl[i]["a"], cl[i]["gi"]))
    zt, prev, pending = z0.copy(), J_cal, []
    for i in order:
        c = cl[i]
        ws = [w - 1 for w in c["on"] + c["off"]]
        zt[c["gi"], ws] = z[c["gi"], ws]
        Jt = J_of(zt)[0]
        if not np.isfinite(Jt):
            c["step"] = 0.0
            pending.append(i)
            continue
        c["step"] = prev - Jt
        c["carried"] = [cl[j]["grid"] + f" {cl[j]['a']}-{cl[j]['b']}" for j in pending]
        prev, pending = Jt, []
    out["step_sum"], out["step_left"] = J_cal - prev, len(pending)
    out["clusters"] = cl
    out["solves"] = solves[0]
    out["z_final"], out["z_ours"] = z.astype(int).tolist(), z0.astype(int).tolist()
    out["seconds"] = round(time.perf_counter() - start, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{task}_{n}.json").write_text(json.dumps(out, indent=1))
    (OUT / f"{task}_{n}.pkl").write_bytes(pickle.dumps({"x": xs.astype(np.float64), "z": z, "z_ours": z0, "x_cal": x_cal[: cal.nn].reshape(T, cal.nc)}))
    return out


def run(task="small", only=None, n_jobs=2, redo=False):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    todo = [n for n in ns if redo or not (OUT / f"{task}_{n}.json").exists()]
    start = time.perf_counter()
    for i in range(0, len(todo), 2 * n_jobs):
        for out in Parallel(n_jobs=n_jobs)(delayed(one)(task, n) for n in todo[i : i + 2 * n_jobs]):
            n, D = out["episode"], out["D"]
            naive = refs[n]["J_naive_cents"] / 100
            sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
            print(
                f"episode {n} ({out.get('src')}): calendar {sc(out.get('J_cal')):.4f} plan {sc(out.get('J_plan')):.4f} after putting back {sc(out.get('J_final')):.4f};"
                f" grid-weeks that differ {out.get('weeks_differ_raw')} -> {out.get('weeks_differ_physical')} -> {out.get('weeks_differ_final')}; clusters {len(out.get('clusters', []))}; {out.get('solves')} LPs, {out.get('seconds')} s",
                flush=True,
            )
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"run": run})
