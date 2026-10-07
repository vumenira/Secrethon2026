"""The calendar model with the simulator's rule for weeks that shed base load made exact.

In a week a grid sheds, the simulator loads every segment fully, so each fuel burns all it can: gas
min(share x power, share x power x last week's stock / threshold, stock on hand), another fuel min(share x power, stock
on hand). The calendar model (common.Cal) only bounds the burn from above, so its plans can keep fuel at a grid through
a shed week. Here, for every grid with fabs, week and fuel with a terminal (gas, crude), binaries pick which of those
limits is the one reached whenever z = 0 (z = 0 now means: the week sheds). Nuclear fuel is left as it was.

    crude:  b = 1: the stock runs out            I_t <= S (z + 1 - b)
            b = 0: the segment is full           G >= cap - cap (z + b)
    gas:    b1 = 1: the stock runs out           I_t <= S (z + 1 - b1)
            b2 = 1: the ration is what limits    G >= (cap / thr) I_{t-1} - (cap S / thr) (z + 1 - b2)
            neither: the segment is full         G >= cap - cap (z + b1 + b2);   b1 + b2 <= 1

Our own play satisfies every row (checked by ``Tight.check``), so the optimum is still a bound for nothing but is a
plan closer to what an agent can do; the proven bound of a time-limited run is weak.

uv run python tight.py run --task=small --only=41,29 --limit=120
"""

import json
import pickle
import time

import fire
import numpy as np
import scipy.sparse as sp

import common as C
import diff as Dm


OUT = C.HERE / "tight"
EPS = 1e-7


class Tight(C.Cal):
    def __init__(self, task, n, entropy=12345, marks=None):
        super().__init__(task, n, entropy, marks)
        from shockbench_flow.dynamics.sim import initial_stock

        inst, mk, T, nc, tmpl = self.inst, self.marks, self.T, self.nc, self.tmpl
        i0 = initial_stock(inst)
        psi = inst.params.psi
        n0 = self.nn + self.nz
        rows, cols, vals, up = [], [], [], []
        r = 0
        self.cases = []  # (gi, t, k, kind, first binary column, slot, cap, thr)
        nb = 0

        def add(entries, rhs):
            nonlocal r
            for j, v in entries:
                rows.append(r)
                cols.append(j)
                vals.append(v)
            up.append(rhs)
            r += 1

        for gi, (go, _fabs, _S) in enumerate(self.grids):
            g = inst.grids[go]
            ga = inst.nodes[g].grid
            Gbar = np.asarray(mk.G_bar)[:, go]
            for k in ga.fuels:
                if inst.commodities[k].id == "nucfuel":
                    continue
                s = inst.slot_index[(g, k)]
                S = float(inst.stock_slots[s].storage)
                rationed = k == ga.rationed and psi * ga.ibar[k] > 0
                thr = psi * ga.ibar[k] if rationed else 0.0
                for t in range(T):
                    cap = float(ga.shares[k] * Gbar[t])
                    jG, jI, jz = t * nc + tmpl[("G", go, k)], t * nc + tmpl[("I", s)], self.nn + gi * T + t
                    b = n0 + nb
                    if not rationed:
                        self.cases.append((gi, t, k, "plain", b, s, cap, 0.0))
                        nb += 1
                        add([(jI, 1.0), (jz, -S), (b, S)], S * (1 + EPS))  # b = 1: the stock runs out
                        add([(jG, -1.0), (jz, -cap), (b, -cap)], -cap * (1 - EPS))  # b = 0: the segment is full
                    else:
                        self.cases.append((gi, t, k, "gas", b, s, cap, thr))
                        nb += 2
                        M = cap * S / thr  # the ration row divided by thr: G >= (cap / thr) I_{t-1} - M (z + 1 - b2)
                        add([(jI, 1.0), (jz, -S), (b, S)], S * (1 + EPS))
                        ent = [(jG, -1.0), (jz, -M), (b + 1, M)]
                        rhs = M
                        if t > 0:
                            ent.append(((t - 1) * nc + tmpl[("I", s)], cap / thr))
                        else:
                            rhs -= cap * float(i0[s]) / thr
                        add(ent, rhs + EPS * max(M, 1.0))
                        add([(jG, -1.0), (jz, -cap), (b, -cap), (b + 1, -cap)], -cap * (1 - EPS))
                        add([(b, 1.0), (b + 1, 1.0)], 1.0)
        self.nb, self.i0 = nb, i0
        extra = sp.csr_matrix((vals, (rows, cols)), shape=(r, n0 + nb))
        A = sp.hstack([self.A.tocsr(), sp.csr_matrix((self.A.shape[0], nb))], format="csr")
        self.A = sp.vstack([A, extra], format="csc")
        self.lo = np.concatenate([self.lo, np.full(r, -np.inf)])
        self.up = np.concatenate([self.up, np.array(up)])
        self.cost = np.concatenate([self.cost, np.zeros(nb)])
        self.lb = np.concatenate([self.lb, np.zeros(nb)])
        self.ub = np.concatenate([self.ub, np.ones(nb)])

    def bounds(self, z=None, fixed=None, values=None, band=C.BAND):
        lb, ub = self.lb.copy(), self.ub.copy()
        if z is not None:
            zz = np.asarray(z, dtype=float).ravel()
            known = ~np.isnan(zz)
            idx = self.nn + np.flatnonzero(known)
            lb[idx], ub[idx] = zz[known], zz[known]
        return lb, ub

    def start(self, x, z):
        """The full column vector of a simulator trajectory (or any plan that obeys the rule): its case binaries."""
        a = np.asarray(x, dtype=float)[: self.nn].reshape(self.T, self.nc)
        b = np.zeros(self.nb)
        n0 = self.nn + self.nz
        for gi, t, k, kind, col, s, cap, thr in self.cases:
            if z[gi, t] > 0.5:
                continue
            go = self.grids[gi][0]
            I = a[t, self.tmpl[("I", s)]]
            G = a[t, self.tmpl[("G", go, k)]]
            Iprev = a[t - 1, self.tmpl[("I", s)]] if t > 0 else float(self.i0[s])
            if G >= cap * (1 - 1e-9):
                continue  # the segment is full
            if kind == "gas" and thr * G >= cap * Iprev * (1 - 1e-9) - 1e-9:
                b[col - n0 + 1] = 1.0
            else:
                b[col - n0] = 1.0
        return np.concatenate([np.asarray(x, dtype=float)[: self.nn], np.asarray(z, dtype=float).ravel(), b])


def one(task, n, limit, free=True):
    start = time.perf_counter()
    refs = C.references(task)[n]
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    d = C.load_traj(task, n)
    cal = Tight(task, n)
    z0, _both = cal.calendar(d["zpi"])
    play = cal.start(d["zpi"], z0)
    out = {"episode": n, "D": D, "J_played": d["J_cents"] / 100, "residual_play": cal.residual(play), "binaries": int(cal.nz + cal.nb)}
    # 1. our calendar, the rule exact
    r1 = cal.milp(*cal.bounds(z=z0), start=play, limit=limit, gap=1e-4)
    x1 = r1.pop("x")
    out["cal"] = r1
    # 2. the realizable plan's calendar (of the looser model), the rule exact
    p = Dm.OUT / f"{task}_{n}.pkl"
    zs = pickle.loads(p.read_bytes())["z"] if p.exists() else None
    if zs is not None and (zs != z0).any():
        r2 = cal.milp(*cal.bounds(z=zs), start=None, limit=min(limit, 30.0), gap=1e-4)
        r2.pop("x")
        out["plan_calendar"] = r2
    # 3. the calendar free
    if free and x1 is not None:
        r3 = cal.milp(*cal.bounds(), start=x1, limit=2 * limit, gap=1e-4)
        x3 = r3.pop("x")
        out["free"] = r3
        if x3 is not None:
            OUT.mkdir(exist_ok=True)
            (OUT / f"{task}_{n}.pkl").write_bytes(pickle.dumps({"x": x3[: cal.nn].reshape(cal.T, cal.nc), "z": np.round(x3[cal.nn : cal.nn + cal.nz]).reshape(len(cal.grids), cal.T), "z_ours": z0}))
    out["seconds"] = round(time.perf_counter() - start, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{task}_{n}.json").write_text(json.dumps(out, indent=1))
    return out


def one_window(task, n, W, limit):
    """Our calendar kept after week W and free in weeks 1..W, the rule exact: what re-timing the opening alone can give."""
    start = time.perf_counter()
    refs = C.references(task)[n]
    D = (refs["J_naive_cents"] - refs["J_oracle_cents"]) / 100
    d = C.load_traj(task, n)
    cal = Tight(task, n)
    z0, _both = cal.calendar(d["zpi"])
    play = cal.start(d["zpi"], z0)
    out = {"episode": n, "D": D, "W": W, "J_played": d["J_cents"] / 100}
    r1 = cal.milp(*cal.bounds(z=z0), start=play, limit=min(limit, 60.0), gap=1e-4)
    x1 = r1.pop("x")
    out["cal"] = r1
    zf = z0.astype(float).copy()
    zf[:, :W] = np.nan
    r2 = cal.milp(*cal.bounds(z=zf), start=x1 if x1 is not None else play, limit=limit, gap=1e-4)
    x2 = r2.pop("x")
    out["window"] = r2
    if x2 is not None:
        zw = np.round(x2[cal.nn : cal.nn + cal.nz]).reshape(len(cal.grids), cal.T)
        out["z_ours"] = ["".join("#" if q else "." for q in row[: W + 4]) for row in z0]
        out["z_window"] = ["".join("#" if q > 0.5 else "." for q in row[: W + 4]) for row in zw]
        OUT.mkdir(exist_ok=True)
        (OUT / f"{task}_{n}_w{W}.pkl").write_bytes(pickle.dumps({"x": x2[: cal.nn].reshape(cal.T, cal.nc), "z": zw, "z_ours": z0}))
    out["seconds"] = round(time.perf_counter() - start, 1)
    OUT.mkdir(exist_ok=True)
    (OUT / f"{task}_{n}_w{W}.json").write_text(json.dumps(out, indent=1))
    return out


def window(task="small", only=None, W=12, limit=120.0, n_jobs=2):
    from joblib import Parallel, delayed

    from report import weights

    refs = C.references(task)
    w = weights(task, sorted(refs))
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    tot = totb = 0.0
    for i in range(0, len(ns), n_jobs):
        for out in Parallel(n_jobs=n_jobs)(delayed(one_window)(task, n, int(W), limit) for n in ns[i : i + n_jobs]):
            n, D = out["episode"], out["D"]
            naive = refs[n]["J_naive_cents"] / 100
            sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
            a, b = out["cal"], out["window"]
            if a["J"] is not None and b["J"] is not None:
                tot += w[n] * (a["J"] - b["J"])
                totb += w[n] * (a["J"] - b["bound"])
            print(f"episode {n}: our calendar, rule exact {sc(a['J']):.4f} [{a['status']}]; weeks 1-{W} free {sc(b['J']):.4f} [{b['status']}, bound {sc(b['bound']):.4f}]; ours {out.get('z_ours')} found {out.get('z_window')}  ({out['seconds']} s)", flush=True)
    print(f"pooled gain of the calendars found over ours on these {len(ns)} episodes: {tot:.4f} (proven bound {totb:.4f}); {time.perf_counter() - start:.0f} s")


def show(out, refs):
    n, D = out["episode"], out["D"]
    naive = refs[n]["J_naive_cents"] / 100
    sc = lambda J: float("nan") if J is None else (naive - J) / D  # noqa: E731
    line = f"episode {n}: played {sc(out['J_played']):.4f} (its rows hold to {out['residual_play']:.1e}); our calendar, rule exact {sc(out['cal']['J']):.4f} [{out['cal']['status']}, bound {sc(out['cal']['bound']):.4f}]"
    if "plan_calendar" in out:
        r = out["plan_calendar"]
        line += f"; the looser plan's calendar, rule exact {sc(r['J']):.4f} [{r['status']}, bound {sc(r['bound']):.4f}]"
    if "free" in out:
        r = out["free"]
        line += f"; calendar free {sc(r['J']):.4f} [{r['status']}, bound {sc(r['bound']):.4f}]"
    print(line + f"  ({out['seconds']} s)", flush=True)


def run(task="small", only=None, limit=120.0, n_jobs=2, free=True):
    from joblib import Parallel, delayed

    refs = C.references(task)
    ns = sorted(refs) if only is None else ([only] if isinstance(only, int) else [int(v) for v in only])
    start = time.perf_counter()
    for i in range(0, len(ns), n_jobs):
        for out in Parallel(n_jobs=n_jobs)(delayed(one)(task, n, limit, free) for n in ns[i : i + n_jobs]):
            show(out, refs)
    print(f"{time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    fire.Fire({"run": run, "window": window})
