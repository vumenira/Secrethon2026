"""Shared pieces of the gap analysis (2026-10-07): worlds, references, the calendar model and its solves.

The calendar model is the earlier analyst's (``milp_real.py``): the package's clairvoyant LP ``build_lp(inst, marks)``
plus one binary z per grid with fabs and week (z = 1: no base load shed; z = 0: the grid's fabs get no energy). Here z
is always fixed to what a recorded play did, so every solve is an LP. A "variant" fixes a set of the LP's columns to
the values of the play (within a relative band of 2e-9, the replay test's own scale) and leaves the rest free.
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "recovery_tune"
AGENT = ROOT / "agents" / "mpc_pulse_rec_v4"
TOTAL = {"small": 64, "full": 16}
CACHED = {"small": "v_r1", "full": "v_r1"}
BAND = 2e-9


def world(task, n, entropy=12345):
    from shockbench_flow.evaluation.cache import default_cache_dir
    from shockbench_flow.policies.naive_fq import REPLICATIONS
    from shockbench_flow_agent.scoring import _world

    return _world(task, entropy, n, REPLICATIONS, str(default_cache_dir()))


def run_dir(task, entropy=12345):
    return OUT / f"{task}-e{entropy}-n{TOTAL[task]}"


def references(task, entropy=12345):
    return {r["episode"]: r for r in json.loads((run_dir(task, entropy) / "references.json").read_text())}


def cached(task, name=None, entropy=12345):
    rows = json.loads((run_dir(task, entropy) / f"{name or CACHED[task]}.json").read_text())["rows"]
    return {r["episode"]: r["J_policy_cents"] for r in rows}


def traj_path(task, n):
    return HERE / "traj" / f"{task}_{n}.pkl"


def load_traj(task, n):
    return pickle.loads(traj_path(task, n).read_bytes())


def pooled(refs, J_cents):
    """The board's pooled score and the score by harm level, for {episode: J in cents} (floats are rounded)."""
    from shockbench_flow.scoring.rss import rss_pooled, rss_stratum

    ns = sorted(J_cents)
    J = [int(round(J_cents[n])) for n in ns]
    naive = [refs[n]["J_naive_cents"] for n in ns]
    oracle = [refs[n]["J_oracle_cents"] for n in ns]
    strata = [refs[n]["stratum"] for n in ns]
    present = sorted(set(strata))
    from shockbench_flow.scoring.rss import STRATUM_WEIGHTS

    weights = [w if (i + 1) in present else 0.0 for i, w in enumerate(STRATUM_WEIGHTS)]
    by = {}
    for s in present:
        idx = [i for i, v in enumerate(strata) if v == s]
        by[s] = rss_stratum([J[i] for i in idx], [naive[i] for i in idx], [oracle[i] for i in idx])
    return rss_pooled(J, naive, oracle, strata, weights), by


# ----- the calendar model ------------------------------------------------------------------------------------------
def fab_grids(inst):
    """[(grid ordinal, its fabs that draw energy, the sum of its segment shares)] for grids served base load first."""
    out = []
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        fabs = [fo for fo in inst.grid_fabs[go] if inst.nodes[inst.fabs[fo]].fab.e > 0]
        if ga.priority == "base_first" and fabs:
            out.append((go, fabs, sum(ga.shares.values())))
    return out


class Cal:
    """One episode's calendar model: the clairvoyant LP with the binaries appended, and a HiGHS instance to solve it."""

    def __init__(self, task, n, entropy=12345):
        from shockbench_flow.oracle.lp import build_lp

        inst, omega, marks, _fallback = world(task, n, entropy)
        self.task, self.n = task, n
        self.inst = inst.at_digest(marks.instance_digest)
        self.marks = marks
        self.model = model = build_lp(inst, marks)
        self.T, self.nc, self.tmpl = model.T, model.meta["nc"], model.meta["template"]
        self.nn = len(model.lb)
        self.grids = fab_grids(self.inst)
        T, nc, tmpl, n0 = self.T, self.nc, self.tmpl, self.nn
        nz = len(self.grids) * T
        rows, cols, vals, hi = [], [], [], []
        r = 0
        self.emax = np.zeros((len(self.grids), T))
        for gi, (go, fabs, S) in enumerate(self.grids):
            ybar, Gbar = np.asarray(marks.y_bar)[:, go], np.asarray(marks.G_bar)[:, go]
            cap = np.zeros(T)
            for fo in fabs:
                fa = self.inst.nodes[self.inst.fabs[fo]].fab
                cap += np.where(np.asarray(marks.R)[:, fo] > 0, fa.e * np.asarray(marks.alpha_bar)[:, fo] * fa.cap0, 0.0)
            cap = np.minimum(cap, np.maximum(Gbar * S - ybar, 0.0))
            self.emax[gi] = cap
            for t in range(T):
                z = n0 + gi * T + t
                rows += [r, r]  # ysh + ybar z <= ybar
                cols += [t * nc + tmpl[("ysh", go)], z]
                vals += [1.0, float(ybar[t])]
                hi.append(float(ybar[t]))
                r += 1
                for fo in fabs:  # sum E - Emax z <= 0
                    rows.append(r)
                    cols.append(t * nc + tmpl[("E", fo)])
                    vals.append(1.0)
                rows.append(r)
                cols.append(z)
                vals.append(-float(cap[t]))
                hi.append(0.0)
                r += 1
        extra = sp.csr_matrix((vals, (rows, cols)), shape=(r, n0 + nz))
        pad = lambda A: sp.hstack([A, sp.csr_matrix((A.shape[0], nz))], format="csr")  # noqa: E731
        self.A = sp.vstack([pad(model.A_ub), pad(model.A_eq), extra], format="csc")
        self.lo = np.concatenate([np.full(model.A_ub.shape[0], -np.inf), model.b_eq, np.full(r, -np.inf)])
        self.up = np.concatenate([model.b_ub, model.b_eq, np.array(hi)])
        self.cost = np.concatenate([model.objective(), np.zeros(nz)])
        self.lb = np.concatenate([model.lb, np.zeros(nz)])
        self.ub = np.concatenate([model.ub, np.ones(nz)])
        self.offset = -float(np.sum(model.meta["salvage_const"]))
        self.nz = nz
        self.h = None
        # column classes of one week's template
        self.keys = [None] * nc
        for key, j in tmpl.items():
            self.keys[j] = key

    def calendar(self, zpi):
        """z of a play's column vector: 1 where the grid shed no base load. Also the grid-weeks it shed and powered."""
        T, nc, tmpl = self.T, self.nc, self.tmpl
        z = np.zeros((len(self.grids), T))
        both = 0
        for gi, (go, fabs, _S) in enumerate(self.grids):
            ybar = np.asarray(self.marks.y_bar)[:, go]
            shed = zpi[np.arange(T) * nc + tmpl[("ysh", go)]]
            E = sum(zpi[np.arange(T) * nc + tmpl[("E", fo)]] for fo in fabs)
            z[gi] = (shed <= 1e-9 * np.maximum(ybar, 1.0)).astype(float)
            both += int(((shed > 1e-6 * ybar) & (E > 1e-6)).sum())
        return z, both

    def residual(self, full):
        """The worst relative row residual of the whole model at a full column vector (the replay test's scale)."""
        Ax = self.A @ full
        A = self.A.tocsr()
        prod = np.abs(A.data * full[A.indices])
        maxterm = np.zeros(A.shape[0])
        nonempty = np.diff(A.indptr) > 0
        maxterm[nonempty] = np.maximum.reduceat(prod, A.indptr[:-1][nonempty])
        scale = np.maximum(1.0, np.maximum(maxterm, np.maximum(np.abs(np.where(np.isfinite(self.up), self.up, 0.0)), np.abs(np.where(np.isfinite(self.lo), self.lo, 0.0)))))
        viol = np.maximum(Ax - self.up, self.lo - Ax) / scale
        return float(viol.max())

    def highs(self):
        from scipy.optimize._highspy import _core as hs

        if self.h is not None:
            return self.h
        A = self.A
        inf = hs.kHighsInf
        lp = hs.HighsLp()
        lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
        lp.col_cost_ = self.cost
        lp.col_lower_ = np.where(np.isinf(self.lb), -inf, self.lb)
        lp.col_upper_ = np.where(np.isinf(self.ub), inf, self.ub)
        lp.row_lower_ = np.where(np.isinf(self.lo), -inf, self.lo)
        lp.row_upper_ = np.where(np.isinf(self.up), inf, self.up)
        lp.offset_ = float(self.offset)
        lp.a_matrix_.format_ = hs.MatrixFormat.kColwise
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = A.shape[1], A.shape[0]
        lp.a_matrix_.start_ = A.indptr.astype(np.int32)
        lp.a_matrix_.index_ = A.indices.astype(np.int32)
        lp.a_matrix_.value_ = A.data.astype(np.float64)
        h = hs._Highs()
        h.setOptionValue("output_flag", False)  # no "threads" option: the scheduler is already up, and run() then errors
        if h.passModel(lp) != hs.HighsStatus.kOk:
            raise RuntimeError("HiGHS refused the model")
        self.h, self.hs = h, hs
        return h

    def solve(self, zpi, z, fixed=None, band=BAND, limit=900.0, fresh=False):
        """The LP with z fixed and the columns of ``fixed`` (a bool mask over the LP's columns) held at ``zpi``.

        Returns (J in USD formed as the simulator forms it or None, the status text, seconds, x or None).
        """
        from shockbench_flow.oracle.lp import lp_usd

        h = self.highs()
        hs = self.hs
        lb, ub = self.lb.copy(), self.ub.copy()
        if fixed is not None:
            idx = np.flatnonzero(fixed)
            v = zpi[idx]
            d = band * np.abs(v)
            new_ub = np.minimum(ub[idx], v + d)
            new_lb = np.minimum(np.maximum(lb[idx], v - d), new_ub)
            lb[idx], ub[idx] = new_lb, np.maximum(new_ub, new_lb)
        lb[self.nn :], ub[self.nn :] = z.ravel(), z.ravel()
        inf = hs.kHighsInf
        n = len(lb)
        h.changeColsBounds(n, np.arange(n, dtype=np.int32), np.where(np.isinf(lb), -inf, lb), np.where(np.isinf(ub), inf, ub))
        if fresh:
            h.clearSolver()
        h.setOptionValue("time_limit", float(limit))
        t0 = time.perf_counter()
        h.run()
        seconds = time.perf_counter() - t0
        info = h.getInfo()
        status = h.modelStatusToString(h.getModelStatus())
        x = np.array(h.getSolution().col_value, dtype=np.float64)
        ok = status == "Optimal" and x.shape == (n,) and bool(np.all(np.isfinite(x)))
        if not ok:
            return None, status, seconds, None
        return float(lp_usd(self.model, x[: self.nn])[0]), status, seconds, x

    # ----- column masks ---------------------------------------------------------------------------------------------
    def mask(self, pick, weeks=None):
        """A bool mask over the LP's columns: template columns where ``pick(key)`` holds, in ``weeks`` (0-based) or all."""
        cols = np.array([j for j, key in enumerate(self.keys) if pick(key)], dtype=int)
        out = np.zeros(self.nn + self.nz, dtype=bool)
        if cols.size == 0:
            return out
        ts = np.arange(self.T) if weeks is None else np.asarray(list(weeks), dtype=int)
        out[(ts[:, None] * self.nc + cols[None, :]).ravel()] = True
        return out


def classes(inst):
    """Predicates on a column key for the decision groups of the split."""
    chk = set(inst.chokepoints)
    K = inst.commodities
    name = {k: c.id for k, c in enumerate(K)}
    fuels = {k for g in inst.grids for k in inst.nodes[g].grid.fuels}
    inputs = {inst.nodes[f].fab.input for f in inst.fabs}
    grids = set(inst.grids)
    is_x = lambda key: key[0] == "x"  # noqa: E731
    tail = lambda key: inst.edges[key[1]].tail  # noqa: E731
    head = lambda key: inst.edges[key[1]].head  # noqa: E731
    out = {
        # the task's partition: flows out of non-strait nodes by good, releases at straits, lot starts, the rest
        "fuel": lambda key: is_x(key) and key[2] in fuels and tail(key) not in chk,
        "mat": lambda key: is_x(key) and key[2] in inputs and tail(key) not in chk,
        "chip": lambda key: is_x(key) and key[2] not in fuels and key[2] not in inputs and tail(key) not in chk,
        "strait": lambda key: is_x(key) and tail(key) in chk,
        "lots": lambda key: key[0] == "p",
        "other": lambda key: key[0] in ("G", "E", "y", "ysh", "xi", "D", "U", "O", "lift"),
        # pieces of the rest
        "grid": lambda key: key[0] in ("G", "E", "y", "ysh"),
        "xi": lambda key: key[0] == "xi",
        "serve": lambda key: key[0] in ("D", "U"),
        "disp": lambda key: key[0] in ("O", "lift"),
        # by good, strait releases included
        "fuel+": lambda key: is_x(key) and key[2] in fuels,
        "mat+": lambda key: is_x(key) and key[2] in inputs,
        "chip+": lambda key: is_x(key) and key[2] not in fuels and key[2] not in inputs,
        # pieces of the fuel flows
        "fuel_grid": lambda key: is_x(key) and key[2] in fuels and head(key) in grids and inst.edges[key[1]].tau == 0,
        "fuel_up": lambda key: is_x(key) and key[2] in fuels and tail(key) not in chk and not (head(key) in grids and inst.edges[key[1]].tau == 0),
        "strait_fuel": lambda key: is_x(key) and tail(key) in chk and key[2] in fuels,
        "strait_rest": lambda key: is_x(key) and tail(key) in chk and key[2] not in fuels,
        "x": is_x,
        "state": lambda key: key[0] in ("I", "Q", "B"),
        "rules": lambda key: key[0] in ("p", "G", "E", "y", "ysh", "xi", "D", "U", "O", "lift"),
        "nonstate": lambda key: key[0] not in ("I", "Q", "B"),
        "all": lambda key: True,
    }
    for k in fuels:
        out[f"{name[k]}+"] = (lambda kk: lambda key: is_x(key) and key[2] == kk)(k)
    return out
