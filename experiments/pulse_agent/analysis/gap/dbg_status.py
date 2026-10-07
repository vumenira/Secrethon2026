"""Two checks of the solves that did not come out "Optimal" at the first try in solve.py.

1. "Unknown" then "Optimal" with a wider band and a cold start: which of the two changes mattered, and does the
   optimum depend on the band? Each (episode, variant) is solved cold with the tight band and cold with the wide one.
2. "all" (every column fixed, the stocks too) infeasible: the rows our own vector violates in absolute terms, and the
   same LP with the stocks left free (every decision and rule fixed).

uv run python dbg_status.py
"""

import json

import numpy as np

import common as C
import solve as S


def unknowns(task, pairs):
    for n, name in pairs:
        d = C.load_traj(task, n)
        cal = C.Cal(task, n)
        zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
        z, _ = cal.calendar(d["zpi"])
        zpi[cal.nn :] = z.ravel()
        fixed = S.variants(cal, task)[name]
        stored = json.loads((C.HERE / "res" / f"{task}_{n}_all.json").read_text())["J"][name]
        Jt, st, sec_t, _ = cal.solve(zpi, z, fixed, band=C.BAND, fresh=True)
        Jw, sw, sec_w, _ = cal.solve(zpi, z, fixed, band=1e-7, fresh=True)
        rel = float("nan") if Jt is None or Jw is None else (Jt - Jw) / abs(Jw)
        rel_s = float("nan") if Jw is None else (stored - Jw) / abs(Jw)
        print(f"episode {n} {name}: cold, tight band: {st} ({sec_t:.1f} s); cold, wide band: {sw} ({sec_w:.1f} s); tight minus wide {rel:+.2e} of the cost; stored minus wide {rel_s:+.2e}", flush=True)


def all_fixed(task, ns):
    for n in ns:
        d = C.load_traj(task, n)
        cal = C.Cal(task, n)
        zpi = np.concatenate([d["zpi"], np.zeros(cal.nz)])
        z, _ = cal.calendar(d["zpi"])
        zpi[cal.nn :] = z.ravel()
        Ax = cal.A @ zpi
        viol = np.maximum(Ax - cal.up, cal.lo - Ax)
        over = np.maximum(zpi - cal.ub, cal.lb - zpi)
        cls = C.classes(cal.inst)
        nonstate = cal.mask(cls["nonstate"])
        J, status, sec, _ = cal.solve(zpi, z, nonstate, band=C.BAND, fresh=True)
        Jp = d["J_cents"] / 100
        print(
            f"episode {n}: our vector's largest absolute row violation {viol.max():.2e} (rows above 1e-7: {int((viol > 1e-7).sum())}), largest bound violation {over.max():.2e};"
            f" every decision and rule fixed, stocks free: {status}, cost minus ours {float('nan') if J is None else (J - Jp) / abs(Jp):+.2e} of the cost ({sec:.1f} s)",
            flush=True,
        )


if __name__ == "__main__":
    unknowns("small", [(63, "freex_fuel"), (63, "fix_fuel+"), (63, "prex_48"), (12, "fix_fuel+"), (23, "win_32_44"), (43, "free_fuel")])
    all_fixed("small", [3, 6, 0, 1])
