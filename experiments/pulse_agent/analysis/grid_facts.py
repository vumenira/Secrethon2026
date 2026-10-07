"""Print the grid and fuel-stock facts of the Small and Full instances (no episodes are played)."""

import numpy as np
from shockbench_flow.dynamics.sim import initial_stock
from shockbench_flow.hosting.tasks import task_generator
from shockbench_flow.policies import lp_common as L
from shockbench_flow.policies.naive_parts import cached_plan


for task in ("small", "full"):
    inst, params = task_generator(task)
    i0 = initial_stock(inst)
    K = inst.commodities
    print(f"\n===== {task}: T={inst.T} nodes={len(inst.nodes)} edges={len(inst.edges)} slots={len(inst.stock_slots)}"
          f" grids={len(inst.grids)} fabs={len(inst.fabs)} demands={len(inst.demands)} psi={inst.params.psi}")
    try:
        from shockbench_flow.policies.registry import PolicyContext
        from shockbench_flow.instance.io import to_static  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print("  (no to_static:", type(exc).__name__, ")")
    print("  commodities:", [(k, c.id, c.v, c.disposal_cost) for k, c in enumerate(K)])
    pis = sorted({(d.pi, d.backlog) for d in inst.demands})
    print("  demand pi (price, backlog):", pis[:6], "..." if len(pis) > 6 else "")
    for go, g in enumerate(inst.grids):
        ga = inst.nodes[g].grid
        print(f"  grid {go} node {inst.nodes[g].id}: base_load={ga.base_load:.4g} deliverable={ga.deliverable:.4g}"
              f" voll={ga.voll:.4g} rationed={ga.rationed} priority={ga.priority} fabs={list(inst.grid_fabs[go])}")
        for k in ga.fuels:
            s = inst.slot_index[(g, k)]
            st = inst.stock_slots[s]
            use = ga.shares[k] * ga.deliverable
            print(f"     fuel {K[k].id:10s} share={ga.shares[k]:.3f} use/wk={use:.4g} ibar={ga.ibar.get(k)}"
                  f" I0={i0[s]:.4g} cover_wk={i0[s] / use if use else float('nan'):.2f}"
                  f" ibar_wk={(ga.ibar.get(k) or 0) / use if use else float('nan'):.2f}"
                  f" storage={st.storage:.4g} storage_wk={st.storage / use if use else float('nan'):.2f}"
                  f" holding={st.holding:.4g} salvage={st.salvage:.4g} v={K[k].v:.4g}")
        if None in ga.shares:
            print(f"     unmodelled share={ga.shares[None]:.3f}")
    taus = [e.tau for e in inst.edges]
    print("  edge tau: max", max(taus), "mean", round(float(np.mean(taus)), 2))
    print("  fab tau:", sorted({inst.nodes[f].fab.tau for f in inst.fabs}),
          "fab e:", sorted({round(inst.nodes[f].fab.e, 4) for f in inst.fabs}))
