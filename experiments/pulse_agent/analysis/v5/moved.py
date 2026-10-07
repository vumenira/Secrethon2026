import sys, json
sys.path.insert(0, r"C:\Users\mykha\AppData\Local\Temp\claude\C--Users-mykha-Claude-code-projects-Secrethon2026\d989db8b-5de3-4e20-b2c1-6278ade0c74d\scratchpad\v5")
import lab, numpy as np
for task, names in (("small", ["ft_s_v5_hc_free_p1", "ft_s_v5_hc_free_p2", "ft_s_v5_hc_pin_p1"]), ("full", ["ft_v5_hc_free_p1", "ft_v5_hc_free_p2", "ft_v5_hc_pin_p1"])):
    inst = lab.world(task, 0)[0]
    grids = lab.fab_grids(inst)
    n = 64 if task == "small" else 16
    total = n * inst.T * len(grids)
    print(task, "fab grids", len(grids), [inst.nodes[inst.grids[g]].id for g in grids], "fabs per grid", [len(inst.grid_fabs[g]) for g in grids], "grid-weeks", total)
    run = lab.OUT / f"{task}-e12345-n{n}"
    for name in names:
        rows = json.loads((run / f"{name}.json").read_text())["rows"]
        sw = sum(r["weeks_switched"] for r in rows)
        w0 = sum(r["weeks_whole"][0] for r in rows); w1 = sum(r["weeks_whole"][1] for r in rows)
        print(f"  {name}: switched {sw} ({sw / total:.3f} of grid-weeks), whole {w0} -> {w1} ({w0 / total:.3f} -> {w1 / total:.3f}), moved {np.mean([r['calendar_moved'] for r in rows]):.4f}, episodes with more than 5 % of grid-weeks switched: {sum(r['weeks_switched'] > 0.05 * inst.T * len(grids) for r in rows)} of {n}")
