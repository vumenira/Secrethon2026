"""What the pulse of agents/ft_buffer_v2 derives from an instance: which fuels it may hold at which grid, sale lead times."""

import json
import sys
from pathlib import Path

import fire


ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")


def main(task="full", agent="ft_buffer_v2"):
    from shockbench_flow.hosting.tasks import task_generator

    sys.path.insert(0, str(ROOT / "agents" / agent))
    import pulse  # noqa: E402

    inst, _ = task_generator(task)
    N, K = inst.nodes, inst.commodities
    params = {"pulse_hold_weeks": 6.0, "pulse_fill": 0.9, "pulse_margin": 1.0, "pulse_run": 1.0, "pulse_partial": 0.5, "pulse_memory": 4.0, "pulse_payback": 0.0}
    params |= json.loads((ROOT / "agents" / agent / "params.json").read_text())
    p = pulse.Pulse(inst, params)
    print(f"{task} (T {inst.T}), params of {agent}: payback {params['pulse_payback']}, tail {params.get('pulse_tail')}")
    psi = inst.params.psi
    for go, g, fabs, fuels, shares in p.grids:
        ga = N[g].grid
        need = sum(N[inst.fabs[fo]].fab.e * N[inst.fabs[fo]].fab.cap0 for fo in fabs)
        room = ga.deliverable * shares - ga.base_load
        line = f"  {N[g].id:9s} fabs need {need:5.0f} GWh/wk, room above base load {room:5.0f} ({room / ga.deliverable:.1%} of output);"
        for k, (share, threshold, feeders, worth_it) in fuels.items():
            burn = share * ga.deliverable
            how = "no terminal" if not feeders else ("held" if worth_it else "not held (payback)")
            line += f" {K[k].id}: burn {burn:.0f}, threshold {threshold / burn if burn else 0:.2f} wk, {how};"
        print(line)
    skipped = [N[g].id for go, g in enumerate(inst.grids) if go not in {x[0] for x in p.grids}]
    print("  grids the pulse leaves alone (no fab, or not base-first):", skipped)
    if p.lead:
        print("  weeks from a lot's start to its chip's earliest sale:", {N[inst.fabs[fo]].id: v for fo, v in p.lead.items()})
        print("  last week a lot can still be sold, by fab lead:", sorted({inst.T - v for v in p.lead.values()}))


if __name__ == "__main__":
    fire.Fire(main)
