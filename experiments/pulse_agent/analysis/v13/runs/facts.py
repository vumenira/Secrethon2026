"""Instance facts for the pulse: grids with fabs, their fuels, thresholds, terminals, fabs. No episode is played."""
import json, math, sys
from pathlib import Path

ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")
AG = ROOT / "agents" / "mpc_pulse_rec_v13r"
sys.path.insert(0, str(AG))
import agent, pulse  # noqa: E402
from sbflow.policies.naive_parts import cached_plan  # noqa: E402
from sbflow.policies.registry import PolicyContext  # noqa: E402

for task in ("full", "small"):
    raw = json.loads((AG / "instances" / f"{task}.json").read_text(encoding="utf-8"))
    inst, plan = cached_plan({"instance": raw}, PolicyContext().fq_quantile)
    P = pulse.Pulse(inst, agent.PARAMS)
    N, K, E = inst.nodes, inst.commodities, inst.edges
    print(f"===== {task}: T {inst.T}, psi {inst.params.psi}")
    for go, g, fabs, fuels, shares in P.grids:
        ga = N[g].grid
        draw = P.full_draw[go]
        print(f"grid {go} {N[g].id}: deliverable {ga.deliverable:.0f} shares {shares:.3f} voll {ga.voll:.0f} draw {draw:.0f} ({draw/ga.deliverable:.3f}) gain/wk {P.gain[go]/1e9:.2f} bn last {P.last[go]} small {[K[k].id for k in P.small[go]]}")
        for k, (share, thr, feeders, worth) in fuels.items():
            burn = share * ga.deliverable
            fs = [(N[tail].id, round(st / burn, 2), round(E[e].u0 / burn, 2)) for e, tail, st, ss in feeders]
            room = sum(st for _e, _t, st, _s in feeders)
            n = 1 + int(max(P.fill * room - thr, 0.0) // burn) if burn > 0 else 0
            gs = inst.stock_slots[inst.slot_index[(g, k)]].storage
            pipes = [(N[src].id, round(inst.stock_slots[slot].supply / burn, 2), round(inst.stock_slots[slot].storage / burn, 2), round(E[e].u0 / burn, 2)) for e, src, slot, ss in P.pipes.get((go, k), [])]
            print(f"   {K[k].id:8s} share {share:.3f} burn {burn:8.0f} thr/burn {thr/burn if burn else 0:.2f} gridstore/burn {gs/burn:.2f} rationed {k == ga.rationed} worth {worth} lean {P.lean_worth.get((go,k))} pipew {P.pipe_worth.get((go,k))} feeders(node,storage/burn,u0/burn) {fs} runs_n {n if thr > burn else '-'} pipes(src,supply,storage,u0) {pipes}")
        for fo in fabs:
            f = inst.fabs[fo]
            fa = N[f].fab
            ws = inst.stock_slots[inst.slot_index[(f, fa.input)]].storage
            cs = inst.stock_slots[inst.slot_index[(f, fa.product)]].storage
            print(f"   fab {fo} {N[f].id}: cap0 {fa.cap0:.0f} e {fa.e:.4f} draw {fa.cap0*fa.e:.0f} input {K[fa.input].id} product {K[fa.product].id} tau {fa.tau} wafer store {ws/fa.cap0:.1f} wk chip store {cs/fa.cap0:.1f} wk lead {P.lead.get(fo)} ways {[(N[t].id, round(c,1)) for c, e, t, ss in P.wafer_edges[fo]]}")
    print("fabs by product:")
    byp = {}
    for fo, f in enumerate(inst.fabs):
        fa = N[f].fab
        byp.setdefault(K[fa.product].id, []).append((fo, N[f].id, fa.cap0, N[fa.grid].id if fa.grid is not None else None))
    for p, v in byp.items():
        print("  ", p, v)
    print("demands:")
    for do, d in enumerate(inst.demands):
        print(f"   {do} {N[d.node].id} {K[d.k].id} dbar {d.dbar:.0f} pi {d.pi:.0f} backlog {d.backlog}")
