"""The trim's forecast of a grid's fab wafers against what the next week found (needs the TRACE lines in pulse.py).

uv run python trace.py --task=small --n=7 --grid=grid_kr --b="ration_trim=true"
"""

import sys
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import probe as P  # noqa: E402


def main(task="small", n=7, grid="grid_kr", b="ration_trim=true", folder="mpc_pulse_rec_v6", entropy=12345, first=1, last=None):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    with without_secret_like():
        factory = load_agent_class(str(C.ROOT / "agents" / folder), "submission_trace")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(P.parse(b) | {"strict": True})
        pulse_mod = sys.modules["pulse"]
        orig = pulse_mod.Pulse.__init__

        def init(self, inst_, params):
            orig(self, inst_, params)
            self.trace = []

        pulse_mod.Pulse.__init__ = init
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    trace = list(shim.agent.pulse.trace)
    unload_agent()
    inst = inst.at_digest(marks.instance_digest)
    name = [nd.id for nd in inst.nodes]
    go = next(i for i, g in enumerate(inst.grids) if name[g] == grid)
    print("week fab: wafers on hand, lots allowed, rho, due next week (pipeline+straits), coming (with sent), alpha, R -> forecast wafers next week | actual wafers next week, actual lots this week")
    for t, g2, fo, wafers, lots, rho, soon, coming, alpha, R in trace:
        if g2 != go or t < first or (last and t > last):
            continue
        f = inst.fabs[fo]
        fa = inst.nodes[f].fab
        s = inst.slot_index[(f, fa.input)]
        rec, nxt = traj.records[t - 1], traj.records[t] if t < len(traj.records) else None
        actual = float(nxt.stock[s]) + float(nxt.lots_started[fo]) if nxt is not None else float("nan")
        print(
            f"{t:3d} {name[f]:18s} {wafers:9.0f} {lots:9.0f} {rho:5.2f} {soon:9.0f} {coming:9.0f} {alpha:4.2f} {R:4.2f} -> {wafers - rho * lots + coming:9.0f} | {actual:9.0f} {float(rec.lots_started[fo]):9.0f}"
            f"  (true alpha {float(marks.alpha_bar[t - 1, fo]):.2f}->{float(marks.alpha_bar[min(t, inst.T - 1), fo]):.2f} R {float(marks.R[t - 1, fo]):.2f}->{float(marks.R[min(t, inst.T - 1), fo]):.2f})"
        )


if __name__ == "__main__":
    fire.Fire(main)
