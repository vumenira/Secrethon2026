"""Play one episode under two settings of PARAMS and print, per grid, the weeks whose shed or fab energy differ.

uv run python diff.py --task=small --n=7 --b="ration_trim=true"
"""

import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402
import probe as P  # noqa: E402


def play(task, n, folder, params, entropy):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    with without_secret_like():
        factory = load_agent_class(str(folder), "submission_diff")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params)
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
    log = list(shim.agent.pulse.log) if shim.agent.pulse is not None else []
    unload_agent()
    return inst.at_digest(marks.instance_digest), marks, traj, log


def main(task="small", n=0, a="", b="ration_trim=true", folder="mpc_pulse_rec_v6", entropy=12345, tol=1.0, grid=None, every=False):
    root = C.ROOT / "agents" / folder
    pa, pb = P.parse(a) | {"strict": True}, P.parse(b) | {"strict": True}
    inst, marks, ta, la = play(task, n, root, pa, entropy)
    _i, _m, tb, lb = play(task, n, root, pb, entropy)
    name = [nd.id for nd in inst.nodes]
    K = [c.id for c in inst.commodities]
    print(f"{task} episode {n}: A {pa} J {ta.J_cents / 1e11:,.2f} bn; B {pb} J {tb.J_cents / 1e11:,.2f} bn; B saves {(ta.J_cents - tb.J_cents) / 1e11:,.2f} bn USD")
    ev = defaultdict(list)
    for w, go, what, moved in lb:
        ev[(w, go)].append(what + ":" + ",".join(f"{K[k] if isinstance(k, int) else k}={q:.0f}" for k, q in moved.items()))
    eva = defaultdict(list)
    for w, go, what, moved in la:
        eva[(w, go)].append(what)
    for go, g in enumerate(inst.grids):
        if grid is not None and name[g] != grid:
            continue
        ga = inst.nodes[g].grid
        fabs = inst.grid_fabs[go]
        r = ga.rationed
        s = inst.slot_index[(g, r)] if r is not None else None
        sa = np.array([float(x.shed[go]) for x in ta.records])
        sb = np.array([float(x.shed[go]) for x in tb.records])
        ea = np.array([sum(float(x.energy[fo]) for fo in fabs) for x in ta.records])
        eb = np.array([sum(float(x.energy[fo]) for fo in fabs) for x in tb.records])
        print(f"{name[g]}: shed A {sa.sum():,.0f} B {sb.sum():,.0f} ({sb.sum() - sa.sum():+,.0f}); fab energy A {ea.sum():,.0f} B {eb.sum():,.0f} ({eb.sum() - ea.sum():+,.0f})")
        for ti in range(len(ta.records)):
            if not every and abs(sa[ti] - sb[ti]) <= tol and abs(ea[ti] - eb[ti]) <= tol:
                continue
            ra, rb = ta.records[ti], tb.records[ti]
            gas = f" gas burn {ra.segment.get((go, r), 0.0):6.0f}|{rb.segment.get((go, r), 0.0):<6.0f} grid stock {float(ra.stock[s]):6.0f}|{float(rb.stock[s]):<6.0f}" if r is not None else ""
            Gb = float(marks.G_bar[ti, go])
            print(
                f"  wk {ti + 1:3d} G {Gb:6.0f} shed {sa[ti]:6.0f}|{sb[ti]:<6.0f} fabE {ea[ti]:5.0f}|{eb[ti]:<5.0f}{gas}"
                f"  A:{','.join(eva.get((ti + 1, go), []))}  B:{' '.join(ev.get((ti + 1, go), []))}"
            )


if __name__ == "__main__":
    fire.Fire(main)
