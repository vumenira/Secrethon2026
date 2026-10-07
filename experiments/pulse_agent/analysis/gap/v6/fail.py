"""Provoke the failure paths of the rules added in agents/mpc_pulse_rec_v6, with ``strict`` off.

1. Each new method of pulse.Pulse raises in one week: the episode ends without an error, one pulse failure is counted.
2. A new method raises every week: the play costs what the play with ``pulse`` false costs, to the cent.
3. params.json with a new name of the wrong kind, or an unknown name: the agent refuses to load.

uv run python fail.py --task=small --n=25
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

ON = {"fuel_cancel": True, "ration_trim": True, "wafers_fit": True, "strict": False}


def play(task, n, folder, params, patch=None, entropy=12345):
    from shockbench_flow.dynamics.env import rollout
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    inst, omega, marks, fallback = C.world(task, n, entropy)
    with without_secret_like():
        factory = load_agent_class(str(folder), "submission_fail")
        module = sys.modules[factory.__module__]
        module.PARAMS.update(params)
        if patch is not None:
            patch(sys.modules["pulse"])
        shim = _metered_shim(factory, None)
        traj = rollout(inst, shim, omega, "standard", _policy_seed(entropy, n, NO_ZIP_SHA256), marks=marks, fallback=fallback)
        errors = list(shim.errors)
    agent = shim.agent
    out = (int(traj.J_cents), errors, agent.pulse_failed, agent.planner.failed, sum(1 for r in traj.records if r.invalid))
    unload_agent()
    return out


def raising(name, weeks):
    def patch(mod):
        orig = getattr(mod.Pulse, name)

        def method(self, obs, *args, **kwargs):
            if weeks is None or int(obs["week"]) in weeks:
                raise RuntimeError(f"provoked in {name}")
            return orig(self, obs, *args, **kwargs)

        setattr(mod.Pulse, name, method)

    return patch


def main(task="small", n=25, folder="mpc_pulse_rec_v6"):
    root = C.ROOT / "agents" / folder
    import contextlib
    import io

    quiet = io.StringIO()
    J_on, errors, pf, lf, _inv = play(task, n, root, ON)
    print(f"all three on, strict off: J {J_on} errors {errors} pulse failures {pf} planner fallbacks {lf}")
    J_off, errors, pf, lf, _inv = play(task, n, root, ON | {"pulse": False})
    print(f"pulse false: J {J_off} errors {errors} pulse failures {pf}")
    for name in ("cancel_unburnable", "trim_rationed", "fit_wafers", "grid_week"):
        with contextlib.redirect_stderr(quiet):
            J, errors, pf, lf, _inv = play(task, n, root, ON, raising_grid(name, {10}) if name == "grid_week" else raising(name, {10}))
        print(f"{name} raises in week 10: J {J} errors {errors} pulse failures {pf} planner fallbacks {lf} (traceback lines printed: {quiet.getvalue().count('provoked')})")
        quiet.truncate(0)
        quiet.seek(0)
    with contextlib.redirect_stderr(quiet):
        J, errors, pf, lf, _inv = play(task, n, root, ON, raising("cancel_unburnable", None))
    print(f"cancel_unburnable raises every week: J {J} ({'equal to' if J == J_off else 'DIFFERS from'} the play with pulse false) errors {errors} pulse failures {pf}")
    try:
        play(task, n, root, ON | {"strict": True}, raising("trim_rationed", {10}))
        print("strict on: NO ERROR RAISED")
    except Exception as exc:  # noqa: BLE001
        print(f"strict on, trim_rationed raises in week 10: the play raises {type(exc).__name__}: {str(exc)[:80]}")
    for bad in ({"ration_trim": 1}, {"fuel_cancel": "yes"}, {"ration_trim_cover": -1.0}, {"ration_trim_margin": True}, {"wafers_fit": None}, {"wafer_fit": True}):
        tmp = Path(tempfile.mkdtemp(prefix="v6fail_", dir=HERE))
        copy = tmp / "agent"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
        (copy / "params.json").write_text(json.dumps(bad))
        try:
            play(task, n, copy, {})
            print(f"params.json {bad}: LOADED (not refused)")
        except Exception as exc:  # noqa: BLE001
            print(f"params.json {bad}: refused, {type(exc).__name__}: {str(exc)[:110]}")
        shutil.rmtree(tmp, ignore_errors=True)


def raising_grid(name, weeks):
    def patch(mod):
        orig = mod.Pulse.grid_week
        box = {"week": 0}
        orig_adjust = mod.Pulse.adjust

        def adjust(self, obs, *args, **kwargs):
            box["week"] = int(obs["week"])
            return orig_adjust(self, obs, *args, **kwargs)

        def method(self, *args, **kwargs):
            if box["week"] in weeks:
                raise RuntimeError(f"provoked in {name}")
            return orig(self, *args, **kwargs)

        mod.Pulse.adjust = adjust
        mod.Pulse.grid_week = method

    return patch


if __name__ == "__main__":
    fire.Fire(main)
