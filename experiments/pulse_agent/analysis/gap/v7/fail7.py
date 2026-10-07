"""Provoke the failure paths of the rules added in agents/mpc_pulse_rec_v7, with ``strict`` off.

1. Each new method of pulse.Pulse raises in one week: the episode ends without an error, one pulse failure is counted.
2. A new method raises every week: the play costs what the play with ``pulse`` false costs, to the cent.
3. With ``strict`` on the same failure is raised out of the play.
4. params.json with a new name of the wrong kind, or an unknown name: the agent refuses to load.

uv run python fail7.py --task=small --n=25
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import fire

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "v6"))
import common as C  # noqa: E402
import fail as F  # noqa: E402

ON = {"ration_trim_release": True, "fuel_trim": True, "osat_fit": True, "strict": False}
BAD = (
    {"ration_trim_release": 1}, {"fuel_trim": "yes"}, {"fuel_trim_cover": -1.0}, {"fuel_trim_margin": True},
    {"osat_fit": None}, {"fuel_trims": True},
)


def main(task="small", n=25, folder="mpc_pulse_rec_v7"):
    root = Path(folder) if Path(folder).is_absolute() else C.ROOT / "agents" / folder
    quiet = io.StringIO()
    J_on, errors, pf, lf, _inv = F.play(task, n, root, ON)
    print(f"all three on, strict off: J {J_on} errors {errors} pulse failures {pf} planner fallbacks {lf}")
    J_off, errors, pf, lf, _inv = F.play(task, n, root, ON | {"pulse": False})
    print(f"pulse false: J {J_off} errors {errors} pulse failures {pf}")
    for name in ("passing", "trim_fuels", "fit_osats"):
        with contextlib.redirect_stderr(quiet):
            J, errors, pf, lf, _inv = F.play(task, n, root, ON, F.raising(name, {10}))
        print(f"{name} raises in week 10: J {J} errors {errors} pulse failures {pf} planner fallbacks {lf} (traceback lines printed: {quiet.getvalue().count('provoked')})")
        quiet.truncate(0)
        quiet.seek(0)
    for name in ("passing", "trim_fuels", "fit_osats"):
        with contextlib.redirect_stderr(quiet):
            J, errors, pf, lf, _inv = F.play(task, n, root, ON, F.raising(name, None))
        print(f"{name} raises every week: J {J} ({'equal to' if J == J_off else 'DIFFERS from'} the play with pulse false) errors {errors} pulse failures {pf}")
        quiet.truncate(0)
        quiet.seek(0)
    for name in ("passing", "trim_fuels", "fit_osats"):
        try:
            with contextlib.redirect_stderr(quiet):
                _J, errors, pf, _lf, _inv = F.play(task, n, root, ON | {"strict": True}, F.raising(name, {10}))
            print(f"strict on, {name} raises in week 10: the week is an error of the play: {errors[:1]} (pulse failures {pf})" if errors else f"strict on, {name}: NO ERROR")
        except Exception as exc:  # noqa: BLE001
            print(f"strict on, {name} raises in week 10: the play raises {type(exc).__name__}: {str(exc)[:80]}")
    for bad in BAD:
        tmp = Path(tempfile.mkdtemp(prefix="v7fail_", dir=HERE))
        copy = tmp / "agent"
        shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
        (copy / "params.json").write_text(json.dumps(bad))
        try:
            F.play(task, n, copy, {})
            print(f"params.json {bad}: LOADED (not refused)")
        except Exception as exc:  # noqa: BLE001
            print(f"params.json {bad}: refused, {type(exc).__name__}: {str(exc)[:110]}")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    fire.Fire(main)
