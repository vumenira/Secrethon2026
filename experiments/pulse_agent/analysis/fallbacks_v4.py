"""``fallbacks5.py`` pointed at agents/mpc_pulse_rec_v4: each fallback provoked once in one episode of Small root 12345
(strict off, strict on, the pulse not buildable), with rule 2 (``premium_limit``) off, as the folder has it, and on;
then the checks of the five new entries of ``PARAMS`` when the agent is loaded.

uv run python fallbacks_v4.py
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
import fallbacks5 as fb  # noqa: E402


fb.AGENT = "mpc_pulse_rec_v4"
fb.WORK = Path(__file__).parent / "smoke_agents" / "mpc_pulse_rec_v4_fb"

LOADS = (
    {"plan_duty": True},
    {"premium_limit": True, "cover_weeks": 6},
    {"pulse_small": False, "premium_small": False},
    {"premium_limit": 1},
    {"pulse_small": "yes"},
    {"premium_small": None},
    {"cover_weeks": "4"},
    {"cover_weeks": -1.0},
    {"premium_limits": True},
)


if __name__ == "__main__":
    for limit in (False, True):
        print(f"--- premium_limit {limit}", flush=True)
        fb.play({"strict": False, "premium_limit": limit})
        fb.play({"strict": True, "premium_limit": limit})
        fb.play({"strict": False, "premium_limit": limit}, no_build=True)
    print("--- the checks at load", flush=True)
    for params in LOADS:
        try:
            module = fb.load(params)
            print(f"{params} -> loaded: {({k: module.PARAMS[k] for k in params})}")
        except ValueError as error:
            print(f"{params} -> ValueError: {str(error)[:120]}")
