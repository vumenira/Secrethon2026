"""Native Windows support: stand-ins for two Unix-only modules the package imports but local scoring never uses.

``shockbench_flow.information.runner`` imports ``fcntl`` and ``termios`` (the scoring container's pipes); local
evaluation does not call them. On Windows this writes two empty stand-ins into a folder of this repository and puts
it on ``sys.path`` (no change to the virtual environment). Import this module before anything from shockbench_flow.
On Linux and macOS it does nothing.
"""

import sys
from pathlib import Path


STUBS = Path(__file__).resolve().parent / "_winstubs"


def install() -> None:
    if sys.platform != "win32":
        return
    STUBS.mkdir(exist_ok=True)
    for name in ("fcntl", "termios"):
        stub = STUBS / f"{name}.py"
        if not stub.exists():
            stub.write_text(f'"""Stand-in for the Unix-only module {name} (see winsupport.py)."""\n')
    if str(STUBS) not in sys.path:
        sys.path.append(str(STUBS))


install()
