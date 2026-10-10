"""Start a command under the machine's evaluation lock as soon as the lock is free.

uv run python outputs/s13/soon.py <command and its arguments>

It runs outputs/recovery_tune/tools/with_lock.py with the command, unchanged: the lock is taken by that tool and
no play runs outside it. The one difference is when the tool is started: this script watches the lock twice a
second and starts the tool the moment the lock is gone, where the tool by itself looks every 15 seconds and loses
the lock every time to a session that queues its plays back to back (seen on 2026-10-10: six plays of other
sessions in a row over 50 minutes with this session's command waiting from the first).
"""
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "recovery_tune" / "tools" / "with_lock.py"
LOCK = TOOL.parents[1] / ".play_lock"


def main():
    waited = 0.0
    while LOCK.exists():
        if waited % 300 < 0.5:
            print(f"soon: the lock is held ({waited:.0f} s so far)", flush=True)
        time.sleep(0.5)
        waited += 0.5
    return subprocess.call([sys.executable, str(TOOL), *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
