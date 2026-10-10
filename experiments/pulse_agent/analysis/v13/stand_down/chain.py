"""Run the commands of a text file one after another (one per line; a line starting with # is skipped).

uv run python outputs/recovery_tune/tools/with_lock.py uv run python outputs/s13/chain.py outputs/s13/<file>.txt

One lock for several plays: the machine's lock is not a queue, so every command queued by itself waits its own
turn among the other sessions. A command that fails does not stop the ones after it.
"""
import shlex
import subprocess
import sys
import time
from pathlib import Path


def main(path):
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        start = time.time()
        print(f"chain: {line}", flush=True)
        code = subprocess.call(shlex.split(line))
        print(f"chain: exit {code} after {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
