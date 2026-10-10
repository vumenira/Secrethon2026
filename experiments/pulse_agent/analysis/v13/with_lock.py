"""Run a command while holding one of the machine's evaluation slots, so that plays of several sessions fit in memory.

uv run python outputs/recovery_tune/tools/with_lock.py <command and its arguments>

The laptop has 16 GB. A play with 2 workers takes about 1.3 GB, so a few can run side by side but not any number:
there are SLOTS slots, and a command also waits until MIN_FREE_GB of memory is free before it starts. Every command
that plays episodes (tune.py, sbf view, sbf evaluate, a script of your own) goes through this file, with --n_jobs=2.
A slot is a folder outputs/recovery_tune/.play_lock*; it is removed when the command ends, however it ends.
"""
import ctypes, os, subprocess, sys, time
from pathlib import Path

HOME = Path(__file__).resolve().parents[1]
SLOTS = 4
MIN_FREE_GB = 2.5
LOCKS = [HOME / ".play_lock"] + [HOME / f".play_lock{i}" for i in range(2, SLOTS + 1)]


def free_gb():
    if sys.platform != "win32":
        return float("inf")

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "free", "page", "page_free", "virtual", "virtual_free", "extended")
        ]

    status = Status()
    status.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.free / 2**30


def acquire(note=""):
    """Take a free slot (waiting for one), then wait for memory; the slot's folder."""
    waited = 0
    spare = int(os.environ.get("WITH_LOCK_SPARE", "0"))  # a low-priority command: start only with this many slots left free
    while True:
        if spare and sum(not lock.exists() for lock in LOCKS) <= spare:
            time.sleep(20)
            waited += 20
            continue
        for lock in LOCKS:
            try:
                os.mkdir(lock)
            except FileExistsError:
                continue
            (lock / "owner.txt").write_text(f"pid {os.getpid()} since {time.ctime()}: {note[:200]}\n")
            while free_gb() < MIN_FREE_GB:
                if waited % 300 == 0:
                    print(f"with_lock: slot taken, waiting for memory ({free_gb():.1f} GB free, {MIN_FREE_GB} wanted)", flush=True)
                time.sleep(15)
                waited += 15
            return lock
        if waited % 300 == 0:
            print(f"with_lock: all {SLOTS} evaluation slots are taken, waiting ({waited} s so far)", flush=True)
        time.sleep(10)
        waited += 10


def release(lock):
    try:
        (lock / "owner.txt").unlink()
    except OSError:
        pass
    try:
        os.rmdir(lock)
    except OSError:
        pass


def main():
    lock = acquire(" ".join(sys.argv[1:]))
    try:
        return subprocess.call(sys.argv[1:])
    finally:
        release(lock)


if __name__ == "__main__":
    sys.exit(main())
