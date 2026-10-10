"""The kit's once-per-process Reset work on this machine (0.28 s of CPU on the scoring host, per hosting/metering.py)."""
import sys, time
c0 = time.process_time()
import numpy as np
from shockbench_flow_agent import shim
from shockbench_flow_agent.shim import DATA_DIR, SCHEMA_FILE, load_instance
print(f"imports: {time.process_time() - c0:.3f} s")
paths = [p for p in sorted(DATA_DIR.glob("*.json")) if p.name != SCHEMA_FILE]
order = sys.argv[1:] or [p.name for p in paths]
mods0 = set(sys.modules)
for name in order + order:
    p = [q for q in paths if q.name == name][0]
    c = time.process_time(); inst = load_instance(p); dt = time.process_time() - c
    print(f"load_instance({p.name}): {dt:.3f} s cpu; new modules so far {len(set(sys.modules) - mods0)}")
