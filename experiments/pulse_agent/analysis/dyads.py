"""Per network: the edges that tell a CN-TW conflict's closure of the Taiwan Strait (no episode is played)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\Users\mykha\Claude code projects\Secrethon2026") / "agents" / "rec_merged"))
import recovery  # noqa: E402
from sbflow.disruption.params import FlagshipMarkLaws, MarkLaws  # noqa: E402
from sbflow.instance.schema import FLAGSHIP_KINDS  # noqa: E402
from shockbench_flow.hosting.tasks import task_generator  # noqa: E402

for task in ("tiny", "small", "full"):
    inst, _params = task_generator(task)
    laws = FlagshipMarkLaws() if inst.kind in FLAGSHIP_KINDS else MarkLaws()
    dyads = recovery.dyad_edges(inst, laws)
    capped = sum(1 for e in inst.edges if e.u0 is not None and e.u0 > 0.0)
    print(f"{task}: kind {inst.kind}, T {inst.T}, edges {len(inst.edges)} ({capped} with a capacity), straits"
          f" {len(inst.chokepoints)}, grids {len(inst.grids)}; strait closures {getattr(laws, 'strait_closures', ())};"
          f" edges per dyad strait {{ordinal: count}} {{{', '.join(f'{k}: {len(v)}' for k, v in dyads.items())}}}")
