import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
for item in ("full:12345:16", "full:777:32", "full:2026:16", "small:12345:64"):
    task, root, N = item.split(":")
    folder = ROOT / "outputs" / "recovery_tune" / f"{task}-e{root}-n{N}"
    line = []
    for name in ("v12_all", "a13_off", "a13_s8", "a13_s26", "a13_s8_src"):
        p = folder / f"{name}.json"
        if p.exists():
            secs = [r["seconds"] for r in json.loads(p.read_text())["rows"]]
            line.append(f"{name} mean {sum(secs) / len(secs):.1f} max {max(secs):.1f}")
    print(item, "; ".join(line))
