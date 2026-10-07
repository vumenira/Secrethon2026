"""Is the code of one agent folder the code of another (comments apart), and which files differ from a third.

uv run python same_code.py --a=agents/mpc_pulse_rec_v7 --b=outputs/recovery_tune/agents/t_v7 --base=agents/mpc_pulse_rec_v6
"""

import ast
import hashlib
from pathlib import Path

import fire

ROOT = Path(r"C:\Users\mykha\Claude code projects\Secrethon2026")


def files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "params.json"}


def main(a="agents/mpc_pulse_rec_v7", b="outputs/recovery_tune/agents/t_v7", base="agents/mpc_pulse_rec_v6"):
    fa, fb, f0 = files(ROOT / a), files(ROOT / b), files(ROOT / base)
    print(f"{a}: {len(fa)} files; {b}: {len(fb)}; {base}: {len(f0)}; names equal: {set(fa) == set(fb) == set(f0)}")
    for name in sorted(fa):
        same_bytes = name in fb and fa[name].read_bytes() == fb[name].read_bytes()
        if same_bytes:
            continue
        if name.endswith(".py") and name in fb:
            ta, tb = ast.parse(fa[name].read_text(encoding="utf-8")), ast.parse(fb[name].read_text(encoding="utf-8"))
            print(f"  {name}: bytes differ between {a} and {b}; the syntax trees (code and docstrings, no comments) are {'equal' if ast.dump(ta) == ast.dump(tb) else 'DIFFERENT'}")
        else:
            print(f"  {name}: differs or is missing in {b}")
    changed = [name for name in sorted(fa) if name not in f0 or hashlib.sha256(fa[name].read_bytes()).digest() != hashlib.sha256(f0[name].read_bytes()).digest()]
    print(f"files of {a} that differ from {base}: {changed}")


if __name__ == "__main__":
    fire.Fire(main)
