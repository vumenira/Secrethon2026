"""Copy the installed shockbench-flow's planner (``mpc_det``) into an agent folder as a local package, ``sbflow``.

uv run python scripts/vendor_planner.py                # into agents/mpc_det_h20/sbflow
uv run python scripts/vendor_planner.py --agent=mine

The server has numpy and SciPy, not shockbench-flow's policies nor the packages they import. This copies every module
``policies.mpc_det`` imports, under another name, and changes only what the server lacks (``PATCHES``): the solver
becomes the HiGHS that SciPy bundles (the same version as the highspy package, so the plans are the same), the
instance file's schema check is dropped, and so are the logging and the parallel loops. A module the planner reaches
only inside a function it never calls is left as a stub that refuses to import. Run it again after a
``uv sync --upgrade-package shockbench-flow``; a patch that no longer matches stops it.
"""

import ast
import shutil
import sys
from importlib import metadata
from pathlib import Path

import fire
import shockbench_flow
from shockbench_flow_agent.submission import missing_imports

from sbf_starter import ROOT


SOURCE, NAME = "shockbench_flow", "sbflow"
ENTRY = ("shockbench_flow.policies.mpc_det",)
PACKAGES = Path(shockbench_flow.__file__).resolve().parent.parent

HIGHS = '''"""HiGHS as SciPy bundles it, under the names the planner uses from highspy (scripts/vendor_planner.py)."""

from scipy.optimize._highspy import _core


globals().update({name: getattr(_core, name) for name in dir(_core) if not name.startswith("_")})
Highs = _core._Highs
'''
STUB = '''"""Not copied: the planner never imports {module} while it plays (scripts/vendor_planner.py)."""

raise ImportError("{module} is not part of this submission")
'''
# module -> (old, new, times it must occur): exact text of the installed release
PATCHES = {
    "shockbench_flow.instance.io": (
        ("import fastjsonschema\n\n", "", 1),
        (
            "    return fastjsonschema.compile(json.loads(_schema_text()), use_default=False, use_formats=False)\n",
            "    return lambda raw: raw  # the scorer's own instance: no schema check (the server has no validator)\n",
            1,
        ),
        ("fastjsonschema.JsonSchemaValueException", "Exception", 2),
    ),
    "shockbench_flow.policies.lp_common": (
        (
            "    import highspy\n\n    return highspy\n",
            "    from shockbench_flow import _highs\n\n    return _highs\n",
            1,
        ),
        (
            "    import highspy\n\n    from shockbench_flow.policies.registry import PolicyContext\n",
            "    from shockbench_flow import _highs as highspy\n"
            "    from shockbench_flow.policies.registry import PolicyContext\n",
            1,
        ),
    ),
    "shockbench_flow.parallel": (
        (
            "    from joblib import Parallel, delayed\n\n"
            "    return list(Parallel(n_jobs=n_jobs, batch_size=BATCH_SIZE)(delayed(fn)(x) for x in xs))\n",
            '    raise RuntimeError("no joblib on the server: n_jobs must be 1")\n',
            1,
        ),
    ),
}


def source_file(module: str) -> Path | None:
    path = PACKAGES / Path(*module.split("."))
    if (path / "__init__.py").is_file():
        return path / "__init__.py"
    return path.with_suffix(".py") if path.with_suffix(".py").is_file() else None


def imported(path: Path) -> list[tuple[str, bool]]:
    """(name, inside a function) of every import of a file; ``from a import b`` gives a and a.b."""
    found = []

    def visit(node, lazy):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Import):
                found.extend((alias.name, lazy) for alias in child.names)
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module, lazy))
                found.extend((f"{child.module}.{alias.name}", lazy) for alias in child.names)
            visit(child, lazy or isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)))

    visit(ast.parse(path.read_text(encoding="utf-8")), False)
    return found


def closure() -> dict[str, bool]:
    """Every module of the package the entry modules reach -> whether only from inside functions."""
    lazy_only: dict[str, bool] = {}
    queue = [(module, False) for module in ENTRY]
    while queue:
        module, lazy = queue.pop()
        if source_file(module) is None or (module in lazy_only and (lazy or not lazy_only[module])):
            continue
        lazy_only[module] = lazy
        parts = module.split(".")
        queue.extend((".".join(parts[:i]), lazy) for i in range(1, len(parts)))  # its packages' __init__
        for name, inner in imported(source_file(module)):
            if name.split(".")[0] == SOURCE:
                queue.append((name, lazy or inner))
    return lazy_only


def patched(module: str, text: str) -> str:
    for old, new, times in PATCHES.get(module, ()):
        if text.count(old) != times:
            raise SystemExit(f"{module}: expected {times} of {old!r}, found {text.count(old)}: update PATCHES")
        text = text.replace(old, new)
    return text


def main(agent: str = "mpc_det_h20") -> None:
    """Write agents/<agent>/sbflow from the installed shockbench-flow."""
    version = metadata.version("shockbench-flow")
    out = ROOT / "agents" / agent / NAME
    if out.exists():
        shutil.rmtree(out)
    modules = closure()
    written = {}
    for module, lazy in sorted(modules.items()):
        src = source_file(module)
        rel = Path(*module.split(".")[1:], "__init__.py") if src.name == "__init__.py" else Path(*module.split(".")[1:])
        rel = rel if rel.suffix == ".py" else rel.with_suffix(".py")
        if module == SOURCE:
            text = (
                f'"""shockbench-flow {version}\'s planner, copied by scripts/vendor_planner.py (MIT, see LICENSE)."""\n'
            )
            text += f'\n__version__ = "{version}"\n'
        elif lazy:
            text = STUB.format(module=module)
        else:
            text = patched(module, src.read_text(encoding="utf-8"))
        written[rel] = text.replace(SOURCE, NAME)
    written[Path("_highs.py")] = HIGHS
    for rel, text in written.items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        (out / rel).write_text(text, encoding="utf-8", newline="\n")
    for file in metadata.files("shockbench-flow"):
        if "licenses" in file.parts:
            shutil.copyfile(file.locate(), out / file.name)

    names = [f"{NAME}/{rel.as_posix()}" for rel in written]
    missing = {n: missing_imports((out.parent / n).read_bytes(), names) for n in names}
    missing = {n: m for n, m in missing.items() if m}
    copied = sum(not lazy for lazy in modules.values())
    size = sum(len(text) for text in written.values()) / 1024
    print(f"{out.relative_to(ROOT)}: shockbench-flow {version}, {copied} modules copied, {len(modules) - copied} stubs")
    print(f"{len(written)} files, {size:.0f} KB")
    if missing:
        for name, mods in missing.items():
            print(f"  {name} still imports {', '.join(mods)}", file=sys.stderr)
        raise SystemExit("imports the server lacks remain: add a patch")


if __name__ == "__main__":
    fire.Fire(main)
