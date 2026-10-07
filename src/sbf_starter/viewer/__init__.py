"""``sbf view``: an interactive page of an agent's episodes (the map, its shipments week by week, costs, events).

The command evaluates the agent on the dev split (as ``sbf evaluate``), records a few telling episodes (its worst and
best against the naive rule, and one where a strait closes) and writes one self-contained ``viewer.html``: data, CSS
and JavaScript inline, no server, no network. The page's Scores tab shows the evaluation by harm level. Everything
lands in ``outputs/view/<agent>_<task>/`` and is reused while the agent's files do not change, so a second run only
rewrites the page.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from sbf_starter import DEFAULT_TASK, ROOT, check_task, env_id


PAGE = Path(__file__).with_name("page.html")
OUT_DIR = ROOT / "outputs" / "view"
CACHE_VERSION = 1  # bump when the records or the evaluation saved here change meaning


def _say(*parts) -> None:
    print(*parts, flush=True)


def agent_hash(path: Path) -> str:
    """SHA-256 of an agent's files (a folder's files by relative path, or one file), caches skipped."""
    h = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for p in files:
        if "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        h.update(str(p.relative_to(path) if path.is_dir() else p.name).encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def _picks(rows: list[dict], env, seed: int) -> list[tuple[int, str]]:
    """(episode, why) of the agent's worst and best episode against naive and the first in which a strait closes."""
    from sbf_starter.play import closes

    def skill(r):  # the share of the attainable saving the agent achieved in this episode
        d = r["J_naive_cents"] - r["J_clairvoyant_cents"]
        return (r["J_naive_cents"] - r["J_policy_cents"]) / d if d > 0 else 0.0

    ranked = sorted(rows, key=skill)
    picks = [(ranked[0]["episode"], "worst vs naive"), (ranked[-1]["episode"], "best vs naive")]
    for r in rows:
        if r["episode"] not in {p for p, _ in picks} and closes(env, r["episode"], 0.5, seed):
            picks.append((r["episode"], "a strait closes"))
            break
    seen, out = set(), []
    for n, why in picks:
        if n not in seen:
            seen.add(n)
            out.append((int(n), why))
    return sorted(out)


def _page(data: dict) -> str:
    text = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return PAGE.read_text(encoding="utf-8").replace("/*__DATA__*/null", text, 1)


def view(
    agent: str,
    task: str = DEFAULT_TASK,
    episode: int | list[int] | None = None,
    episodes: str | int | list[int] = "dev",
    entropy: int = 0,
    quick: bool = False,
    n_jobs: int = -1,
    seed: int = 0,
    fresh: bool = False,
    out: str | None = None,
) -> Path:
    """Write the viewer page of an agent's episodes and return its path (module docstring).

    Args:
        agent: an agent's name (a folder of agents/), a submission folder or zip, or an agent.py.
        task: tiny, small (the public board's) or full (the private board's).
        episode: one episode (or a list) to record without evaluating; default: evaluate, then pick 3.
        episodes: the evaluation's episodes: dev (20, 5 per harm level), a count k or a list.
        entropy: 0 for the public dev episodes; any other integer for scenarios of your own.
        quick: a rough naive rule and 4 episodes, in seconds (not the board's numbers).
        n_jobs: workers of a first run's reference computation (-1: all cores).
        seed: the reset's seed of the recorded episodes.
        fresh: ignore what an earlier run saved and play everything again.
        out: the folder to write to (default: outputs/view/<agent>_<task>[_e<entropy>][_quick]).

    """
    import gymnasium as gym
    from shockbench_flow_agent import NAIVE_REPLICATIONS, QUICK
    from shockbench_flow_gym import dashboard

    from sbf_starter.agents import load, resolve
    from sbf_starter.scoring import as_dict, evaluate
    from sbf_starter.viewer import export

    check_task(task)
    path = resolve(agent)
    if path.is_dir():
        name = path.name
    else:  # agents/mine/agent.py is "mine", a zip its stem
        name = path.parent.name if path.name == "agent.py" else path.stem
    suffix = (f"_e{entropy}" if entropy else "") + ("_quick" if quick else "")
    folder = Path(out) if out else OUT_DIR / f"{name}_{task}{suffix}"
    shown_folder = folder.relative_to(ROOT) if folder.is_relative_to(ROOT) else folder
    key = {
        "version": CACHE_VERSION,
        "agent": agent_hash(path),
        "task": task,
        "entropy": entropy,
        "quick": quick,
        "seed": seed,
    }
    stamp = folder / "cache.json"
    if fresh or not stamp.exists() or json.loads(stamp.read_text()) != key:
        if folder.exists():
            for old in [*folder.glob("record_ep*.npz"), folder / "eval.json"]:
                old.unlink(missing_ok=True)
        folder.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps(key, indent=1) + "\n")
    else:
        _say(f"{name} unchanged since the last run: reusing what {shown_folder} holds")

    env = gym.make(env_id(task), entropy=entropy) if entropy else gym.make(env_id(task))
    evaluation, rows = None, {}
    ev_file = folder / "eval.json"
    if episode is None:
        if ev_file.exists() and json.loads(ev_file.read_text()).get("episodes_arg") == episodes:
            evaluation = json.loads(ev_file.read_text())
        else:
            _say(f"evaluating {name} on {task} ({'quick' if quick else episodes} episodes) ...")
            start = time.perf_counter()
            result = evaluate(str(path), task, episodes, quick=quick, entropy=entropy, n_jobs=n_jobs)
            _say(str(result))
            evaluation = as_dict(result) | {"episodes_arg": episodes}
            ev_file.write_text(json.dumps(evaluation, indent=1, default=str) + "\n")
            _say(f"evaluated in {time.perf_counter() - start:.0f} s")
        rows = {r["episode"]: r for r in evaluation["per_episode"]}
        chosen = _picks(evaluation["per_episode"], env, seed)
    else:
        chosen = [(int(n), "chosen") for n in (episode if isinstance(episode, (list, tuple)) else [episode])]

    replications = QUICK["fq_replications"] if quick else NAIVE_REPLICATIONS
    cls = load(path)
    shown = []
    for n, why in chosen:
        file = folder / f"record_ep{n}.npz"
        if file.exists():
            rec = dashboard.load_record(file)
        else:
            _say(f"recording episode {n} ({why}) ...")
            rec = dashboard.record_episode(
                env, cls, seed=seed, options={"episode": n}, naive_replications=replications, n_jobs=n_jobs
            )
            dashboard.save_record(rec, file)
            rec = dashboard.load_record(file)
        row = rows.get(n)
        if row and row["J_policy_cents"] != rec["meta"]["J_cents"]:
            _say(
                f"  note: episode {n} replayed at a different cost than in the evaluation (an agent seeded differently)"
            )
        shown.append(export.episode(rec, why, row))
    data = {
        "agent": name,
        "task": task,
        "entropy": entropy,
        "quick": quick,
        "made": time.strftime("%Y-%m-%d %H:%M"),
        "net": export.network(rec["static"]),
        "episodes": shown,
        "eval": {k: v for k, v in evaluation.items() if k != "episodes_arg"} if evaluation else None,
        "stats": export.score_stats(evaluation) if evaluation else None,
    }
    page = folder / "viewer.html"
    page.write_text(_page(data), encoding="utf-8")
    _say(f"written {page} ({page.stat().st_size / 1e6:.1f} MB): open it in a browser (xdg-open {page})")
    return page


__all__ = ["agent_hash", "view"]
