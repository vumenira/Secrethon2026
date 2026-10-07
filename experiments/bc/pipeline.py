"""The BC pipeline, end to end: collect the teacher's data, train variants, pick the best, confirm it, publish it.

    uv run python experiments/bc/pipeline.py --task=tiny --run=tiny_laptop      # see run_tiny_laptop.sh
    uv run python experiments/bc/pipeline.py --task=small --run=small_pc        # see run_small_pc.ps1

Five stages, each resumable: run the same command again and finished work is skipped (chunks of data on disk,
trained candidates on disk), so a crash, a reboot or Ctrl+C loses at most the chunk or variant in progress.

1. **collect.** Teacher C (``replan_lib.Replanner``: a linear program re-solved every week from the episode's real
   state with the true future and the game's rules) plays training episodes of root ``--entropy``, once exactly and
   ``--noisy_copies`` times with noise. Every week is recorded: the agent's features, including the LP planner's own
   plan (``agent_template/``), and the teacher's answer. Chunks of ``--chunk`` episodes go to ``<run>/chunks/``.
   Collection stops at ``--episodes`` or after ``--collect_hours``, whichever comes first.
2. **train.** Each variant of ``--variants`` trains a flow and a release network (``nets.py``; the GPU when PyTorch
   sees CUDA, else every CPU thread). ``plain`` variants drop the LP columns; ``lp`` variants keep them, and each
   also gives a ``_floor`` candidate (the same networks, never shipping or releasing less than the planner). Each
   candidate is a complete agent folder in ``<run>/candidates/<name>/``.
3. **select.** Every candidate, ``heuristic`` and ``mpc_det_h20`` play ``--val_episodes`` fresh episodes of root
   ``--val_root``; the best candidate by score is kept.
4. **confirm.** The best candidate, heuristic and mpc_det_h20 on the dev episodes and on ``--confirm_episodes`` of
   root ``--confirm_root``, never used to choose; paired differences against both.
5. **publish.** The best candidate is copied to ``agents/<--publish>/``; ``<run>/REPORT.md`` and ``results.json``
   hold every number.

Native Windows works (``winsupport.py``); ``sbf check`` needs Linux, so check the published agent on Linux or WSL.
The first evaluation on a new root computes its reference costs (minutes on Tiny, 15 to 30 on Small); they are cached
in ``~/.cache/shockbench-flow`` for later runs.
"""

import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

import winsupport  # noqa: F401 - first: Windows stand-ins for two Unix-only modules (does nothing elsewhere)


# isort: split
import fire
import numpy as np
import replan_lib as R


HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "agent_template"
BASELINES = ("heuristic", "mpc_det_h20")
DEFAULT_VARIANTS = (
    # name, use the LP plan as input, "too little" weight for flows and releases, hidden units, hidden layers
    {"name": "plain_u4", "use_lp": False, "flow_under": 4.0, "rel_under": 2.0, "hidden": 128, "layers": 3},
    {"name": "lp_u4", "use_lp": True, "flow_under": 4.0, "rel_under": 2.0, "hidden": 128, "layers": 3},
    {"name": "lp_u2", "use_lp": True, "flow_under": 2.0, "rel_under": 2.0, "hidden": 128, "layers": 3},
    {"name": "lp_u8", "use_lp": True, "flow_under": 8.0, "rel_under": 2.0, "hidden": 128, "layers": 3},
    {"name": "lp_u4_big", "use_lp": True, "flow_under": 4.0, "rel_under": 2.0, "hidden": 256, "layers": 4},
)
LP_FLOW_COLS, LP_REL_COLS = 1, 2  # the LP columns are the last ones of each feature table (agent_template/agent.py)


def say(text: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {text}", flush=True)


# ----- the machine ---------------------------------------------------------------------------------------------------
def free_memory_gb() -> float:
    """Memory available to new processes, in GB (Linux, Windows; 8 when unknown)."""
    try:
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = Status()
            st.dwLength = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullAvailPhys / 1e9
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return info["MemAvailable"] / 1e6
    except Exception:  # noqa: BLE001 - unknown platform: a modest guess
        return 8.0


def auto_workers(task: str) -> int:
    """One worker per CPU thread but two, and no more than free memory allows (about 0.5 GB each on Small)."""
    per_worker = {"tiny": 0.35, "small": 0.5, "full": 1.5}.get(task, 0.5)
    by_cpu = max(1, (os.cpu_count() or 2) - 2)
    by_mem = max(1, int((free_memory_gb() - 1.0) / per_worker))
    return min(by_cpu, by_mem)


# ----- 1. collect ----------------------------------------------------------------------------------------------------
def collect(run, task, episodes, first_episode, entropy, noise, noisy_copies, chunk, workers, collect_hours, seed):
    chunks = run / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)
    starts = list(range(first_episode, first_episode + episodes, chunk))
    todo = [s for s in starts if not (chunks / f"chunk_{s:06d}.npz").exists()]
    say(
        f"collect: {len(starts) - len(todo)} of {len(starts)} chunks already on disk; {len(todo)} to go, "
        f"{workers} workers, at most {collect_hours} h"
    )
    if not todo:
        return
    deadline = time.time() + collect_hours * 3600
    t0, done_plays = time.time(), 0
    ctx = mp.get_context("spawn")  # fresh processes on every platform: they import replan_lib, not torch
    with ctx.Pool(workers) as pool:
        for c, start in enumerate(todo, start=1):
            if time.time() > deadline:
                say(f"collect: the {collect_hours} h budget is spent; stopping before chunk {start}")
                break
            stop = min(start + chunk, first_episode + episodes)
            jobs = [
                {
                    "agent": str(TEMPLATE),
                    "task": task,
                    "entropy": entropy,
                    "n": n,
                    "copy": k,
                    "seed": seed * 1_000_003 + n * 101 + k,
                    "player": "teacher",
                    "noise": noise if k else 0.0,
                    "oracle": k == 0,
                }
                for n in range(start, stop)
                for k in range(1 + noisy_copies)
            ]
            results = [r for r in pool.imap_unordered(R.run_episode_safe, jobs)]
            failed = [r for r in results if "failed" in r]
            kept = sorted((r for r in results if "failed" not in r), key=lambda r: (r["n"], r["copy"]))
            for r in failed:
                say(f"  episode {r['n']} (play {r['copy']}) skipped: {r['failed']}")
            if kept:
                exact = [r for r in kept if "oracle_usd" in r]
                np.savez_compressed(
                    chunks / f"chunk_{start:06d}.npz",
                    X=np.concatenate([r["X"] for r in kept]).astype(np.float32),
                    Y=np.concatenate([r["Y"] for r in kept]).astype(np.float32),
                    episode=np.concatenate([np.full(len(r["Y"]), r["n"]) for r in kept]),
                    XR=np.concatenate([r["XR"] for r in kept]).astype(np.float32),
                    YR=np.concatenate([r["YR"] for r in kept]).astype(np.float32),
                    episode_r=np.concatenate([np.full(len(r["YR"]), r["n"]) for r in kept]),
                    teacher_usd=np.array([r["cost_usd"] for r in exact]),
                    oracle_usd=np.array([r["oracle_usd"] for r in exact]),
                    failed=np.array(len(failed)),
                )
            done_plays += len(jobs)
            rate = (time.time() - t0) / done_plays
            left = sum(min(s + chunk, first_episode + episodes) - s for s in todo[c:]) * (1 + noisy_copies) * rate
            say(
                f"collect: chunk {c}/{len(todo)} (episodes {start}..{stop - 1}) saved; "
                f"{rate * workers:.0f} s per play per worker; about {left / 60:.0f} min left for the rest"
            )


def load_data(run):
    files = sorted((run / "chunks").glob("chunk_*.npz"))
    if not files:
        raise SystemExit("no data: run the collect stage first")
    parts = [dict(np.load(f)) for f in files]
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("X", "Y", "episode", "XR", "YR", "episode_r")}
    teacher = np.concatenate([p["teacher_usd"] for p in parts])
    oracle = np.concatenate([p["oracle_usd"] for p in parts])
    data["teacher_gap"] = float(teacher.mean() / oracle.mean() - 1) if len(oracle) else float("nan")
    data["failed"] = int(sum(int(p["failed"]) for p in parts))
    return data


# ----- 2. train ------------------------------------------------------------------------------------------------------
def make_candidate(folder: Path, weights: dict) -> None:
    """A complete agent folder: the template's code (agent.py, lp_planner.py, sbflow/) and weights.npz."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE / "agent.py", folder / "agent.py")
    shutil.copy2(TEMPLATE / "lp_planner.py", folder / "lp_planner.py")
    if (folder / "sbflow").exists():
        shutil.rmtree(folder / "sbflow")
    shutil.copytree(TEMPLATE / "sbflow", folder / "sbflow", ignore=shutil.ignore_patterns("__pycache__"))
    np.savez(folder / "weights.npz", **weights)


def train(run, data, variants, epochs, patience, batch, lr, val_share, seed, device):
    import torch  # main process only
    from nets import train_plain

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 1)
    say(
        f"train: device {device}; {len(data['Y']):,} flow rows, {len(data['YR']):,} release rows, "
        f"{len(np.unique(data['episode']))} episodes"
    )
    module, _ = R.load_agent_module(str(TEMPLATE))
    episodes = np.unique(data["episode"])
    val = np.random.default_rng(seed).choice(episodes, max(1, int(len(episodes) * val_share)), replace=False)
    out = {}
    for v in variants:
        folder = run / "candidates" / v["name"]
        log = folder / "train_log.json"
        if (folder / "weights.npz").exists() and log.exists():
            say(f"train: {v['name']} already trained, skipped")
            out[v["name"]] = json.loads(log.read_text())
            continue
        say(f"train: {v['name']} ({v})")
        X = data["X"] if v["use_lp"] else data["X"][:, :-LP_FLOW_COLS]
        XR = data["XR"] if v["use_lp"] else data["XR"][:, :-LP_REL_COLS]
        common = {
            "module": module,
            "hidden": v["hidden"],
            "layers": v["layers"],
            "epochs": epochs,
            "lr": lr,
            "batch": batch,
            "seed": seed,
            "device": device,
            "patience": patience,
        }
        t0 = time.time()
        flows, hist = train_plain(
            X, data["Y"], data["episode"], val, **common, under_weight=v["flow_under"], label=f"{v['name']} flows"
        )
        rel, hist_r = train_plain(
            XR, data["YR"], data["episode_r"], val, **common, under_weight=v["rel_under"], label=f"{v['name']} releases"
        )
        weights = flows | {f"rel_{k}": a for k, a in rel.items()} | {"use_lp": np.array([int(v["use_lp"])])}
        make_candidate(folder, weights)
        if v["use_lp"]:  # the same networks, never below the planner's plan: no extra training
            make_candidate(run / "candidates" / f"{v['name']}_floor", weights | {"lp_floor": np.array([1])})
        best = lambda h: min(h, key=lambda e: e["val_loss"])  # noqa: E731
        info = {
            "variant": v,
            "minutes": round((time.time() - t0) / 60, 1),
            "flows": best(hist),
            "releases": best(hist_r),
            "history": {"flows": hist, "releases": hist_r},
        }
        log.write_text(json.dumps(info, indent=1, default=str))
        out[v["name"]] = info
        say(
            f"train: {v['name']} done in {info['minutes']} min; held-out error flows "
            f"{info['flows']['val_mae']:.3f}, releases {info['releases']['val_mae']:.3f} x capacity"
        )
    return out


# ----- 3 and 4. select and confirm -----------------------------------------------------------------------------------
def score_set(task, episodes, entropy, agents: dict, workers, against=()):
    """Scores of ``agents`` (name -> folder) on one episode set, and paired differences of the first vs ``against``."""
    from shockbench_flow_agent import EpisodeSet

    es = EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=workers, verbose=False)
    scores = {}
    for name, folder in agents.items():
        s = es.score(str(folder), name=name, n_jobs=workers)
        scores[name] = {
            "rss": s.rss,
            "interval": s.interval,
            "by_harm": s.rss_by_stratum,
            "fallback_weeks": s.fallback_weeks,
            "cost_usd": s.cost_usd,
        }
        say(f"  {name}: {s.rss:.3f} ({s.interval[0]:.2f} to {s.interval[1]:.2f})")
    first = next(iter(agents))
    diffs = {}
    for b in against:
        if b in agents and b != first:
            c = es.compare(str(agents[first]), str(agents[b]), n_jobs=workers)
            diffs[b] = {"diff": c.diff, "interval": c.interval}
            say(f"  {first} minus {b}: {c.diff:+.3f} ({c.interval[0]:+.2f} to {c.interval[1]:+.2f})")
    return {"scores": scores, "paired": diffs}


def baselines() -> dict:
    root = R.ROOT / "agents"
    return {b: root / b for b in BASELINES if (root / b / "agent.py").exists()}


# ----- 5. report -----------------------------------------------------------------------------------------------------
def write_report(run, task, results):
    lines = [f"# BC pipeline report: {task} ({time.strftime('%Y-%m-%d %H:%M')})", ""]
    d = results.get("data")
    if d:
        lines += [
            f"Data: {d['episodes']} episodes, {d['flow_rows']:,} flow rows, {d['release_rows']:,} release "
            f"rows; teacher C {d['teacher_gap']:+.1%} over the oracle; {d['failed']} plays skipped.",
            "",
        ]
    lines += [
        "## Training",
        "",
        "| Variant | LP input | too little (flows) | network | held-out error flows / releases | minutes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, info in results.get("train", {}).items():
        v = info["variant"]
        lines.append(
            f"| {name} | {'yes' if v['use_lp'] else 'no'} | {v['flow_under']}x | {v['layers']} x "
            f"{v['hidden']} | {info['flows']['val_mae']:.3f} / {info['releases']['val_mae']:.3f} | "
            f"{info['minutes']} |"
        )
    for title, key in (
        ("Selection", "select"),
        ("Confirmation: dev", "confirm_dev"),
        ("Confirmation: fresh root", "confirm_fresh"),
    ):
        block = results.get(key)
        if not block:
            continue
        lines += ["", f"## {title} ({block['label']})", "", "| Agent | Score | 90 % interval |", "| --- | --- | --- |"]
        for name, s in block["scores"].items():
            lines.append(f"| {name} | {s['rss']:.3f} | {s['interval'][0]:.2f} to {s['interval'][1]:.2f} |")
        for b, p in block.get("paired", {}).items():
            lines.append(f"\nBest minus {b}: {p['diff']:+.3f} ({p['interval'][0]:+.2f} to {p['interval'][1]:+.2f})")
    if results.get("best"):
        lines += ["", f"**Best: `{results['best']}`**, published to `agents/{results.get('published')}/`."]
    (run / "REPORT.md").write_text("\n".join(lines) + "\n")
    (run / "results.json").write_text(json.dumps(results, indent=1, default=str))


def main(
    task: str = "tiny",
    run: str | None = None,
    stages: str = "collect,train,select,confirm,publish",
    episodes: int = 2000,
    first_episode: int = 0,
    entropy: int = 12345,
    collect_hours: float = 2.0,
    chunk: int | None = None,
    noise: float = 0.2,
    noisy_copies: int = 1,
    workers: int | None = None,
    variants: str | None = None,
    epochs: int = 40,
    patience: int = 6,
    batch: int = 4096,
    lr: float = 1e-3,
    val_share: float = 0.1,
    device: str = "auto",
    val_root: int = 777,
    val_episodes: int = 20,
    confirm_root: int = 999,
    confirm_episodes: int = 20,
    publish: str | None = None,
    seed: int = 0,
) -> None:
    """Run the BC pipeline (module docstring); every stage resumes from what is on disk.

    Args:
        task: tiny, small (or full).
        run: the run's folder name under outputs/bc/ (default pipeline_<task>); keep it to resume.
        stages: which stages to run, comma-separated: collect, train, select, confirm, publish.
        episodes: training episodes to collect at most (episodes first_episode .. first_episode + episodes - 1).
        first_episode, entropy: where the training episodes come from (never the dev root 0).
        collect_hours: stop collecting after this long (finishes the chunk in progress).
        chunk: episodes per saved chunk (default 100 on Tiny, 20 on Small).
        noise, noisy_copies: the noisy plays of each episode (lognormal spread; extra plays).
        workers: parallel processes (default: CPU threads - 2, limited by free memory).
        variants: names of DEFAULT_VARIANTS to train, comma-separated (default: all).
        epochs, patience, batch, lr, val_share: network training (patience: stop after that many epochs without
            a better held-out loss).
        device: auto (CUDA if PyTorch sees it), cuda or cpu.
        val_root, val_episodes: the fresh episodes that choose the best variant.
        confirm_root, confirm_episodes: other fresh episodes, with dev, that confirm it (never used to choose).
        publish: the agent folder name for the best (default bc_<task>_best).
        seed: noise, validation split and torch.
    """
    run_dir = R.ROOT / "outputs" / "bc" / (run or f"pipeline_{task}")
    run_dir.mkdir(parents=True, exist_ok=True)
    as_list = lambda x: [str(i).strip() for i in (x if isinstance(x, (list, tuple)) else str(x).split(","))]  # noqa: E731
    stages = set(as_list(stages))  # fire turns "a,b" into a tuple: accept both
    workers = workers or auto_workers(task)
    chunk = chunk or (100 if task == "tiny" else 20)
    wanted = as_list(variants) if variants else None
    chosen = [v for v in DEFAULT_VARIANTS if not wanted or v["name"] in wanted]
    if not chosen:
        raise SystemExit(f"no variant matches {variants}: choose from {[v['name'] for v in DEFAULT_VARIANTS]}")
    publish = publish or f"bc_{task}_best"
    results_file = run_dir / "results.json"
    results = json.loads(results_file.read_text()) if results_file.exists() else {}
    say(
        f"run folder {run_dir}; {workers} workers; {os.cpu_count()} CPU threads; "
        f"{free_memory_gb():.1f} GB free memory; stages {sorted(stages)}"
    )

    if "collect" in stages:
        collect(
            run_dir, task, episodes, first_episode, entropy, noise, noisy_copies, chunk, workers, collect_hours, seed
        )
    if "train" in stages:
        data = load_data(run_dir)
        results["data"] = {
            "episodes": int(len(np.unique(data["episode"]))),
            "flow_rows": int(len(data["Y"])),
            "release_rows": int(len(data["YR"])),
            "teacher_gap": data["teacher_gap"],
            "failed": data["failed"],
        }
        results["train"] = train(run_dir, data, chosen, epochs, patience, batch, lr, val_share, seed, device)
        del data
        write_report(run_dir, task, results)
    names = [n for v in chosen for n in ((v["name"], f"{v['name']}_floor") if v["use_lp"] else (v["name"],))]
    candidates = {n: run_dir / "candidates" / n for n in names if (run_dir / "candidates" / n / "weights.npz").exists()}
    if "select" in stages:
        say(
            f"select: {len(candidates)} candidates and {list(baselines())} on {val_episodes} episodes of root "
            f"{val_root}"
        )
        block = score_set(task, val_episodes, val_root, candidates | baselines(), workers)
        block["label"] = f"{val_episodes} episodes of root {val_root}"
        results["select"] = block
        results["best"] = max(candidates, key=lambda n: block["scores"][n]["rss"])
        say(f"select: best is {results['best']} ({block['scores'][results['best']]['rss']:.3f})")
        write_report(run_dir, task, results)
    if "confirm" in stages:
        best = results.get("best")
        if not best:
            raise SystemExit("no best candidate yet: run the select stage first")
        agents = {best: candidates[best]} | baselines()
        say(f"confirm: {best} on the dev episodes")
        results["confirm_dev"] = score_set(task, "dev", 0, agents, workers, BASELINES) | {"label": "dev, 20"}
        say(f"confirm: {best} on {confirm_episodes} episodes of root {confirm_root}")
        results["confirm_fresh"] = score_set(task, confirm_episodes, confirm_root, agents, workers, BASELINES) | {
            "label": f"{confirm_episodes} episodes of root {confirm_root}"
        }
        write_report(run_dir, task, results)
    if "publish" in stages and results.get("best"):
        target = R.ROOT / "agents" / publish
        make_candidate(target, dict(np.load(candidates[results["best"]] / "weights.npz")))
        results["published"] = publish
        write_report(run_dir, task, results)
        say(f"publish: {results['best']} -> {target}")
        if sys.platform != "win32":
            say(f"next: uv run sbf check {publish} --task={task}")
        else:
            say("next: copy the agent folder to Linux (or WSL) and run sbf check there")
    say(f"report: {run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    fire.Fire(main)
