"""Play variants of an agent folder on a root of our own and compare them, paired, with every result cached.

uv run python experiments/approaches/tune.py run --variants=variants.json --task=small --entropy=12345 --episodes=64
uv run python experiments/approaches/tune.py report --base=mpc_det_h20 --task=small --entropy=12345 --episodes=64

A variant is {name: {"base": an agent folder, "params": its params.json, "truth": kinds played on their true
recoveries (agents with a recovery.py only)}}; {} plays the agent <name> as it is. An agent folder is looked up in
agents/ and then in experiments/pulse_agent/versions/. Results are cached by name in
outputs/recovery_tune/<task>-e<entropy>-n<episodes>/<name>.json, so a name is played once: change the name, or --redo.
"""

import json
import shutil
import sys
import time
from pathlib import Path

import fire
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "recovery_tune"
VERSIONS = ROOT / "experiments" / "pulse_agent" / "versions"  # every version before the final agent
TRUTH_FIELDS = {"closure": "o", "capacity": "u", "grid": "G_bar", "supply": "supply", "fab": "R", "osat": "R_osat"}


def wait_for_memory(min_gb=1.5, poll=30, give_up=600):
    """Hold a run while the machine is short of free memory (several runs share it); Windows only."""
    if sys.platform != "win32":
        return
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in ("total", "free", "page", "page_free", "virtual", "virtual_free", "extended")
        ]

    start = time.perf_counter()
    while time.perf_counter() - start < give_up:
        status = Status()
        status.length = ctypes.sizeof(Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if status.free >= min_gb * 2**30:
            return
        print(f"waiting: {status.free / 2**30:.1f} GB of memory free, {min_gb} wanted", flush=True)
        time.sleep(poll)


def agent_dir(name):
    """An agent folder by name: agents/<name>, else experiments/pulse_agent/versions/<name> (the earlier versions)."""
    for home in (ROOT / "agents", VERSIONS):
        if (home / name / "agent.py").is_file():
            return home / name
    raise FileNotFoundError(f"{name}: no agent.py under agents/ or {VERSIONS.relative_to(ROOT).as_posix()}/")


def folder(name, spec):
    """The variant's agent folder: a copy of its base agent with its params.json."""
    if "base" not in spec:
        return agent_dir(name)
    out = OUT / "agents" / name
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(agent_dir(spec["base"]), out, ignore=shutil.ignore_patterns("__pycache__", "params.json"))
    if spec.get("params"):
        (out / "params.json").write_text(json.dumps(spec["params"]))
    return out


def truth_series(self, kind, element, event, value, nominal, t, H):
    """The element's true value in each window week until it is back to nominal, nominal after (no new onsets)."""
    if kind not in self.truth_kinds:
        return self.model_series(kind, element, event, value, nominal, t, H)
    field = TRUTH_FIELDS[kind]
    avg = np.asarray(getattr(self.truth, field))[t - 1 : t - 1 + H, element].astype(float).copy()
    now = getattr(self.truth, f"{field}_now", None)
    T = self.truth.T
    tol = 1e-9 * max(1.0, abs(nominal))
    for h in range(H):
        week = t + h + 1  # the week whose observation shows the instant at the end of window week h
        if now is not None and week <= T:
            back = now[week - 1, element] >= nominal - tol
        else:
            back = avg[h] >= nominal - tol
        if back:
            avg[h + 1 :] = nominal
            break
    return avg


def play_truth(root, spec, episodes, kinds):
    """As scoring._play, with the recovery forecast replaced by the scenario's own recoveries."""
    from shockbench_flow.dynamics.env import rollout, took_fallback
    from shockbench_flow.hosting.docker import without_secret_like
    from shockbench_flow_agent.local_eval import NO_ZIP_SHA256
    from shockbench_flow_agent.scoring import _metered_shim, _policy_seed, _world
    from shockbench_flow_agent.shim import load_agent_class, unload_agent

    task, entropy, regime, fq, cache = spec
    rows = []
    for n in episodes:
        inst, omega, marks, fallback = _world(task, entropy, n, fq, cache)
        with without_secret_like():
            factory = load_agent_class(root, "submission_truth")
            rec = sys.modules["recovery"].Recovery
            rec.truth, rec.truth_kinds, rec.model_series, rec.series = marks, set(kinds), rec.series, truth_series
            shim = _metered_shim(factory, None)
            start = time.perf_counter()
            seed = _policy_seed(entropy, n, NO_ZIP_SHA256)
            traj = rollout(inst, shim, omega, regime, seed, marks=marks, fallback=fallback)
            seconds = time.perf_counter() - start
        planner = getattr(shim.agent, "planner", None)
        failed = getattr(planner, "failed", None)
        unload_agent()
        rows.append(
            {
                "episode": n,
                "J_policy_cents": traj.J_cents,
                "fallback_weeks": sum(took_fallback(r) for r in traj.records),
                "first_error": f"week {shim.errors[0][0]}: {shim.errors[0][1]}" if shim.errors else None,
                "forecast_failed": failed,
                "seconds": round(seconds, 3),
                "omega_hash": omega.hash,
            }
        )
    return rows


def episode_set(task, entropy, episodes, n_jobs):
    from shockbench_flow_agent import EpisodeSet

    return EpisodeSet.build(task, episodes, entropy=entropy, n_jobs=n_jobs, verbose=True)


def run_dir(task, entropy, episodes):
    return OUT / f"{task}-e{entropy}-n{episodes}"


def run(variants, task="small", entropy=12345, episodes=64, n_jobs=3, redo=False):
    """Play every variant of the JSON file ``variants`` ({name: {base, params, truth}}) not played yet."""
    from joblib import Parallel, delayed

    es = episode_set(task, entropy, episodes, n_jobs)
    out = run_dir(task, entropy, episodes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "references.json").write_text(json.dumps(list(es.references), indent=1))
    specs = json.loads(Path(variants).read_text())
    for name, spec in specs.items():
        target = out / f"{name}.json"
        if target.exists() and not redo:
            print(f"{name}: cached", flush=True)
            continue
        wait_for_memory()
        start = time.perf_counter()
        root = folder(name, spec)
        if spec.get("truth"):
            ns = list(es.episodes)
            chunks = [c for c in np.array_split(np.array(ns), min(len(ns), n_jobs)) if c.size]
            parts = Parallel(n_jobs=n_jobs)(
                delayed(play_truth)(str(root), es._spec, [int(n) for n in c], spec["truth"]) for c in chunks
            )
            rows = [r for part in parts for r in part]
        else:
            rows = es.play(str(root), n_jobs=n_jobs)
        target.write_text(json.dumps({"spec": spec, "rows": rows}, indent=1))
        table = es.rss([r["J_policy_cents"] for r in rows])
        bad = sum(r["fallback_weeks"] for r in rows)
        errors = [r["first_error"] for r in rows if r.get("first_error")]
        failed = sum(r.get("forecast_failed") or 0 for r in rows)
        slow = max(r["seconds"] for r in rows)
        print(
            f"{name}: score {table['rss']:.4f} by level {table['rss_by_stratum']} fallback weeks {bad}"
            f" forecast failures {failed} errors {errors[:2]} slowest episode {slow:.0f} s"
            f" ({time.perf_counter() - start:.0f} s)",
            flush=True,
        )


def report(base="mpc_det_h20", task="small", entropy=12345, episodes=64, n_boot=4000):
    """Every cached variant against ``base``: score, the paired difference and its 90 % interval."""
    from shockbench_flow_agent.scoring import _boot_stats

    es = episode_set(task, entropy, episodes, 1)
    out = run_dir(task, entropy, episodes)
    refs = list(es.references)
    costs = {}
    for p in sorted(out.glob("*.json")):
        if p.stem != "references":
            costs[p.stem] = [r["J_policy_cents"] for r in json.loads(p.read_text())["rows"]]
    strata = [r["stratum"] for r in refs]
    counts = [strata.count(s) for s in (1, 2, 3, 4)]
    excluded = sum(r["excluded"] is not None for r in refs)
    print(f"{task}, root {entropy}, {episodes} episodes; by harm level {counts}; excluded {excluded}")
    names = [base] + [n for n in costs if n != base]
    boots = dict(zip(names, _boot_stats(refs, [costs[n] for n in names], True, n_boot, 0)))
    base_rss = es.rss(costs[base])["rss"]
    for n in names:
        table = es.rss(costs[n])
        d = boots[n] - boots[base]
        lo, hi = np.quantile(d[np.isfinite(d)], [0.05, 0.95])
        delta = (np.array(costs[base], dtype=float) - np.array(costs[n], dtype=float)) / 1e11  # bn USD saved
        by = " ".join(f"{table['rss_by_stratum'].get(s, float('nan')):.3f}" for s in (1, 2, 3, 4))
        print(
            f"{n:26s} {table['rss']:.4f}  diff {table['rss'] - base_rss:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            f"  P(better) {float((d > 0).mean()):.2f}  levels {by}"
            f"  better/worse episodes {int((delta > 1e-6).sum())}/{int((delta < -1e-6).sum())}"
        )


if __name__ == "__main__":
    fire.Fire({"run": run, "report": report})
