"""Train a BC agent from teacher C, the re-planning teacher (run 4), optionally with DAgger rounds (run 5).

    uv run python experiments/bc/train_replan.py --agent=bc_tiny_v4                       # run 4
    uv run python experiments/bc/train_replan.py --agent=bc_tiny_dagger --dagger_rounds=3 \\
        --dataset=outputs/bc/<run 4>/dataset_round0.npz                                  # run 5, reusing run 4's data

Teacher C (``replan_lib.Replanner``): every week it re-plans from the episode's real state with the true future,
following the game's rules, and answers with this week's flows and strait releases.

Round 0 (as run 1 to 3, with the new teacher): the teacher plays each training episode, once exactly and
``--noisy_copies`` times with noise on its decisions; every week is recorded with the teacher's answer for that week's
real state (so the noisy plays now get correct answers). Then the flow and release networks are trained.

DAgger rounds 1..``--dagger_rounds``: the *student* (the networks just trained) plays ``--dagger_episodes`` new
episodes; every week the teacher says what it would do in the student's state; those weeks join the data and both
networks are trained again on everything. The student learns to recover from its own mistakes.

``--workers`` processes play episodes in parallel (each about 0.4 GB of memory). The agent folder gets weights.npz;
outputs/bc/<date_time>/ gets the datasets, each round's weights and train_log.json.
"""

import json
import multiprocessing as mp
import time
from pathlib import Path

import fire
import numpy as np
import replan_lib as R


def play(jobs: list[dict], workers: int, label: str) -> list[dict]:
    """Run the episodes of ``jobs`` on ``workers`` processes, printing progress."""
    results, start = [], time.time()
    ctx = mp.get_context("spawn")  # fresh processes: they import replan_lib only (no torch)
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(R.run_episode, jobs), start=1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {label}: {i}/{len(jobs)} episodes ({time.time() - start:.0f} s)", flush=True)
    return sorted(results, key=lambda r: (r["n"], r.get("copy", 0)))


def to_dataset(results: list[dict]) -> dict:
    """Stack the recorded weeks; each row keeps its episode index (for the validation split)."""
    return {
        "X": np.concatenate([r["X"] for r in results]).astype(np.float32),
        "Y": np.concatenate([r["Y"] for r in results]).astype(np.float32),
        "episode": np.concatenate([np.full(len(r["Y"]), r["n"]) for r in results]),
        "XR": np.concatenate([r["XR"] for r in results]).astype(np.float32),
        "YR": np.concatenate([r["YR"] for r in results]).astype(np.float32),
        "episode_r": np.concatenate([np.full(len(r["YR"]), r["n"]) for r in results]),
    }


def merge(a: dict, b: dict) -> dict:
    return {k: np.concatenate([a[k], b[k]]) for k in ("X", "Y", "episode", "XR", "YR", "episode_r")}


def fit(data: dict, fit_args: dict, val_share: float, seed: int, net: dict | None = None):
    """Train the flow and release networks on ``data``; (numpy weights, histories).

    ``net`` None or model "classic": runs 4 to 8's networks (2 x ``hidden``, ``train_bc.train``). Otherwise
    ``nets.py``: model "plain" (``layers`` x ``hidden``) or "shared" (the flow network shares summaries between routes,
    ``net["share"]`` its averaging matrices; the release network stays plain).
    """
    episodes = np.unique(data["episode"])
    n_val = max(1, int(len(episodes) * val_share))
    val = np.random.default_rng(seed).choice(episodes, n_val, replace=False)
    if net is not None and net["model"] != "classic":
        return fit_nets(data, fit_args, val, net)
    from train_bc import arrays, train  # torch is imported here, in the main process only

    under = fit_args.pop("under_weight")
    flow_under = fit_args.pop("flow_under_weight")
    print(f"  flow network (answering too little counts {flow_under} x)")
    net, mean, std, hist = train(data["X"], data["Y"], data["episode"], val, **fit_args, under_weight=flow_under)
    print(f"  release network (answering too little counts {under} x)")
    net_r, mean_r, std_r, hist_r = train(data["XR"], data["YR"], data["episode_r"], val, **fit_args, under_weight=under)
    fit_args["under_weight"], fit_args["flow_under_weight"] = under, flow_under
    weights = arrays(net, mean, std) | arrays(net_r, mean_r, std_r, "rel_")
    return weights, {"flows": hist, "releases": hist_r}


def fit_nets(data: dict, fit_args: dict, val: np.ndarray, net: dict):
    """``fit`` with nets.py's networks (runs 9 to 12)."""
    from nets import train_plain, train_shared  # torch: main process only

    common = {k: fit_args[k] for k in ("module", "hidden", "epochs", "lr", "batch", "seed")} | {"layers": net["layers"]}
    under, flow_under = fit_args["under_weight"], fit_args["flow_under_weight"]
    print(f"  flow network ({net['model']}, {net['layers']} x {fit_args['hidden']}; too little counts {flow_under} x)")
    if net["model"] == "shared":
        flows, hist = train_shared(
            data["X"],
            data["Y"],
            data["episode"],
            val,
            net["share"],
            **common,
            summary=net["summary"],
            under_weight=flow_under,
            label="flows",
        )
    else:
        flows, hist = train_plain(
            data["X"], data["Y"], data["episode"], val, **common, under_weight=flow_under, label="flows"
        )
    print(f"  release network (plain, {net['layers']} x {fit_args['hidden']}; too little counts {under} x)")
    rel, hist_r = train_plain(
        data["XR"], data["YR"], data["episode_r"], val, **common, under_weight=under, label="releases"
    )
    weights = flows | {f"rel_{k}": v for k, v in rel.items()}
    return weights, {"flows": hist, "releases": hist_r}


def share_matrices(module, task: str, entropy: int) -> np.ndarray:
    """The agent's averaging matrices of the shared network, from one reset's config."""
    import gymnasium as gym
    from shockbench_flow_gym.play import agent_config_from_reset

    env = gym.make(R.env_id(task), entropy=entropy)
    obs, info = env.reset(options={"episode": 0})
    return module.SlotFeatures(agent_config_from_reset(env, obs, info)).share_matrices


def best(history: list[dict]) -> dict:
    return min(history, key=lambda h: h["val_loss"])


def main(
    agent: str = "bc_tiny_v4",
    task: str = "tiny",
    episodes: int = 300,
    entropy: int = 12345,
    noise: float = 0.2,
    noisy_copies: int = 1,
    dagger_rounds: int = 0,
    dagger_episodes: int = 100,
    epochs: int = 60,
    hidden: int = 64,
    lr: float = 1e-3,
    batch: int = 2048,
    under_weight: float = 2.0,
    flow_under_weight: float = 1.0,
    model: str = "classic",
    layers: int = 2,
    summary: int = 16,
    val_share: float = 0.1,
    seed: int = 0,
    workers: int = 5,
    dataset: str | None = None,
    first_episode: int = 0,
    merge_with: str | None = None,
    out: str | None = None,
) -> None:
    """Collect teacher C's answers, train, and optionally run DAgger rounds; write agents/<agent>/weights.npz.

    Args:
        agent: the agent folder that receives weights.npz (needs SlotFeatures and ReleaseFeatures).
        task: tiny (the weights fit the network they were trained on).
        episodes: round 0's training episodes (indices 0 .. episodes - 1 of root ``entropy``).
        entropy: the root of the training episodes; keep it away from 0, the dev episodes the test uses.
        noise, noisy_copies: round 0's noisy plays (spread of the random factor; extra plays per episode).
        dagger_rounds: rounds where the student plays and the teacher labels (0: plain behaviour cloning).
        dagger_episodes: new episodes per DAgger round (never the ones of round 0).
        epochs, hidden, lr, batch: the networks' training.
        under_weight: the release network's loss counts answering too little this many times more.
        flow_under_weight: the same for the flow network (above 1: it learns to ship more than the teacher).
        model: "classic" (runs 4 to 8), "plain" (any depth, ``layers``) or "shared" (routes share summaries, run 11).
        layers: hidden layers of each network (models plain and shared).
        summary: the shared network's summary size per route.
        val_share: share of the episodes held out to measure the networks on states they never saw.
        seed: seeds the noise, the validation split and torch.
        workers: processes that play episodes in parallel.
        dataset: a dataset_round0.npz of an earlier run: skip round 0's collection.
        first_episode: round 0 plays episodes first_episode .. first_episode + episodes - 1.
        merge_with: a dataset_round0.npz of an earlier run (same agent features) added to round 0's new data.
        out: the run folder (default: outputs/bc/<date_time>).
    """
    module, folder = R.load_agent_module(agent)
    if not hasattr(module, "ReleaseFeatures"):
        raise ValueError(f"{agent}'s agent.py has no ReleaseFeatures: use train_bc.py for a flows-only agent")
    out = Path(out or R.ROOT / f"outputs/bc/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    base = {"agent": str(folder), "task": task, "entropy": entropy}
    fit_args = {"epochs": epochs, "hidden": hidden, "lr": lr, "batch": batch, "seed": seed, "module": module}
    fit_args["under_weight"], fit_args["flow_under_weight"] = under_weight, flow_under_weight
    net = {"model": model, "layers": layers, "summary": summary}
    if model == "shared":
        net["share"] = share_matrices(module, task, entropy)
    log = {"args": {k: v for k, v in locals().items() if isinstance(v, (int, float, str, type(None)))}, "rounds": []}

    # ----- round 0: the teacher plays (exactly, then with noise) -----
    if dataset:
        print(f"round 0: reusing {dataset}")
        with np.load(dataset) as f:
            data = {k: f[k] for k in f.files}
        teacher_gap = None
    else:
        last = first_episode + episodes - 1
        print(f"round 0: teacher C plays episodes {first_episode}..{last} of root {entropy} ({1 + noisy_copies} each)")
        jobs = [
            base
            | {
                "n": n,
                "copy": c,
                "seed": seed * 1_000_003 + n * 101 + c,
                "player": "teacher",
                "noise": noise if c else 0.0,
                "oracle": c == 0,
            }
            for n in range(first_episode, first_episode + episodes)
            for c in range(1 + noisy_copies)
        ]
        results = play(jobs, workers, "round 0")
        exact = [r for r in results if "oracle_usd" in r]
        teacher_gap = np.mean([r["cost_usd"] for r in exact]) / np.mean([r["oracle_usd"] for r in exact]) - 1
        print(f"  teacher C costs {teacher_gap:+.1%} vs the oracle on these episodes (the best a perfect copy can do)")
        data = to_dataset(results)
        np.savez_compressed(out / "dataset_round0.npz", **data)
        if merge_with:
            with np.load(merge_with) as f:
                old = {k: f[k] for k in f.files}
            if np.isin(np.unique(old["episode"]), np.unique(data["episode"])).any():
                raise ValueError(f"{merge_with} shares episodes with this run: choose another --first_episode")
            data = merge(old, data)
            print(f"  merged with {merge_with}")
    print(f"  {len(data['Y']):,} flow rows, {len(data['YR']):,} release rows")
    print("training")
    weights, hist = fit(data, fit_args, val_share, seed, net)
    np.savez(out / "weights_round0.npz", **weights)
    log["rounds"].append({"round": 0, "rows": int(len(data["Y"])), "teacher_gap": teacher_gap, "history": hist})

    # ----- DAgger rounds: the student plays, the teacher labels -----
    for r in range(1, dagger_rounds + 1):
        first = first_episode + episodes + (r - 1) * dagger_episodes  # new episodes each round
        print(f"DAgger round {r}: the student plays episodes {first}..{first + dagger_episodes - 1}")
        jobs = [
            base | {"n": n, "seed": seed + n, "player": "student", "weights": weights, "noise": 0.0, "oracle": True}
            for n in range(first, first + dagger_episodes)
        ]
        results = play(jobs, workers, f"round {r}")
        student_gap = np.mean([x["cost_usd"] for x in results]) / np.mean([x["oracle_usd"] for x in results]) - 1
        print(f"  the student (round {r - 1}'s networks) costs {student_gap:+.1%} vs the oracle on these episodes")
        new = to_dataset(results)
        np.savez_compressed(out / f"dataset_round{r}.npz", **new)
        data = merge(data, new)
        print(f"training on {len(data['Y']):,} flow rows, {len(data['YR']):,} release rows")
        weights, hist = fit(data, fit_args, val_share, seed, net)
        np.savez(out / f"weights_round{r}.npz", **weights)
        log["rounds"].append(
            {"round": r, "rows": int(len(data["Y"])), "student_gap_before": student_gap, "history": hist}
        )

    np.savez(folder / "weights.npz", **weights)
    (out / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    last = log["rounds"][-1]["history"]
    print(
        f"done: flows {best(last['flows'])['val_mae']:.3f}, releases {best(last['releases'])['val_mae']:.3f} "
        "mean error x capacity on held-out episodes"
    )
    print(f"weights: {folder / 'weights.npz'}   run folder: {out}")
    print(f"next: uv run python experiments/bc/test_bc.py --agent={Path(agent).name}")


if __name__ == "__main__":
    fire.Fire(main)
