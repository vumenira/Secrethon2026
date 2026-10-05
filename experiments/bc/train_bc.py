"""Train a behaviour cloning (BC) agent: copy the clairvoyant oracle's weekly flows (and its strait releases).

    uv run python experiments/bc/train_bc.py --agent=bc_tiny_rel          # flows + strait releases, 300 episodes
    uv run python experiments/bc/train_bc.py                               # bc_tiny: flows only (run 1)
    uv run python experiments/bc/train_bc.py --episodes=50 --epochs=20     # a quick try
    uv run python experiments/bc/train_bc.py --dataset=outputs/bc/<run>/dataset.npz   # retrain, skip step 1

Three steps:
1. collect: for each training episode (your own root, ``--entropy``, never the dev episodes) solve the oracle's
   linear program, which knows the whole episode in advance, and read its planned flow for every week and slot (and,
   if the agent learns releases, its tanker release for every week and release group). Play the episode with the
   oracle's decisions and record what an agent would see each week next to what the oracle did. ``--noisy_copies``
   replays each episode with random noise on the decisions, so the networks also see states a little off the
   oracle's path (the oracle's plan stays the label there: it is not re-planned, DAgger's job later).
2. train: a small MLP (PyTorch) learns features -> oracle flow / nominal capacity, one row per slot and week; a second
   one the same for releases, one row per release group and week.
3. export: the weights go to ``agents/<agent>/weights.npz`` (plain numpy arrays, read by the agent without torch).

The agent decides what is learned: an agent.py with a ``ReleaseFeatures`` class (``bc_tiny_rel``) gets the release
network too; one without it (``bc_tiny``) gets the flow network only.

Files: outputs/bc/<date_time>/ holds dataset.npz, weights.npz (a copy) and train_log.json.
"""

import importlib.util
import json
import time
from pathlib import Path

import fire
import gymnasium as gym
import numpy as np
import torch
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.marks import compute_marks
from shockbench_flow.oracle.lp import build_lp, lp_flow_key, solve_oracle
from shockbench_flow_gym import play_episode  # also registers the ShockBench/* environments

from sbf_starter import env_id


ROOT = Path(__file__).resolve().parents[2]


def load_agent_module(agent: str):
    """The agent's agent.py as a module (for its feature classes, ``MAX_SHARE`` and ``normalise``)."""
    folder = Path(agent) if Path(agent).exists() else ROOT / "agents" / agent
    spec = importlib.util.spec_from_file_location("bc_agent_module", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, folder


# ----- step 1: collect the oracle's decisions ----------------------------------------------------------------------
def release_groups(inst) -> list[tuple[int, int, int]]:
    """(strait, good, out edge) of every release group, in the order of their first override slot (as the agent)."""
    groups = []
    for c, k, e, _lane in inst.override_slots:
        if (c, k, e) not in groups:
            groups.append((c, k, e))
    return groups


def oracle_plan(task: str, n: int, entropy: int) -> tuple[np.ndarray, np.ndarray, float]:
    """The oracle's plan of episode n: flows (T, slots), releases (T, release groups), and its cost in USD."""
    inst, _params = task_generator(task)
    model = build_lp(inst, compute_marks(inst, scenario(task, n, entropy=entropy)))
    result = solve_oracle(model)
    if result.x is None:
        raise RuntimeError(f"episode {n}: the oracle's LP was not solved ({result.message})")
    groups = release_groups(inst)
    flows = np.zeros((inst.T, len(inst.action_slots)))
    releases = np.zeros((inst.T, len(groups)))
    for t in range(1, inst.T + 1):  # the LP's weeks are 1..T, like observation["week"]
        for s, (e, k, lane) in enumerate(inst.action_slots):
            flows[t - 1, s] = max(result.x[model.index[("x", t, *lp_flow_key(inst, e, k, lane))]], 0.0)
        for i, (_c, k, e) in enumerate(groups):  # a release leaves the strait by edge e, its lanes summed
            j = model.index.get(("x", t, e, k, None))
            releases[t - 1, i] = max(result.x[j], 0.0) if j is not None else 0.0
    return flows, releases, float(result.J_usd)


class Teacher:
    """An agent that plays the oracle's decisions (times some noise) and records (features, label) every week."""

    def __init__(self, config, plan, rng, noise, module):
        self.flows, self.releases = plan
        self.rng, self.noise, self.max_share = rng, noise, module.MAX_SHARE
        self.features = module.SlotFeatures(config)
        self.release_features = module.ReleaseFeatures(config) if hasattr(module, "ReleaseFeatures") else None
        if self.release_features is not None:
            assert len(self.release_features.groups) == self.releases.shape[1], "release groups differ from the LP's"
        self.X, self.Y, self.XR, self.YR = [], [], [], []

    def noisy(self, qty):
        return qty * self.rng.lognormal(0.0, self.noise, qty.shape) if self.noise > 0 else qty

    def act(self, observation):
        week = int(observation["week"][0])
        flows = self.flows[week - 1]
        self.X.append(self.features(observation))
        self.Y.append(np.clip(flows / self.features.u0, 0.0, self.max_share))  # the label: share of capacity
        action = {"flows": self.noisy(flows) * observation["action_mask"]}
        if self.release_features is not None:
            rf, releases = self.release_features, self.releases[week - 1]
            self.XR.append(rf(observation, self.features.global_part))
            self.YR.append(np.clip(releases / rf.u0, 0.0, self.max_share))
            action["override_qty"], action["release_mode"] = rf.release_action(self.noisy(releases), observation)
        return action  # without a release network: the default tanker release, as that agent plays


def collect(task, episodes, entropy, noise, noisy_copies, seed, module):
    env = gym.make(env_id(task), entropy=entropy)
    rng = np.random.default_rng(seed)
    parts = {"X": [], "Y": [], "XR": [], "YR": [], "episode": [], "episode_r": []}
    gaps = []
    start = time.time()
    for n in range(episodes):
        flows, releases, oracle_cost = oracle_plan(task, n, entropy)
        for copy in range(1 + noisy_copies):  # copy 0: exactly the oracle's decisions; the others: with noise
            teachers = []

            def make(config, copy=copy, teachers=teachers):
                teachers.append(Teacher(config, (flows, releases), rng, noise if copy else 0.0, module))
                return teachers[-1]

            cost = play_episode(env, make, n)
            if copy == 0:
                gaps.append((oracle_cost, cost))
            t = teachers[0]
            for name in ("X", "Y", "XR", "YR"):
                parts[name] += getattr(t, name)
            parts["episode"] += [n] * sum(len(y) for y in t.Y)  # one row per slot and week
            parts["episode_r"] += [n] * sum(len(y) for y in t.YR)  # one row per release group and week
        if (n + 1) % 10 == 0 or n + 1 == episodes:
            print(f"  collected {n + 1}/{episodes} episodes ({time.time() - start:.0f} s)", flush=True)
    data = {
        "X": np.concatenate(parts["X"]).astype(np.float32),
        "Y": np.concatenate(parts["Y"]).astype(np.float32),
        "episode": np.asarray(parts["episode"]),
        "oracle_usd": np.array([g[0] for g in gaps]),
        "replay_usd": np.array([g[1] for g in gaps]),
    }
    if parts["XR"]:
        data |= {
            "XR": np.concatenate(parts["XR"]).astype(np.float32),
            "YR": np.concatenate(parts["YR"]).astype(np.float32),
            "episode_r": np.asarray(parts["episode_r"]),
        }
    return data


# ----- step 2: train a network -------------------------------------------------------------------------------------
def train(X, Y, episode, val_episodes, epochs, hidden, lr, batch, seed, module, under_weight=1.0):
    """One MLP features -> share of capacity; the best epoch on the held-out episodes is kept.

    ``under_weight`` > 1 punishes answering less than the label that many times more than answering more (1: even).
    """
    torch.manual_seed(seed)
    is_val = np.isin(episode, val_episodes)  # whole episodes held out, so validation is honest
    mean, std = X[~is_val].mean(0), X[~is_val].std(0)
    std = np.where(std < 1e-3, 1.0, std)  # a feature (almost) constant in training is left unscaled
    to_t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    Xt, Yt = to_t(module.normalise(X[~is_val], mean, std)), to_t(Y[~is_val])
    Xv, Yv = to_t(module.normalise(X[is_val], mean, std)), to_t(Y[is_val])

    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    huber = torch.nn.HuberLoss(delta=0.5, reduction="none")

    def loss_fn(pred, target):
        weight = torch.where(pred < target, under_weight, 1.0)
        return (huber(pred, target) * weight).mean()

    best, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        net.train()
        order = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = order[i : i + batch]
            loss = loss_fn(net(Xt[idx])[:, 0], Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pred = net(Xv)[:, 0]
            val = loss_fn(pred, Yv).item()
            val_mae = (pred.clamp(min=0) - Yv).abs().mean().item()
        history.append({"epoch": epoch, "train_loss": total / len(Xt), "val_loss": val, "val_mae": val_mae})
        marker = ""
        if val < best:
            best, marker = val, "  <- best so far"
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        print(
            f"  epoch {epoch:3d}  train {total / len(Xt):.4f}  validation {val:.4f}  "
            f"(mean error {val_mae:.3f} x capacity){marker}",
            flush=True,
        )
    net.load_state_dict(best_state)
    return net, mean, std, history


# ----- step 3: export ----------------------------------------------------------------------------------------------
def arrays(net, mean, std, prefix=""):
    """A trained network as numpy arrays named as the agent reads them ("" the flows', "rel_" the releases')."""
    layers = [m for m in net if isinstance(m, torch.nn.Linear)]
    out = {prefix + "mean": mean, prefix + "std": std}
    for i, layer in enumerate(layers, start=1):
        out[f"{prefix}W{i}"] = layer.weight.detach().numpy().T.astype(np.float32)  # numpy: x @ W
        out[f"{prefix}b{i}"] = layer.bias.detach().numpy().astype(np.float32)
    return out


def summary(history):
    best = min(history, key=lambda h: h["val_loss"])
    return f"best epoch {best['epoch']}, mean error {best['val_mae']:.3f} x capacity on held-out episodes"


def main(
    task: str = "tiny",
    agent: str = "bc_tiny",
    episodes: int = 300,
    entropy: int = 12345,
    noise: float = 0.2,
    noisy_copies: int = 1,
    epochs: int = 60,
    hidden: int = 64,
    lr: float = 1e-3,
    batch: int = 2048,
    val_share: float = 0.1,
    seed: int = 0,
    dataset: str | None = None,
    out: str | None = None,
) -> None:
    """Collect the oracle's decisions, train the network(s), write agents/<agent>/weights.npz.

    Args:
        task: tiny (the weights fit the network they were trained on).
        agent: the agent folder that receives weights.npz (a name in agents/ or a path); bc_tiny_rel learns releases.
        episodes: training episodes (each one oracle solve and 1 + noisy_copies plays).
        entropy: the root of the training episodes; keep it away from 0, the dev episodes the test uses.
        noise: spread of the random factor on the noisy copies' decisions (0.2: most within about +-40 %).
        noisy_copies: extra plays per episode with noise (0: pure behaviour cloning).
        epochs, hidden, lr, batch: the networks' training.
        val_share: share of the episodes held out to measure the networks on states they never saw.
        seed: seeds the noise, the validation split and torch.
        dataset: a dataset.npz of an earlier run: skip step 1 (it must hold releases if the agent learns them).
        out: the run folder (default: outputs/bc/<date_time>).
    """
    module, folder = load_agent_module(agent)
    learns_releases = hasattr(module, "ReleaseFeatures")
    out = Path(out or ROOT / f"outputs/bc/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    out.mkdir(parents=True, exist_ok=True)
    what = "flows and strait releases" if learns_releases else "flows"

    if dataset:
        print(f"1/3 dataset: reusing {dataset}")
        with np.load(dataset) as f:
            data = {k: f[k] for k in f.files}
        if learns_releases and "XR" not in data:
            raise ValueError(f"{dataset} has no releases (an older run): collect again without --dataset")
    else:
        print(f"1/3 collecting the oracle's {what} on {episodes} episodes of root {entropy} ({task})")
        data = collect(task, episodes, entropy, noise, noisy_copies, seed, module)
        np.savez_compressed(out / "dataset.npz", **data)
    gap = data["replay_usd"].mean() / data["oracle_usd"].mean() - 1
    print(f"    flows: {len(data['Y']):,} rows (slot x week), {data['X'].shape[1]} features each")
    if learns_releases:
        print(f"    releases: {len(data['YR']):,} rows (release group x week), {data['XR'].shape[1]} features each")
    print(
        f"    replaying the oracle's {what} costs {gap:+.1%} vs the oracle's own plan (the best a perfect copy can do)"
    )

    all_episodes = np.unique(data["episode"])
    n_val = max(1, int(len(all_episodes) * val_share))
    val_episodes = np.random.default_rng(seed).choice(all_episodes, n_val, replace=False)
    fit = {"epochs": epochs, "hidden": hidden, "lr": lr, "batch": batch, "seed": seed, "module": module}

    print(f"2/3 training the flow network ({epochs} epochs)")
    net, mean, std, history = train(data["X"], data["Y"], data["episode"], val_episodes, **fit)
    weights = arrays(net, mean, std)
    histories = {"flows": history}
    if learns_releases:
        print(f"    training the release network ({epochs} epochs)")
        net_r, mean_r, std_r, history_r = train(data["XR"], data["YR"], data["episode_r"], val_episodes, **fit)
        weights |= arrays(net_r, mean_r, std_r, "rel_")
        histories["releases"] = history_r

    print("3/3 exporting")
    np.savez(folder / "weights.npz", **weights)
    np.savez(out / "weights.npz", **weights)
    log = {
        "args": {
            "task": task,
            "agent": agent,
            "episodes": episodes,
            "entropy": entropy,
            "noise": noise,
            "noisy_copies": noisy_copies,
            "epochs": epochs,
            "hidden": hidden,
            "lr": lr,
            "batch": batch,
            "val_share": val_share,
            "seed": seed,
            "dataset": dataset,
        },
        "rows": int(len(data["Y"])),
        "features": int(data["X"].shape[1]),
        "release_rows": int(len(data["YR"])) if learns_releases else 0,
        "replay_gap": float(gap),
        "history": histories["flows"],
        "history_releases": histories.get("releases"),
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=2, default=str))
    for name, h in histories.items():
        print(f"done, {name}: {summary(h)}")
    print(f"weights: {folder / 'weights.npz'}   run folder: {out}")
    print(f"next: uv run python experiments/bc/test_bc.py --agent={agent}")


if __name__ == "__main__":
    fire.Fire(main)
