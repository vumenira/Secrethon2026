"""Teacher C, the re-planning teacher, and the episode loop that records what it teaches.

Every week the teacher looks at the episode's real state, solves the planning-rules linear program from this week to
the end with the episode's true future, and answers with this week's flows and strait releases. Its answers are
always possible from where the episode really is, whoever played the weeks before (the teacher itself, with noise, or
the student: that is DAgger).

The pieces come from the package: ``lp_common.rolled_lp`` (the LP from the current state, as ``mpc_det`` builds it)
fed with the true weekly values of the episode (``compute_marks``) instead of a forecast. With the rules off, at
week 1 it is the oracle's own LP (the package's "no-drift" property): ``check_week1`` tests that.

No PyTorch here: worker processes import this module only, to keep their memory small.
"""

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
from shockbench_flow.hosting.tasks import scenario, task_generator
from shockbench_flow.marks import FabHit, compute_marks
from shockbench_flow.oracle.lp import build_lp, solve_oracle
from shockbench_flow.policies import lp_common as L
from shockbench_flow_gym.play import agent_config_from_reset

from sbf_starter import env_id


ROOT = Path(__file__).resolve().parents[2]


def load_agent_module(agent: str):
    """The agent's agent.py as a module (its feature classes, ``MAX_SHARE``, ``normalise``, ``mlp_forward``)."""
    folder = Path(agent) if Path(agent).exists() else ROOT / "agents" / agent
    spec = importlib.util.spec_from_file_location("bc_agent_module", folder / "agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, folder


def release_groups(inst) -> list[tuple[int, int, int]]:
    """(strait, good, out edge) of every release group, in the order of their first override slot (as the agent)."""
    groups = []
    for c, k, e, _lane in inst.override_slots:
        if (c, k, e) not in groups:
            groups.append((c, k, e))
    return groups


class Replanner:
    """Teacher C for one episode: ``plan(wire_obs)`` -> (flows per slot, release per group) for this week."""

    def __init__(self, task: str, n: int, entropy: int, planning_rules: bool = True):
        self.inst, _params = task_generator(task)
        self.marks = compute_marks(self.inst, scenario(task, n, entropy=entropy))
        self.groups = release_groups(self.inst)
        self.rules = planning_rules
        self.solves = 0

    def window_arrays(self, t: int) -> dict:
        """The true weekly values of weeks t..T (every field the window LP reads)."""
        return {name: np.array(getattr(self.marks, name)[t - 1 :]) for name in L.WINDOW_FIELDS}

    def window_hits(self, t: int) -> tuple:
        """Fab outages that start in weeks t..T, their onset renumbered from the window's start (as the package)."""
        T = self.inst.T
        return tuple(
            FabHit(fab=h.fab, onset=h.onset - (t - 1), severity=h.severity)
            for h in self.marks.fab_hits
            if h.onset > t - 1 and t <= h.onset_week <= T
        )

    def model(self, wire_obs: dict):
        t = int(wire_obs["week"])
        H = self.inst.T - t + 1  # to the end of the episode
        return L.rolled_lp(
            self.inst, wire_obs, self.window_arrays(t), H, fab_hits=self.window_hits(t), planning_rules=self.rules
        )

    def plan(self, wire_obs: dict) -> tuple[np.ndarray, np.ndarray]:
        model = self.model(wire_obs)
        result = solve_oracle(model)
        self.solves += 1
        if result.x is None:
            raise RuntimeError(f"week {wire_obs['week']}: the teacher's LP was not solved ({result.message})")
        x, tmpl = result.x, model.meta["template"]  # tmpl: the week-1 column of each key
        flows = np.array([max(x[tmpl[("x", e, k, lane)]], 0.0) for e, k, lane in self.inst.action_slots])
        releases = np.array(
            [max(x[tmpl[("x", e, k, None)]], 0.0) if ("x", e, k, None) in tmpl else 0.0 for _c, k, e in self.groups]
        )
        return flows, releases


def check_week1(task: str = "tiny", n: int = 0, entropy: int = 12345) -> tuple[float, float]:
    """(oracle J, teacher C's week-1 J with the rules off): equal if the window is fed correctly."""
    inst, _ = task_generator(task)
    marks = compute_marks(inst, scenario(task, n, entropy=entropy))
    J_oracle = solve_oracle(build_lp(inst, marks)).J_usd
    env = gym.make(env_id(task), entropy=entropy)
    _obs, info = env.reset(options={"episode": n})
    teacher = Replanner(task, n, entropy, planning_rules=False)
    return float(J_oracle), float(solve_oracle(teacher.model(info["obs"])).J_usd)


# ----- one episode: someone plays, the teacher labels every week --------------------------------------------------
_WORKER = {}  # per process: the agent module and one environment per (task, entropy)


def _setup(agent: str):
    if "module" not in _WORKER:
        _WORKER["module"], _ = load_agent_module(agent)
    return _WORKER["module"]


def _env(task: str, entropy: int):
    key = (task, entropy)
    if key not in _WORKER:
        _WORKER[key] = gym.make(env_id(task), entropy=entropy)
    return _WORKER[key]


def run_episode(job: dict) -> dict:
    """Play episode ``job["n"]`` and record (features, teacher's answer) every week.

    ``job["player"]``: "teacher" plays the teacher's answers (times lognormal noise of spread ``job["noise"]``),
    "student" plays the agent with ``job["weights"]`` (DAgger), "none" only measures the teacher (no recording).
    Returns the arrays, the episode's cost in USD and the number of LP solves.
    """
    module = _setup(job["agent"])
    env = _env(job["task"], job["entropy"])
    rng = np.random.default_rng(job["seed"])
    teacher = Replanner(job["task"], job["n"], job["entropy"], planning_rules=job.get("planning_rules", True))
    obs, info = env.reset(options={"episode": job["n"]})
    config = agent_config_from_reset(env, obs, info)
    sf, rf = module.SlotFeatures(config), module.ReleaseFeatures(config)
    student = None
    if job["player"] == "student":
        module.WEIGHTS = job["weights"]
        student = module.Agent(config)
    X, Y, XR, YR = [], [], [], []
    cents, done = 0, False
    while not done:
        flows, releases = teacher.plan(info["obs"])  # info["obs"]: the simulator's own view of this week's state
        x = sf(obs)
        xr = rf(obs, sf.global_part)
        X.append(x)
        Y.append(np.clip(flows / sf.u0, 0.0, module.MAX_SHARE))
        XR.append(xr)
        YR.append(np.clip(releases / rf.u0, 0.0, module.MAX_SHARE))
        if student is not None:
            action = student.act(obs)
        else:
            noise = job["noise"]
            shake = (lambda q: q * rng.lognormal(0.0, noise, q.shape)) if noise > 0 else (lambda q: q)
            action = {"flows": shake(flows) * obs["action_mask"]}
            action["override_qty"], action["release_mode"] = rf.release_action(shake(releases), obs)
        obs, _r, terminated, truncated, info = env.step(action)
        cents += int(info["reward_cents"])
        done = terminated or truncated
    out = {"n": job["n"], "cost_usd": -cents / 100, "solves": teacher.solves}
    if job.get("oracle"):  # the oracle's own cost of this episode, to measure how close the player came
        out["oracle_usd"] = float(solve_oracle(build_lp(teacher.inst, teacher.marks)).J_usd)
    if job["player"] != "none":
        out |= {"X": np.concatenate(X), "Y": np.concatenate(Y), "XR": np.concatenate(XR), "YR": np.concatenate(YR)}
    return out
