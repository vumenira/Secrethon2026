# Hypotheses: learning from the clairvoyant oracle (2026-10-04)

Ideas for an agent that learns from the clairvoyant plan (the "oracle", score 1), the linear program that knows every
disruption of an episode in advance. Nothing here has been built or run yet. The numbering follows our first discussion
(4, AggreVaTe, and 6, hindsight consensus, were dropped as too expensive or over the CPU budget).

The bar to beat is the kit's `heuristic` agent: **0.40** on Small's 20 dev episodes ([heuristic.md](heuristic.md)).

## Verdicts at a glance

| #   | Hypothesis                                   | Verdict                                                                 |
| --- | -------------------------------------------- | ----------------------------------------------------------------------- |
| 1   | Behaviour cloning (BC) of the oracle         | **Suitable as a first test.** Cheap and fast; unlikely to be the final agent |
| 2   | DAgger with a clairvoyant re-planner          | **Suitable: the best pure imitation option.** Costs one LP solve per training label |
| 3   | Privileged teacher, then RL fine-tuning      | **Not now.** Only worth it if 1 or 2 work and then plateau              |
| 5   | Learned forecast of disruptions + LP planner | **Suitable, likely the strongest.** More code; must fit the CPU budget   |

## The problem shared by every action-imitation idea

The oracle is good because it **knows the future**, not because it reasons better. It may stock up in week 10
because a strait closes in week 14; the agent sees at most a noisy warning in week 10. Copying it, the student learns
"what the oracle does on average in situations that look like this". That works as a hedge, but it can't reach the
oracle: the score will land between naive and 1. The literature calls this the *imitation gap*
([ADVISOR](https://alphaxiv.org/abs/2007.12173), [Swamy et al. 2022](https://arxiv.org/pdf/2208.02225)).

Two traps for any learned policy here:

- **The private board scores the same submission on Full** (395 action slots, against Small's 108). A network with
  fixed input and output sizes can't run there. Use one small network shared across routes (each route's own
  features in, its flow out), or separate weights per network.
- **Predict a multiple of the normal plan for each route**, not raw quantities, which span orders of magnitude.

---

## 1. Behaviour cloning (BC)

**Verdict: suitable as a first test.** A few hours of work to learn whether the imitation gap kills the idea. Expect
it to beat naive, but probably not the heuristic by much.

**How.** Solve the oracle on many scenarios of our own root (`--entropy=12345`). Record (observation, oracle action)
for every week. Train a network by plain supervised regression to output the oracle's action from the observation.

**Pros**

- The simplest option: standard supervised learning, nothing interactive.
- Data generation is one oracle solve per episode (cheap on Tiny, seconds on Small).
- Inference is a small network: far under the 2 s / 4 s CPU budget, and only numpy or torch at run time.

**Cons**

- **Covariate shift.** The network only sees states the oracle reached. After one mistake (too little stock, a
  shipment into a closed strait) it is in a state it never trained on, and errors pile up over 52 or 104 weeks.
- The full imitation gap: the oracle's actions often react to information the observation doesn't hold.
- The oracle's per-week actions have to be pulled out of its LP solution (the `x` columns), which needs some digging in
  `shockbench_flow/oracle/lp.py`.

## 2. DAgger with a clairvoyant re-planner

**Verdict: suitable, the best pure imitation option.** The natural next step if BC shows promise.

**How.** Start from the BC network. Let the student play episodes, and at every state it reaches ask the expert what it
would do there. Add those labels to the dataset, retrain, and repeat for a few rounds
([Ross et al. 2011](https://arxiv.org/pdf/1811.06711)). The expert is a *clairvoyant re-planner*: the LP rebuilt from
the student's current state (`lp_common.rolled_lp`, which `mpc_det` uses) with the scenario's true future in place
of `mpc_det`'s "today's disruptions persist" forecast. The code looks like it allows this; not verified yet.

**Pros**

- Fixes BC's covariate shift: the network learns to recover from its own mistakes.
- Theory says on-policy methods like DAgger handle a privileged teacher better than BC
  ([Swamy et al. 2022](https://arxiv.org/pdf/2208.02225)).
- Same cheap network at run time as BC.

**Cons**

- Each label is one LP solve: trivial on Tiny, around a second or more per week on Small (52 per episode per round).
  Overnight runs on this laptop; Full is much heavier.
- The re-planner has to be built and checked first (it should match the oracle's cost when started from the oracle's
  own states).
- The imitation gap remains: DAgger fixes *where* the student is trained, not *what* it can see.

## 3. Privileged teacher, then RL fine-tuning

**Verdict: not now.** Only if 1 or 2 work and plateau below the heuristic or MPC.

**How.** Imitate first (1 or 2), then fine-tune with reinforcement learning, so the student learns where copying the
oracle hurts and does better on its own. Variants: Learning by Cheating, ADVISOR (switch between imitation and RL by
state), asymmetric actor-critic (the critic sees the true future, the actor doesn't)
([oracle-planner imitation in games](https://arxiv.org/pdf/2012.12186)).

**Pros**

- The only option here that can, in principle, close part of the imitation gap.
- Starting RL from an imitated policy is far easier than RL from scratch.

**Cons**

- RL on this task is heavy: large observations, episode costs that vary by orders of magnitude, 52 to 104 weeks per
  episode. The PPO attempt (`agents/1ppo_tiny`) showed how fiddly it is.
- Long training on a weak PC; many hyperparameters; hard to tell noise from progress on 20 episodes.
- Every extra step adds a way to end up worse than the imitated policy.

## 5. Learned forecast of disruptions + LP planner

**Verdict: suitable, likely the strongest.** It imitates the oracle's *knowledge* instead of its *actions*.

**How.** Train a model to predict the network for the next few weeks (each strait's open fraction, capacities,
sanctions, tariffs) from the observation's warnings, messages and pending prohibitions. Labels come free: the true
future of each training scenario. Feed that forecast into the rolling LP planner (`mpc_det`'s LP), in place of its
"nothing changes" assumption. Related to *predict-then-optimise*.

**Pros**

- The LP keeps actions feasible and coherent (stock, capacity, lead times): no impossible or wasteful flows.
- An easier learning problem: "does this strait close in 3 weeks?" is far smaller than 395 flows at once.
- Each strait and route is predicted the same way, so it moves to Full more easily.
- Easy to read: we can check the forecasts directly, and stepping back to plain `mpc_det` is easy.

**Cons**

- An LP solve every week at run time: it must stay under 2 s (Small) and 4 s (Full) of CPU, one thread. Check the
  timings of `agents/mpc_det_h20` first.
- More code inside the package's LP helpers (`lp_common.persistence_arrays`, `window_marks`), and SciPy's
  `linprog` in `agent.py` (the server has no `highspy`).
- Its ceiling is set by the planner: a perfect forecast inside a short window is still not the full oracle.
- Warnings are noisy and some announcements are false alarms: the forecast must give probabilities, and the planner
  has to decide how to act on a "40 % chance of closure".

---

## Plan

1. Build BC on Tiny, with the shared-per-route network that predicts multiples of the normal plan. Compare it with
   `heuristic` and `template` (`uv run sbf compare`).
2. If it beats naive clearly, add DAgger rounds; then move to Small.
3. In parallel or instead, if BC shows the student can't tell when disruptions come, try 5.
