# Behaviour cloning (BC) on Tiny

Hypothesis 1 of [../hypotheses.md](../hypotheses.md): a small network learns to copy a teacher that knows the whole
episode in advance. The network sees only what an agent sees, so it learns "what the teacher does, on average, in
situations that look like this one".

**Result (2026-10-05):** the final model, **BC baseline v12** (`agents/bc_tiny_v12`), **ties the kit's `heuristic` on
Tiny**. It's a little ahead on two of the three episode sets, but not significantly. It's the checkpoint new agents
are compared against, and the starting point for Small.

| Episode set                         | bc_tiny_v12 | heuristic | template | final minus heuristic (90 %) |
| ----------------------------------- | ------------- | --------- | -------- | ---------------------------- |
| dev (20 episodes)                   | **0.780**     | 0.762     | 0.756    | +0.02 (-0.04 to 0.08)        |
| root 999 (40, fresh)                | 0.681         | **0.690** | 0.633    | -0.01 (-0.06 to 0.05)        |
| root 2026 (40, seen by no choice)   | **0.795**     | 0.779     | 0.723    | +0.02 (-0.02 to 0.05)        |

Score: 0 = the naive rule, 1 = the clairvoyant oracle. `sbf check` passes: 3 ms of CPU per week (of 2 s), 392 KB.

## BC baseline v12 (`agents/bc_tiny_v12`)

**The checkpoint from now on.** A new idea counts as progress when it beats v12 on the same episodes:

```bash
uv run sbf compare <new agent> bc_tiny_v12 --entropy=999 --episodes=40   # fresh episodes, paired interval
uv run sbf compare <new agent> bc_tiny_v12                               # the 20 dev episodes
```

### At a glance

| | |
| --- | --- |
| Files | `agent.py` (numpy only) and `weights.npz` (312 KB) |
| Networks | flows: 47 inputs, 3 x 128, 1 output (39,297 weights); releases: 31 inputs, 3 x 128, 1 output (37,249 weights) |
| CPU per week (`sbf check`, Tiny) | about 4 ms median, 10 ms in week 1 (budget 2 s) |
| Trained on | 2,000 episodes of root 12345 (Tiny), 2 plays each: 2.08 M flow rows, 312 k release rows |
| Score | dev 0.780, root 999 0.681, root 2026 0.795 (heuristic: 0.762, 0.690, 0.779) |
| By harm level, 1 calm to 4 harmful | dev 0.69 / 0.83 / 0.85 / 0.84; root 999 0.72 / 0.66 / 0.64 / 0.55; root 2026 0.72 / 0.85 / 0.64 / 0.93 |

### What it sees

Each week, one row of numbers per route and good (an action slot) for the flow network, and one per release group
(strait, good, route out) for the release network. Quantities are divided by the route's nominal capacity `u0`, so a
1 means "a normal week of this route". Large ones go through `log1p` so they stay readable. Then every number is
standardised with the training data's mean and spread, and clipped to ±10.

**Flow network: 47 numbers per slot on Tiny (43 + the number of goods):**

| Group | Numbers |
| ----- | ------- |
| The route this week (13) | capacity / nominal; freight cost / nominal; lead time / nominal; highest tariff on its edges; `action_mask`; lowest openness of its straits; highest warning score of its straits; lowest capacity / nominal of its later edges; whether it is a sea lane; transport mode (sea, air, pipeline, grid) |
| Stocks and flows (6) | stock of its good at the source and at the destination; cargo in transit on it; last week's requested and executed flow; the normal plan's weekly flow |
| Announcements on the route (4) | a sanction announced and not yet in force (flag, weeks until it starts); a closure of its strait with an announced end (flag, weeks until it reopens) |
| Its good (one per good) | which good the route carries |
| Shared by every slot (9) | week / T; weeks left / T; mean demand forecast, next 2 weeks, weeks 4–7 (each over week 1's); backlog; last week's lost sales; highest warning score anywhere; lowest strait openness anywhere |
| Other routes (2) | open capacity of the other routes of its good from the same source, and into the same destination |
| What lies ahead (3) | supply available at its source; health of the factory, grid or packaging plant at its source and at its destination (against week 1) |
| Messages (3) | threats on its edges; sanction notices on its edges; tariff notices on its source's or destination's region |
| History (3) | its straits' openness 1 and 2 weeks ago; their warning score 1 week ago |

**Release network: 31 numbers per release group on Tiny (27 + goods):**

| Group | Numbers |
| ----- | ------- |
| The queue (3) | cargo waiting to leave by this route; all cargo of its good waiting at the strait; weeks the oldest has waited |
| Arriving cargo (3) | cargo reaching the strait this week, next week, and this week by lanes that leave by this route |
| The strait and the route out (10) | openness; throughput; capacity / nominal and cost / nominal of the route out; its tariff; whether an override slot is open; warning score; war-risk class; a closure with an announced end (flag, weeks until it reopens) |
| Where the cargo goes (2) | stock at the route's end; cargo in transit on it |
| Its good, shared numbers | as for flows |

**Not read:** region and dyad warnings (only the highest one), last week's cost by component, the messages'
stated effective weeks, anything older than two weeks.

### What it decides

- `flows`: share x nominal capacity x `action_mask`, with the share clipped to [0, 2].
- Strait releases: the release network's share x the route out's nominal capacity, clipped to [0, 2]. While the
  strait is at least half open, a (strait, good) pair releases at least half of what waits plus what arrives this
  week (the floor). Each group's quantity goes on its first override slot that's open this week, with
  `release_mode` 1. A pair with no open slot keeps the default rule (0).

### The tricks, and what each was worth

| Trick | Why | Evidence |
| ----- | --- | -------- |
| **Teacher C**: re-plan every week, from the real state, with the true future and the game's rules | the oracle's own plan also controls factories and power, so it can't be copied | a perfect copy scores 0.805 on dev; the oracle's replay 0.694 (run 3, teachers table) |
| **Noisy copies**: each episode played again with noise (spread 0.2) on the teacher's decisions, still labelled by the teacher | the copy must see states a little off the teacher's path, or one mistake loses it | without them: 0.003 instead of 0.491 (run 2) |
| **One network shared by every route** | runs on any network size; every route's week is a training example | the same code runs on Small and Full |
| **Learned strait releases, with arriving cargo** | releases decide whether tankers wait; 78 % of the teacher's releases are of cargo that arrives and leaves in the same week | teacher with releases +5 %, without +33 % over the oracle (run 1); arriving-cargo features: release error 0.093 to 0.057 (run 4) |
| **Release floor** (half of waiting + arriving) | a release that's too small holds the cargo, with no safety net | dev 0.492 / 0.517 / 0.548 for floors 0 / 0.5 / 1 (run 4; small, inside the noise) |
| **Ship more than the teacher**: "too little" counts 4x for flows, 2x for releases | the teacher knows the future and ships just enough; shipping too little costs far more than too much | +0.23 on root 999, the largest gain (run 7) |
| **Context about other routes and what lies ahead** | the copy imitates the teacher for the right reason more often | +0.09 on root 999 (run 6) |
| **Bigger networks, more data**: 3 x 128 (was 2 x 64), 2,000 episodes (was 300) | more room to learn, and more varied episodes, including the rare harmful ones | +0.06 (run 9), then +0.01 to +0.02 (run 12) |
| **Ratios and first-week levels** as inputs | the same numbers mean the same thing on every route and network | (design choice, not measured) |

### Training recipe

| Setting | Value |
| ------- | ----- |
| Episodes | 2,000 of root 12345: 0–999 (run 9's data) and 1000–1999 |
| Plays per episode | 1 exact + 1 noisy (lognormal factor, spread 0.2, on flows and releases) |
| Label | the teacher's answer for that week's real state / nominal capacity, clipped to [0, 2] |
| Teacher's cost | +5.3 % (0–999) and +5.8 % (1000–1999) over the oracle |
| Loss | Huber (delta 0.5); "less than the teacher" counts 4x (flows), 2x (releases) |
| Optimiser | Adam, learning rate 0.001, batches of 2,048 rows, 60 epochs |
| Validation | 10 % of the episodes held out whole; the epoch with the best held-out loss is kept (flows: 34, releases: 14) |
| Held-out error | flows 0.043, releases 0.046 x capacity |
| Time | about 18 min to collect each 1,000 episodes (7 workers); about 5 min to train |

### How it plays

Over the 20 dev episodes (mean per episode):

| Agent       | Cargo sent | Overrides | Shortage, M USD | Power shed, M USD | Tariff, M USD | Disposal, M USD |
| ----------- | ---------- | --------- | --------------- | ----------------- | ------------- | --------------- |
| bc_tiny_v12 | 45,800     | 436       | 1,873           | 151               | 51            | 40              |
| heuristic   | 148,000    | 0         | 1,868           | 125               | 80            | 61              |

It matches heuristic's shortage while shipping a third as much: it saves on tariffs and disposal, and gives back
slightly more power cuts. It takes over the strait's release (`release_mode` 1) every week.

### Limits

- **Tiny weights only.** On Small the inputs grow to 51 and 35 (8 goods), and `Agent` refuses the weights. The code
  runs there unchanged; it needs teacher C's data on Small and retraining.
- **Weaker in harmful episodes** on root 999 (harm 3: 0.64, harm 4: 0.55). Some losses are beyond any agent: on
  dev episodes 4, 15 and 20 even teacher C loses about 0.9 bn USD to the oracle.
- **Below its teacher**: 0.780 on dev against teacher C's 0.805, a copy of a teacher that sees the future.
- `RELEASE_FLOOR` (0.5) and the 4x / 2x weights were not tuned.

### Reproduce

```bash
# run 9: 1,000 episodes into bc_tiny_v9 (about 20 min with 7 workers)
uv run python experiments/bc/train_replan.py --agent=bc_tiny_v9 --episodes=1000 --model=plain --layers=3 \
    --hidden=128 --flow_under_weight=4 --workers=7
# v12: 1,000 more episodes, merged with run 9's dataset (replace <run 9> by its run folder)
uv run python experiments/bc/train_replan.py --agent=bc_tiny_v12 --episodes=1000 --first_episode=1000 \
    --merge_with=outputs/bc/<run 9>/dataset_round0.npz --model=plain --layers=3 --hidden=128 \
    --flow_under_weight=4 --workers=7

uv run python experiments/bc/test_bc.py --agent=bc_tiny_v12 --others='[heuristic,template]'   # dev
uv run python experiments/bc/test_bc.py --agent=bc_tiny_v12 --entropy=999 --episodes=40        # fresh episodes
uv run sbf check bc_tiny_v12
```

Each worker process needs about 0.25 GB of memory. Training writes `agents/<agent>/weights.npz` and a run folder
`outputs/bc/<date_time>/` (datasets, each round's weights, `train_log.json`). `outputs/` is not committed, so the
run folders named below exist only on the machine that made them. Each run's agent and weights are committed.

## Files

| File | What it does |
| ---- | ------------ |
| `replan_lib.py` | teacher C and the episode loop: someone plays (the teacher, with noise, or the student), the teacher labels every week; no PyTorch, so worker processes stay small |
| `train_replan.py` | runs 4 to 12: collects teacher C's data on several processes (`--workers`), trains, optional DAgger rounds (`--dagger_rounds`); `--model classic/plain/shared`, `--layers`, `--hidden`, `--flow_under_weight`, `--dataset` (reuse), `--merge_with` |
| `nets.py` | runs 9 to 12's networks: `plain` (an MLP of any depth) and `shared` (routes decide together, run 11), with the asymmetric loss |
| `train_bc.py` | runs 1 to 3: teacher A (the oracle's one plan, solved once per episode); also the training function of runs 4 to 8 |
| `test_bc.py` | scores an agent and others on the same episodes, with a paired check against the first other |
| `agents/bc_tiny*` | one folder per run (table below): `agent.py` and its `weights.npz` |

## What was tested

Every run on Tiny, in order. "Root 999" is 40 fresh episodes that were never used for training. Dev flattered
several agents there, so it's the main check from run 4 on.

| Run | Agent             | What changed                                                      | Dev (20)  | Root 999 (40) | Verdict |
| --- | ----------------- | ----------------------------------------------------------------- | --------- | ------------- | ------- |
| 1   | `bc_tiny`         | teacher A (the oracle's plan), flows only, noisy copies            | 0.491     |               | below heuristic |
| 2   | `bc_tiny_nonoise` | run 1 without the noisy copies                                     | 0.003     |               | much worse: noise stays |
| 3   | `bc_tiny_rel`     | + a release network for the strait                                 | 0.391     |               | no gain: investigated |
| 4   | `bc_tiny_v4`      | teacher C, "arriving at the strait" features, release floor        | 0.517     | 0.403         | better |
| 5   | `bc_tiny_dagger`  | run 4 + 3 DAgger rounds                                            | 0.238     |               | worse |
| 6   | `bc_tiny_v6`      | run 4 + 13 context numbers per route                               | 0.722     | 0.496         | better |
| 7   | `bc_tiny_v4_ship` | run 4, flow loss counts "too little" 4x                            | 0.781     | 0.636         | **biggest gain** |
| 8   | `bc_tiny_v8`      | run 6 + "too little" 4x                                            | **0.820** | 0.612         | best on dev only |
| 9   | `bc_tiny_v9`      | run 8 + networks 3 x 128 (were 2 x 64), 1,000 episodes (were 300)  | 0.772     | 0.671         | better |
| 10  | `bc_tiny_v10`     | run 9 + 10 own-state numbers per route                             | 0.715     | 0.671         | no change |
| 11  | `bc_tiny_v11`     | run 10 + routes deciding together (shared network)                 | 0.713     | 0.625         | worse |
| 12  | `bc_tiny_v12`     | run 9's recipe on 2,000 episodes: **the baseline**                 | 0.780     | **0.681**     | **final** |
|     | heuristic         | the kit's rule                                                     | 0.762     | 0.690         | |
|     | template          | send the maximum                                                   | 0.756     | 0.633         | |

### Lessons

1. **The oracle's own plan can't be copied.** Its linear program also decides when factories start work and how
   the grids share power, which no agent controls. That makes it a bound for the score, not a target. Replaying its
   decisions scores only 0.694 on dev, below heuristic. The teacher to copy is teacher C (0.805).
2. **Shipping too little is the main failure.** The teacher knows the future and ships just enough. Without the
   future, the copy comes up short whenever something goes wrong. Counting "too little" 4x in training gave the
   largest gain of any change (+0.23 on root 999).
3. **Noisy copies are essential** (run 2): trained only on the teacher's own path, the copy drifts off after one
   mistake.
4. **DAgger, as run here, hurts** (run 5): in the student's states the teacher, knowing nothing bad will happen,
   asks for even less.
5. **A closer copy isn't a better agent** (runs 5 and 11): a teacher who knows the future can be copied too
   faithfully.
6. **What helped:** context about the other routes and what lies ahead (run 6), a bigger network on more episodes
   (runs 9 and 12). What didn't: more inputs about the agent's own state (run 10), routes deciding together (run 11).
7. **Tiny has little room**: send-the-maximum already scores 0.63 to 0.76. The case for BC is on Small, where
   heuristic scores 0.40.

## Run log

Short notes per run; the run folders are the local `outputs/bc/<date_time>/`.

### Runs 1 to 3: teacher A, the oracle's plan

`train_bc.py`: per episode, solve the oracle's linear program once and replay its weekly decisions (300 episodes,
1 noisy copy with spread 0.2, MLP 2 x 64, 60 epochs).

- **Run 1** (`bc_tiny`, flows only): 0.491 on dev, heuristic minus 0.27 (-0.44 to -0.11). The network copied its
  teacher well (error 0.042 x capacity), but replaying the oracle's flows costs **+37 %** over the oracle, as much as
  naive. On 12 training episodes:

  | Played in the simulator                    | Cost vs the oracle |
  | ------------------------------------------ | ------------------ |
  | the oracle's flows only                    | +33 %              |
  | the oracle's flows and its tanker releases | **+5 %**           |
  | template (send the maximum)                | +17 %              |

  The oracle times when the tankers queued at the strait leave; run 1 always used the default release rule.
- **Run 2** (`bc_tiny_nonoise`, `--noisy_copies=0`): 0.003, minus 0.49 against run 1 (-0.94 to -0.25). This is
  covariate shift: without the noisy plays, one small mistake puts the network in states it never saw.
- **Run 3** (`bc_tiny_rel`, + a release network): 0.391, no clear difference from run 1 (-0.10, -0.28 to 0.07).
  Mixing BC and the oracle's replay on dev showed why:

  | Flows         | Releases      | Dev score |
  | ------------- | ------------- | --------- |
  | oracle replay | oracle replay | **0.694** |
  | oracle replay | default rule  | -0.135    |
  | BC            | oracle replay | 0.404     |
  | oracle replay | BC            | 0.441     |
  | BC            | BC            | 0.391     |

  1. Even a perfect copy (0.694) is below heuristic. The oracle's plan idles the fabs to save the grids' fuel. Fed
     the same flows, the simulator's fabs run at full speed, and on dev episode 4 `grid_eu` runs dry in week 8 and
     sheds power for the rest of the episode (about 850 M USD).
  2. 78 % of the oracle's releases happen when the queue the agent sees is empty: the cargo arrives and leaves in the
     same week, and the features didn't show arriving cargo.
  3. A release that is too small holds the cargo with no safety net (dev episode 5 alone: -22.95).

### The three teachers

| Teacher | What it is | A perfect copy scores (dev) | Cost on Tiny |
| ------- | ---------- | --------------------------- | ------------ |
| A, the oracle | one plan before the game; may also control factories and power | 0.694 | 1 LP per episode (0.05 s) |
| B, oracle with the game's rules | one plan before the game; the rules hold in week 1 only | 0.774 | the same |
| C, re-planning teacher | a new plan every week from the real state, true future, game's rules | **0.805** | 26 LPs per episode (1.7 s) |

Even teacher C loses about 0.9 bn USD to the oracle on dev episodes 4, 15 and 20: no agent can idle the fabs.

### Run 4: teacher C, arriving cargo, release floor

`bc_tiny_v4`, `train_replan.py` (300 episodes, 5 workers, 5.5 min to collect). Teacher C, three release features for
the cargo arriving at the strait (this week, next week, and this week by the group's route), release floor 0.5,
release loss "too little" 2x. Teacher +4.8 % over the oracle; release error 0.057 (run 3: 0.093). With the new
features, every week the teacher releases has a visible queue or arriving cargo behind it.

Dev 0.517, root 999 0.403; heuristic minus 0.25 on dev. Release floor checked afterwards on dev: 0 gives 0.492,
0.5 gives 0.517, 1.0 gives 0.548. Small and inside the noise, so the floor stayed at 0.5.

### Run 5: DAgger

`bc_tiny_dagger`: run 4's data, then 3 rounds where the student plays 100 new episodes and teacher C labels its
states (`--dagger_rounds=3 --dagger_episodes=100`, about 12 min). The dev score fell with every round: 0.517, 0.457,
0.323, **0.238** (minus 0.28 against run 4, -0.63 to -0.07). Over the 20 dev episodes:

| Agent          | Cargo sent per episode | Shortage, M USD | Power shed, M USD | Freight, tariff, holding, disposal, M USD |
| -------------- | ---------------------- | --------------- | ----------------- | ----------------------------------------- |
| heuristic      | 148,000                | 1,868           | 125               | 152                                       |
| bc_tiny_v4     | 32,000                 | 1,906           | 313               | 78                                        |
| bc_tiny_dagger | 22,600                 | 1,955           | 400               | 57                                        |

The student ships far less than heuristic, and DAgger made it ship even less. In the student's states the teacher
asks for less (mean flow label 0.123 x capacity in run 4's data, 0.094 to 0.104 in the DAgger rounds), because it
knows nothing bad will happen there. This led to run 7.

### Run 6: more context per route

`bc_tiny_v6`: run 4 plus 13 numbers per route (the other routes, what lies ahead, messages, the straits' history; see
"What it sees"). Dev 0.722, root 999 0.496: better than run 4 (dev +0.21, 0.07 to 0.37), still under
heuristic on fresh episodes (-0.195, -0.29 to -0.10). Same copy error as run 4: the gain comes from better-informed
decisions.

### Runs 7 and 8: ship more than the teacher

`--flow_under_weight=4` on the existing data (3 min each).

- **Run 7** (`bc_tiny_v4_ship`, run 4's data): dev 0.781, root 999 **0.636**. Over run 4: dev +0.26 (0.11 to 0.49),
  root 999 +0.23 (0.15 to 0.32), the largest gain of any change.
- **Run 8** (`bc_tiny_v8`, run 6's data): dev **0.820**, the first to beat heuristic on dev (+0.06, 0.03 to 0.10),
  but root 999 0.612: a tie with run 7 (-0.02) and below heuristic (-0.08, -0.14 to -0.00).

### Runs 9 to 12: bigger, more inputs, routes together, final

One change per run, measured mainly on root 999. The final run combines what helped and is confirmed once on root
2026.

| Run | Change | Dev | Root 999 | Against the run before, on root 999 |
| --- | ------ | --- | -------- | ----------------------------------- |
| 9 (`bc_tiny_v9`) | networks 3 x 128, 1,000 episodes, 7 workers (18 min to collect) | 0.772 | 0.671 | +0.06 (-0.00 to 0.12) over run 8: **helped** |
| 10 (`bc_tiny_v10`) | + 10 exact numbers per route: weeks of cover at the destination, cargo on its way there in 1, 2 and 4 weeks, work in progress at fabs, a grid's base load, last week's shortfall there, sanctions coming on the other routes to it | 0.715 | 0.671 | 0.00 (-0.05 to 0.05) over run 9: no change |
| 11 (`bc_tiny_v11`) | the flow network decides together: each route summarises its situation (16 numbers), then reads the mean summary of the routes with the same source, destination and good (`--model=shared`) | 0.713 | 0.625 | -0.05 (-0.10 to 0.00) against run 10: worse, though a closer copy (error 0.035, against 0.041) |
| 12 (`bc_tiny_v12`) | run 9's recipe on 2,000 episodes (run 9's 1,000 plus 1,000 new) | 0.780 | 0.681 | +0.01 over run 9; root 2026: +0.02 (0.00 to 0.04) |

Run 11 left out one proposed link: a release network that reads this week's flows. Cargo sent into the strait this
week arrives next week at the earliest, so it can't change this week's release.

By harm level (1 calm to 4 most harmful), the final model scores: dev 0.69 / 0.83 / 0.85 / 0.84, root 999
0.72 / 0.66 / 0.64 / 0.55, root 2026 0.72 / 0.85 / 0.64 / 0.93.

## Next

- **Small.** Start from v12's code (`agents/bc_tiny_v12/agent.py`; the same code as `bc_tiny_v9`). It
  already runs on Small: the features read the network from the config, and the release features handle Small's
  dense queue table. What's missing is teacher C's data on Small (52 weeks, bigger linear programs) and retraining.
  First, time one teacher C episode on Small to size the run.
- Tune "ship more" (2x, 4x, 8x) and the release floor on a root of our own, not on dev.
- DAgger again, with the ship-more loss or with the teacher's answers mixed into what is played (beta-DAgger).
