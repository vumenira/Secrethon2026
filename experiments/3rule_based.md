# 3rule_based: ship what is needed, on the best open route (2026-10-04)

`agents/3rule_based/agent.py`. A hand-written rule agent, no training, no weights. Instead of sending the maximum on
every route, every week it works out what each place needs and ships that, on the best route still open, and holds
back plants whose products cannot leave. Local runs only; nothing was uploaded.

## Results

|                                    | Small                                     | Full                         | Tiny                  |
| ---------------------------------- | ----------------------------------------- | ---------------------------- | --------------------- |
| Score, 20 dev episodes             | **0.5510** (90 %: 0.46 to 0.63)           | not run (references too slow) | 0.8199               |
| Score by harm level 1 to 4         | 0.50, 0.58, 0.62, 0.59                    |                              |                       |
| vs heuristic (paired, dev)         | **+0.151** (90 %: +0.104 to +0.202)       |                              |                       |
| vs template (paired, dev)          | **+0.142** (90 %: +0.090 to +0.204)       |                              | +0.064 (-0.03 to +0.16, holds 0) |
| Mean cost per episode, dev, bn USD | agent 2,950; naive 3,485; clairvoyant 2,555 |                            |                       |
| Fresh root 12345, mean cost, bn USD | 2,989 vs heuristic 3,119 (wins 12 of 12) | 7,080 vs heuristic 7,648 (wins 4 of 4) |          |
| Weeks played by the naive rule     | 0 of 1,040                                |                              |                       |
| `sbf check`: first / median week   | 0.015 / 0.0024 s of 2 s                   | 0.048 / 0.011 s of 4 s       |                       |

The heuristic agent's baseline is 0.4004 on Small (experiments/heuristic_agent/README.md).

How much to trust it: the rules were designed while looking at dev episodes 0, 5, 10 and 15, so the dev score may be
slightly optimistic. The fresh-root rows are the check against that: episodes of root 12345 that were never looked at,
compared on raw cost (no clairvoyant plan needed), where the agent beats the heuristic in every episode, on Small and on
Full.

## What the cost is made of (Small, dev episodes 0, 5, 10, 15, send-the-maximum)

| Component                 | bn USD per episode | Share  |
| ------------------------- | ------------------ | ------ |
| Grid power shed           | 1,863              | 60 %   |
| Unserved chip demand      | 1,172              | 38 %   |
| Everything else together (freight, tariffs, holding, queues, disposal) | 55 | 2 % |

Missing a leading-edge chip costs 50,440 USD; holding one costs 16 USD a week. Freight is a rounding error. So the game
is: keep the grids fuelled and get chips to the markets; stock is cheap, routes are not worth saving money on.

The clairvoyant plan sheds about as much power as anyone on these episodes (much of the shed is unavoidable); its real
edge is chips: it loses fewer of them while starting half as many fab lots, because it sends them where they can be
sold.

## How it works

Every week, for every node that holds goods (sources, terminals, fabs, OSATs), the agent splits its stock between the
places that stock can go.

1. **Need.** Each destination (node, good) has a target: `cover` weeks of its use, at most `fill` of its storage, plus
   what it uses while a shipment travels. Its position is its stock plus everything already on its way (the pipeline
   and the cargo queued at straits). Need = target - position. Use per week comes from the observation: the demand
   forecast at markets, a grid's fuel share of its generation, a fab's effective capacity, otherwise the normal plan's
   inflow. A market that does not buy a good needs none of it.
2. **Route.** Routes to a destination are tried best first, ranked by freight + tariff + `lead_cost` per week of lead
   time + `open_penalty` for a partly closed strait. A route carries at most the smallest free capacity along all its
   edges, after the cargo already queued at a strait (that cargo leaves first), times its straits' open fraction. A
   route is skipped when a strait on it is below `min_open`, or when an announced sanction will block one of its edges
   before the cargo arrives (`pending_prohibitions`).
3. **Backpressure.** A plant is fed only what its product can leave by: an OSAT's raw chips, then a fab's wafers, are
   capped by the capacity of the plant's usable routes to places that use its product, and cut to 0 as its product
   piles up (from `pile` to 2 `pile` weeks of that exit). An OSAT's throughput goes to its dearest chip first, because
   packaging splits it pro rata over the raw stock on hand.
4. **Push.** What is left after the needs is pushed (`push`) towards destinations with free storage, so sources, whose
   unshipped supply is lost, never sit on stock.
5. **Warnings** (`warn_level`, `warn_boost`): a destination whose route crosses a strait or region with a high warning
   score gets more cover. It currently changes nothing (see below).

`release_mode` and `override_qty` are left at the default release.

## How it was built: the mistakes that each rule fixes

Each step was measured on dev episodes 0, 5, 10, 15 (mean cost, bn USD; template 3,087, heuristic 3,083).

| Version                                                  | Mean cost | What went wrong / what it fixed |
| -------------------------------------------------------- | --------- | ------------------------------- |
| Need + route, first edge's capacity only                 | 3,375     | Worse: it filled lanes whose later legs were cut (Suez to the EU at 41 of 2,668), and the cargo queued |
| Route capacity = the smallest along all its edges        | 3,096     | About level with the template |
| Partly open straits used, ranked lower (`min_open` 0.5 to 0.05) | 3,015 | At 0.5 it stopped every Qatar LNG route when Hormuz was 35 % open (no alternative): Taiwan and Korea grids starved |
| Backpressure on plants                                   | 2,968     | 4 million finished chips were thrown away at osat_my (its exits blocked) while markets starved |

Send-the-maximum's own big mistake: it sends up to 258,000 leading-edge chips a week to `sink_cn`, which buys none
(storage 1 unit): they are thrown away, and the stock they took is missing at the US, EU and Japan markets.

## Parameters

In `PARAMS` at the top of `agent.py`; a `params.json` beside it overrides any of them.

| Parameter      | Default | Meaning | Sweep on root 12345, 12 Small episodes (vs default) |
| -------------- | ------- | ------- | --------------------------------------------------- |
| `cover`        | 6       | weeks of use a destination aims to hold | 3 and 10: within 2 bn (storage caps bind first) |
| `fill`         | 0.95    | share of storage it may fill | 0.8: 4 bn better (noise level) |
| `min_open`     | 0.05    | below this open fraction a route is not used | 0.3: 1 bn worse |
| `open_penalty` | 0.5     | ranking charge for a closed strait, share of the good's value | not swept |
| `open_power`   | 1.0     | capacity scale by open fraction | not swept |
| `lead_cost`    | 0.02    | ranking charge per week of lead time, share of value | not swept |
| `push`         | 1.0     | share of leftover stock pushed downstream | 0: 4 bn worse |
| `pile`         | 2.0     | weeks of product a plant may pile before its input is cut | 1: 40 bn worse; 3: 4 bn better; 5: 2 bn worse |
| `warn_level`   | 0.5     | warning score that counts | 0.2 with boost 2: no change |
| `warn_boost`   | 1.0     | extra cover at a full warning | 0: no change |

Differences under about 10 bn on 12 episodes are noise. `pile` is the only number that clearly matters so far.

## How to tune it

Tune on a root of your own, never on the dev episodes, and keep the dev episodes for the final `sbf compare`.

1. Write the numbers to try in `agents/3rule_based/params.json`, e.g. `{"pile": 3.0, "fill": 0.8}`.
2. Score on your own root:
   `uv run sbf evaluate 3rule_based --task=small --entropy=12345 --episodes=16`. The first run on a new root builds
   that root's references (the clairvoyant plan per episode, a few seconds each on Small), and uses much of the
   memory: run one job at a time.
3. Cheaper, without references: compare raw episode costs against the heuristic on the same episodes of a root
   (lower is better). That is how the sweep above was made: `gym.make("ShockBench/Small-v0", entropy=12345)`, play
   episodes 0 to 11 with each variant (set `PARAMS` before building `Agent(config)`), sum
   `obs["last_week.cost_components"]` over the weeks and subtract `info["salvage_cents"] / 100` at the end.
4. Confirm a winner on the dev episodes: `uv run sbf compare 3rule_based heuristic --task=small` (and against the
   previous version of this agent). Keep a change only if the paired interval stays above 0.
5. For an automatic search, copy `examples/06_policy_search.py`: it already writes `params.json` candidates, but for
   the heuristic agent's numbers; point its `HEURISTIC` at this `agent.py` and replace `start_params` and `mutate` with
   this agent's parameters.

## How to make it better

Where the money still goes, in order of size:

- **Chips (the largest gap to the clairvoyant plan).** Rank destinations by the shortage penalty (`sinks.pi`:
  leading-edge 50k, mature 10k) when stock is scarce, not by weeks of cover. Feed fabs whose product reaches markets
  with high penalties first. Check where chips still get thrown away (`disposal`) or lost per market.
- **Warnings that act.** The warning rule changes nothing today because storage caps cap the extra cover. Better use of
  `warning.score`, `messages.*` and `closure_end.*`: before a strait closes, shift traffic to the alternative routes
  early, and push stock past the strait (to terminals and grids behind it) instead of only raising cover.
- **Strait releases.** `release_mode` / `override_qty` are unused: holding tanker cargo when its exit is blocked, or
  releasing to the fuller route, could cut queue costs and reroute fuel.
- **Grid shed.** Largely unavoidable on the dev episodes, but in episode 15 the rules shed 158 bn against the oracle's
  139: compare fuel targets at terminals with the grids' rationing buffer (`ibar`, `psi`) and the nuclear fuel's
  8-week pipeline.
- **A planner.** The need-and-route logic is a greedy version of what an LP over the next few weeks would do;
  approach 3 (MPC with `linprog`) can reuse its tables (routes, rates, positions, capacities).

## How it was run

```
uv run sbf compare 3rule_based heuristic --task=small
uv run sbf compare 3rule_based template --task=small
uv run sbf evaluate 3rule_based --task=small
uv run sbf compare 3rule_based template          # Tiny
uv run sbf check 3rule_based --task=small
uv run sbf check 3rule_based --task=full
```

The cost breakdowns, the comparison with the clairvoyant plan and the parameter sweep came from throwaway scripts (not
in the repository): the clairvoyant plan of an episode is `build_lp(inst, marks)` and `solve_oracle` from
`shockbench_flow.oracle.lp`, on the episode the environment holds after a reset (`env.unwrapped.core._ep`).
