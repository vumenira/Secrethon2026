# How the environment works, and where the score is (2026-10-06)

A read of the `shockbench-flow` 0.1.2 source (simulator, cost, disruption generator, information layer, naive rule,
scoring), plus an EDA of the scenarios and of naive against the clairvoyant plan. Everything is on **Small**, the
public board's network, unless noted. Nothing was uploaded and no agent was trained.

**Bottom line.** Almost all of the money is in **disruptions that are already in force at week 1 and last the whole
episode** (sanctions, material outages, long strait closures). The clairvoyant plan's edge over naive is mostly
**re-planning around what is visible now**: getting about twice as many wafers into the fabs over routes naive never
uses, and keeping the Korean and Taiwanese grids fuelled. Forecasting new events matters much less. The warning scores
are close to useless; the messages are useful in a few specific ways.

## 1. How the environment works (from the source)

### One week, in order (`dynamics/sim.py`)

1. **Strait releases.** Cargo that reached a strait joins its queue. Your `override_qty` / hold is applied first,
   then the default release (oldest cargo first) up to the strait's throughput this week.
2. **Clip.** Your `flows` are cut to the mask, to each edge's remaining capacity (after strait releases), to stock on
   hand (shared pro rata over everything drawing on the same stock) and to the fleet limit.
3. **Dispatch.** Executed flows leave **last week's** stock and enter the pipeline (arrival = now + lead time).
4. **Arrivals** join stock at every node.
5. **Produce.**
   - Sources refill up to their storage. **Supply you do not ship is lost**, and goods cost nothing to buy.
   - Grids: each fuel feeds its **own segment** (`shares`), so LNG cannot be replaced by crude. LNG output is
     **rationed** when last week's LNG stock is below `psi * ibar` = 0.6 x `ibar`, which is about 0.85 to 1.2 weeks
     of LNG use. Base load is served before fabs (`base_first` on every Small grid).
   - Fabs start `min(capacity x restoration, wafers on hand)`. A fab runs whatever wafers it has, and chips come out
     6 to 8 weeks later. OSATs (packaging plants) package raw chips in 2 weeks.
6. **Serve** markets. On Small every market is **lost-sales** (no backlog): demand you miss this week is gone.
7. **Charge:** freight, war risk, tariff (rate x value), holding, queue holding, shortage, **disposal** (stock above
   storage at non-source nodes, 10 % of value per unit), power shed. At T the leftover stock is credited at salvage.

The simulator draws no random numbers. A scenario (omega) is drawn before the episode, and **nothing the agent does
changes the disruptions**.

### The economics (Small)

| Item | Value | So |
| ---- | ----- | -- |
| Power shed (VOLL) | 4.1 M USD per GWh, **100x** the value of LNG | never let a grid run out of a fuel |
| Unmet leading-edge chip (`chip_le`) | 50,440 USD (value 20,000) | the most valuable good to deliver |
| Unmet mature chip (`chip_mat`) | 10,360 USD | 5x less than leading edge |
| Holding | 16 USD per chip-week, 33 USD per GWh LNG-week | stock is almost free |
| Salvage at T | 3 to 18 USD per chip, 500 to 1,500 per GWh LNG | leftover stock is worth almost nothing |
| Storage | 4 weeks of demand at markets and of LNG use at grids; 104 weeks of nuclear fuel | buffers are capped: excess is thrown away at a cost |
| `sink_cn` | buys **no** `chip_le` | anything sent there is lost |

### The naive rule (`policies/naive.py`), score 0

A fixed **order-up-to** plan computed once from the nominal network: target = (lead time + 1) weeks of the route's
nominal flow, plus a closure buffer from the generator's queue statistics, plus the grid buffer. It never re-plans: it
keeps its reset-time routes, drops a route only when it is observed closed or sanctioned (switching to that route's
listed duplicate), and stops shipping goods that could not reach a market by T. It ignores warnings and messages.

### The score (`scoring/rss.py`)

Per harm stratum s: RSS_s = sum(naive - agent) / sum(naive - oracle), in USD. The board combines strata as
`sum_s p_s mean(g_s) / sum_s p_s mean(D_s)` with p = 50 / 30 / 15 / 5 %. It is a **ratio of sums**: an episode
counts in proportion to its naive-oracle gap D, not equally. D is similar across strata (0.8 to 1.5 bn USD on
average), so the effective weights stay close to the nominal ones: 45 / 33 / 18 / 4 % on dev, 48 / 26 / 20 / 5 % on
root 999. **Calm episodes carry about half the score.**

### What the agent sees (`information/theta.py`, regime `standard`)

- The **whole network as it is this week**: capacities, open fractions, sanctions, tariffs, supply, fab and grid
  health. No blackouts, no coverage limit.
- **No `closure_end`** (`chi = False`): you are not told when a strait reopens.
- `warning.score` = 0.604 X + 0.797 W, one week late, where X is the latent risk (an AR(1)) that tilts the hazard of
  military events and conflicts, and W is AR(1) noise.
- `messages`: tariff proposals (formal or informal) then final notices; sanction threats (TIES) then legal
  publications; military threats (MID) against a strait. Decoys are shadow threads: a decoy is withdrawn at its
  effect date and never acts. Target false shares: 25 / 36 / 54 / 30 % (formal, informal, TIES, MID).
- `pending_prohibitions`: the pairs of every published sanction not yet in force, with its effective week. Decoy
  publications appear here too.
- `demand_forecast`: 8 weeks ahead. It includes a term for **last week's conflict state** of the market's region, so
  forecast jumps leak a little hidden state.

## 2. EDA results

Scripts in this folder; numbers from 120 Small episodes of root 12345 (scenarios only) and 6 to 10 dev episodes
(naive played, oracle solved).

### 2.1 The disruptions: mostly there from week 1, and they stay

Per episode, Small (`env_eda.py`, `events.csv`):

| Type | Carried in at week 1 | Starting in the episode | Median length of in-episode ones | In-episode ones lasting to T | Share of generator harm |
| ---- | -------------------- | ----------------------- | -------------------------------- | ---------------------------- | ----------------------- |
| sanction | 20.4 | 3.5 | 95 weeks | 78 % | **84 %** |
| militarised closure | 0.66 | 0.42 | 38 weeks | 65 % | 6 % |
| material outage (wafer supplier) | 1.0 | 0.27 | 93 weeks | 66 % | 3 % |
| regional conflict | 0.93 | 0.20 | 47 weeks | 67 % | 3 % |
| tariff | 5.6 | 3.2 | 47 weeks | 73 % | 2 % |
| port strike | 0.10 | 1.15 | 2.4 weeks | 9 % | 1 % |
| weather closure | 0.03 | 1.26 | 0.8 weeks | 2 % | 0.2 % |
| energy shock (grid) | 0.12 | 1.25 | 1.6 weeks | 7 % | 0.1 % |
| piracy | 0.07 | 0.31 | 12 weeks | 46 % | 0 % |

- **About 90 % of harm comes from events already in force at week 1** (93 % for sanctions, 81 to 90 % for closures,
  outages and conflicts), and most of them outlast the episode. "What I see now will last" is right most of the time.
- Events cluster (a self-exciting Hawkes process): 2.3 events start in the 4 weeks after an event, against 0.9 in any
  4 weeks. Part of this is several targets of one sanction package, so read it as "an onset raises near-term risk".
- New events start evenly across the episode (372 / 333 / 344 / 342 per quarter over 120 episodes).
- The sanctions that matter most hit fuel routes: `bypass.tb.src_gulf_crude.chk_malacca`, `src_ru_gas -> term_jp`,
  `src_us_crude -> chk_panama`, `src_au_lng -> term_kr / term_jp`, `src_kz_uranium -> grid_eu`.

### 2.2 Straits (`signals.py`)

About 2 impaired spells per episode. They come in two kinds:

| | |
| --- | --- |
| Short (1 to 2 weeks) | 47 % of spells: weather. **Cape, Panama, Turkish only ever have these** (median 2 weeks, none over 8) |
| Long (to T) | 30 % of spells last to the end; 29 % already run in week 1. Malacca (median 13.5 weeks), Suez, Taiwan, Hormuz |
| How closed | mean open fraction 0.38 to 0.54 inside a spell: usually partly open, not shut |

**Every one of the 51 in-episode militarised closures was announced by a real MID threat** against that strait. The
lead is short: median 1 week, 10th percentile 0 (shown the week it starts), 90th percentile 9.7 weeks; 51 % have at
least one week. MID threads: 0.53 per episode, 20 % decoys.

### 2.3 Messages (`env_eda.py`)

| Thread | Per episode | Decoys | Lead before the effect (median, p10 to p90) |
| ------ | ----------- | ------ | ------------------------------------------- |
| sanction: TIES threat that **never** gets a publication | 1.5 | **100 %** | |
| sanction: TIES threat **then** legal publication | 3.4 | **3.5 %** | threat 3.2 weeks (0 to 90) |
| sanction: legal publication with no earlier threat | 0.48 | 40 % | publication 0 weeks (-0.4 to 4.3) |
| tariff, formal proposal | 2.4 | 27 % | 9.8 weeks (3.0 to 33) |
| tariff, informal proposal | 2.1 | 31 % | 3.2 weeks (0.4 to 14) |
| tariff final notice | | does **not** confirm (27 to 30 % decoys remain) | 0.9 weeks |
| MID threat (strait) | 0.53 | 20 % | 1.4 weeks (0 to 9.6) |

So a sanction publication usually arrives at the onset (lead 0): `pending_prohibitions` gives little time. The TIES
threat gives a few weeks, and a threat that is later published is almost surely real.

### 2.4 Warnings and forecast

| Signal | Target | AUROC |
| ------ | ------ | ----- |
| strait warning | a militarised closure of that strait within 8 weeks | 0.58 (base rate 0.7 %) |
| strait warning | any closure within 1 to 8 weeks | 0.54 |
| region warning | any event in the region within 1 to 8 weeks | 0.52 |
| region warning | region in conflict (minor or war) now | 0.66 |
| dyad warning | a military event in either region | 0.49 |

The warning correlates 0.59 with the latent risk, as designed, but the risk only tilts a small part of the events.
**For timing events, warnings are close to noise.**

Demand forecast: median absolute error 7 % at every horizon from 0 to 7 weeks; demand spans 0.83 to 1.20 x nominal
(5th to 95th percentile).

### 2.5 Where the oracle beats naive (`cost_decomp.py`, `oracle_vs_naive.py`, dev episodes)

Mean over 6 dev episodes, bn USD:

| Component | Naive | Oracle | Gap | Same episodes, disruptions removed: naive / oracle |
| --------- | ----- | ------ | --- | -------------------------------------------------- |
| power shed | 1,981 | 1,437 | **543** | 0 / 0 |
| shortage | 1,146 | 766 | **379** | 59 / 5 |
| everything else | 21 | 14 | 8 | 10 / 7 |
| total | 3,148 | 2,217 | **931** | 70 / 11 |

- **Without disruptions both cost almost nothing** (70 vs 11 bn). The gap is how disruptions are handled.
- Even the oracle loses 2,200 bn: most of the damage cannot be avoided, and the score measures only the avoidable part.

Gap by place (10 dev episodes, bn USD per episode):

| Where | Naive | Oracle | Gap |
| ----- | ----- | ------ | --- |
| `sink_us` chip_le shortage | 628 | 353 | **276** |
| `grid_kr` shed | 359 | 163 | **196** |
| `grid_tw` shed | 255 | 97 | **158** |
| `grid_eu` shed | 849 | 704 | 145 |
| `grid_jp` shed | 713 | 615 | 99 |
| `sink_eu` chip_le shortage | 252 | 200 | 53 |
| `sink_jp` chip_le shortage | 195 | 162 | 33 |
| every chip_mat shortage together | 107 | 80 | 27 |

What the oracle does differently:

- **Starts about twice as many fab lots**: 10.0 M wafers per episode against naive's 4.8 M. Korean memory fab 4.1 M
  vs 1.1 M, Taiwan leading 2.0 M vs 0.4 M, Japan memory 0.9 M vs 0.4 M. EU mature gets *fewer* (1.7 M vs 2.1 M):
  scarce wafers go to leading-edge chips.
- **Fabs are far below capacity even for the oracle** (10 M of 47 M a year). The bottleneck is **moving wafers**,
  not fab capacity or wafer supply: `mat_jp_wafer` has 18.8 M available, naive lifts 2.7 M, the oracle 4.1 M.
- **It uses routes naive never touches**: `lombok.*` (around Malacca), `cape.*` (around Suez), `east.*`, and **air**
  for raw chips (`air.ct.fab_kr_memory_1.osat_my`, `air.ct.fab_jp_memory_1.osat_my`). Then it flies about 2x more
  finished chip_le from the OSATs to `sink_us`.
- **The gap grows through the episode**: chip shortage gap by quarter 6 / 65 / 147 / 169 bn. The chip pipeline is
  long (fab 6 to 8 weeks, OSAT 2 weeks, plus transit), so lost wafer weeks early show up as lost sales 10+ weeks later.
- Shed: the oracle lifts more LNG from `src_au_lng` and `src_us_lng`. Korea and Taiwan have the biggest relative
  gaps (55 to 62 %).
- Naive also floods `grid_eu` with Kazakh nuclear fuel (420 k vs 47 k units): wasteful but cheap (holding only).
  Not where the money is.

## 3. Hypotheses, ranked by expected gain

1. **Re-plan on what you see, assuming it persists.** About 90 % of harm is visible at week 1 and lasts. A weekly
   re-plan on the observed network (the package's `mpc_det` idea, or rules) should capture most of the gap without
   any forecasting. Teacher C (true future) scored 0.80 on Small. **Test first:** `agents/mpc_det_h20` has never been
   scored on Small. Run `uv run sbf compare mpc_det_h20 3rule_based --task=small` and time it with `sbf check`.
2. **Maximise fab starts on leading-edge fabs.** Move wafers over every usable route, alternative lanes and air
   included, to `fab_kr_memory_1` and `fab_tw_leading_1` first (chip_le: 50 k per missed unit against 10 k for
   chip_mat). Then fly finished chip_le to `sink_us` first. This is the largest single gap (`sink_us` chip_le alone
   is 276 bn of 931). Measure: lots started per fab against the oracle's numbers above.
3. **Keep each grid's LNG stock high**, near its 4-week storage and never below 0.6 x `ibar`, where rationing starts.
   This matters most at `grid_kr` and `grid_tw`. Energy shocks (1.25 per episode, 1.6 weeks, unannounced) and
   weather closures (1 to 2 weeks) can only be met with stock, and stock is almost free (33 USD per GWh-week against
   4.1 M per GWh shed).
4. **Tell long closures from short ones**:
   - A closure at Cape, Panama or Turkish, with no MID threat: weather. Expect it to reopen within 1 to 3 weeks:
     keep the route and lean on buffers.
   - A closure that follows a MID threat against that strait, or is already running in week 1 at Malacca, Taiwan,
     Suez or Hormuz: treat it as permanent (median 38 weeks, 65 % last to T). Re-route at once to `lombok` / `cape`
     / air.
   - A MID threat (80 % real, half have at least one week of lead): top up the stock beyond that strait and start
     the bypass now.
5. **Read sanctions by thread**:
   - A TIES threat with a publication after it: treat the ban as certain (96.5 %), and stop sending cargo whose lane
     will cross the banned edge after its effective week. Otherwise the cargo waits at the strait.
   - A threat with no publication yet: about 2/3 real at first, falling the longer no publication comes. Hedge, do
     not commit.
   - A publication with no earlier threat: 40 % decoys, so it is weaker evidence.
6. **Don't spend effort on warning scores.** AUROC 0.52 to 0.58 for event timing. At most use them as a small prior
   on strait risk, or region warning as a hint of conflict (0.66).
7. **Tariffs and strait releases are small money** on Small: tariff harm 2 %, queue holding 2.6 vs 0.2 bn. Simple
   rules are enough: avoid tariffed routes when an untaxed one has capacity, and keep the default release unless a
   rule clearly wins.
8. **End of episode:** wafers shipped after about week T - 12 can't become sold chips, and leftover stock is worth
   almost nothing. Only freight is at stake, so this is small.
9. **For learned agents** (BC, RL, forecast + LP): add per-route features "impaired since week 1", "MID threat
   against a strait on this route", "TIES threat / publication on this edge", and capacity on alternative lanes. A
   learned disruption forecast (hypothesis 5 in [../hypotheses.md](../hypotheses.md)) works on about 10 % of the
   harm. Persistence plus the rules of 4 and 5 may get most of that for free.

## 4. Other useful facts

- **No purchase cost.** Goods from sources are free; the cost of shipping more is freight, tariff, holding, and
  disposal above storage. Unshipped source supply is lost every week. Over-shipping is cheap unless it overflows
  storage or goes where nobody buys it.
- **Dispatch draws on last week's stock**, so cargo arriving this week can only leave next week (one week per hop).
- **Lanes through a strait** wait in the strait's queue when it is partly closed, and pay queue holding (2,000 to
  7,200 USD per unit-week for tanker fuel at war-risk classes). The default release is oldest first.
- **Dev episodes are 5 per stratum**; the board weights strata 50 / 30 / 15 / 5. `sbf evaluate` already applies the
  weights. Episode-level numbers are noisy: an unweighted mean over dev is not the score.
- Small and Full share the generator family. Full has 104 weeks, so persistent events weigh even more there
  (not measured here).
- The package ships its own baselines (`policies/`): `mpc_det` (rolling LP, today's network persists), `mpc_scen`
  (LP over sampled scenarios of the public generator), `greedy_lp`, `sz_state_base_stock`, `human_ref`. They need
  `highspy` and the package, which the server lacks, but their logic can be ported (SciPy's `linprog` exists there).
- Seeing the hidden truth for training: `scenario(task, n, entropy)` gives the omega, `compute_marks` the true weekly
  network, `build_view(..., theta)` the agent's view, and `messages[i].decoy` the truth about each message. That is
  free labels for any forecasting model.

## Reproduce

```bash
uv run python experiments/eda/env_eda.py                 # 120 Small episodes of root 12345: events, messages, warnings (2 min, 4 workers)
uv run python experiments/eda/signals.py                 # closure spells, MID threats, what warnings track (2 min)
uv run python experiments/eda/cost_decomp.py             # naive vs oracle by cost component, with and without disruptions (1 min)
uv run python experiments/eda/oracle_vs_naive.py         # per grid, market, fab, supply node and route (1 min)
```

`env_eda.py` and `cost_decomp.py` write to `outputs/eda/`. The runs above:
`outputs/eda/small_2026-10-06_12-45-05/` and `outputs/eda/costs_small_2026-10-06_12-48-49/`. The cost scripts read
naive's reset-time table from the kit's cache (`~/.cache/shockbench-flow`); without it the first run computes it
(slow).

Limits: the cost comparisons use 6 to 10 dev episodes (noisy, and dev was looked at before); the scenario statistics
use 120 episodes of root 12345. Full was not analysed.
