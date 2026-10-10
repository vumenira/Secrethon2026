# The pulse agent: every version and measurement (2026-10-06 to 2026-10-10)

This is the detailed log behind our agent. For the short version, read [SOLUTION.md](../../SOLUTION.md) first.

- **The final agent is `agents/mpc_pulse_rec_v13`** (0.8972 on the public board; 0.918 on 64 untouched Full
  episodes. The agent before it, `agents/mpc_pulse_rec_v12`, has 0.8910 and 0.916). Every earlier version is
  kept, runnable, under `versions/<name>` beside this file
  (`experiments/pulse_agent/versions/`) or, for the last few, in `agents/`; a version named below without a path is
  one of the `versions/` folders.
- **Part one** (down to "Risks and limits") describes the first version, `versions/ft_buffer`: the benchmark's rolling LP
  planner with a value for goods at the window's end, a fuel reserve at grids, a "pulse" that delivers scarce fuel in
  whole weeks so fabs can run, and a price that sends scarce fuel to the grid that loses most. It reads no warnings and
  no announcements and has no learned parts. Uploaded on 2026-10-06, it scored 0.8289.
- **Part two** ("Later versions") adds one entry per version, v2 to v13, each with what it changed, what was measured
  and what was dropped, and the analyses of 2026-10-09 and 2026-10-10 behind v11, v12 and v13.
- **The code explains itself** in module docstrings: `agent.py` (overview, every setting commented in `PARAMS`),
  `pulse.py` (the pulse and the rules after it), `reserve.py` (the LP changes, with formulas), `recovery.py` (the
  forecast).
- **The scripts behind the analyses** are in `experiments/pulse_agent/analysis/`, the harness in
  `experiments/approaches/` (`tune.py`, `foresight.py`).

## Scores of the first version

Local scores, every agent on the same episodes. Roots 12345 were used for tuning; roots 2026 were played only to
confirm. Small sets have 64 episodes, Full sets 16.

| Agent | Small 12345 | Small 2026 | Full 12345 | Full 2026 |
| --- | --- | --- | --- | --- |
| `mpc_det_h20` (the package's planner, 20-week window) | 0.6856 | 0.6692 | 0.3290 | 0.2899 |
| `mpc_buffer`, end value alone | 0.7508 | 0.7434 | 0.6607 | 0.6187 |
| `mpc_combined`, end value + recovery forecast | 0.7623 | 0.7568 | 0.6738 | 0.6329 |
| **`ft_buffer` (the first upload)** | 0.8247 | 0.8214 | 0.8358 | 0.8567 |

- **Fresh episodes confirm it.** Against `mpc_combined`, `ft_buffer` gains +0.065 on Small 2026 (90 % paired interval
  +0.045 to +0.084, 47 episodes better / 17 worse) and +0.224 on Full 2026 (+0.170 to +0.274, 15 / 1).
- **No week fell back** to the naive rule and no error occurred in any of these plays.
- **CPU:** `sbf check ft_buffer --task=full` under Ubuntu: slowest week 1.77 s of the 4 s limit, median 0.47 s (one dev
  episode, machine loaded). Against `mpc_det_h20` on Full the mean week costs 1.05 times as much and the slowest 1.28.

## How it works

### 1. The base: a rolling LP that plans on "nothing changes"

Each week the planner (`sbflow/`, the package's `mpc_det`, unchanged) assumes the network stays as observed and demand
follows the forecast, solves a linear program for the cheapest flows over a 20-week window, and sends the first week's.
It has three blind spots, and each of the changes below answers one:

- It prices whatever is left when the window ends at the end-of-episode salvage price.
- It holds no fuel against events that have not started.
- From the window's second week it plans to shed base load to power a fab, which the simulator never does.

### 2. End value (`end_value` 2.0, in `reserve.py`)

While the window ends before the episode does, goods in stock, in transit or in process at the window's end are
credited a share of their customs value instead of the salvage price. Without it the planner stops caring about
anything that pays off after week 20.

- **It is the largest single change on the plain planner:** +0.065 on Small and +0.33 on Full (root 12345), +0.074 and
  +0.33 on the fresh roots, better in every one of the 32 Full episodes.
- **Full gains more.** A likely reason, not tested: the window covers a fifth of its 104 weeks, against two fifths of
  Small's 52, so far more of the episode lies beyond it.
- **Level:** 0.5 loses 0.019 against 1.0 on Full; 2.0 gains 0.003; 3.0 gains nothing. On Small 1.0 and 2.0 are equal.
- **A longer window is no substitute.** With the end value on, 24 weeks is no better than 20 on Small and worse on
  Full (-0.009 next to the pulse).

### 3. Fuel reserve and ramp (`reserve_weeks` 8, `reserve_feeders`, `shed_ramp` 0.002)

- **Reserve:** a soft minimum on the fuel each grid holds, in weeks of its own burn, priced at 0.001 of the grid's value
  of lost load per unit-week missing. With `reserve_feeders` the cover the grid's storage cannot hold is asked of the
  terminal that feeds it.
- **Ramp:** shed base load costs slightly more the earlier in the window it happens, so of two plans that shed the same
  the planner takes the one that sheds later. It also stops the reserve from paying for early shedding.
- **Worth:** next to the pulse on Full, removing both loses 0.014 (interval -0.021 to -0.008). On Small without the
  pulse the reserve was a coin flip.

### 4. The pulse (`pulse.py`)

**The problem.** At every grid with fabs, nominal output exceeds base load by exactly what the fabs draw (0.2 % of
output at one Full grid, 9 % at another). The simulator serves base load first. So a fuel that is a few per cent short
takes all the fabs' energy, every week the shortage lasts. On Full, after week 13, the planner with end value and
reserve starts lots at 0.14 to 0.29 of the fabs' capacity; the clairvoyant plan runs them at 0.49 to 0.61, by shedding
base load to power them, which no agent can do directly.

**The idea.** Do over time what the clairvoyant plan does within a week. Hold the short fuel at the terminal for some
weeks (the grid sheds that fuel's share of output) and deliver a full week's burn in others (no shedding, the fabs
run). The energy burnt is the same. Base load shed rises by exactly the energy the fabs get, and that energy is worth
3 to 9 times as much in chips.

**The weekly rule,** for each grid with fabs, after the planner has chosen its flows:

- **ON week:** the grid can serve base load plus the fabs' draw. While the grid is being pulsed, a fuel the terminal
  cannot keep supplying is delivered as needed and no more.
- **OFF week:** it cannot. A pulse starts only if the fabs would get less than half their draw as things are
  (`pulse_partial` 0.5); a mild shortage that leaves them most of it is left alone. The short fuel is then held at the
  terminal. The rationed fuel (LNG) gets one delivery to lift the grid's stock to the rationing threshold, once the
  terminal holds that plus four whole weeks (`pulse_run` 4).
- **Nothing is held** when the short fuel has no terminal (it comes by pipe or from a source), when its tank would take
  more than 12 weeks to fill (`pulse_hold_weeks`), or when 12 whole weeks of the grid's fabs would not pay for a tank
  stranded when the shortage ends (`pulse_payback`).

**What it touches.** Only the quantities on terminal-to-grid flows the planner already sends; never more than the
terminal's stock or the edge's observed capacity. It reads the week's observation and the planner's memory of the
observed network, not the LP or its window.

**Effect on Full (root 12345).** Fab energy rises from 36,200 to 64,400 GWh per episode (clairvoyant 90,000). As shares
of the naive-to-clairvoyant gap: unserved demand -0.203, base load shed +0.042. The first, untuned pulse already gave
+0.137 over the planner with end value and reserve; the uploaded settings give +0.160.

### 5. Shortfall premium (`short_premium` 1.0, in `reserve.py`)

The planner prices a fuel shortfall the same at every grid. The premium prices it by the chips it costs under the
pulse, so scarce fuel goes to the grid that loses most. Worth +0.010 on Full (interval touches 0) and +0.003 on Small.

## The settings of the first version (`versions/ft_buffer/params.json`)

| Setting | Value | Meaning |
| --- | --- | --- |
| `h_weeks` | 20 | the planning window |
| `end_value` | 2.0 | share of customs value credited at the window's end |
| `reserve_weeks`, `reserve_feeders` | 8.0, true | weeks of burn a grid should hold, the excess at its terminal |
| `reserve_price`, `reserve_fuels` | 0.001, "all" | price of missing reserve (share of the value of lost load); which fuels |
| `shed_ramp` | 0.002 | how much dearer shedding is per window week earlier |
| `pulse` | true | the pulse is on |
| `pulse_run`, `pulse_hold_weeks` | 4, 12 | LNG is charged when the terminal holds this many whole weeks; slowest tank worth holding |
| `pulse_partial`, `pulse_payback` | 0.5, 12 | a pulse starts when fabs get under this share; weeks of fabs that must pay for a stranded tank |
| `short_premium` | 1.0 | share of the lost chips' value priced on a fuel shortfall |

Nothing reads which task is played; the same settings serve Small and Full.

## What was tried and left out

| Change | Result |
| --- | --- |
| Base load before fabs priced in every window week of the LP | -0.02 to -0.04 on Full: it prices unavoidable shedding far above its cost |
| No end value for fuel at grids (against nuclear hoarding) | -0.18 on Full: the planner runs nuclear stock down |
| Window 24 with the pulse | -0.009 on Full |
| Window 24 to 40 without the end value | +0.031 to +0.040 on Small, but no gain once the end value is on |
| Demand scale 1.05 to 1.2, throughput scale 0.8 to 0.9 | +0.003 at most; the throughput scale loses |
| Reserve price and ramp ten times higher; lots valued as chips at the window's end | -0.001 at most on Full |

## Why the signal-based approaches were dropped

A separate analysis fed the planner the scenario's ground truth (Small, root 12345, old 20-week planner):

- **Perfect early warning is worth almost nothing:** every event known two weeks ahead with its true duration adds
  +0.003; a warning without a duration makes the planner worse.
- **Decoy sanctions and true demand:** +0.0004 and -0.0002, both within noise.
- **Even full knowledge to the episode's end reached only 0.773.** The cause, traced in three episodes and estimated
  for the rest: the clairvoyant plan powers fabs ahead of base load. That is the gap the pulse goes after, and why it
  beats the full-knowledge planner.

So approaches 2, 4, 6, 7 and 8 of `experiments/approaches/README.md` were not built.

## Risks and limits

- **Tuned on few Full episodes.** Four pulse settings were chosen on the 16 episodes of Full root 12345, two of them
  after looking at single losing episodes. Together they are worth about 0.006, which may be optimistic.
- **Where it loses.** On Small root 12345 it loses in 12 of 64 episodes against the planner without the pulse (worst
  -0.069), for a total of 218 bn USD against gains of 4,883 bn. On Full it loses in 1 of 16 (-0.002).
- **Known causes of losses:** chips made too late to be sold before the episode ends (fixed in v2, below); chips that
  are not the bottleneck; fuel held at the terminal when a shortage ends (up to about two weeks of that fuel's burn).
- **It depends on the networks' energy margin:** output barely covering base load plus fabs. That holds on Small and
  Full. At a grid without fabs, without base-first priority, or whose short fuel has no terminal, it does nothing.
- **Failure paths:** a failed pulse sends the planner's flows unchanged; a failed reserve plans as `mpc_det` would; a
  failed planner sends the maximum. None occurred in any play.
- **Out of reach:** the remaining gap on Full is 0.16, almost all unserved chip demand. 58 % of the clairvoyant plan's
  fab energy after week 13 is drawn in weeks when it sheds base load at that grid, so part of that gap is closed to any
  agent.

## Later versions (v2 to v9, in the order they were built)

- **v2, `versions/ft_buffer_v2`, confirmed:** one more rule, `pulse_tail` 0: a fab stops counting toward a pulse once a
  lot started now could not reach a sink before the episode ends (12 to 14 weeks). Against the uploaded agent:

  | Set | Uploaded | v2 | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8247 | 0.8327 | +0.0081 (+0.0054 to +0.0111), 44 episodes better / 1 worse |
  | Small 2026 (fresh) | 0.8214 | 0.8291 | +0.0077 (+0.0054 to +0.0102) |
  | Full 12345 (tuning) | 0.8358 | 0.8371 | +0.0012 (+0.0006 to +0.0018), 13 / 0 |
  | Full 2026 (fresh) | 0.8567 | 0.8590 | +0.0022 (+0.0017 to +0.0028) |

  `sbf check ft_buffer_v2` passes under Ubuntu: slowest week 0.31 s of 2 s on Small, 1.54 s of 4 s on Full. Not
  uploaded.
- **`versions/mpc_pulse_rec` and `versions/mpc_pulse_rec_v2`, confirmed:** the pulse agent plus the recovery forecast of
  `mpc_combined` (visible impairments are planned to end as their duration laws say; `recovery.py`). The first is built
  on the uploaded pulse, the second on v2. Both merges reproduce their parents to the cent with either part switched
  off, and the gains of the forecast and of the tail rule add up to within 0.001.

  | Set | Uploaded `ft_buffer` | `mpc_pulse_rec` | `mpc_pulse_rec_v2` | v2 merge against uploaded (90 % interval) |
  | --- | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8247 | 0.8337 | 0.8416 | +0.0169 (+0.0130 to +0.0211), 55 episodes better / 7 worse |
  | Small 2026 (fresh) | 0.8214 | 0.8318 | 0.8389 | +0.0174 (+0.0139 to +0.0212), 59 / 4 |
  | Full 12345 (tuning) | 0.8358 | 0.8409 | 0.8418 | +0.0060 (+0.0040 to +0.0080), 13 / 3 |
  | Full 2026 (fresh) | 0.8567 | 0.8633 | 0.8667 | +0.0100 (+0.0056 to +0.0160), 16 / 0 |

  Next to the pulse the forecast is worth about +0.009 on Small and +0.005 to +0.008 on Full (it was +0.013 without the
  pulse). All three kinds (closures, capacity, grids) are kept; the grid kind is the smallest, about +0.001.
  `sbf check mpc_pulse_rec_v2` passes under Ubuntu: slowest week 0.30 s of 2 s on Small, 1.46 s of 4 s on Full.
  On the 20 dev episodes (`sbf compare`) it scores 0.8666 against 0.8534 for `ft_buffer` on Small (+0.0133, interval
  +0.0085 to +0.0191) and 0.8740 against 0.8674 on Full (+0.0066, +0.0040 to +0.0096).
  **Uploaded on 2026-10-06: 0.8429 on the public board, no fallback week.**
- **v3, `versions/ft_buffer_v3` and `versions/mpc_pulse_rec_v3`, confirmed:** a trace of v2 on Full put 0.098 of its
  remaining 0.158 out of reach of any rule for holding the fuel we deliver. The largest reachable pieces were wafers
  missing at fabs in weeks they had power and fuel still held when lots can no longer be sold. v3 adds three things to
  `pulse.py`, each switched by a setting:
  - **Wafer top-up (`pulse_wafers` 1.0):** the planner expects an idle fab to run from next week, so it keeps one week
    of that plan's lots at hand and sends no more; a whole week then finds less than the fab can process. The top-up
    raises the wafer flows the planner already sends so that a week of capacity is at hand. Worth +0.003 on Full and
    under +0.001 on Small, where it loses a little in more episodes than it wins.
  - **Run-down (`pulse_rundown`):** near the last week a lot can still be sold, the gas charge no longer waits for four
    whole weeks, and no fuel is held that cannot give a whole week in time. Worth +0.001 to +0.002.
  - **Two spare weeks (`pulse_tail` 2):** the end-of-sales rule assumed the fastest route; most chips travel slower.
    Worth +0.001 to +0.002.

  `mpc_pulse_rec_v3` is the uploaded `mpc_pulse_rec_v2` with v3's `pulse.py` and `reserve.py` dropped in:

  | Set | `mpc_pulse_rec_v2` (uploaded) | `mpc_pulse_rec_v3` | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8416 | 0.8463 | +0.0048 (+0.0029 to +0.0068), 40 episodes better / 14 worse |
  | Small 2026 (fresh) | 0.8389 | 0.8440 | +0.0051 (+0.0037 to +0.0067), 51 / 6 |
  | Full 12345 (tuning) | 0.8418 | 0.8469 | +0.0052 (+0.0036 to +0.0067), 15 / 0 |
  | Full 2026 (fresh) | 0.8667 | 0.8706 | +0.0038 (+0.0025 to +0.0049), 12 / 3 |

  `sbf check mpc_pulse_rec_v3` passes under Ubuntu: slowest week 0.31 s of 2 s on Small, 1.59 s of 4 s on Full.
  On the 20 dev episodes (`sbf compare` against `mpc_pulse_rec_v2`) it scores 0.8754 against 0.8666 on Small (+0.0087,
  interval +0.0044 to +0.0135) and 0.8774 against 0.8740 on Full (+0.0034, +0.0021 to +0.0049).
  **Uploaded on 2026-10-06: 0.8498 on the public board, no fallback week.**
- **The ceiling, and what it suggested:** the clairvoyant plan with "base load before fabs"
  made exact (one binary per fab grid and week) is a valid upper bound for any agent with perfect foresight. On Small
  root 12345 (64 episodes) the best plan found scores 0.939 and the proven bound is 0.955, against 0.833 for v2 as
  played; with v2's own whole weeks kept and everything else optimised it is 0.887. On Full the search timed out; v2's
  calendar optimised gives 0.894 against 0.837. The better plan burns nearly the same fuel in the right weeks, runs
  fabs without a small fuel segment where that segment is below their draw, and puts more LNG and less crude on the
  full edges into the Korean terminal. That gave three rules to try: a small fuel segment must not block the pulse;
  the shortfall premium on the limiting fuel only; the pulse's duty told to the planner.
- **v4, `versions/ft_buffer_v4` and `versions/mpc_pulse_rec_v4`, confirmed:** of the three rules, one ships.
  - **Rule 1, shipped (`pulse_small`, `premium_small`):** a fuel whose share of the grid's power is below the fabs'
    draw (computed from the instance; crude at two grids on both networks) no longer makes the pulse stand down. What
    it misses is taken off the energy the fabs can get, and the week is ON when every other fuel can run in full.
    Before, a missing small fuel left the fabs at zero for as long as it was missing (19 weeks in one Small episode).
  - **Rule 2, built, off (`premium_limit`):** the shortfall premium in full only for a grid's scarcest fuel. On top of
    rule 1 it is +0.0014 and +0.0001 on the tuning roots and +0.0004 and -0.0003 on the fresh ones, every interval
    holding 0, and it makes more episodes worse than the uploaded agent.
  - **Rule 3, built, off (`plan_duty`):** a flat cap on planned lots at the pulse's duty. It loses 0.001 to 0.003.

  `mpc_pulse_rec_v4` is the uploaded `mpc_pulse_rec_v3` with v4's `pulse.py` and `reserve.py` dropped in and rule 1
  on. With the new switches off it reproduces the uploaded agent to the cent on all 64 Small and all 16 Full episodes.

  | Set | `mpc_pulse_rec_v3` (uploaded) | `mpc_pulse_rec_v4` | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8463 | 0.8568 | +0.0104 (+0.0069 to +0.0141), 30 episodes better / 12 worse |
  | Small 2026 (fresh) | 0.8440 | 0.8495 | +0.0055 (+0.0026 to +0.0087), 22 / 12 |
  | Full 12345 (tuning) | 0.8469 | 0.8522 | +0.0052 (+0.0017 to +0.0093), 8 / 4 |
  | Full 2026 (fresh) | 0.8706 | 0.8730 | +0.0025 (+0.0014 to +0.0038), 9 / 3 |

  The fresh roots confirm it at about half the gain of the tuning roots, and on Full 2026 two episodes carry most of
  the saving. `sbf check mpc_pulse_rec_v4` passes under Ubuntu: slowest week 0.28 s of 2 s on Small, 1.32 s of 4 s on
  Full. On the 20 dev episodes (`sbf compare` against `mpc_pulse_rec_v3`) it scores 0.8866 against 0.8754 on Small
  (+0.0112, interval +0.0039 to +0.0205) and 0.8805 against 0.8774 on Full (+0.0031, interval -0.0002 to +0.0079,
  which holds 0). On a third fresh Full set (root 777, 32 episodes) v4 and v3 both score 0.8268 (difference 0.0000,
  interval -0.0001 to +0.0001): rule 1 only matters in episodes where a small fuel goes missing, and that set has
  almost none. So on Full, v4 is worth between nothing and +0.005 over v3, depending on the episodes, and does no
  harm. **Uploaded on 2026-10-07: 0.8577 on the public board, no fallback week.**
- **v5, `versions/ft_buffer_v5`, researched and not built into an agent:** telling the planner which weeks a pulsed
  grid's fabs will run. The ceiling was measured first: the true calendar of a finished play told for all 20 window
  weeks, with that play's fuel deliveries replayed so the calendar stays true, gains +0.0007 on Small and +0.0011 on
  Full (both intervals end below +0.003). Told without the replay it loses 0.002 to 0.008, because the pulse's ON runs
  shift once the plan changes. So this is not worth building, and the folder plays as v4.
  - **Why the flat cap of v4's rule 3 lost:** it capped each fab separately, while at a grid with two fabs the plan
    gives nearly all the room to one of them. As one row per grid it no longer loses, and no longer gains either.
  - **Why a correct belief can hurt:** the pulse counts a fab's draw by the wafers it holds. A plan that knows a fab
    will sit idle stops sending it wafers, and the pulse then never holds fuel for it.
  - **The calendar is predictable** from the observation (92 % of grid-weeks right four weeks ahead on Small, against
    about 30 % for the planner's belief that every week is ON); it is just not what limits the score.
- **Where the rest of the gap sits, and v6 (being built in `versions/mpc_pulse_rec_v6`):** the exact model was re-solved
  for `mpc_pulse_rec_v4` with our own fab calendar kept and groups of our decisions fixed. The best plan with our
  calendar scores 0.9014 against 0.8568 played on Small root 12345, and 0.9003 against 0.8522 on Full. About 0.03 of
  that 0.045 is chips, wafers and fuel routed with knowledge of sanctions and outages to come, or the simulator's own
  rules (the optimum starts fewer lots in weeks 1 to 4, which no agent can decline to do). The end game is not the
  problem: the best finish from week 44 gains 0.0013 on Small. What needs no foresight:
  - **Gas burnt in slack weeks (about 0.011 on Small, 0.006 on Full, from the records, untested in play):** when the
    fabs draw less than the grid's margin, the simulator throttles every segment alike, so the segment that needs no
    fuel is throttled and gas burns in its place; that gas is then missing in a later week that sheds.
  - **Fuel that can never be burnt (0.0025 on Small, 0.0035 on Full, certain):** nuclear fuel shipped to grids whose
    stock already covers every remaining week, hoarded because of the window-end value.
  - **Wafers into a full fab store (about 0.001 on Full).**

- **v6, `versions/mpc_pulse_rec_v6`, confirmed:** the three levers above, built in `pulse.py` of a copy of the uploaded
  agent, each behind a switch. They only lower flows the planner already sends, after the pulse has run.
  - **Gas trim (`ration_trim`):** `Pulse.grid_week` models one grid-week as the simulator does (gas rationed on last
    week's stock, base load first, one load factor for all segments). In a week without shedding, the gas delivered
    from the terminal is cut to this week's burn plus the closing stock that lets next week's needed gas through the
    rationing rule; the rest stays at the terminal for a week that would otherwise shed. Worth +0.0086 on Small and
    +0.0061 on Full (tuning roots), close to the estimate from the records.
  - **Unburnable fuel (`fuel_cancel`):** a dispatch from a source straight into a grid is cut so that stock, fuel
    under way and the dispatch never exceed the full-rate burn of the weeks left. Worth +0.0025 on Small and +0.0035
    on Full; it changes only freight, tariff, holding and salvage, and no episode gets worse.
  - **Wafers into a full store (`wafers_fit`):** only at grids that shed and are not being pulsed (at pulsed grids the
    fabs need the wafers gathered in OFF weeks, and cutting them there lost 0.003). Worth +0.0006 on Full, nothing on
    Small.

  With the three switches off it reproduces the uploaded agent to the cent on all 64 Small and all 16 Full episodes,
  and the three gains add up to within 0.0001.

  | Set | `mpc_pulse_rec_v4` (uploaded) | `mpc_pulse_rec_v6` | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8568 | 0.8678 | +0.0111 (+0.0092 to +0.0133), 61 episodes better / 3 worse |
  | Small 2026 (fresh) | 0.8495 | 0.8652 | +0.0157 (+0.0130 to +0.0184), 62 / 2 |
  | Full 12345 (tuning) | 0.8522 | 0.8624 | +0.0102 (+0.0089 to +0.0117), 16 / 0 |
  | Full 2026 (fresh) | 0.8730 | 0.8819 | +0.0089 (+0.0077 to +0.0102), 16 / 0 |
  | Full 777 (fresh, 32 episodes) | 0.8268 | 0.8375 | +0.0107 (+0.0093 to +0.0121), 32 / 0 |

  Unlike earlier versions the gain did not halve on fresh episodes, and it is not carried by a few of them. Known
  weak spots: an energy shock that starts at a trimmed grid costs about a week of the trim, and `wafers_fit` is on
  from weak evidence. CPU per week is unchanged from v4. `sbf check mpc_pulse_rec_v6` passes under Ubuntu: slowest
  week 0.25 s of 2 s on Small, 1.08 s of 4 s on Full. On the 20 dev episodes (`sbf compare` against
  `mpc_pulse_rec_v4`) it scores 0.9024 against 0.8866 on Small (+0.0158, interval +0.0127 to +0.0195) and 0.8936
  against 0.8805 on Full (+0.0132, +0.0101 to +0.0161).
  **Uploaded on 2026-10-07: 0.8714 on the public board, no fallback week.**
- **v7, `versions/mpc_pulse_rec_v7`, confirmed:** what was left of the slack that v6's gas trim went after. Replaying
  v6 showed 0.0146 (Small) and 0.0212 (Full) of fuel-free energy still unused in weeks without shedding, but 85 % to
  95 % of it is worth nothing: it sits at grids with plenty of gas, at a grid whose gas comes straight from a source,
  behind full terminals, or in the last weeks, with no later shedding to spend it in. The reachable part:
  - **Strait releases in the trim's forecast (`ration_trim_release`):** the trim's forecast of next week's fab draw
    counted the whole queue at any strait that is not shut as arriving (60,000 to 120,000 wafers where the strait
    passed 1,296 a week). `Pulse.passing` now works out what the straits pass on with the simulator's own release
    rule. Worth +0.0018 on Small and +0.0007 on Full.
  - **Trim for fuels that are not rationed (`fuel_trim`):** such a fuel is delivered so that it has on hand what its
    segment burns once the week's slack is taken off; the rest waits at the terminal. It needs no forecast, since the
    delivery arrives before the grid runs. Worth +0.0011 on Small and +0.0018 on Full.
  - **Built and left off (`osat_fit`):** raw chips toward a packaging plant whose store is full. The disposal fee is
    the same at the fab, so cutting pays only if the chips sell elsewhere; it nets -0.0001.

  With the new switches off it reproduces v6 to the cent on all 64 Small and all 16 Full episodes.

  | Set | `mpc_pulse_rec_v6` (uploaded) | `mpc_pulse_rec_v7` | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8678 | 0.8707 | +0.0029 (+0.0023 to +0.0036), 58 episodes better / 6 worse |
  | Small 2026 (fresh) | 0.8652 | 0.8676 | +0.0024 (+0.0019 to +0.0030), 57 / 7 |
  | Full 12345 (tuning) | 0.8624 | 0.8651 | +0.0027 (+0.0013 to +0.0042), 15 / 1 |
  | Full 2026 (fresh) | 0.8819 | 0.8848 | +0.0029 (+0.0012 to +0.0050), 13 / 3 |
  | Full 777 (fresh, 32 episodes) | 0.8375 | 0.8401 | +0.0026 (+0.0018 to +0.0035), 29 / 3 |

  On Full the gain is concentrated: without its two best episodes it is +0.0010 to +0.0020. After v7 the bank
  estimate of what this lever still holds is 0.0002 to 0.0004, so it is used up. CPU per week is unchanged.
  `sbf check mpc_pulse_rec_v7` passes under Ubuntu: slowest week 0.22 s of 2 s on Small, 1.03 s of 4 s on Full.
  On the 20 dev episodes (`sbf compare` against `mpc_pulse_rec_v6`) it scores 0.9072 against 0.9024 on Small
  (+0.0048, interval +0.0035 to +0.0066) and 0.8961 against 0.8936 on Full (+0.0025, +0.0019 to +0.0033). Not
  uploaded.
- **What is left above v6, and v8 (being built in `versions/mpc_pulse_rec_v8`):** an analysis of the opening weeks and
  of which weeks the fabs run (Small root 12345). v6 plays 0.8678 against 0.9017 for the best plan with its own
  calendar.
  - **The opening's flows hold nothing without foresight.** The optimum's opening is not the same from episode to
    episode; its gain comes from knowing when events already running in week 1 will end. An opening planned on week
    1's network alone gains +0.0006 (interval holds 0). About 0.0035 of the opening's 0.0086 is closed to any agent
    (week-1 lots are forced by wafers already on the way).
  - **A correction:** the "realizable clairvoyant" of the ceiling analysis (0.939) is not realizable. That model let
    a plan keep fuel at a grid in a week that sheds, where the simulator burns it all. The gap from choosing other
    weeks, about 0.037, is therefore an upper limit, not a target.
  - **One thing of size needs no foresight:** the better plan burns less gas in weeks 1 to 5 at a grid that will be
    short later and spends it later in whole weeks, because chips are in surplus early. A play test (gas held at the
    terminal in weeks 2 to 4 at grids where the planner's week-1 plan already expects a shortage) gained +0.0051 on
    Small (interval +0.0025 to +0.0079), untuned, on the tuning root only and with the grids picked by name.
- **v8, `versions/mpc_pulse_rec_v8` (rule off) and `versions/mpc_pulse_rec_v8_on` (rule on):** the opening hold as a rule
  (`open_hold`). The episode's first plan decides: a grid is held in weeks 2 to 4 if the plan leaves its gas short by
  at least 0.1 of the segment over window weeks 4 to 12, the pulse can hold that gas, and no other short fuel can
  take all the fabs' energy (that last condition is what excludes the grid whose shortages are of crude, without
  naming it). A hold ends early when the terminal is full. Re-checking the trigger every week was measured and is
  off: it released nothing in the known false alarm and started late holds on small shortfalls.

  | Set | `mpc_pulse_rec_v7` | With the opening hold on | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Small 12345 (tuning) | 0.8707 | 0.8760 | +0.0053 (+0.0031 to +0.0076), 35 episodes better / 17 worse |
  | Small 2026 (fresh) | 0.8676 | 0.8741 | +0.0065 (+0.0033 to +0.0099), 33 / 20 |
  | Full 12345 (tuning) | 0.8651 | 0.8651 | -0.0000 (-0.0011 to +0.0011), 8 / 6 |
  | Full 2026 (fresh) | 0.8848 | 0.8863 | +0.0014 (+0.0002 to +0.0025), 7 / 7 |
  | Full 777 (fresh, 32 episodes) | 0.8401 | 0.8402 | +0.0001 (-0.0011 to +0.0012), 12 / 14 |

  **It helps on Small and does nothing reliable on Full.** On Small the gain held on fresh episodes without
  shrinking. On Full the early shedding is recovered by about week 9 and the rest is the phase of the pulse's cycle
  over 104 weeks; Full's opening weeks are also better used to begin with (the Korean fabs draw 0.66 of full in weeks
  2 to 4 against 0.45 on Small). The builder left the rule off by default for that reason, so `mpc_pulse_rec_v8` plays
  exactly as v7; `mpc_pulse_rec_v8_on` is the same code with `{"open_hold": true}` in a `params.json`.
  `sbf check mpc_pulse_rec_v8_on` passes under Ubuntu: slowest week 0.22 s of 2 s on Small, 1.01 s of 4 s on Full.
  On the 20 dev episodes (`sbf compare` against `mpc_pulse_rec_v7`) the rule on scores 0.9089 against 0.9072 on
  Small (+0.0017, interval +0.0001 to +0.0036) and 0.8961 against 0.8961 on Full (-0.0000, -0.0014 to +0.0011): a
  smaller gain on Small than on the two roots, and again nothing on Full.
  **`mpc_pulse_rec_v8_on` uploaded on 2026-10-07: 0.8768 on the public board, no fallback week** (v7 alone was not
  uploaded).
- **v9, `versions/mpc_pulse_rec_v9`, confirmed (2026-10-08):** how a run of whole weeks starts and ends. The plan was to
  replace the pulse's timing rules with a schedule optimised each week; the ceiling was measured first, on Full.
  - **A full scheduler is not worth building.** With the arrivals known in advance, the best schedule of each grid's
    deliveries is worth +0.010 on Full. Re-solved each week from what the agent sees it is at or below zero (-0.0004,
    -0.0015 and -0.0032 with three ways of forecasting arrivals): the planner's later weeks are intentions (it forecast
    0.85 of a week's gas at one grid where 1.04 arrived, and wafers that never came), so the search sheds load ahead of
    shortages that do not happen.
  - **One part of the hindsight schedule needs no forecast,** and that is what was built, behind the switch
    `schedule`. At a pulsed grid, for the rationed fuel: the last week of a run is delivered its burn alone, so the
    stock that only serves to lift the rationing threshold waits at the terminal instead of burning for base load with
    the fabs nearly idle; and the charge that starts a run is made as soon as one whole week can follow, without
    waiting for four. It reads only the week's observation and the instance and has no setting to tune.

  | Set | `mpc_pulse_rec_v8_on` (uploaded) | `mpc_pulse_rec_v9` | Difference (90 % interval) |
  | --- | --- | --- | --- |
  | Full 12345 (tuning) | 0.8651 | 0.8700 | +0.0050 (+0.0024 to +0.0075), 11 episodes better / 5 worse |
  | Full 2026 (fresh) | 0.8863 | 0.8940 | +0.0078 (+0.0032 to +0.0131), 14 / 2 |
  | Full 777 (fresh, 32 episodes) | 0.8402 | 0.8448 | +0.0046 (+0.0020 to +0.0073), 18 / 11 |
  | Small 12345 (tuning) | 0.8760 | 0.8804 | +0.0044 (+0.0022 to +0.0065), 34 / 20 |
  | Small 2026 (fresh) | 0.8741 | 0.8766 | +0.0025 (+0.0007 to +0.0043), 28 / 25 |

  With the switch off it reproduces the uploaded agent to the cent on all 16 Full and all 64 Small episodes. One
  setting was scored, once per set. On Full 2026 one episode carries half the gain (+0.0032 without the two best).
  Where it loses: at a grid whose threshold exceeds a week's burn every short run wastes part of the gas, and when
  arrivals collapse right after a charge the tank is spent on partly powered weeks. Two refinements follow from that
  (charge at once only where the threshold is no more than a week's burn, and only when the wafers are there); they
  were not built, because the rule was fixed before the fresh sets were played. `sbf check mpc_pulse_rec_v9` passes
  under Ubuntu: slowest week 0.22 s of 2 s on Small, 1.01 s of 4 s on Full.
  **Uploaded on 2026-10-08: 0.8827 on the public board, no fallback week.**
- **Can event ends be predicted better? No (analysis of 2026-10-08, nothing built).** The forecast uses an event's
  type and how long it has been seen. Whether anything else the agent observes says how much longer a running event
  will last was checked in the generator's source and on 400 sampled scenarios each of Full and Small.
  - **In the source:** every duration is drawn from its type's law on a random stream of its own. Severity, target,
    tariff rate and lead times use other streams, child events are drawn from a parent's start and never its end, and
    no message is generated from an end time (`closure_end` is empty under the scored regime).
  - **On samples:** for the five classes we forecast, adding 16 observables to the law changes the Brier score of
    "still on after h weeks" by 0.0000 to 0.003, zero or worse within noise, and none of 91 correlations per task
    stands out. The test has power: the generator's hidden mixture component, which the agent cannot see, shows at
    once. The law itself is calibrated.
  - **Sanctions and tariffs,** which we keep for the whole window: nothing observable says which one ends, and the law
    alone would only tell the planner that a cut edge has 0.26 to 0.30 of its capacity where it has 0.25.
  - **The one exception is the regional-conflict family:** a war's second-stage capacity cut ends exactly 52 weeks
    after it began, and its first stage and the closure of its strait never end before 52 weeks. By exposure that is
    about a tenth of the capacity and strait families, a guess of +0.001 to +0.002 on Full, below what our episode
    sets can resolve. Not built.

  So the forecast is at the limit of what the observation contains, and the rest of what perfect knowledge of ends
  would give (about +0.007 on Full) is luck of the draw.
- **Was anything fitted? A clean test on untouched episodes (2026-10-09).** A new root, 9001 (32 Full and 64 Small
  episodes), was built and played once per agent, with nothing chosen on it. It stays reserved for confirming a
  finished version: no analysis and no tuning there.

  | Set | Role | `mpc_pulse_rec_v2` | `mpc_pulse_rec_v9` | Gain (90 % interval) | Better / worse |
  | --- | --- | --- | --- | --- | --- |
  | Full 9001 (32) | untouched | 0.8807 | 0.9010 | +0.0202 (+0.0169 to +0.0239) | 32 / 0 |
  | Full 777 (32) | confirmation from v4 on | 0.8239 | 0.8448 | +0.0209 (+0.0178 to +0.0240) | 31 / 1 |
  | Full 2026 (16) | confirmation, reused | 0.8667 | 0.8940 | +0.0273 (+0.0205 to +0.0349) | 16 / 0 |
  | Full 12345 (16) | tuning | 0.8418 | 0.8700 | +0.0282 (+0.0240 to +0.0326) | 16 / 0 |
  | Small 9001 (64) | untouched | 0.8502 | 0.8890 | +0.0388 (+0.0329 to +0.0448) | 62 / 2 |
  | Small 2026 (64) | confirmation | 0.8389 | 0.8766 | +0.0377 (+0.0324 to +0.0435) | 61 / 3 |
  | Small 12345 (64) | tuning | 0.8416 | 0.8804 | +0.0388 (+0.0327 to +0.0451) | 62 / 2 |

  - **Small: nothing was fitted.** Untouched less used is +0.0005 (-0.0068 to +0.0077).
  - **Full: about a quarter of the gain the two old sets showed is not there.** Untouched less roots 12345 and 2026
    is -0.0076 (-0.0129 to -0.0021); root 777 agrees with the untouched set. On 777, where every version was played,
    the shortfall sits in v3, v4 and v9; v6 and v7 hold in full.
  - **Expect v9 near 0.875 on Full:** its score over all 96 Full episodes (0.858 to 0.891); the four sets give 0.870,
    0.894, 0.845 and 0.901.
  - **The searched settings are not fitted noise.** `versions/mpc_pulse_rec_v9_safe` is v9 with every setting that was
    chosen by search put back to its starting value (`end_value` 1.0, `reserve_weeks` 4 without feeders,
    `pulse_partial` 1.0, `pulse_payback` 0, `pulse_tail` 0, the two forecast rules found on Small, `open_hold` and
    `wafers_fit` off). It is below v9 on all seven sets: -0.0052 (-0.0080 to -0.0025) over the 80 Full episodes its
    values were not chosen on, -0.0054 (-0.0077 to -0.0034) over 192 Small episodes. The searched values alone are
    worth +0.0047 on the Full episodes they were chosen on and +0.0043 on the 80 they were not.
    **Uploaded on 2026-10-08: 0.8793 on the public board** (v9: 0.8827), so the stripped version is not safer.
  - **An audit of every `PARAMS` entry** sorted them into derived, confirmed on fresh episodes, chosen on tuning
    episodes only, and never varied. Chosen on tuning episodes only: the reserve's size, the end value's level,
    `pulse_partial` with `pulse_payback`, `pulse_tail`, `pulse_hold_weeks`, the opening hold's five details,
    `wafers_fit` and three forecast rules. All but `pulse_hold_weeks` and one forecast rule are reset in the
    stripped version above.
- **Where the shortfall sits on Full (2026-10-09, from the cached costs of 96 episodes).** The score is weighted
  0.50 / 0.30 / 0.15 / 0.05 over harm levels 1 to 4, and v9 scores 0.873 / 0.870 / 0.892 / 0.869 there, so the calm
  level carries half the shortfall. The typical episode is fine (per-episode median 0.91 at every level); the worst
  fifth of level-1 episodes hold 48 % of that level's loss, with single episodes at 0.53 to 0.75. Four of the six
  worst level-1 episodes are in root 777, which is why that set scores 0.845.
- **Closed on 2026-10-09 without a build:**
  - **The 104 action entries per Full episode that the simulator refuses** are one override slot a week, on a route
    prohibited for the whole episode, sent at quantity 0 because the action format sends every slot of a pair. The
    package's own `mpc_det_h20` played natively scores 0.3290 on Full root 12345, as the port does; the largest
    difference in an episode is 1.5e-6 of its cost.
  - **Weeks the agent does not fully observe** do not occur under the scored regime (0 of 1,664 Full weeks and 0 of
    3,328 Small weeks on root 12345), so the blackout branches never run.
  - **Warnings and announcements, on Full with the pulse:** on the v2 pulse agent (Full root 12345), the true course
    of the whole 20-week window is worth +0.022, of which the ends of what is already visible are +0.015. Every
    onset known perfectly is therefore about +0.007, before decoys and short leads.
  - **Finer window-end values:** the Full score is flat in `end_value` between 1 and 3 (0.829, 0.832, 0.829 at the
    `ft_q` stage).
  - **The war family's certain ends: built, worth +0.0001 on Full, left out.** Read in the generator and checked on
    800 sampled scenarios: a regional conflict cuts the two regions' edges to 0.40 for max(its duration, 52) weeks
    with the strait shut if it is the Taiwan Strait, then to 0.27 for exactly 52 weeks. So the second stage ends
    at least 104 weeks after the onset, and on Full only conflicts that began before week 1 end one inside the
    episode. The forecast already keeps the first stage and the strait for the whole window (721 of 721
    strait-weeks), so the only correction is releasing a second-stage cut on its known week. The truth for the
    whole family is worth +0.0011 on Full root 12345, the certain part +0.0004; the rule as played gave +0.00009
    over 64 Full episodes (+0.00003 to +0.00017; it acts in 18, three carry the gain) and cannot act on Small.
    Knowing weeks ahead that capacity returns is worth nothing to the planner; all of the effect is the week the
    cut ends in. Not worth 150 lines in the submitted agent.
  - **Costs other than shortages, and buying nuclear fuel later.** On 16 recorded Full episodes of v10, weighted as
    the score is, freight, tariffs, holding, war risk, disposal and salvage are 58 bn USD per episode against the
    clairvoyant plan's 36: 0.007 of v10's gap of 0.105. All the rest is unserved demand (0.101); v10 sheds slightly
    less base load than the plan. Two thirds of the 0.007 is nuclear fuel: v10 buys on both of a grid's routes
    until stock covers the weeks left and a unit waits 37 weeks at the grid, where the plan, knowing which routes
    stay open, waits a year and buys just in time. That stock is insurance that gets used: 21 % of nuclear routes
    are prohibited in week 1 and in 12 % of episodes a grid later loses both. A ledger of purchase rules over 464
    scenarios (it reproduces v10's played nuclear costs within 0.05 bn) gives buying just in time -0.046, and
    covering only the loss of the largest route +0.0012 (+0.0002 to +0.0020) with single episodes losing up to
    624 bn. Applied only to a route that pays a tariff this week it is +0.0007 (+0.0006 to +0.0008) with no loss
    in 464 scenarios; that form is the only one worth adding, and it changes nothing on Small.
- **v10, `agents/mpc_pulse_rec_v10`, confirmed (2026-10-09):** two rules for when a run of whole weeks may start or
  go on, found while tracing v9's losing episodes. Neither has a setting of its own.
  - **Every fuel counts (`schedule_fuels`).** v9's tests looked at the rationed fuel alone, but a week is whole only
    if every fuel's segment runs. At one grid 5 of 15 gas charges were made the week before a crude-short week and
    burnt for base load; at another, gas and crude were full in alternate weeks for 30 weeks. The rule works out in
    how many weeks the other fuels next allow a whole week (from flows, stocks, shipments due and the pulse's own
    arrival forecast) and charges, or continues a run, only if the terminal can keep the rationed segment full
    until then.
  - **The stand-down test counts a whole week as the simulator does (`schedule_hold`):** the larger of burn and
    threshold, plus what the grid's stock burns that week at the rationed rate, where it counted burn plus
    threshold.
  - **Measured and left off:** waiting for `pulse_run` whole weeks where the threshold exceeds a week's burn
    (`schedule_left`: +0.0001 on Full, -0.0009 on Small next to the two rules) and charging only when the wafers
    are there (`schedule_wafers`: -0.0001 on Full). A schedule for the end of sales was measured before building:
    +0.0001 once free only after the last sellable week; the +0.009 seen earlier was the payback of fuel held
    earlier.

  | Set | `mpc_pulse_rec_v9` | `mpc_pulse_rec_v10` | Difference (90 % interval) | Better / worse |
  | --- | --- | --- | --- | --- |
  | Full 12345 (tuning) | 0.8700 | 0.8718 | +0.0018 (+0.0007 to +0.0031) | 8 / 2 |
  | Full 2026 (fresh) | 0.8940 | 0.8977 | +0.0037 (+0.0015 to +0.0067) | 7 / 5 |
  | Full 777 (fresh, 32 episodes) | 0.8448 | 0.8509 | +0.0062 (+0.0030 to +0.0098) | 19 / 4 |
  | Full 9001 (untouched, 32 episodes) | 0.9010 | 0.9020 | +0.0011 (+0.0004 to +0.0018) | 10 / 4 |
  | Small 12345 (tuning) | 0.8804 | 0.8843 | +0.0039 (+0.0022 to +0.0057) | 24 / 15 |
  | Small 2026 (fresh) | 0.8766 | 0.8796 | +0.0030 (+0.0016 to +0.0046) | 17 / 16 |
  | Small 9001 (untouched) | 0.8890 | 0.8922 | +0.0032 (+0.0018 to +0.0046) | 19 / 17 |

  With the two switches off it reproduces v9 to the cent on all 16 Full and all 64 Small tuning episodes. The form
  of the fuels rule was changed once after its first scored round, from traces of three losing tuning episodes; the
  fresh and untouched sets were played once, after that. On Full 12345 three episodes carry +0.0016 of the +0.0018.
  CPU per week on Full is unchanged (mean 0.27 s, slowest about 1.0 s in the same process as v9).
  `agents/mpc_pulse_rec_v10_on` is a snapshot from the evening before (the first form of the fuels rule, all four
  switches on through a `params.json`): **uploaded on 2026-10-09, 0.8854 on the public board.**
- **Why the worst Full episodes are worse (analysis of 2026-10-09, on v9; roots 12345, 2026 and 777).** The twelve
  worst of the 64 episodes (per-episode score 0.53 to 0.81) were recorded and compared with the clairvoyant plan,
  with five typical episodes as a contrast.
  - **It is Korea, and the state is there in week 1.** In 11 of the twelve the Korean grid gets a fraction of the
    gas it burns (its gas segment runs at 0.05 to 0.65 of full, median 0.3) for most of the episode, and its fabs
    stand idle: 60 % of all lots lost in the twelve. In all 13 episodes of the 64 where the edges into Korea's
    terminal could carry under half a week's gas, the Australia-to-Korea gas edge (0.66 of Korea's burn) is banned,
    in 11 of them for weeks 1 to 90, and a strait route is cut or closed as well. Those 13 episodes hold 43 % of all
    money lost.
  - **Most of that is closed to any agent.** The clairvoyant plan runs the Korean memory fab at 0.5 to 1.0 by
    shedding base load. A bound that respects "base load before fabs" (the clairvoyant LP plus, per grid and week, a
    variable that ties the fabs' share of their draw to the scarcest fuel's segment; it still knows all 104 weeks
    and lets fuel wait at a grid that sheds, so it is optimistic) puts at least 55 % of the twelve's loss out of
    reach, 67 to 72 % in the four worst. In typical episodes the closed part is 0 to 2 % of attainable. Over the 40
    episodes the bound was solved for, closed is 0.047 of the pooled shortfall of 0.141.
  - **Foresight is not what is missing there:** the true network in all 20 window weeks recovers 6 % of the
    twelve's loss, and next to nothing in the four worst.
  - **What is reachable is the pulse's handling of grids with two short fuels or a trickle.** Five rules without a
    tuned number, play-tested in a copy of v9 (`versions/mpc_pulse_rec_v9_tail`), gained +0.0086 on the 64 Full
    episodes (+0.0059 to +0.0116; 48 episodes better, 16 worse), more than half of it in ordinary episodes:
    - **sync** (half the gain): in a week that is not whole, also hold every fuel that could run in full, so it is
      not burnt for base load while the week waits for the other fuel; charge the gas only if the other fuels will
      have a week's burn at their terminals. At Japan, gas and crude had their whole weeks in different weeks and
      the fabs ran one week in 14 where the fuel carried one in 3 to 4.
    - **idle:** in an energy shock at a pulsed grid, keep holding instead of releasing the tank to base load.
    - **lack and wait:** count what a tank still lacks as the simulator burns it, and let a tank take as long as
      lots can still be sold instead of 12 weeks, so a trickle of 0.1 of burn is collected instead of burnt.
    - **lapse:** apply the end-of-run test after long runs too, which v9 skipped after four whole weeks.
    - **runs** (a sixth, +0.0007): where the threshold exceeds a week's burn, wait until the tank carries as many
      whole weeks as it can hold.
  - **Tried and dropped:** holding China's gas once its shortage has lasted long enough to pay for a tank (-0 to
    -25 bn per episode), `premium_limit` next to the five (nothing on the whole set).
  - **A correction to how the score pools:** it is the sum of weight times mean saving over the sum of weight times
    mean attainable, not the weighted mean of the levels' ratios.
  - The rules overlap with v10's two and came from reading these three sets, so they were rebuilt on v10 as v11
    (below) and confirmed on the untouched roots.
- **A small fuel outbids the scarce one on a shared edge (2026-10-09, on v10; follows from the analysis above).**
  - **The mechanism.** Two of the three edges into Korea's terminal carry gas and crude. Korea burns 2,750 GWh of
    gas and 220 of crude a week and its fabs draw 456, so crude is a "small" fuel, and `premium_small` prices its
    shortfall over the fabs' draw: 33.7 M USD per GWh against 5.6 for gas, in every window week. Over weeks 1 to 90
    of the 13 Korea-starved episodes a shared edge is at capacity in 992 of 1,170 weeks, and crude takes 156 to 228
    GWh of it: its segment runs at 0.64 to 1.00 while gas runs at 0.05 to 0.63, with gas waiting at the strait
    (1,370 GWh at Malacca for 84 weeks in one episode). Taiwan's Panama edge has the same split; no other grid
    with fabs has an edge shared by two fuels.
  - **The rule (`premium_whole`, in `reserve.py`).** A small fuel is fab energy only in a whole week, and under the
    pulse the weeks that are not whole are the share of its burn a big fuel is short by. So the small fuel's
    shortfall is free up to that share of its own burn: one LP column per grid and small fuel, priced at minus the
    fuel's premium, bounded each window week by the fuel's own shortfall and by the other fuels' shortfalls scaled
    by the ratio of shares. It reads the plan's own shortfalls; no forecast, memory or setting. In one episode the
    plan on the Taiwan edge moves from 216 gas and 220 crude a week to 449 and 36, equal shares of burn.
  - **It pays only with a hold of the small fuel in the pulse.** v10's pulse holds a small fuel only in a week it
    is short of a week's burn; otherwise the planner's flow stands and is burnt at once, the week before a whole
    week too (57 % of the crude burnt outside whole weeks, and whole weeks found 0.82 of their crude). Five lines
    in `Pulse.adjust` hold a small fuel whose arrivals are below its burn in a week that is not whole.

  | Against v10 | LP rule alone | Hold alone | Both |
  | --- | --- | --- | --- |
  | Full, 64 episodes (roots 12345, 2026, 777) | +0.0012 (+0.0002 to +0.0022) | +0.0018 (+0.0009 to +0.0027) | +0.0045 (+0.0028 to +0.0062), 40 better / 24 worse |
  | Small, 128 episodes (roots 12345, 2026) | -0.0009 (-0.0021 to +0.0003) | +0.0007 (+0.0003 to +0.0013) | +0.0027 (+0.0016 to +0.0040), 62 / 66 |

  One form of each piece was built and neither was altered. Capping crude into Korea at a fixed share of its burn
  by hand did not work (+116 bn over nine starved episodes at 0.30, -1 at 0.15): a fixed share is wrong wherever
  the gas share differs from it. Of the pair's 934 bn on Full, 705 are in the 13 starved episodes. The largest
  loser is -28 bn in one Full episode. CPU per week rises by 5 % in the mean. Both pieces go into v11, and like the
  five rules above they still need the untouched roots: a second untouched Full set, root 5150 (32 episodes), was
  built for that (v10 scores 0.8949 on it).
- **v11, `agents/mpc_pulse_rec_v11`, confirmed on Full (2026-10-09):** v10 plus the rules of the two entries above,
  seven switches, none with a setting of its own.
  - **From the worst-episode analysis (stage one), in `pulse.py`:**
    - `hold_together`: in a week that is not whole, also hold the other scarce fuels that could run in full (the
      "sync" rule's hold; its charge timing was dropped in favour of v10's `schedule_fuels`).
    - `shock_hold`: keep holding through an energy shock at a pulsed grid.
    - `schedule_ends`: `schedule`'s end-of-run test also at a grid that no longer counts as pulsed, so long runs do
      not end with a partial week and a dead week.
    - `schedule_runs`: where the threshold exceeds a week's burn, wait until the tank carries as long a run as the
      terminal's storage holds (it differs from `schedule_left`, which waits `pulse_run` weeks and stays off).
    - `pulse_hold_sales`: a tank may take as long as lots can still be sold. Thin evidence: +0.0004 on Full next
      to the pair (+0.0001 to +0.0008), most of it one episode; nothing on Small.
    - The "lack" rule turned out to be v10's `schedule_hold`, whose form is the more exact one; nothing was added.
  - **From the shared-edge analysis (stage two):** `premium_whole` in `reserve.py` and `hold_small` in `pulse.py`,
    on or off together.
  - **Leave one out, stage one, on Full roots 12345 and 777 as one (48 episodes):** `hold_together` +0.0029
    (+0.0014 to +0.0046), `schedule_ends` +0.0023 (+0.0009 to +0.0040), `shock_hold` +0.0014 (+0.0005 to +0.0025),
    `schedule_runs` +0.0004 (-0.0000 to +0.0009), `pulse_hold_sales` +0.0001 (-0.0000 to +0.0004).

  | Set | `mpc_pulse_rec_v10` | `mpc_pulse_rec_v11` | Difference (90 % interval) | Better / worse |
  | --- | --- | --- | --- | --- |
  | Full 12345 (rules found here) | 0.8718 | 0.8835 | +0.0116 (+0.0059 to +0.0173) | 10 / 6 |
  | Full 2026 (rules found here) | 0.8977 | 0.9052 | +0.0075 (+0.0047 to +0.0104) | 13 / 3 |
  | Full 777 (rules found here, 32 episodes) | 0.8509 | 0.8597 | +0.0088 (+0.0052 to +0.0123) | 24 / 8 |
  | **Full 9001 (untouched, 32 episodes)** | 0.9020 | 0.9069 | +0.0049 (+0.0022 to +0.0078) | 23 / 9 |
  | **Full 5150 (untouched, 32 episodes)** | 0.8949 | 0.9034 | +0.0085 (+0.0047 to +0.0127) | 21 / 11 |
  | **Full, the two untouched sets as one (64)** | 0.8988 | 0.9050 | +0.0062 (+0.0041 to +0.0086) | 44 / 20 |
  | Small 12345 and 2026 as one (128) | 0.8831 | 0.8885 | +0.0054 (+0.0037 to +0.0071) | 69 / 59 |
  | **Small 9001 (untouched)** | 0.8922 | 0.8930 | +0.0008 (-0.0023 to +0.0037) | 38 / 26 |

  - **On Full two thirds of the gain holds on untouched episodes** (+0.0062 against +0.0093 on the 64 episodes the
    rules were found on); stage one and stage two each give about half of it there. The largest loss on the
    untouched Full sets is 18 bn USD in an episode, the largest gain 136.
  - **On Small the untouched set cannot tell v11 from v10.** One episode loses 69 bn there, four times the largest
    loss seen on the Small sets the rules were measured on (17 bn); it was not opened, to keep the set untouched.
    Against v9 on the same set v11 is +0.0040 (+0.0002 to +0.0075).
  - With all seven switches off v11 reproduces v10 to the cent on all 16 Full and 64 Small tuning episodes; the
    plays that ran during a memory incident on 2026-10-09 (the solver failing with the machine out of commit) were
    replayed and are equal to the cent, and every measurement play was made with failures raising.
  - `sbf check mpc_pulse_rec_v11` passes under Ubuntu on a quiet machine: slowest week 0.24 s of 2 s on Small, 1.07
    s of 4 s on Full (median 0.30 s). In the same process as v10 the mean week on Full costs 3 % more.
  - Losing episodes on the sets used: about 25 bn in three Full episodes (one of them from `shock_hold`), up to 17
    bn on Small. None was traced week by week.
- **Nuclear fuel on a tariffed route (`fuel_cancel_tariff`, 2026-10-09, on v11).** The safe form of the purchase
  rules costed that morning: a dispatch from a source straight into a grid on a route that pays a tariff this week
  is cut so that stock, fuel under way and the dispatch do not exceed what the grid would lack to the episode's end
  if its largest route were cut from next week on. Against v11 on the 64 Full episodes of roots 12345, 2026 and 777:
  +0.0007 (+0.0004 to +0.0009), 34 episodes better, none worse, 30 equal to the cent; 143 bn saved, within 1.2 bn
  of the ledger's prediction. On Small every episode is equal to the cent (each nuclear grid there has one route).
- **Why v11 sometimes loses 20 to 40 bn in a Small episode (2026-10-09; a Small set of our own, root 6006, 128
  episodes).** On that set v11 is +0.0039 over v10 (+0.0018 to +0.0060), 64 episodes better and 64 worse; four
  lose more than 25 bn. There is no single cause: three of v11's rules each hold fuel, or end a run, for a whole
  week that does not need it.
  - **`hold_together` holds the rationed fuel while the grid waits for another (fixed, check "together").** At
    Japan crude reached the terminal at a ninth of its burn while gas arrived as fast as it burnt; the gas was
    held, its stock burnt down and the next delivery was a charge in a dark week: a dark gas week every second
    week until the tank was full, for gas that was never needed. `schedule_fuels` had just decided the same gas
    could run through to the crude's week. The check applies that test in the week `hold_together` would hold.
    Not specific to Small: it is `hold_together`'s known 35 bn loss in a Full episode too.
  - **`shock_hold` holds a tank through a shock that outlasts the sellable weeks (fixed, check "shock").** A fuel
    is now held only while the grid and its terminal hold less than the sellable weeks left can burn beyond
    their own arrivals. Small has 36 sellable weeks against 88 on Full, so a shock reaches the end far more often.
  - **`schedule_ends` ends a run that was not ending (diagnosed, not fixed).** Its test asks whether next week's
    segment can run in full, which a whole week needs only when the fabs draw their capacity; at a grid whose
    fabs were wafer-limited the test failed every twelve weeks and cost a dark week each time. A check for it
    took 43 bn off the eight episodes it was read from and added 97 in six others, and was left off.
  - **Not fixed either:** `schedule_runs` waiting 30 weeks for a three-week run the tank could never carry, and
    a shift of scarce wafers between fabs under the pair in one episode (not traced).

  | Checks "together" and "shock" against v11 | Difference (90 % interval) | Better / worse | Worst episode |
  | --- | --- | --- | --- |
  | Small 6006 (128), the set they were built on | +0.0010 (+0.0004 to +0.0018) | 20 / 11 | 3.6 bn |
  | Small 12345 and 2026 as one (128) | +0.0007 (+0.0001 to +0.0013) | 12 / 6 | 3.4 bn |
  | Full 12345, 2026 and 777 as one (64) | +0.0003 (+0.0001 to +0.0005) | 16 / 2 | 0.7 bn |

  It trims the tail and moves the score little. On the analysis set the largest loss against v10 falls from 39 to
  26 bn. The episode that loses 69 bn on the untouched Small set was never opened. Built as `hold_checks` in
  `mpc_pulse_rec_v12b`.
- **A second pass over what v11 loses on Full (2026-10-09; `mpc_pulse_rec_v12a`).** Two rules for a grid the
  pulse stood down at because a short fuel "cannot be held"; neither has a setting of its own.
  - **Where v11's loss sits.** Its shortfall on the 64 Full episodes of roots 12345, 2026 and 777 is 0.127. Over
    the 40 episodes with a base-first bound, 0.047 is closed and 0.060 open (0.073 for v9). On 28 recorded
    episodes about 85 % of the open shortage falls in the second year, and it splits into grids whose nuclear
    fuel ran short (0.013; closed in fact, since the bound stretches a stock that a shedding grid burns), grids
    whose gas comes by pipe or is not worth a tank (China, Europe, SEA, the US: 0.016, the piece the two rules
    address) and a remainder at Taiwan, Korea and Japan that looks like foresight (0.008).
  - **Tried and dropped:** topping up wafers at powered fabs. The planner does under-load them (1,589 bn of lots
    on the 28 episodes), but both forms lost on the episodes picked for it (-24 and -31 bn over eight): lots move
    between fabs instead of rising.
  - **`pipe_pulse`.** The first pass called a fuel that comes straight from a source unreachable. But a supply
    node is a tank too: the simulator refills it each week up to its storage and keeps what is not shipped. So a
    fuel that reaches a grid from a supply node over an edge one week long can wait there as a terminal's fuel
    waits at the terminal, with its delivery decided a week ahead. Where such a pipe carried 0.72 of a week's gas
    the grid shed the difference every week and its fabs stood dark for 95 of 104 weeks in one episode. With the
    rule the pipes hold back what the next charge takes and ship the charge, then a week's burn at a time, once a
    whole week can follow. +0.0070 on the 64 Full episodes (+0.0045 to +0.0097), 28 better and 5 worse.
  - **`lean_hold`.** `pulse_payback` leaves a fuel out when whole weeks do not pay for a full tank of it, and a
    grid short of such a fuel was not pulsed at all. A pulse does not need a full tank: the fuel is held up to
    one charge, and only if whole weeks pay for that. +0.0021 on top (+0.0009 to +0.0034), 14 better and 5 worse;
    its largest loss is 46 bn, an episode where the chips of the whole weeks it made were not short.

  | Both rules against v11 | Difference (90 % interval) | Better / worse |
  | --- | --- | --- |
  | Full 12345 (16) | +0.0090 (+0.0024 to +0.0154) | 9 / 4 |
  | Full 2026 (16) | +0.0069 (+0.0042 to +0.0101) | 9 / 0 |
  | Full 777 (32) | +0.0097 (+0.0062 to +0.0136) | 21 / 2 |
  | Full, the three as one (64) | +0.0091 (+0.0066 to +0.0120) | 39 / 6 |
  | Small 12345 and 2026 (128) | equal to the cent in every episode | 0 / 0 |

  On Small no pipe can carry a charge and one charge pays at no grid, so neither rule acts there. These three
  sets are where the rules were found, through three revisions of the folder; the confirmation on untouched
  episodes is in the v12 entry below. Two caveats from the builder: the final count in `lean_hold`'s payback test
  (one whole charge) was settled after a first form lost on two Small sets, and `pipe_pulse`'s payback test still
  subtracts the supply nodes' weekly lift, the reasoning that failed for `lean_hold`. The largest loss, 46 bn in
  one episode, is `lean_hold` running a fab for twelve more whole weeks whose chips were not short: the planner
  then sent fewer wafers to another fab of the same product. The pulse values every whole week as if its chips
  were needed, in v11's rules too.
- **v12, `agents/mpc_pulse_rec_v12`, confirmed (2026-10-10).** v11 with the three pieces above merged into
  `pulse.py` and `agent.py` by a three-way merge on v11: `pipe_pulse` and `lean_hold` (v12a), `hold_checks` with
  "together" and "shock" (v12b), `fuel_cancel_tariff` (v12c). The three folders are kept in `versions/` as they
  were measured. The six merge conflicts were two pieces adding text or code at the same place, and both sides were
  kept.
  - **The merge is exact.** With all four new switches off the merged folder reproduces v11 to the cent on all 16
    episodes of Full 12345 and all 64 of Small 12345. With one piece on at a time it reproduces that piece's own
    measured play to the cent on Full 12345 (all three) and on Small 12345 (`hold_checks`; the other two cannot
    act on Small).
  - **The pieces add up.** On Full 12345 they measured +0.0090, +0.0008 and +0.0007 separately and +0.0101
    together (0.8835 to 0.8936).

  | Set | `mpc_pulse_rec_v11` | `mpc_pulse_rec_v12` | Difference (90 % interval) | Better / worse |
  | --- | --- | --- | --- | --- |
  | Full 12345 (pieces found here) | 0.8835 | 0.8936 | +0.0101 (+0.0037 to +0.0162) | 12 / 3 |
  | Full 2026 (pieces found here) | 0.9052 | 0.9128 | +0.0076 (+0.0047 to +0.0108) | 12 / 0 |
  | Full 777 (pieces found here, 32 episodes) | 0.8597 | 0.8702 | +0.0105 (+0.0071 to +0.0143) | 29 / 1 |
  | Full, those three as one (64) | 0.8730 | 0.8830 | +0.0100 (+0.0075 to +0.0128) | 53 / 4 |
  | **Full 9001 (untouched, 32 episodes)** | 0.9069 | 0.9194 | +0.0125 (+0.0084 to +0.0167) | 27 / 1 |
  | **Full 5150 (untouched, 32 episodes)** | 0.9034 | 0.9128 | +0.0094 (+0.0051 to +0.0143) | 25 / 4 |
  | **Full, the two untouched sets as one (64)** | 0.9050 | 0.9161 | +0.0111 (+0.0081 to +0.0142) | 52 / 5 |
  | Small 12345 and 2026 as one (128) | 0.8885 | 0.8892 | +0.0007 (+0.0001 to +0.0013) | 12 / 6 |
  | **Small 9001 (untouched)** | 0.8930 | 0.8935 | +0.0005 (+0.0000 to +0.0010) | 4 / 1 |

  - **On Full the gain did not shrink on untouched episodes** (+0.0111 against +0.0100 where the pieces were
    found). The largest loss in an untouched Full episode is 24 bn USD, the largest gain 200 bn. Against v10 on
    the same 64 episodes v12 is +0.0173 (+0.0140 to +0.0208), 0.8988 to 0.9161.
  - **On Small it changes little,** as its pieces said it would.
  - `sbf check mpc_pulse_rec_v12` passes under Ubuntu on Small and Full. Timed back to back with v11 on the same
    Full episode on a loaded machine: slowest week 1.55 s of 4 s and median 0.48 s, against 1.87 s and 0.50 s for
    v11, so it costs no more CPU.
  - **Uploaded on 2026-10-09: 0.8910 on the public board, no fallback week** (the best upload before it:
    `mpc_pulse_rec_v10_on`, 0.8854).

- **Closed on 2026-10-10 before anything was built (Full root 12345, 16 episodes, against v12; 90 % paired
  intervals).** Two probes that need no code, one rule that was built and dropped, and what recorded episodes show.
  - **The ends of visible events.** v12 with the true end of every closure, capacity cut and grid shortage it sees
    (`tune.py`'s `truth` option): +0.0025 (+0.0011 to +0.0040, 13 episodes better / 3 worse). Hedging end dates
    with scenarios can only take a part of that, and two scenarios in the window take 6.6 to 7 s for a first week.
  - **The window-end value.** `end_value_fuel` 4 and 8: -0.0000 and -0.0001. `end_value` 1.0 and 0.5 with the
    fuel's kept at 2: -0.0014 (-0.0020 to -0.0009) and -0.0097 (-0.0121 to -0.0069). A value per good and node
    from the clairvoyant plan's shadow prices was not built: in the window every chip's price is pinned at the
    credit itself, and the exact cost-to-go (the plan told everything to the episode's end) had already measured
    below the scalar. A window with coarse far weeks was not built either: a truthful 32-week window is -0.0047.
  - **A buffer of chips at sinks: built, nothing.** All 16 Full sinks lose what they cannot serve, and the plan
    sends a sink its forecast and no more. As a soft minimum in the window the plan met it by "not selling" in
    its first week, which is not the agent's to decide (a linear program cannot say that a sink sells what it
    holds). As a top-up after the planner, from packaged chips the plan leaves unused for four weeks and within the
    capacity the week's flows leave: -0.0000 (6 / 10). The routes into sinks are full while chips pile up at the
    OSATs, so there is nothing to hold a buffer with (`analysis/v13/probes/sink_buffer.py`).
  - **What three recorded episodes show** (root 12345, episodes 0, 3 and 7; `analysis/v13/probes/`). Japan dark in
    four weeks of five for the whole episode: both edges that carry crude into its terminal are cut to 0.21 of its
    burn from week 1, and the pulse runs at that duty. Europe dark for 101 weeks: its crude is banned for 104.
    Powered fabs in the US and Europe at 4 to 50 % of capacity: the routes out of the fab carry 10 to 40 % of what
    it makes, or its one material source has no supply. None of it is the agent's doing.
  - **A sweep of 31 settings on v12 was started and not finished** (three values played on two sets, all within
    0.0005 of zero; the variants are `w13_*`).
- **The week's CPU (2026-10-10).** What a week over the budget costs, how near the budget the agent comes, and a
  guard for the week that runs out all the same (`guard` and `preload` in `agent.py`; `analysis/v13/cpu/`).
  - **A week played by the naive rule is dear.** With one week handed to the planner's own naive rule, as the
    server does with a week over its CPU budget (Full root 12345, 16 episodes, every episode worse each time):
    week 1 -0.0073 (-0.0089 to -0.0057), weeks 1 and 2 -0.0132 (-0.0155 to -0.0111), week 30 -0.0110 (-0.0130 to
    -0.0091). That is the size of a whole version.
  - **The slowest week is the first:** a solve without a basis to start from, 10 to 15 thousand simplex iterations
    on Full where a later week takes 2 to 4 thousand. Nothing makes it faster and keeps the plan: a starting
    basis shipped with the agent halves the iterations and saves no time (a basis switches presolve off), no solver
    option gains more than a third, and the same optimum reached another way plays other flows (the window has
    ties: two optimal first weeks differ by 100,000 units on some slots).
  - **The scoring host is no slower than the laptop.** The package says its `mpc_scen` fits 1 s a week on Small at
    3 scenarios and not at 4 on that host (`shockbench_flow/policies/mpc_scen.py`, `S_BUDGET`). On the laptop,
    under the load of nine play workers, they take 1.2 to 1.45 s and 1.3 to 1.8 s (the p95 of an episode's weeks),
    and v13's first Full week 1.3 to 3.3 s over the 20 dev episodes in the same conditions. So the host runs at
    0.56 to 0.83 of those times: the first week about 1.2 s there for the median episode and 2.4 s at most, of 4 s.
    (An earlier figure of ours, the host two to two and a half times slower, came from one unrelated number.)
  - **The guard.** Every solve is given what is left of 0.8 of the week's budget (HiGHS's `time_limit`). A week
    whose solve is cut plays, through the pulse, last week's plan for it if there is one, else the first week of a
    5-week window solved by 0.92 of the budget (a fifth of the full window's time), else an older plan, else the
    naive rule. A solve cut with presolve on returns no basis, so the weeks after it solve without presolve from
    the basis the last cut run reached, until one ends. With the limit not reached the agent plays as v12, to the
    cent; with the solve cut on purpose a cut costs -0.0006 to -0.0014 in week 1 (stand-in windows of 5 to 16
    weeks, no order in it), -0.0012 in weeks 1 and 2, -0.0009 in week 30: a tenth to a fifth of the naive week.
  - **The preload.** The instance and its naive plan are built at import, which the server does not meter, from the
    two public instance files in the agent's `instances/`: 0.10 to 0.15 s off the first week on the laptop.
- **Three more passes over what v12 loses on Full (2026-10-10).** Each was built in a work folder of its own on v12
  and measured against it on the 64 Full episodes of roots 12345, 2026 and 777; the scripts are in `analysis/v13/`.
  - **Where the pulse leaves a grid as it is (`stand_down/`).** A census of every grid-week of those 64 plays in
    which the fabs got less than their draw (46,144 grid-weeks), by the branch of the pulse that left the week as
    it was, with the chips missed that week as what is at stake (an upper bound: the parts add to 0.29 where the
    real gap is 0.117, because not every chip is needed):

    | What the pulse did | Grid-weeks | Of the score, at most |
    | --- | --- | --- |
    | Holding, dark by design (terminal, pipe, one charge) | 10,001 | 0.217 |
    | Stood down: the fuel is cut to nothing, or has no terminal (nuclear) | 2,356 | 0.034 |
    | Stood down: not worth a tank (`pulse_payback`) | 1,948 | 0.009 |
    | Energy shock, held or not | 695 | 0.019 |
    | Opening hold | 193 | 0.007 |
    | Left alone by choice (`pulse_partial`) | 251 | 0.0010 |
    | Whole weeks with a small fuel missing | 23,537 | 0.004 |

    The one stand-down the pulse makes by choice is worth a thousandth. In every week a grid was stood down for a
    fuel that cannot be held, that fuel takes the fabs' whole draw by itself, so nothing is gained by holding the
    others. Two rules were built: `mild_start` (a grid left alone by `pulse_partial` is pulsed once the chips
    missed since its last whole week are worth more than the dark week a start costs; no setting) gains +0.0002
    (+0.0000 to +0.0004), 4 episodes better and none worse, all of it crude at China; `mild_ends` (a partly powered
    week that a dark week must follow is held instead) gains +0.0001 (14 / 6) and is left off.
  - **Runs of whole weeks and the wafer top-up (`runs/`).** `schedule_ends` ended 67, 69 and 234 runs at grids
    that are not pulsed (Full 12345, Full 2026, Small 6006), and all but 2, 2 and 9 of them are grids whose gas
    arrives at 0.8 to 0.9 of its burn, where a dark week in six is the cycle; a dark week is nearly free where the
    fabs are short of wafers, which is why the earlier "ends" check lost. Three fixes, all left off: `ends_needed`
    +0.0001 (-0.0001 to +0.0003; 4 / 4, 56 equal); `runs_reach` loses about as often as it gains; `wafers_plan`
    (the top-up aims at the lots the window plan itself starts) +0.0000 on Full (-0.0002 to +0.0002) and +0.0006
    on Small (+0.0002 to +0.0011 over 128 episodes). The known 46 bn loss turned out to be a grid pulsed for chips
    in surplus (lost sales flat at 24 to 38 % while their stock rose from 2.7 to 9 weeks of demand): a pulse that
    knew would gain about 0.001 with hindsight, and neither signal tried (the fab's chip store, the plan's own
    stock growth) tells such an episode from the pulse's best.
  - **The twelve worst episodes against the clairvoyant plan (`worst_episodes/`).** Their shortfall is 11,573 bn
    USD. A bound that keeps base load before fabs for every subset of a grid's fabs puts 7,588 of it (66 %) out of
    any agent's reach. Of the 3,985 left, about 2,000 is fab energy per unit of the scarcest fuel (Korea 1,040,
    Japan 509, China 215: the pulse's ground, see the census) and about 1,300 is fuel in the wrong place when a
    route is cut. The plan bridges a long closure with fuel it parked beyond the strait beforehand (30,000 GWh of
    gas and 34,000 of crude at one strait in one episode); v12 holds nothing there, because its reserve ends where
    the storage of a grid and its terminal does, at a few weeks of burn.
- **v13, `agents/mpc_pulse_rec_v13`, confirmed (2026-10-10).** v12 with the reserve at straits, `mild_start`, the
  guard and the preload; `mild_ends`, `ends_needed`, `runs_reach`, `wafers_plan` and `reserve_sources` are in the
  code and off.
  - **The reserve at straits (`strait_reserve`, in `reserve.py`).** Tanker cargo waits at a strait until the agent
    releases it, and a strait's queue has no limit. The soft minimum of the grids' fuel reserve is put on a fuel's
    queue at the last strait before the terminals, wherever the fuel reaches that strait through another one:
    26 weeks of what the strait can pass on to the grids behind it (each edge into a grid's terminal at its
    capacity in the window, up to the burn it feeds, within the strait's throughput), at the reserve's own price,
    tapering to nothing at the episode's end and absent in a week the queue is charged at a war-risk rate. It
    names no strait, grid or fuel. Alone on the 64 Full episodes: +0.0046 with 26 weeks (+0.0018 to +0.0080;
    43 / 21) and +0.0036 with 8 (+0.0017 to +0.0060; 46 / 18); only those two were played. The 26 weeks earn
    their extra in four episodes with a long closure (one gains 268 bn against 183 with 8) and cost the rest a
    little holding. Its largest loss, 58 bn in one episode, is not the reserve's own fuel: the plan curtailed
    another supply of a grid whose stores were full, and that route was cut later. `reserve_sources` (the same
    reserve at supply nodes) takes that loss back and loses as much elsewhere; it is off.
  - **The merge is exact.** With the new switches off the merged folder equals v12 to the cent on Full 12345 and
    Small 12345, and with one piece on at a time it equals that piece's own play (the reserve on Full 12345,
    `mild_start` on Full 2026, `wafers_plan` and `ends_needed` with `runs_reach` on Full 12345). The final files
    differ from the merged ones in comments and in the reserve's default.

  | Set | `mpc_pulse_rec_v12` | `mpc_pulse_rec_v13` | Difference (90 % interval) | Better / worse |
  | --- | --- | --- | --- | --- |
  | Full 12345 (pieces found here) | 0.8936 | 0.8948 | +0.0012 (-0.0001 to +0.0025) | 12 / 4 |
  | Full 2026 (pieces found here) | 0.9128 | 0.9146 | +0.0018 (-0.0007 to +0.0046) | 10 / 6 |
  | Full 777 (pieces found here, 32 episodes) | 0.8702 | 0.8780 | +0.0078 (+0.0033 to +0.0136) | 21 / 11 |
  | Full, those three as one (64) | 0.8830 | 0.8877 | +0.0047 (+0.0020 to +0.0081) | 43 / 21 |
  | **Full 9001 (untouched, 32 episodes)** | 0.9194 | 0.9204 | +0.0009 (+0.0001 to +0.0018) | 21 / 11 |
  | **Full 5150 (untouched, 32 episodes)** | 0.9128 | 0.9153 | +0.0025 (+0.0002 to +0.0050) | 20 / 12 |
  | **Full, the two untouched sets as one (64)** | 0.9161 | 0.9182 | +0.0021 (+0.0008 to +0.0036) | 41 / 23 |
  | Small 12345 and 2026 as one (128) | 0.8892 | 0.8946 | +0.0054 (+0.0032 to +0.0079) | 91 / 37 |
  | **Small 9001 (untouched)** | 0.8935 | 0.8983 | +0.0048 (+0.0018 to +0.0078) | 40 / 24 |

  - **On Full a bit under half of the gain holds on untouched episodes** (+0.0021 against +0.0047). The reserve
    pays in episodes where a strait or a route is cut for long after the episode has begun, and the sets it was
    found on have more of them (four carry the 26 weeks' extra). The largest loss in an untouched Full episode is
    25 bn USD, the largest in a used one 58.
  - **On Small nearly all of it holds** (+0.0048 against +0.0054), the first gain there since v10.
  - **What it costs.** 3 to 6 % more simplex iterations a week. `sbf check mpc_pulse_rec_v13` passes under Ubuntu on Small and Full: on a quiet machine the slowest week is the first, 1.09 s of 4 s on Full (median 0.33 s) and 0.13 s of 2 s on Small; v12 timed back to back takes 1.09 s (median 0.34 s), so v13 costs no more CPU.
  - **Uploaded on 2026-10-10: 0.8972 on the public board, no fallback week** (v12: 0.8910; the gain there, +0.0062,
    is the one the Small sets showed).

## Reproducing a number

Every row above was cached under `outputs/recovery_tune/<task>-e<root>-n<episodes>/<name>.json` on the machine that
played it (not committed). To replay one, write a variants file and run the harness; a `base` is looked up in
`agents/` and then in `versions/`:

```
{"c_pulse": {"base": "ft_buffer", "params": { ...the folder's params.json... }}}

uv run python experiments/approaches/tune.py run --variants=<file> --task=full --entropy=2026 --episodes=16 --n_jobs=2
uv run python experiments/approaches/tune.py report --base=mpc_det_h20 --task=full --entropy=2026 --episodes=16
```

`report` prints every cached variant against the base with a 90 % paired bootstrap interval.

## After v8 (2026-10-08)

- **`agents/mpc_pulse_rec_v9b`:** `mpc_pulse_rec_v8_on` with `pulse_run` 6 (was 4), no code change. About +0.0005
  over eight local sets (one significant gain, no significant loss), close to noise. **Uploaded on 2026-10-08: 0.8767
  on the public board** (v8_on 0.8768), no fallback week. It came out of the experiments in
  [experiments/window_end/README.md](../window_end/README.md): longer and shorter windows, a tapered or final
  window-end value, sweeps of the settings on Small and Full, and the cost-part comparison against the clairvoyant
  plan. None of them gained.
