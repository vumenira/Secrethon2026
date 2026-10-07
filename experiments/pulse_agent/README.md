# The pulse agent: every version and measurement (2026-10-06 to 2026-10-07)

This is the detailed log behind our agent. For the short version, read [SOLUTION.md](../../SOLUTION.md) first.

- **The final agent is `agents/mpc_pulse_rec_v8_on`** (0.8768 on the public board). Every earlier version is kept,
  runnable, under `versions/<name>` beside this file (`experiments/pulse_agent/versions/`); a version named below
  without a path is one of those folders.
- **Part one** (down to "Risks and limits") describes the first version, `versions/ft_buffer`: the benchmark's rolling LP
  planner with a value for goods at the window's end, a fuel reserve at grids, a "pulse" that delivers scarce fuel in
  whole weeks so fabs can run, and a price that sends scarce fuel to the grid that loses most. It reads no warnings and
  no announcements and has no learned parts. Uploaded on 2026-10-06, it scored 0.8289.
- **Part two** ("Later versions") adds one entry per version, v2 to v8, each with what it changed, what was measured
  and what was dropped.
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

## Later versions (v2 to v8, in the order they were built)

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
- **v8, `versions/mpc_pulse_rec_v8` (rule off) and `agents/mpc_pulse_rec_v8_on` (rule on):** the opening hold as a rule
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
