# Agent approaches the package's policies do not cover (2026-10-04, first results 2026-10-05)

Eight ways to improve on the `mpc_det` port, from a read of the installed `shockbench-flow` 0.1.2 source: which
observation fields its policies use, and how the generator makes the signals they ignore. All eight are upgrades to
`agents/mpc_det_h20`, not new agent families. The order is a judgement from the source and the baseline scores; only
approach 1 has been measured (see [Measured so far](#measured-so-far-approach-1)).

Both boards play the same generator settings (gamma 0.62, `flagship_profile`, regime `standard`), so the generator
facts below apply to Small and to Full. Every score quoted is from Small.

## What the package's policies already cover

| Family      | Policies                                                             | Idea                                             |
| ----------- | -------------------------------------------------------------------- | ------------------------------------------------ |
| Stock rules | `naive`, `nd`, `sz_state_base_stock`, `human_ref`                    | order up to a target level per destination       |
| One-week LP | `greedy_lp`                                                          | cheapest flows this week toward naive's targets  |
| Rolling LP  | `mpc_det` and its window, demand x1.1 and throughput-scale variants  | plan on "what I see now stays as it is"          |
| Scenario LP | `mpc_scen`, `hindsight_consensus`                                    | average over a few draws of the generator        |
| Kit         | `heuristic`, PPO from scratch, evolutionary search on the heuristic's numbers |                                         |

## What they leave out

- **Warnings and announcements:** no policy reads `warning.score` or `messages.*` (`policies/scenarios.py` says so). A
  regime filter on the score, `hamilton_filter`, is named in `sz_state_base_stock.py` but exists nowhere.
- **Recovery times:** `mpc_det` plans every impairment it sees as lasting the whole window. Only the scenario policies
  use the duration laws, by sampling.
- **Onsets that depend on what was seen:** the scenario policies draw future events from one fixed public library, the
  same draws in every episode.
- **Decoys:** `pending_prohibitions` lists sanctions that never take effect, yet `mpc_det` treats every entry as
  certain.
- **The value of stock at the window's end:** it is the static end-of-episode salvage price per unit.

## The signals in numbers

Source column: **checked** was read directly in the package source; **helper** comes from a helper agent's read of
`disruption/` and `information/`, with some values computed from the package's closed-form functions, and was not
re-checked.

| Signal | Fact | Source |
| --- | --- | --- |
| `warning.score` | `S = 0.604 X + 0.797 W`, one week late; X is a hidden hazard level, W is noise | checked |
| | X and W have the same week-to-week correlation (0.95 straits, 0.9991 regions), so the history of S adds nothing to this week's S | checked |
| | It shifts the chance that an event starts, never how long it lasts | helper |
| | Closure rate multiplier at S = -2 / 0 / +2: 0.12 / 0.86 / 2.6 for a strait | helper |
| | Militarised closures: 0.43 per year in total (Malacca 0.18, Taiwan 0.13, Hormuz and Suez 0.058 each) | helper |
| | The scores of Cape, Panama and the Turkish straits predict nothing | helper |
| `messages.*` | Share of threads that never take effect: formal tariff 0.25, informal tariff 0.36, sanction threat 0.538, closure threat 0.298 | checked |
| | No field separates a decoy from a real thread; only its withdrawal at the effect time reveals it | helper |
| | Closure threat (`channel` 5, `target_kind` 0): 16 % give no lead, the rest a median of 11 days (nine in ten within 81); no date is stated | helper |
| | Formal tariff proposal: median lead 66 days, the exact week stated | helper |
| | A real threat with no date stays listed after its event starts | helper |
| `pending_prohibitions.*` | Built from every shown sanction publication, decoys included | checked |
| | About 9.5 % of sanctions are published before taking effect, 2 to 180 days ahead (median 32) | helper |
| `closure_end.*` | Always empty under the scored regime | checked |
| Durations | Weather closure: median 6 days, nine in ten over within 17; shows as `open` 0 with war-risk class 0 | helper |
| | Militarised closure: median 31 weeks (one in ten under 3.9, one in ten over 217); partial, `open` 0.03 to 0.73 | helper |
| | A CN-TW conflict closes the Taiwan Strait fully for at least 52 weeks | helper |
| | Sanctions last years on average, tariffs a median of one year; no duration law is memoryless, so age matters | helper |
| Week 1 | About 0.7 closures, 22 sanctions and 6 tariffs are already active (10 episodes) | helper |
| Demand | The shown forecast is already the demand process's own expectation, and disruptions do not move demand | helper |
| `config` | Holds no generator parameters; an agent must ship them as tables | checked |

## The approaches

### 1. Recovery forecast by impairment type and age

- **Gap:** `mpc_det_h20` plans a closed strait, a struck port or a grid short of power as staying so for all 20 weeks.
- **Idea:** in the same single LP, let each impairment end as the duration law of its event type says, given how long
  it has been seen. A storm's closure is mostly over by next week; a militarised one is not.
- **What it reads:** `graph_now.open` with `graph_now.war_risk` (the closure's type), edge capacity, grid power, supply
  caps, fab and OSAT restoration. No warnings, no announcements.
- **Why it might pay:** short closures are the common ones, and several impairments are already active at week 1.
- **Cost and risk:** no extra CPU. It acts only on what is already in force, so decoys cannot fool it. A full closure
  of the Taiwan Strait can be a storm or a conflict of a year or more; the war-risk class should tell them apart, which
  is not verified.
- **Status (2026-10-06):** built and confirmed. The final rules are in `experiments/pulse_agent/versions/mpc_combined` (`recovery.py`): next to
  the end value the forecast adds +0.013 on fresh episodes of Small and of Full. See [Outcome](#outcome-2026-10-06).

### 2. Act on closure threats

- **Gap:** a threat aimed at a strait (`messages.channel` 5) is ignored until the strait closes.
- **Idea:** when a threat appears, lower the strait's planned open fraction in the coming weeks by the chance the threat
  is real times the chance the closure has started by then. Cargo is then sent early or by another lane instead of
  arriving at a closed strait and queueing.
- **Why it might pay:** about 70 % of these threats are real, with a median lead of 11 days.
- **Cost and risk:** no extra CPU. The 30 % false alarms cost money in calm episodes, which carry half the score
  weight. The agent must track each thread's age, because a real threat stays listed after its closure starts.
- **How to test first:** give the planner the true start week of every threatened closure and compare on our own root.
  If that ceiling is small, drop it.
- **Status (2026-10-06):** dropped. Its ceiling was measured: every closure known two weeks ahead with its true
  duration adds +0.0008; fed to the planner without a duration it loses 0.002.

### 3. Fuel buffers and tuned planner knobs

- **Gap:** nothing in the planner asks for a reserve of fuel at grids. Its demand x1.1 variant scored 0.733 against
  0.729 on the 20 dev episodes and does not build one.
- **Idea:** give fuel held at grids a value in the LP (or a minimum level), then search the planner's numbers (window,
  demand scale, throughput scale, stock value) on our own root.
- **Why it might pay:** shed power and unserved demand are about 2,900 of the heuristic's 2,944 bn USD per episode;
  freight, tariffs and holding together are 43. Holding stock is cheap next to running out.
- **Cost and risk:** a search can fit the episodes it sees, and every knob must be free of the network's scale to
  carry over to Full.
- **Status (2026-10-06):** built (`reserve.py` in `experiments/pulse_agent/versions/mpc_buffer` and `experiments/pulse_agent/versions/ft_buffer`). The reserve is worth
  little by itself next to the end value, and 0.014 on Full next to the pulse. Demand and throughput scales did nothing.

### 4. Discount pending sanctions for decoys

- **Gap:** `mpc_det` reroutes for every entry of `pending_prohibitions` as if it were certain.
- **Idea:** plan a pending sanction as taking effect with its real probability: a second scenario in the LP, used only
  in weeks with a pending entry.
- **Why it might pay:** perhaps half the entries are decoys. That 0.538 is an estimate from the sanction threat's decoy
  share, not a measured rate.
- **Cost and risk:** small upside, since under one sanction in ten is published ahead at all.
- **Status (2026-10-06):** dropped. Its ceiling (decoys removed from the pending list) is +0.0004, within noise.

### 5. A learned value for stock at the window's end

- **Gap:** stock left at the end of the planning window is priced at the end-of-episode salvage price, so a short
  window undervalues it. The score falls from 0.73 with a 24-week window to 0.71 at 20 and 0.54 at 12 (20 dev
  episodes).
- **Idea:** price that stock by node and good from the clairvoyant plan's shadow prices, solved offline on generator
  draws.
- **Cheap first step:** set `H_WEEKS = None` (24 weeks) and try longer windows, a one-line change that shows whether
  there is any room.
- **Cost and risk:** offline solves only; no extra CPU per week. A longer window may already capture the gain.
- **Status (2026-10-06):** built in its simplest form, with nothing learned: goods at the window's end are credited a
  share of their customs value (`end_value`). It is worth +0.074 on Small and +0.33 on Full on fresh episodes. Windows
  of 24 to 40 weeks gain +0.031 to +0.040 without it and nothing with it.

### 6. A hazard model trained on generator draws

- **Gap:** nothing estimates the chance that an event starts soon. The warning score alone is weak.
- **Idea:** the generator is public, so labelled data is unlimited. Train a small model offline that predicts, per
  strait and region, the chance of an onset in the next weeks from what the agent sees: the score, live threats,
  recent events (onsets cluster for about 3 weeks in a region), a new tariff's rate (0.25 against 0.10 reveals
  tension), capacity cuts between two regions (a conflict).
- **Cost and risk:** ships as numpy weights. It earns nothing by itself; it only feeds approaches 2 and 7.
- **Status (2026-10-06):** dropped. Every onset inside the 20-week window, known perfectly, is worth +0.013; at a
  two-week lead +0.003.

### 7. Weighted contingency scenarios

- **Gap:** `mpc_scen` hedges against draws that ignore what was seen. It gains 0.018 over `mpc_det` on the 20 dev
  episodes, within noise, at about 5 times the CPU; its first week took up to 8.5 s on this laptop, over the 2 s
  budget.
- **Idea:** one extra scenario aimed at the live threat ("the threatened strait closes"), weighted by its probability,
  with this week's shipments shared by both scenarios.
- **Cost and risk:** CPU. Add the scenario only in weeks with a live threat.
- **Status (2026-10-06):** dropped, for the same reason as approach 6.

### 8. Distil the hindsight plan into a network

- **Gap:** `hindsight_consensus` is over budget (2.8 to 22 s per week, per its source).
- **Idea:** train a network to predict the planner's knobs from features per strait, so the same network runs on Small
  and on Full.
- **Cost and risk:** the highest risk of the eight for the days left.
- **Status (2026-10-06):** dropped. With full knowledge to the episode's end the planner reaches 0.773 on Small; the
  pulse agent, with no foresight, scores 0.82.

## Outcome (2026-10-06)

The largest gains came from two things that are not on the list above, both found by measuring where the planner's
cost exceeds the clairvoyant plan's:

- **The window's end** (approach 5 in its simplest form): +0.074 on Small, +0.33 on Full.
- **The pulse:** scarce fuel delivered to a grid in whole weeks, so its fabs run in some weeks instead of never. Next
  to the end value and the forecast it adds +0.065 on Small and +0.224 on Full, on fresh episodes.

The agent with both, `experiments/pulse_agent/versions/ft_buffer`, scored 0.8289 on the public board. How it works, every measurement and what
was tried and left out are in [experiments/pulse_agent/README.md](../pulse_agent/README.md). The harness that played
the variants is `tune.py` here; `foresight.py` feeds the planner ground truth to measure ceilings.

## Measured so far: approach 1

Small, 64 episodes of our own (root 12345; 31, 20, 8 and 5 episodes at harm levels 1 to 4), every variant on the same
episodes. The difference is against `mpc_det_h20`, with a 90 % paired bootstrap interval.

| Variant | Score | Difference (90 % interval) | Episodes better / worse |
| --- | --- | --- | --- |
| `mpc_det_h20` | 0.6856 | | |
| Forecast, expected share of each week still impaired (`mode` "mean") | 0.6799 | -0.0057 (-0.0142 to +0.0021) | 25 / 39 |
| Forecast, impaired until the median of its remaining life (`mode` "quantile") | 0.6954 | +0.0098 (+0.0039 to +0.0159) | 37 / 27 |
| **Ceiling:** true recovery times, every kind | 0.7120 | +0.0264 (+0.0207 to +0.0327) | 61 / 2 |
| Ceiling, straits only | 0.6932 | +0.0076 (+0.0054 to +0.0101) | 39 / 2 |
| Ceiling, edge capacity only | 0.6947 | +0.0091 (+0.0071 to +0.0112) | 58 / 3 |
| Ceiling, grids only | 0.6940 | +0.0084 (+0.0041 to +0.0133) | 15 / 1 |
| Ceiling, supply caps only | 0.6856 | 0.0000 | 2 / 1 |

- **The whole idea is worth at most 0.026 on this root.** A planner told exactly when each impairment it sees ends
  scores 0.712. The rest of the distance to the clairvoyant plan lies elsewhere: events that have not started yet, and
  the window's end.
- **Straits, edge capacity and grids share that ceiling about equally;** supply caps add nothing.
- **How the forecast enters the LP matters.** Fading each impairment out in expectation is no better than the
  baseline. Holding it until its median remaining life and then dropping it gains 0.010, with an interval above 0.
- **Not yet done:** the ceiling for fabs and OSATs, the forecast for one kind at a time, a confirmation on the 20 dev
  episodes, `sbf check` on Small and Full. No variant ran on Full.
- **Caution:** the median variant is one of two forecasts tried on one root. Confirm it on other episodes before
  trusting 0.010.

The costs per episode are in `outputs/recovery_tune/small-e12345-n64/` (not committed). The script that played the
variants is `tune.py` in this folder. This table is from the first version of the forecast; the later measurements,
with a corrected classification of capacity cuts, are summarised in `experiments/pulse_agent/README.md`.

## Not worth building

- **Smoothing or Kalman-filtering the warning score:** the signal and its noise have the same week-to-week
  correlation, so this week's score already holds everything the history does.
- **The warning score as a trigger by itself:** a score of +2 multiplies a strait's closure rate by about 2.6, on base
  rates of 0.06 to 0.18 closures per strait-year.
- **A better demand forecast:** the shown forecast is already the conditional mean, and disruptions do not move demand.
- **`closure_end.*`:** always empty on the scored regime.

## How far to trust this

- **Checked in the source:** which fields the policies read, the score formula, the decoy shares, how pending
  sanctions are built, the empty `closure_end`.
- **From a helper agent's read, not re-checked:** the duration laws, the announcement lead times, the base rates, the
  week-1 counts.
- **Measured:** only the table above, plus the baseline scores quoted from the 20 dev episodes of Small (one standard
  error there is 0.17 to 0.20).
- **Package version:** everything is from 0.1.2; re-read the source if it changes.
