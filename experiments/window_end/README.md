# Window length and the window-end value (2026-10-08)

## Summary (read this first)

Experiments on `agents/mpc_pulse_rec_v8_on` (0.8768 on the public board), about 4 hours of compute on this laptop,
everything played paired on the same episodes, tuned on root 12345 and confirmed on root 2026 and the dev episodes.

| Question | Answer |
| --- | --- |
| Longer planning window (24, 32, 48 weeks)? | **No.** -0.0004, -0.0073, -0.0160 on Small; worst at harm level 1. Shorter is worse too (16: -0.003, 12: -0.020): 20 is a sharp optimum. |
| Plan to the episode's end, leftovers worth nothing? | **Already so**, and confirmed: the window shrinks to the weeks left, and leftovers then get salvage only. Crediting them more loses in 55+ of 64 episodes (-0.0013 to -0.0019). A tapered end value before that point is neutral (+-0.0003). |
| How to cut shed? | **Shed is not the gap.** The clairvoyant plan sheds as much (111,416 vs our 111,318 bn over 64 episodes). The whole gap is unserved chips (+8,515 bn), mostly from fabs on grids 2 and 3, and 49-78 % of the clairvoyant plan's fab energy comes from shedding base load for fabs, which the simulator forbids. |
| Any setting worth changing? | **Not really.** The best, `pulse_run` 6 (was 4), averages +0.0004 on Small and +0.0005 on Full over four sets each (one significant gain, one small loss on Full root 777); within the noise. Everything else in two sweeps (27 settings on Small, 13 on Full) is flat or did not hold on fresh episodes. v8 is at a local optimum of its settings. |

**What was implemented:** `agents/mpc_pulse_rec_v9b` = `mpc_pulse_rec_v8_on` with `{"open_hold": true,
"pulse_run": 6.0}` in `params.json` (no code change; `v8_on` is untouched). `sbf check` passes: slowest week 0.16 s of
2 s on Small, 0.80 s of 4 s on Full (`--docker` not run). Expected gain about +0.0005, close to noise. **Uploaded on
2026-10-08 (submission 970176): 0.8767 on the public board, no fallback week**, against 0.8768 for `v8_on`: the
same agent within noise, as the local sets said. `v8_on` remains the reference agent.

**New tools here:** `report.py` (paired differences by harm level), `gap.py` (our costs against the clairvoyant
plan's per week, component, grid and fab), `lots.py` (why fabs start fewer lots). The experimental agent with the two
end-value switches is `experiments/pulse_agent/versions/mpc_pulse_rec_v9b_x` (both off = `v8_on` to the cent).

**Leads for later** (not built; see the end of this file).

The rest of this file is the log, in the order the experiments ran. No agent was changed until a variant won on the
tuning root and held on fresh episodes.

## Setup

- **Harness:** `experiments/approaches/tune.py` (results cached in `outputs/recovery_tune/<task>-e<root>-n<N>/`),
  variants in `variants.json`. `report.py` here breaks a comparison down by harm level, with a 90 % paired interval
  per level.
- **Roots:** tune on Small 12345 (64 episodes), confirm on Small 2026 (64) and the 20 dev episodes, then Full.
- **Machine:** i5-1135G7 (4 cores, 8 threads), 7.6 GB RAM; 3 workers on Small, 2 on Full.
- **Experimental agent:** `experiments/pulse_agent/versions/mpc_pulse_rec_v9b_x` is `agents/mpc_pulse_rec_v8_on` with
  two more settings, `end_value_taper` and `end_value_last` (0 = off, both). With them off the folder reproduces
  `v8_on` to the cent (variants `v9b_x_off` and `v9b_x_off2`).

## Replication of the submitted agent (dev episodes, `sbf view`)

| Task | Logged | Replicated here | Fallback weeks |
| --- | --- | --- | --- |
| Small | 0.9089 | 0.9089 (0.878 to 0.936) | 0 of 1,040 |
| Full | 0.8961 | 0.8961 (0.879 to 0.913); levels 0.902 / 0.911 / 0.871 / 0.846 | 0 of 2,080 |

Full reports "2080 action entries ignored as invalid": one per week, `overrides[53]: slot 53 masked this week` (the
planner fills every strait-queue override slot; Full's slot 53 is always masked). The simulator drops that entry and
plays the rest, so it costs nothing.

## Experiment 1: longer planning windows

The question: with CPU to spare, does planning 24, 32 or 48 weeks ahead beat 20? (Tested on v1/v2 in
`experiments/pulse_agent/README.md`: no gain with the end value on, -0.009 on Full at 24 weeks.)

## Experiment 2: a tapered window-end value

The question: the end value credits goods at the window's end at 2.0 times their customs value until the window
reaches the episode's end (week 33 on Small), then they drop to salvage. Stock at the window's end is worth less the
fewer weeks follow it. `end_value_taper` W scales the share by `min(1, weeks after the window's end / W)`.

Already in the agent (no experiment needed): the window shrinks to the weeks left (`H_t = min(H, T - t + 1)`,
`sbflow/policies/mpc_det.py`), and the end value is off once the window reaches the episode's end. Goods left then
keep the salvage price, which the episode's cost really credits (cost minus salvage), so it stays.

## Results

### Small, root 12345 (tuning, 64 episodes: 31 / 20 / 8 / 5 by harm level), 2026-10-08

All 8 variants in 16 min with 3 workers. No fallback week, no error. `v9b_x_off` equals `v8_on` to the cent on all 64
episodes; `v8_on` reproduces the log's 0.8760.

| Variant | Score | Diff vs v8_on (90 % interval) | Better / worse episodes | Level 1 | Level 2 | Level 3 | Level 4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `v8_on` (h 20) | 0.8760 | | | 0.8772 | 0.8676 | 0.8750 | 0.9284 |
| `h24` | 0.8755 | -0.0004 (-0.0018 to +0.0009) | 33 / 31 | -0.0005 | -0.0000 | -0.0013 | +0.0002 |
| `h32` | 0.8687 | -0.0073 (-0.0106 to -0.0044) | 22 / 42 | -0.0100 | -0.0064 | -0.0034 | +0.0003 |
| `h48` | 0.8600 | -0.0160 (-0.0212 to -0.0114) | 14 / 50 | -0.0211 | -0.0124 | -0.0116 | -0.0041 |
| `taper8` | 0.8761 | +0.0001 (-0.0002 to +0.0003) | 46 / 12 | +0.0002 | -0.0003 | +0.0005 | +0.0002 |
| `taper16` | 0.8758 | -0.0002 (-0.0007 to +0.0002) | 43 / 16 | -0.0001 | -0.0010 | +0.0007 | +0.0002 |
| `taper26` | 0.8757 | -0.0003 (-0.0010 to +0.0004) | 40 / 23 | -0.0003 | -0.0013 | +0.0014 | +0.0002 |

(Level columns: difference in that level's score; per-level intervals from `report.py`. Significant: h32 at levels
1-2, h48 at every level; the tapers at level 3, +0.0005 to +0.0014.)

- **Longer windows lose, more the longer.** The loss is largest in the calm episodes (level 1, half the board's
  weight). A likely reason: the planner assumes the network it sees now stays (save the recovery forecast), and over
  32 to 48 weeks that wrong forecast drives more of the plan; it also switches the end value off earlier (from week
  21 at h 32, week 5 at h 48), which was the largest single gain of the agent. Slowest episode 5 s at h 20, 7 s at
  h 32 and h 48 (CPU is not the limit on Small).
- **The taper does nothing overall.** It moves the score by at most 0.0003; the small gain at harm level 3 is paid
  back at level 2. The end value's flat share is already about right near the episode's end.
- **Decision:** neither goes into the agent. The trend suggested the opposite question, a shorter window (below).

### Follow-ups on Small 12345: shorter windows, and an end value at the episode's end

`end_value_last` (added to `mpc_pulse_rec_v9b_x`, 0 = off): the share credited to goods left when the window reaches the
episode's end (normally they get the salvage price only, which is what the episode's cost credits). `v9b_x_off2`
(both new switches off, after the change) again equals `v8_on` to the cent.

| Variant | Score | Diff vs v8_on (90 % interval) | Better / worse episodes |
| --- | --- | --- | --- |
| `h12` | 0.8563 | -0.0197 (-0.0237 to -0.0158) | 10 / 54 |
| `h16` | 0.8731 | -0.0029 (-0.0046 to -0.0014) | 24 / 40 |
| `h18` | 0.8729 | -0.0031 (-0.0042 to -0.0021) | 18 / 46 |
| `last05` (h 20, share 0.5 at the end) | 0.8747 | -0.0013 (-0.0018 to -0.0008) | 7 / 57 |
| `last1` | 0.8744 | -0.0016 (-0.0023 to -0.0010) | 9 / 55 |
| `last2` | 0.8741 | -0.0019 (-0.0027 to -0.0012) | 9 / 55 |
| `h32_last2` (h32 alone: -0.0073) | 0.8703 | -0.0057 (-0.0080 to -0.0037) | 17 / 47 |
| `h48_last2` (h48 alone: -0.0160) | 0.8670 | -0.0090 (-0.0120 to -0.0061) | 17 / 47 |

- **20 weeks is a sharp optimum:** 16, 18 and 24 all lose (24 within noise). The window is not a lever.
- **At the episode's end, leftovers are best valued at salvage only**, as the agent already does: any share above it
  loses in 55 of 64 episodes or more. This is the participant's hypothesis 2 ("the experiment ends, leftovers are
  worth nothing to us") confirmed as far as it goes; the salvage credit itself is real money and stays.
- **Why long windows lose, split:** giving back the end value at the window's (= episode's) end recovers about half
  of h48's loss and a fifth of h32's. The rest is the long window itself: more of the plan rests on the forecast
  that today's network persists.

## Question 3: where the cost goes, against the clairvoyant plan (Small 12345, 64 episodes)

`gap.py` plays the agent and solves each episode's clairvoyant LP, saving per week the cost components, shed per
grid, lots, fab energy and wafers on hand; `lots.py` and the inline analysis below read them. Our cost per episode
reproduces the cached one to the cent.

**Shed is not the gap.** Sums over the 64 episodes, bn USD:

| Component | Ours | Clairvoyant | Gap |
| --- | --- | --- | --- |
| shortage (chip demand not served) | 53,435 | 44,920 | **+8,515** |
| shed (base load not served) | 111,318 | 111,416 | -99 |
| tariff | 312 | 419 | -107 |
| disposal | 114 | 26 | +88 |
| holding, freight, queue holding, war risk | 498 | 494 | +4 |
| total (salvage aside) | | | 8,400 |

- Shed is 2/3 of every episode's cost, but the clairvoyant plan sheds as much: it is the disruptions' doing. The
  whole gap is unserved chips, 90 % of it in weeks 25 to 52 (lots are about 12 weeks from start to sale).
- **We start 82 % of the clairvoyant plan's lots.** Fabs 3 (grid 2) and 5 (grid 3) run at 51 % and 50 %; fab 0 at
  108 %, fabs 1, 2 at 87 to 89 %. Fabs 3 and 5 hold 94M of the 138M missing lots, and they are energy-bound in 39 to
  41 weeks per episode (the grid sheds base load, so they get nothing).
- **Most of that is out of reach.** 49 % to 78 % of the clairvoyant plan's fab energy is drawn in weeks when it sheds
  base load at the same grid, which the simulator never does (base load first). Against the clairvoyant energy that
  respects the rule, ours is already 1.4 to 2.6 times as much at every fab: the pulse wins that trade.
- **Wafers are never the limit in a week the grid has power to spare** except at grids 0 and 1, where the
  clairvoyant plan leaves its fabs under capacity too (chips are not the bottleneck there).
- **One traced case (episode 3, grid 2):** no LNG at all in weeks 8 to 28 (nothing to pulse), then a steady 865 a week
  against 5,400 needed. The pulse stays off because one whole week's tank (burn plus rationing threshold) takes longer
  than `pulse_hold_weeks` (12) to fill, and from about week 37 the end-of-sales rule stops the fab counting anyway.
  `pulse_hold_weeks` is in the sweep below.

So the answer to "how can we cut shed": we cannot, much. The shed is the price of disruptions, and the pulse already
raises it on purpose where it buys chips. What remains is foresight and the base-load-first rule, as the earlier log
concluded; tuning the rules that exist is the practical lever.

## Experiment 3: one-at-a-time sweep of the agent's settings (Small 12345, 64 episodes)

Each variant moves one setting of `v8_on` (`variants_sweep.json`, 27 variants, 62 min). Diff vs `v8_on`, 90 % paired
interval:

| Setting (default) | Value: diff |
| --- | --- |
| `pulse_run` (4) | 2: -0.0017 (-0.0034 to -0.0001); **6: +0.0006 (-0.0000 to +0.0014)** |
| `pulse_margin` (1.0) | **1.1: +0.0005 (+0.0001 to +0.0010)** |
| `open_hold_last` (4) | 3: -0.0004; **6: +0.0004 (+0.0000 to +0.0009)** |
| `pulse_hold_weeks` (12) | 8: -0.0006 (-0.0014 to -0.0001); 18: +0.0003; 26: +0.0003 |
| `pulse_fill` (0.9) | 0.8: -0.0020 (-0.0036 to -0.0006); 1.0: +0.0002 (-0.0010 to +0.0015) |
| `pulse_partial` (0.5) | 0.3: -0.0004 (-0.0007 to -0.0001); 0.7: -0.0001 |
| `pulse_memory` (4) | 2: -0.0001; 8: -0.0000 |
| `pulse_payback` (12) | 6: -0.0032 (-0.0050 to -0.0016); 24: 0.0000 (identical) |
| `end_value` (2.0) | 1.5: -0.0000; **3.0: -0.0119 (-0.0163 to -0.0081)** |
| `reserve_weeks` (8) | 4: -0.0013 (-0.0021 to -0.0005); 12: +0.0000 |
| `short_premium` (1.0) | 0.5: -0.0001; 2.0: -0.0016 (-0.0025 to -0.0007) |
| `caution` (1.0) | 0.8: -0.0003 (-0.0007 to +0.0000) |
| `ration_trim_cover` (6) | 4: -0.0022 (-0.0029 to -0.0015); 10: -0.0000 |
| `fuel_trim_cover` (6) | 4: -0.0000; 10: -0.0001 |

- **The agent sits at a local optimum.** No move gains more than 0.0006; most are flat, and moving any setting
  "tighter" (less fuel held, shorter runs, smaller reserves, stricter payback) loses 0.001 to 0.003.
- **`end_value` 3.0 loses 0.012 on Small** (the log measured it on Full only, "gains nothing"); 1.5 and 2.0 tie.
  The window-end value is a cliff above 2.
- **Candidates** (each upper interval end near +0.001, and with 27 tries the best few are expected to look good by
  chance): `pulse_run` 6, `pulse_margin` 1.1, `open_hold_last` 6, `pulse_hold_weeks` 18. They point one way: hold
  a little more, deliver a little more per whole week. Next: their combination on this root, then root 2026.

### Candidates combined, and confirmed on Small 2026 (fresh, 64 episodes: 28 / 25 / 9 / 2 by level)

| Variant | Small 12345 (tuning) | Small 2026 (fresh) |
| --- | --- | --- |
| `s_run6` (`pulse_run` 6) | +0.0006 (-0.0000 to +0.0014), 12 / 18 | **+0.0011 (+0.0003 to +0.0021), 18 / 9** |
| `s_run8` | +0.0003 (-0.0007 to +0.0012) | +0.0007 (-0.0003 to +0.0018) |
| `s_margin1.1` | +0.0005 (+0.0001 to +0.0010) | -0.0001 (-0.0004 to -0.0000) |
| `s_margin1.2` | +0.0005 (+0.0001 to +0.0011) | -0.0002 |
| `s_ohlast6` / `s_ohlast8` | +0.0004 / +0.0004 | +0.0002 / +0.0003 |
| `s_hold18` | +0.0003 | 0.0000 (3 episodes differ) |
| `c_run_margin` (run 6, margin 1.1) | +0.0013 (+0.0003 to +0.0024) | +0.0010 (+0.0001 to +0.0020) |
| `c_combo` (run 6, margin 1.1, open_hold_last 6, hold 18) | +0.0016 (+0.0006 to +0.0028) | +0.0011 (+0.0002 to +0.0020) |

`v8_on` on 2026: 0.8741 (the log's number). By level on 2026, `s_run6`: +0.0022 (+0.0008 to +0.0037) at level 1,
-0.0003 and -0.0005 at levels 2 and 3 (intervals hold 0), +0.0050 at level 4 (2 episodes).

- **Only `pulse_run` 6 holds on fresh episodes**; the other candidates' gains on the tuning root were the luck of
  picking the best of 27. The combination is worth no more than `pulse_run` 6 alone on 2026.
- **What it does:** the rationed fuel (LNG) of a pulsed grid is charged only once the terminal holds the charge
  plus 6 whole weeks (4 before), so fabs run in fewer, longer spells. Small and safe, about +0.001.
- Next: Small dev and Full dev (no Full tuning was done, so both are confirmation).

### `pulse_run` 6 on the dev episodes (Small and Full, never tuned on)

| Set | `v8_on` | `s_run6` | Diff (90 % interval) | Better / worse |
| --- | --- | --- | --- | --- |
| Small dev (20) | 0.9089 | 0.9087 | -0.0002 (-0.0013 to +0.0010) | 5 / 6 |
| Full dev (20) | 0.8961 | 0.8971 | +0.0010 (-0.0003 to +0.0027) | 8 / 7 |

Both `v8_on` numbers are the log's. Full plays fast once its references are cached: 20 episodes in 5 min with 2
workers (slowest episode 31 s). Full root 2026 (32 episodes) is the decider; see below.

### What the opening hold trades (gap profile with `open_hold` off and on, Small 12345)

Base load shed above the clairvoyant plan's, bn USD summed over 64 episodes: weeks 1-4 +177 (off) vs +1,481 (on),
weeks 5-8 +539 vs +255, weeks 9-12 +849 vs -452; total gap 8,763 vs 8,400. The hold pays 1,300 bn of shed in weeks
2-4 for about 1,660 bn back later (less shed in weeks 5-12, more chips). A finer hold (part of the segment, or only
in weeks the fabs would draw little) might keep more of the return; `open_hold_draw` was tried in the log with
`open_hold_recheck` and cut gains and losses alike. Left as a lead.

### Opening hold settings (Small 12345)

| Variant | Score | Diff vs v8_on |
| --- | --- | --- |
| `open_hold_draw` 0.75 (no recheck) | 0.8753 | -0.0007 |
| `open_hold_draw` 1.0 | 0.8757 | -0.0003 |
| `open_hold_short` 0.05 | 0.8759 | -0.0001 |
| `open_hold_short` 0.2 | 0.8756 | -0.0004 |

Nothing to gain from its knobs either; the defaults are the best measured.

### `pulse_run` 6 on Full root 2026 (fresh, 32 episodes: 12 / 13 / 5 / 2 by level)

References: 23 min with 2 workers (cached now). `v8_on` 0.8866; `s_run6` 0.8866, diff +0.0001 (-0.0007 to +0.0009),
10 / 12 episodes. By level: +0.0002, -0.0001, -0.0007, +0.0016 (all but level 4 hold 0).

**Verdict on `pulse_run` 6** over the five sets: +0.0006 (Small tuning), +0.0011 (Small 2026, significant), -0.0002
(Small dev), +0.0010 (Full dev, holds 0), +0.0001 (Full 2026). Never significantly worse, a little better on Small.
A safe, small change (about +0.0005 to +0.001 expected), not a step.

## Experiment 4: a sweep on Full (root 12345, 16 episodes, the log's Full tuning set)

The private board plays Full, and no setting had been swept there on v8. References 10 min; each variant about 3 min
with 2 workers (slowest episode 27 s). `v8_on` 0.8651 (the log's number).

| Variant | Diff vs v8_on (90 % interval) | Better / worse |
| --- | --- | --- |
| `pulse_run` 8 | +0.0015 (+0.0002 to +0.0030) | 9 / 4 |
| `pulse_run` 6 | +0.0013 (-0.0000 to +0.0028) | 7 / 5 |
| `end_value` 2.5 | +0.0008 (-0.0001 to +0.0016) | 9 / 7 |
| `pulse_partial` 0.7 | +0.0005 (+0.0002 to +0.0009) | 7 / 2 |
| `short_premium` 0.5 | +0.0004 (+0.0002 to +0.0006) | 10 / 6 |
| `pulse_fill` 1.0 | +0.0002 (-0.0013 to +0.0017) | 12 / 4 |
| `open_hold` off | +0.0000 (-0.0011 to +0.0011) | 6 / 8 |
| `ration_trim_cover` 10, `pulse_margin` 1.1, `pulse_hold_weeks` 18 | 0.0000 | |
| `short_premium` 1.5, `reserve_weeks` 12, `end_value` 1.5 | -0.0001 to -0.0003 | |

- `pulse_run` 6 or 8 gains on Full too: the sixth set pointing the same way.
- `end_value` 2.5 is not followed up: on Small 3.0 lost 0.012, a cliff too close to risk on the private board.
- `pulse_partial` 0.7 and `short_premium` 0.5 are small Full-only gains (flat on Small 12345); checked together with
  `pulse_run` 6 on both 2026 roots below.

### The combination on the fresh roots

`pulse_run` 6 + `pulse_partial` 0.7 + `short_premium` 0.5 (`c_run6_pp`): Full 2026 0.8865 (`v8_on` 0.8866, `pulse_run`
6 alone 0.8866), Small 2026 0.8752 (`pulse_run` 6 alone 0.8752). The two Full-only gains did not hold: **`pulse_run`
6 alone** is the change.

## The candidate: `agents/mpc_pulse_rec_v9b`

`mpc_pulse_rec_v8_on` with `params.json` `{"open_hold": true, "pulse_run": 6.0}`. `uv run sbf check
mpc_pulse_rec_v9b --task=small`: passes, slowest week 0.164 s of 2 s; `--task=full`: passes, 0.795 s of 4 s.

| Set | `v8_on` | `v9b` | Diff (90 % interval) |
| --- | --- | --- | --- |
| Small 12345 (tuning, 64) | 0.8760 | 0.8766 | +0.0006 (-0.0000 to +0.0014) |
| Small 2026 (64) | 0.8741 | 0.8752 | +0.0011 (+0.0003 to +0.0021) |
| Small dev (20) | 0.9089 | 0.9087 | -0.0002 (-0.0013 to +0.0010) |
| Full 12345 (16) | 0.8651 | 0.8663 | +0.0013 (-0.0000 to +0.0028) |
| Full 2026 (32) | 0.8866 | 0.8866 | +0.0001 (-0.0007 to +0.0009) |
| Full dev (20) | 0.8961 | 0.8971 | +0.0010 (-0.0003 to +0.0027) |
| Small 777 (fresh, 64) | 0.8852 | 0.8853 | +0.0001 (-0.0006 to +0.0008), 12 / 16 |
| Full 777 (fresh, 32) | 0.8402 | 0.8397 | -0.0005 (-0.0012 to +0.0001), 9 / 14 |

Over the eight sets: mean +0.0004 on Small, +0.0005 on Full; one significant gain (Small 2026), no significant loss.
A third fresh root (777, played last) took away most of the case: treat it as a coin flip with a slight edge.

## Leads for later (hypotheses, not built)

1. **A finer opening hold.** The hold buys about 1,660 bn of later savings with 1,300 bn of extra shed in weeks 2-4
   (Small 12345, summed). Holding only part of the gas segment, sized from the first plan's shortfall, might keep the
   return at less cost. Its existing knobs (`open_hold_draw`, `open_hold_short`, `open_hold_last`) are flat.
2. **Chips are the only gap, and they are late.** 90 % of the shortage gap is in weeks 25-52. A diagnostic worth
   building before any rule: per product (the two chips sell at 50,400 and 10,300) and per fab, which lots the
   clairvoyant plan starts in weeks with no shedding that we do not.
3. **Long windows lose half their ground through the end value** (h48 with the end value kept at the episode's end
   recovers half its loss). A long window whose far weeks use a cruder model (aggregated weeks) could plan the end
   of the episode without paying the forecast error; CPU allows it on both networks.
4. **`end_value` above 2 is a cliff on Small (3.0: -0.012) but +0.0008 at 2.5 on Full.** Not worth the risk on the
   private board without a much larger Full test.

## Reproducing

Variants files are in this folder; results cache under `outputs/recovery_tune/<task>-e<root>-n<N>/<name>.json`.

```
uv run python experiments/approaches/tune.py run --variants=experiments/window_end/variants.json --task=small --entropy=12345 --episodes=64 --n_jobs=3
uv run python experiments/approaches/tune.py report --base=v8_on --task=small --entropy=12345 --episodes=64
uv run python experiments/window_end/report.py --base=v8_on --task=small --entropy=12345 --episodes=64
uv run python experiments/window_end/gap.py run --agent=agents/mpc_pulse_rec_v8_on --task=small --entropy=12345 --episodes=64
uv run python experiments/window_end/gap.py report --agent=mpc_pulse_rec_v8_on
uv run python experiments/window_end/lots.py
```

Full uses `--n_jobs=2` (8 GB of RAM) and `--episodes=16` (root 12345), `32` (root 2026) or `dev` (root 0).

## Upload

`mpc_pulse_rec_v9b` uploaded on 2026-10-08 at the participant's request (submission 970176): **0.8767 on the public
board, no fallback week**, against 0.8768 for `mpc_pulse_rec_v8_on`. As the local sets said, the two are the same
agent within noise.
