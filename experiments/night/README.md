# Night of 2026-10-09: experiments on top of the teammate's v9

> **Naming:** the candidate built here was renamed `agents/mpc_pulse_rec_v10b` (and the experimental copy
> `mpc_pulse_rec_v10b_x`) on 2026-10-09, so that our agents carry a `b` (the teammate's are v9, v10, ...). Below,
> "v10" in the text means that agent; the variant labels in the tables (`v10_sync`, `v10b_pay0`, ...) are the names
> the results were cached under and are kept as they were.

## Summary (read this first)

**New candidate: `agents/mpc_pulse_rec_v10b`** = the teammate's v9 (0.8827 on the public board) with two new rules in
`pulse.py` and two settings changed, all found by tracing v9's worst Full episodes:

- **`pulse_sync`:** at a grid short of two fuels that can each stop the fabs (gas and crude in a long shortage), the
  pulse timed each fuel on its own clock, so the gas ran its whole week with the crude missing (burnt for base load
  alone). The gas charge now waits until the other fuel will have a whole week at its terminal.
- **`pulse_hold_scarce`:** in a week that cannot be whole, a scarce second fuel (arriving slower than it burns) is
  held for the next whole week instead of being burnt for base load.
- **`pulse_hold_weeks` 26** (was 12): a slow tank (up to half a year to fill for one whole week) is now worth holding.
- **`pulse_payback` 0** (was 12): with the fuels pulsed together, the grids the payback test barred (Full grids 3, 5,
  6; Small grid 3) now run in whole weeks.

| Set | v9 | v10 | Diff (90 % interval) |
| --- | --- | --- | --- |
| Full 12345 (16) | 0.8700 | 0.8722 | +0.0022 (-0.0027 to +0.0066) |
| Full 2026 (32) | 0.8922 | 0.8982 | +0.0060 (+0.0025 to +0.0099) |
| Full 777 (32) | 0.8448 | 0.8541 | +0.0093 (+0.0053 to +0.0137) |
| Full dev (20) | 0.9037 | 0.9167 | +0.0130 (+0.0075 to +0.0188) |
| Small 12345 (64) | 0.8804 | 0.8824 | +0.0020 (-0.0003 to +0.0044) |
| Small 2026 (64) | 0.8766 | 0.8784 | +0.0018 (-0.0000 to +0.0037) |
| Small 777 (64) | 0.8890 | 0.8896 | +0.0006 (-0.0021 to +0.0031) |
| Small dev (20) | 0.9126 | 0.9157 | +0.0031 (-0.0005 to +0.0070) |
| *Brand-new, played once at the end:* | | | |
| Full 31337 (16, mostly calm) | 0.9062 | 0.9062 | +0.0000 (-0.0046 to +0.0047) |
| Full 4242 (24) | 0.8869 | 0.8895 | +0.0025 (-0.0006 to +0.0062) |
| Small 31337 (64) | 0.8816 | 0.8848 | **+0.0032 (+0.0013 to +0.0049)** |

**Full (the private board): +0.0055 on average over six sets, never worse; Small +0.0021 over five, never worse.**
Each step was chosen on one or two sets and confirmed on the others (tables below). The two Full sets that played no
part in choosing anything gain less (0.0000 and +0.0025) than the sets the steps were confirmed on: an honest
expectation for Full is about **+0.002 to +0.005**, larger in episodes with long multi-fuel crises. `v10` reproduces the experimental variants to the cent,
and with the four changes undone it reproduces v9 to the cent. `sbf check` passes on Small and Full (slowest week
0.16 s of 2 s, 0.78 s of 4 s). Not committed. Uploaded on 2026-10-09 as `mpc_pulse_rec_v10b` (submission 971275): **0.8845 on the public board**, no
fallback week (v9 0.8827). (The 3 a.m. submission window had closed before anything
was confirmed). If it becomes the agent to submit, add `agents/mpc_pulse_rec_v10b` to the lint `extend-exclude` in
`pyproject.toml` as v9 is (its inherited long lines; the lines added tonight pass).

**Dead ends tonight** (each logged below): the teammate's two refinements of `schedule` (null); a finer opening hold
(holding part of the gas loses; the hold works because it is whole); capping nuclear stock (-0.097 on Full: it is
insurance against supply loss); a longer nuclear reserve (the routes cap supply); steering energy between fabs
(bound 0.0014); a long window with a coarse far end (not built: a new LP builder, and half its loss is forecast
error it would keep). The chip gap is 92 % one product (the expensive chip) and upstream (lots not started), as the
day before.

The rest of this file is the log, in the order things were run.

Baseline: `agents/mpc_pulse_rec_v9` (from `origin/pulse-agent-v9`, copied into the working tree untracked; 0.8827 on
the public board). Goal: a better score on Full (the private board) without losing on Small. Nothing committed.

- **Harness:** `experiments/approaches/tune.py` (cached results in `outputs/recovery_tune/`), variants files in this
  folder, `experiments/window_end/report.py` for harm levels.
- **Sets:** tuning Small 12345 (64) and Full 12345 (16); fresh Small 2026, 777 (64 each), Full 2026, 777 (32 each);
  dev episodes last. All references were cached the day before.
- **Experimental agent:** `experiments/pulse_agent/versions/mpc_pulse_rec_v10b_x` = v9 plus switches, each off by
  default (`v10_off` must equal `v9` to the cent):
  - `schedule_thr_wait`: the charge that starts a run waits `pulse_run` weeks again (as before v9's `schedule`) at
    grids whose rationing threshold is above a week's burn (grids 2-3 on Small, 2-6 on Full: threshold 1.2 weeks).
  - `schedule_wafers` x: the charge is made at once only while the fabs' wafers ask for at least x of their full
    draw.
  - `open_hold_share` x: the opening hold keeps x of the planner's gas at the terminal (1: all, as now).

## Log

### v9 baselines (no fallback week anywhere)

| Set | v8_on | v9 |
| --- | --- | --- |
| Small 12345 (64) | 0.8760 | 0.8804 |
| Small 2026 (64) | 0.8741 | 0.8766 |
| Small 777 (64) | 0.8852 | 0.8890 |
| Full 12345 (16) | 0.8651 | 0.8700 |
| Full 2026 (32) | 0.8866 | 0.8922 |
| Full 777 (32) | 0.8402 | 0.8448 |

The teammate's numbers reproduce (Small 12345, 2026; Full 12345, 777); Small 777 and Full 2026 with 32 episodes are
new: v9 gains there too.

### Lead 2: the chip gap by product (v9, Small 12345, `products.py`)

`gap.py` now also saves each week's unserved demand per market (ours: the record's `lost`; clairvoyant: the LP's
`U` columns). v9's cost above the clairvoyant plan is 8,123 bn over 64 episodes (v8_on: 8,400), and:

- **92 % of it is product 6** (the chip sold at 50,400): +7,505 bn; product 7 (10,300) +665 bn. We leave 50.7 % of
  product-6 demand unserved, the clairvoyant plan 42.7 %. Which market is short does not matter (the prices differ by
  under 0.1 %); 66 % of the gap shows at node 34.
- **It is upstream, not routing.** Chips served per lot started at the four product-6 fabs: 1.79 ours, 1.81
  clairvoyant (stock at the start included). So nothing is lost between the fab and the market.
- **The lots:** in weeks 9-40 we start 384M at the product-6 fabs against 510M; in weeks 1-8, 123M against 75M
  (the lots forced by wafers already on the way, and the opening). The gap grows from week 25 on, about 12 weeks
  after the lots it comes from.

So the chip gap is the fabs' energy (yesterday's analysis), concentrated on the expensive product. A rule that ranks
grids by the product their fabs make is already in the agent (`short_premium` prices the chips a shortfall costs).

### The teammate's two refinements of `schedule` (tuning roots)

| Variant | Small 12345 (64) | Full 12345 (16) |
| --- | --- | --- |
| `v10_off` | 0.8804 (= v9 to the cent) | 0.8700 (= v9 to the cent) |
| `schedule_thr_wait` | +0.0001 (-0.0007 to +0.0008), 6 / 8 | +0.0001 (-0.0008 to +0.0013), 1 / 3 |
| `schedule_thr_wait`, `pulse_run` 2 | -0.0005 (-0.0011 to +0.0000) | -0.0010 (-0.0028 to +0.0001) |
| `schedule_wafers` 0.5 | +0.0004 (+0.0001 to +0.0007), 17 / 3 | -0.0004 (-0.0014 to +0.0001), 2 / 1 |
| `schedule_wafers` 0.8 | +0.0003 (-0.0002 to +0.0008) | -0.0001 (-0.0005 to +0.0003) |
| both (`thr_wait`, `wafers` 0.5) | +0.0005 (-0.0003 to +0.0013), 20 / 9 | +0.0002 (-0.0008 to +0.0013), 3 / 3 |

**Both are null.** The losses the teammate traced (short runs at grids with a threshold above a week's burn; a tank
spent on partly powered weeks after arrivals collapse) are real but rare: on Full the rules change 2 to 6 of 16
episodes. Not pursued.

### Lead 1: a finer opening hold (`open_hold_share`, tuning roots)

| Variant | Small 12345 | Full 12345 |
| --- | --- | --- |
| `open_hold_share` 0.75 | -0.0005 (-0.0016 to +0.0004), 21 / 31 | +0.0009 (+0.0003 to +0.0015), 10 / 4 |
| `open_hold_share` 0.5 | -0.0017 (-0.0029 to -0.0007), 16 / 36 | -0.0009 (-0.0018 to -0.0002), 4 / 10 |
| `open_hold_share` 0.25 | -0.0025 (-0.0038 to -0.0013), 15 / 37 | -0.0024 (-0.0041 to -0.0008), 5 / 9 |
| `open_hold` off (Small only) | -0.0016 (-0.0033 to -0.0000), 24 / 28 | |

**The hold works because it is whole.** A part held leaves the grid short anyway (the fabs get nothing that week)
and saves less gas for later; the less held, the more it loses. 0.75 on Full alone is not a direction (Small says
the opposite, and it sits between two losses). The opening hold stays as v9 has it.

### Where v9's gap sits on Full (12345, 16 episodes, `gap.py`)

Cost above the clairvoyant plan: 7,163 bn over 16 episodes (the naive-to-clairvoyant gap is 56,752 bn).

- Unserved chips +7,421 bn: product 6 +5,122, **product 7 +2,299** (on Full the cheap chip is a third of it; on
  Small a twelfth). Shed -547 (we shed less than the clairvoyant plan).
- **Holding +238 bn (516 against 278), tariff +33, disposal +22**: about 0.005 of score, unlike Small (+25 on
  holding). Being traced below.
- Lots: 78.5 % of the clairvoyant plan's. Fabs at grid 3 (fab 10: 217M against 373M) and grid 6 (fab 13: 21M
  against 73M) run furthest below it, energy-bound 70 and 87 weeks of 104.

### The holding gap on Full is nuclear fuel (`holding.py`, per-slot stock from `gap.py`)

Holding by kind of stock (Full 12345, 16 episodes, bn USD, ours / clairvoyant / gap): nuclear fuel (good 2) at
grids 415 / 224 / **+191**; everything else +46. Nuclear is the dearest good (customs value 206,264; holding 167 per
unit-week). Episode 0, grid 4's nuclear stock (ours | clairvoyant): week 1 748,838 | 748,838; week 41 816,362 |
393,325; week 65 562,455 | 206,508; week 104 2,183 | 0.

- Every grid starts with 52 weeks of its nuclear burn. **We keep that level (even raise it) for 45 weeks**, until
  the stock covers the weeks left and `fuel_cancel` stops shipments; the clairvoyant plan runs it down from week 9
  (about 27 weeks of cover at mid-episode). On Small (52 weeks) `fuel_cancel` binds from week 1, so the gap is a
  Full thing.
- Nuclear is the only fuel a source ships straight into a grid (8-week lead, one source edge per grid on Small, two
  on Full). One week of a grid's nuclear burn costs about 2.4M a week to hold; a week without it sheds about 59 bn at
  grid 4. Some cover is worth holding, 52 weeks is not.
- **`fuel_cover_cap` W** (in v10x, with `fuel_cancel`): a source's dispatch into a grid is also cut so that stock,
  shipments under way and the dispatch stay within W weeks of the fuel's full-rate burn. Played at W = 13, 20, 26,
  39.

**Result: nuclear stock is insurance, and capping it is a large loss.** `fuel_cover_cap` 26 on Full 12345: 0.7733
against 0.8700 (-0.097), worse in 14 of 16 episodes. Traced in episode 14 (harm level 1): nuclear supply to grid 4
fails in mid-episode; v9 itself runs dry there from week 79 (10,678 of base load shed a week, about 44 bn a week);
with the cap it runs dry 9 weeks earlier, and grid 1 runs dry too (+568 bn in that episode). A full year of nuclear
stock costs about 13 bn of holding per grid and episode; one dry week costs more than three times that. The
clairvoyant plan holds less only because it knows which grids lose their supply. The other caps (13, 20, 39) were
stopped: a tighter cap can only lose more. The holding gap is the price of a hedge that pays.

### Full grids whose fabs sit idle (3, 5, 6), and the pulse's payback test

On Full the fabs of grid 3 get no energy in 71 weeks of 104, grid 5's in 51, grid 6's in 92 (the grid sheds base
load in nearly all of them). The pulse's payback test (`pulse_payback` 12) bars holding gas at grids 3, 5 and 6.

| `pulse_payback` (Full 12345) | Opens the pulse at | Diff vs v9 |
| --- | --- | --- |
| 24 | grid 3 | -0.0006 |
| 48 | grids 3, 5 | -0.0005 |
| 0 (no test) | grids 3, 5, 6 | -0.0005 |

Nothing to gain: these grids are short of **crude**, not gas. Grid 6 (episode 0): crude burns 750 a week and about
150 reach its terminal; crude's segment (0.03 of output) exceeds the fabs' margin (125), so any crude shortfall stops
them, while its gas tanks sit full (30,000 at the grid and at the terminal). The pulse already gathers crude for
whole weeks (23 ON weeks). Crude delivered into each grid and terminal over the episode matches the clairvoyant plan
within a few per cent (grid 6: 30,001 both): crude is scarce at the source, and the clairvoyant plan runs those fabs
by shedding base load for them, which the simulator forbids.

### More nuclear stock, as insurance (`reserve_slow_weeks`)

If nuclear stock is insurance, more of it may pay. The two Full nuclear sources ship 25,738 a week each (51,475)
against a total burn of 41,180, so about a quarter is spare while supply is open, yet our stocks rise only slowly
(grid 4, episode 0: +1,600 a week). A year of extra stock at grid 4 would cost about 3 bn of holding per episode
and would have covered the 25 dry weeks of episode 14 (about 1,100 bn).

`reserve_slow_weeks` W (v10x, `reserve.py`): the planner's soft minimum on fuel stock at grids (`reserve_weeks` 8,
priced at 0.001 of VOLL per unit-week missing, capped by storage, shrinking to the weeks left) is set to W weeks of
burn for a fuel whose nominal stock lasts 26 weeks or more (nuclear on both networks). Played at W = 60, 80, 104.

**Result: no effect** (Full 12345: 0.8700 at W = 60 and 80, per-episode differences of 0.2 bn at most; the rest of
the batch was stopped). The planner already ships nuclear near the limit of the routes: each Full grid is fed by one
8-week edge from each of the two sources, and at grid 4 they carry 8,640 a week each, 17,280 against a burn of
14,400. A target cannot buy what the edges do not carry. The nuclear dry spells are set by the network and the
draw of events.

### Steering energy between fabs of one grid (bound only, not built)

In a partly powered week the simulator gives every fab of a grid the same share of its feasible lots, so a fab that
needs more energy per lot takes energy from a more efficient one (grid 1 on Full: fabs 3 and 4 make the same chip,
0.002 against 0.00126 energy per lot). Starving the less efficient fab of wafers before such weeks would steer the
energy. Upper bound, from v9's own partly powered weeks (Full 12345): grid 1, 4.4 weeks per episode, 0.87M more lots
(about 79 bn, 0.0014 of the gap); grid 0, 0.89M; grid 5, nothing. A bound that needs perfect wafer control weeks
ahead: not worth building.

### The worst Full episodes, and two short fuels pulsed on two clocks (`pulse_sync`)

v9's worst Full episodes are calm ones (harm level 1-2): Full 2026 episode 3 (0.528), Full 777 episode 26 (0.536),
against medians of 0.90 to 0.92. In both our fabs barely run (777/26: fab 10 0.7M lots against the clairvoyant
plan's 25.5M, fab 4 1.5M against 17.7M) while we shed less base load than the clairvoyant plan: a long gas and
crude shortage at grids 1, 2, 3, 5 and 6.

Traced at grid 2 (777/26, week by week): gas (rationed, threshold 6,480, burn 5,400) arrives at about 1,790 a week,
crude (burn 900, more than the fabs' margin of 181) at about 390 a week into a terminal that holds 1.2 weeks of it.
The pulse times each fuel on its own: gas is charged every 4-5 weeks and runs its week with the crude missing
(weeks 10, 15, 24, 29, 38, 43: gas 5,400, crude 0), crude overflows into the grid in other weeks (5, 8, 11, 14,
...). Only weeks 20 and 34 had both, and only then did the fab run. Each gas charge out of step burns about 5,400
for base load alone. Example: the charge of week 9 met a crude terminal of 783 (< 900) at the start of week 10; one
week later it would have held 1,174.

**`pulse_sync`** (v10x): the charge of the rationed fuel waits until every other short fuel that can take all the
fabs' energy (not a `pulse_small` one) will hold a whole week's burn at its terminal when the charged week begins
(`tank + inflow >= burn`; the fuel is held this week, so that is next week's feed). Not when the gas tank is full
(it would overflow anyway). On 777/26: grid 2's fab runs in weeks 11, 13, 18, 20, 29, 34, 36, 45, 50, 52, 57 (v9:
20, 34, plus weeks 2-3 in both), the episode costs 11,947 bn instead of 12,032 (+0.024 of its gap).

**Result (v10x `pulse_sync` against v9):**

| Set | v9 | `pulse_sync` | Diff (90 % interval) | Better / worse |
| --- | --- | --- | --- | --- |
| Full 12345 (tuning, 16) | 0.8700 | 0.8706 | +0.0006 (+0.0000 to +0.0013) | 2 / 0 |
| Full 2026 (32) | 0.8922 | 0.8927 | +0.0005 (-0.0001 to +0.0012) | 3 / 3 |
| Full 777 (32) | 0.8448 | 0.8462 | **+0.0014 (+0.0002 to +0.0031)** | 7 / 0 |
| Small 12345 (tuning, 64) | 0.8804 | 0.8808 | +0.0004 (-0.0004 to +0.0012) | 4 / 2 |

`v10_off4` (the switch off) equals v9 to the cent on both tuning sets. The rule changes few episodes (2 to 7 of 16-32
on Full), always where two short fuels meet; it is the first change of the night that gains on every set. Full 777,
the set of the worst episode traced, gains most. Small 2026, 777 and the dev episodes follow.

(The 3 a.m. submission window closed with no upload: nothing had been confirmed by then.)

`pulse_sync` with the payback test loosened (Full 12345): `pulse_payback` 0 or 48 on top of sync is +0.0004 over sync
alone (-0.0019 to +0.0026), 4 episodes better and 7-8 worse; alone it was -0.0005. Noisy, not taken: sync keeps the
payback of 12.

`pulse_sync` on the fresh Small sets: Small 2026 0.8771 against 0.8766, +0.0005 (+0.0002 to +0.0009), 5 / 3; Small
777 0.8887 against 0.8890, -0.0003 (-0.0009 to +0.0002), 2 / 5. A small loss on one Small set.

A gap in the rule: it counts the other fuel's terminal stock and arrivals, not stock already at the grid, so a grid
holding enough crude itself still delays the gas. `pulse_sync_stock` (v10x) also counts what that stock leaves after
this week; played as a separate variant on all six sets. Every synced terminal holds at least 3.6 weeks of its fuel's
burn, so the rule cannot wait forever on a tank too small to fill.

`pulse_sync` on the dev episodes (never tuned on): Small 0.9129 against 0.9126, +0.0003 (-0.0000 to +0.0006), 3 / 3;
**Full 0.9042 against 0.9037, +0.0005 (+0.0002 to +0.0010), 4 / 0.**

| `pulse_sync` against v9 | Full | Small |
| --- | --- | --- |
| 12345 (tuning) | +0.0006 | +0.0004 |
| 2026 | +0.0005 | +0.0005 |
| 777 | +0.0014 | -0.0003 |
| dev | +0.0005 | +0.0003 |

Positive on every Full set (mean +0.0008), on three of four Small sets (mean +0.0002). Small, steady, and it changes
only the episodes where two short fuels meet.

### `pulse_hold_weeks` with sync (Full)

The pulse holds a fuel only if its tank fills for a whole week within `pulse_hold_weeks` (12); grid 3's crude in Full
2026 episode 3 needs about 12 weeks (150 a week against 1,800). With `pulse_sync` on:

| Variant | Full 12345 (16) | Full 777 (32): against sync | against v9 |
| --- | --- | --- | --- |
| `pulse_hold_weeks` 18 | 0.8706 (= sync) | +0.0010 (+0.0004 to +0.0017), 12 / 11 | +0.0024 (+0.0007 to +0.0047) |
| `pulse_hold_weeks` 26 | 0.8706 (= sync) | +0.0016 (+0.0005 to +0.0029), 11 / 15 | +0.0030 (+0.0010 to +0.0059) |

12345 and 777 are now its tuning sets; Full 2026, both dev splits and the three Small sets confirm.

`pulse_sync_end` (with `schedule`, a run of whole weeks also ends when another short fuel cannot follow): over sync,
+0.0001 on Full 12345, +0.0002 on Full 777, 0.0000 on Small 12345. Dropped: not worth the extra rule.

`pulse_sync_stock`: identical to `pulse_sync` on all 64 episodes of Small 777 (the grids never hold crude left over
when the rule checks), so the rest of its batch was stopped. Dropped.

**Confirmed.** `pulse_sync` with `pulse_hold_weeks` 26, against v9 (90 % paired intervals):

| Set | `pulse_sync` | `pulse_sync` + hold 26 |
| --- | --- | --- |
| Full 12345 (16, tuning) | +0.0006 (+0.0000 to +0.0013) | +0.0006 (-0.0000 to +0.0013) |
| Full 2026 (32) | +0.0005 (-0.0001 to +0.0012) | **+0.0012 (+0.0001 to +0.0024)** |
| Full 777 (32, tuning for hold) | +0.0014 (+0.0002 to +0.0031) | **+0.0030 (+0.0010 to +0.0059)** |
| Full dev (20) | +0.0005 (+0.0002 to +0.0010) | **+0.0010 (+0.0004 to +0.0017)** |
| Small 12345 (64) | +0.0004 (-0.0004 to +0.0012) | +0.0008 (-0.0001 to +0.0017) |
| Small 2026 (64) | +0.0005 (+0.0002 to +0.0009) | +0.0006 (+0.0002 to +0.0010) |
| Small 777 (64) | -0.0003 (-0.0009 to +0.0002) | -0.0002 (-0.0008 to +0.0003) |
| Small dev (20) | +0.0003 (-0.0000 to +0.0006) | +0.0002 (-0.0001 to +0.0005) |

Mean over the four sets: Full +0.0015, Small +0.0004. The longer hold adds to sync on every set but Small dev
(-0.0001). Scores: Full 0.8706 / 0.8934 / 0.8478 / 0.9047, Small 0.8812 / 0.8772 / 0.8888 / 0.9128.

## The candidate: `agents/mpc_pulse_rec_v10b`

v9 plus only what held up: the `pulse_sync` rule in `pulse.py` (on in `PARAMS`) and `pulse_hold_weeks` 26 (was 12).
No other change; the experimental switches of v10x are not in it. On both tuning sets it equals
`v10_sync_hold26` to the cent on every episode, and with `pulse_sync` false and `pulse_hold_weeks` 12 it equals v9 to
the cent (Full 12345, 16 episodes; Small 12345, 64).

### Holding the scarce fuel too (`pulse_hold_scarce`)

In the grid 2 trace crude was also delivered, and burnt for base load, in weeks the gas was short (5, 8, 11, ...):
the pulse holds only the fuels short *this* week, and in those weeks the crude could run in full, so the planner's
delivery stood. `pulse_hold_scarce` (v10x): in a week that cannot be whole at a pulsed grid, every other fuel that
can take all the fabs' energy and arrives at its terminal slower than it burns is held as well (the terminal's
storage still caps it). On top of v10 (sync, hold 26):

| Set | v10 | + `pulse_hold_scarce` | Diff (90 % interval) |
| --- | --- | --- | --- |
| Full 12345 (16) | 0.8706 | 0.8711 | +0.0005 (-0.0011 to +0.0020), 5 / 5 |
| Full 777 (32) | 0.8478 | 0.8502 | **+0.0024 (+0.0004 to +0.0044), 11 / 8** |

Confirmation on the sets it was not chosen on, and the whole picture (90 % paired intervals):

| Set | + `pulse_hold_scarce` | vs v9 | vs v10 (sync, hold 26) | Better / worse vs v10 |
| --- | --- | --- | --- | --- |
| Full 12345 (16) | 0.8711 | +0.0010 (-0.0009 to +0.0028) | +0.0005 (-0.0011 to +0.0020) | 5 / 5 |
| Full 2026 (32) | 0.8940 | +0.0018 (-0.0002 to +0.0040) | +0.0006 (-0.0009 to +0.0021) | 11 / 9 |
| Full 777 (32) | 0.8502 | **+0.0055 (+0.0021 to +0.0095)** | +0.0024 (+0.0004 to +0.0044) | 11 / 8 |
| Full dev (20) | 0.9076 | **+0.0039 (+0.0011 to +0.0069)** | +0.0029 (+0.0004 to +0.0056) | 9 / 6 |
| Small 12345 (64) | 0.8819 | +0.0015 (-0.0007 to +0.0036) | +0.0008 (-0.0009 to +0.0024) | 11 / 15 |
| Small 2026 (64) | 0.8769 | +0.0004 (-0.0014 to +0.0021) | -0.0003 (-0.0019 to +0.0014) | 7 / 13 |
| Small 777 (64) | 0.8893 | +0.0002 (-0.0020 to +0.0023) | +0.0004 (-0.0018 to +0.0026) | 12 / 11 |
| Small dev (20) | 0.9141 | +0.0014 (-0.0004 to +0.0032) | +0.0013 (-0.0004 to +0.0029) | 3 / 7 |

Against v9: Full mean **+0.0031**, Small mean +0.0009, positive on all eight sets. It moves more episodes both ways
than `pulse_sync` (it changes flows in every OFF week of a pulsed grid with a scarce second fuel), so the Small
intervals are wide; on Full the gain is largest on the two sets with the long multi-fuel crises (777, dev). Taken
into v10.

`agents/mpc_pulse_rec_v10b` now carries `pulse_hold_scarce` too (on in `PARAMS`). Reproduction: equal to the
experimental variant (`v10_sh26_scarce`) to the cent on Full 12345 and Small 12345, and with `pulse_sync`,
`pulse_hold_scarce` false and `pulse_hold_weeks` 12 equal to v9 to the cent. `uv run sbf check mpc_pulse_rec_v10b`:
passes on Small (slowest week 0.171 s of 2 s) and Full (0.812 s of 4 s), the same as v9 (`--docker` not run).

### A last round on top of v10 (tuning sets)

| Variant (diff vs v10) | Full 777 (32) | Full 12345 (16) | Small 12345 (64) |
| --- | --- | --- | --- |
| `pulse_payback` 0 (pulse at Full grids 3, 5, 6 and Small grid 3) | **+0.0039 (+0.0021 to +0.0059)** | +0.0012 (-0.0032 to +0.0051) | +0.0005 (-0.0005 to +0.0015) |
| `pulse_partial` 0.7 | +0.0002 (+0.0000 to +0.0004) | +0.0008 (+0.0002 to +0.0014) | +0.0000 |
| `pulse_hold_weeks` 40 | -0.0000 | +0.0000 | +0.0000 |

With the fuels now pulsed together, the payback test no longer protects anything on Full 777: the grids it barred
(gas short together with crude) can now be run in whole weeks. Before v10 it was -0.0005 alone and +0.0004 (noisy)
with sync. On Full 12345 it moves many episodes both ways. Confirmation (with `pulse_payback` 48 as the milder
alternative) on Full 2026, Full dev and the three other Small sets follows. `pulse_partial` 0.7 and hold 40: too
small, dropped.

**`pulse_payback` 0 confirmed** (v10 + payback 0, 90 % paired intervals):

| Set | Score | vs v9 | vs v10 (sync, hold 26, scarce) | Better / worse vs v10 |
| --- | --- | --- | --- | --- |
| Full 12345 (16) | 0.8722 | +0.0022 (-0.0027 to +0.0066) | +0.0012 (-0.0032 to +0.0051) | 6 / 10 |
| Full 2026 (32) | 0.8982 | **+0.0060 (+0.0025 to +0.0099)** | +0.0041 (+0.0018 to +0.0068) | 11 / 19 |
| Full 777 (32) | 0.8541 | **+0.0093 (+0.0053 to +0.0137)** | +0.0039 (+0.0021 to +0.0059) | 18 / 13 |
| Full dev (20) | 0.9167 | **+0.0130 (+0.0075 to +0.0188)** | +0.0091 (+0.0044 to +0.0141) | 11 / 7 |
| Small 12345 (64) | 0.8824 | +0.0020 (-0.0003 to +0.0044) | +0.0005 (-0.0005 to +0.0015) | 16 / 42 |
| Small 2026 (64) | 0.8784 | +0.0018 (-0.0000 to +0.0037) | +0.0014 (+0.0006 to +0.0023) | 14 / 29 |
| Small 777 (64) | 0.8896 | +0.0006 (-0.0021 to +0.0031) | +0.0003 (-0.0005 to +0.0012) | 8 / 40 |
| Small dev (20) | 0.9157 | +0.0031 (-0.0005 to +0.0070) | +0.0016 (-0.0010 to +0.0049) | 3 / 12 |

Against v9: **Full mean +0.0076, Small mean +0.0019**, positive on all eight sets. `pulse_payback` 48 (Full 2026
+0.0048, dev +0.0093 over v10; identical to 0 on Small, where both open the same grid) is the same within noise; 0 is
kept (no test at all, more evidence). **A caution:** on Small it makes many episodes slightly worse and a few much
better (Small 777: 8 better, 40 worse, +0.0003 in all): holding gas at Small grid 3 costs a little in many episodes.
On Full the counts are balanced and the gains large. Taken into `agents/mpc_pulse_rec_v10b` (`pulse_payback` 0).

## Final checks of `agents/mpc_pulse_rec_v10b` (all four changes)

- **Reproduction:** equal to `v10b_pay0` to the cent on Full 12345 and Small 12345; with `pulse_sync` and
  `pulse_hold_scarce` false, `pulse_hold_weeks` 12 and `pulse_payback` 12 equal to v9 to the cent.
- **`sbf check`:** passes on Small (slowest week 0.161 s of 2 s) and Full (0.775 s of 4 s). `--docker` not run.
- **A brand-new root (31337), never played before tonight's final agent was fixed:**

| Set | v9 | v10 | Diff (90 % interval) | Better / worse |
| --- | --- | --- | --- | --- |
| Small 31337 (64) | 0.8816 | 0.8848 | **+0.0032 (+0.0013 to +0.0049)** | 23 / 27 |
| Full 31337 (16; levels 9 / 5 / 3 / 0, pooled) | 0.9062 | 0.9062 | +0.0000 (-0.0046 to +0.0047) | 6 / 10 |

  Full 31337 has no harm-level-4 episode and is mostly calm (9 of 16 at level 1), and v10's gains come from long
  multi-fuel crises; still, it says the Full gain varies a lot from one draw of episodes to the next. (`tune.py
  report` fails on a set with an empty harm level; this row was computed with all episodes pooled alike.) Over the
  five Full sets v10 gains +0.0061 on average (+0.0022, +0.0060, +0.0093, +0.0130, 0.0000). A sixth Full set (root
  4242, 24 episodes) follows.

### A sixth Full set (root 4242, 24 episodes, never played before)

v9 0.8869, v10 0.8895: +0.0025 (-0.0006 to +0.0062), 12 / 11 (pooled alike: +0.0023, -0.0008 to +0.0059). Over the
six Full sets: +0.0022, +0.0060, +0.0093, +0.0130, 0.0000, +0.0025 (mean +0.0055).

## Leads for later

1. **More of the same mechanism.** Every gain tonight came from the pulse treating a grid's fuels as one system
   (sync, scarce, payback). Left: crude held upstream of a small terminal (it overflows into the grid in weeks the
   gas is not ready); a grid whose nuclear is also short (the pulse stands down: nothing can be held for a fuel a
   source ships straight in).
2. **Small's many small losses** under `pulse_payback` 0 (Small 777: 8 better, 40 worse, still +0.0003 in all):
   holding gas at Small grid 3 costs a little in many calm episodes. A payback test that counts the second fuel
   (rather than none) might keep the gains and drop the losses.
3. **Noise:** gains of +0.003 sit inside the spread between Full sets of 16-32 episodes. A larger Full confirmation
   (64 episodes, about 40 min of references and 10 min a variant here) would pin v10's gain down.
