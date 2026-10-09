# The pulse agent after v8: v9b, v9 and v10b (2026-10-08 to 2026-10-09)

What was tried on top of the uploaded pulse agent (`mpc_pulse_rec_v8_on`, 0.8768 on the public board) over two
sessions, what was kept, and where each number comes from. Agents built in these sessions carry a `b` (`v9b`,
`v10b`); `v9` is the teammate's.

| Agent | What it is | Public board (Small) |
| --- | --- | --- |
| `mpc_pulse_rec_v8_on` | the agent before these sessions | 0.8768 |
| `agents/mpc_pulse_rec_v9b` | v8_on with `pulse_run` 6 (was 4), a setting only | 0.8767 |
| `agents/mpc_pulse_rec_v9` (teammate, `origin/pulse-agent-v9`) | v8_on with `schedule`: runs of whole weeks start sooner and end cleanly | 0.8827 |
| **`agents/mpc_pulse_rec_v10b`** | v9 with `pulse_sync`, `pulse_hold_scarce`, `pulse_hold_weeks` 26, `pulse_payback` 0 | **0.8845** |

The detailed logs: [experiments/window_end/README.md](../window_end/README.md) (day session, v9b) and
[experiments/night/README.md](../night/README.md) (night session, v10b).

## How the numbers were made

- **Paired comparisons** with `experiments/approaches/tune.py`: every agent on the same episodes, a 90 % paired
  bootstrap interval, and the count of episodes better and worse.
- **Sets:** tuning on root 12345 (Small 64 episodes, Full 16), confirmation on roots 2026 and 777 (Small 64, Full
  32) and the dev episodes (root 0, 20 per network); two roots never touched before the final agent was fixed,
  31337 (Small 64, Full 16) and 4242 (Full 24).
- **Merges proven exact:** each new rule is a switch; with the switches off an agent reproduces its predecessor to
  the cent on every episode before it is measured.
- **Full first:** the private board plays Full, so a change was taken if it gained on Full and did not lose on
  Small.
- Machine: a 4-core laptop with 8 GB; 3 workers on Small, 2-3 on Full. Reference costs are cached for Full roots
  0, 2026, 12345, 777, 31337, 4242 and Small roots 0, 2026, 12345, 777, 31337, 999.

## Session 1 (2026-10-08, day): window, end value, settings, and v9b

The participant's three questions, answered on Small root 12345 and confirmed elsewhere:

- **A longer planning window loses.** 24, 32 and 48 weeks: -0.0004, -0.0073, -0.0160; shorter is worse too (16:
  -0.003, 12: -0.020). 20 weeks is a sharp optimum. Half of the loss of a long window is losing the window-end value
  (the planner's best single rule), half is planning further on the forecast that today's network persists.
- **Planning to the episode's end, with leftovers worth nothing, is already how the agent works:** the window shrinks
  to the weeks left, and leftovers then get the salvage price, which the episode's cost really credits. Crediting
  them more loses in 55 of 64 episodes or more; a tapered end value is neutral.
- **Shed is not the gap.** The clairvoyant plan sheds as much base load as we do; the whole gap is unserved chips,
  mostly at fabs whose grids are short of energy, and 49-78 % of the clairvoyant plan's fab energy comes from
  shedding base load for fabs, which the simulator forbids.
- **Settings:** one-at-a-time sweeps of 27 settings on Small and 13 on Full. Only `pulse_run` 6 held on fresh
  episodes, at about +0.0005 over eight sets: `mpc_pulse_rec_v9b`, uploaded at the participant's request, **0.8767**
  (v8_on 0.8768): the same agent within noise, as the local sets said.

## Between the sessions: the teammate's v9

`schedule` (in `pulse.py`): at a pulsed grid, the last whole week of a run is delivered only what it burns, and a
run starts as soon as one whole week can follow. 0.8827 on the board; locally +0.0044 to +0.0078 on Full, +0.0025 to
+0.0044 on Small over v8_on. It replaces `pulse_run`'s wait, so v9b's setting does not carry over.

## Session 2 (2026-10-09, night): v10b

### What did not work

| Idea | Result |
| --- | --- |
| The teammate's two refinements of `schedule` (wait at grids whose threshold exceeds a week's burn; charge only with wafers) | within +-0.0005 on both tuning sets: the losses they target are rare |
| A finer opening hold (hold part of the gas in weeks 2-4) | -0.0005 to -0.0025: the hold works because it is whole |
| Capping nuclear stock at grids (the largest holding gap on Full, +191 bn) | -0.097 on Full: the stock is insurance against a nuclear supply cut |
| A longer nuclear reserve | no effect: the routes, not the planner, cap nuclear supply |
| Steering energy between fabs of one grid | bound 0.0014 of the gap, with perfect wafer control: not built |
| A long window with a coarse far end | not built: a new LP builder, and half the long-window loss is forecast error it would keep |
| `pulse_partial` 0.7, `pulse_hold_weeks` 40, `pulse_sync_end`, `pulse_sync_stock` | +0.0008 at most, or identical |

The chip gap was broken down by product: 92 % is the expensive chip, and it is upstream (lots not started, not
chips lost on the way).

### What worked: the fuels of a grid as one system

The worst Full episodes (0.53 against medians of 0.90) were long gas and crude shortages. Traced week by week at one
grid: gas (needed for a whole week: its 6,480 threshold plus a 5,400 burn, 1,790 arriving a week) and crude (900 a
week, 390 arriving, a terminal of about a week) were pulsed on separate clocks. The gas ran its whole weeks with the
crude missing, burnt for base load alone, while the crude reached the grid in other weeks; the fabs ran in 2 weeks
of 60. Four changes followed, each a switch, each confirmed on sets it was not chosen on:

1. **`pulse_sync`:** the gas charge waits until every other short fuel that can stop the fabs will have a whole
   week's burn at its terminal.
2. **`pulse_hold_scarce`:** in a week that cannot be whole, such a fuel arriving slower than it burns is held for the
   next whole week instead of being burnt for base load.
3. **`pulse_hold_weeks` 26** (was 12): a tank that takes up to half a year to fill for one whole week is now worth
   holding.
4. **`pulse_payback` 0** (was 12): with the fuels pulsed together, the grids the payback test barred (Full grids 3, 5,
   6; Small grid 3) now run in whole weeks.

In the worst episode the traced grid's fabs run 25 whole weeks instead of about 4.

### v10b against v9

| Set | v9 | v10b | Diff (90 % interval) |
| --- | --- | --- | --- |
| Full 12345 (16) | 0.8700 | 0.8722 | +0.0022 (-0.0027 to +0.0066) |
| Full 2026 (32) | 0.8922 | 0.8982 | +0.0060 (+0.0025 to +0.0099) |
| Full 777 (32) | 0.8448 | 0.8541 | +0.0093 (+0.0053 to +0.0137) |
| Full dev (20) | 0.9037 | 0.9167 | +0.0130 (+0.0075 to +0.0188) |
| Full 31337 (16, new) | 0.9062 | 0.9062 | +0.0000 (-0.0046 to +0.0047) |
| Full 4242 (24, new) | 0.8869 | 0.8895 | +0.0025 (-0.0006 to +0.0062) |
| Small 12345 (64) | 0.8804 | 0.8824 | +0.0020 (-0.0003 to +0.0044) |
| Small 2026 (64) | 0.8766 | 0.8784 | +0.0018 (-0.0000 to +0.0037) |
| Small 777 (64) | 0.8890 | 0.8896 | +0.0006 (-0.0021 to +0.0031) |
| Small dev (20) | 0.9126 | 0.9157 | +0.0031 (-0.0005 to +0.0070) |
| Small 31337 (64, new) | 0.8816 | 0.8848 | +0.0032 (+0.0013 to +0.0049) |

- **Full +0.0055 on average over six sets, Small +0.0021 over five; never worse.**
- The two Full sets that played no part in choosing anything gain less (0.0000 and +0.0025) than the sets the steps
  were confirmed on, so expect about **+0.002 to +0.005 on Full**, more in episodes with long multi-fuel crises.
- On Small the payback change makes many episodes slightly worse and a few much better (Small 777: 8 better, 40
  worse, still +0.0003 in all).
- **Exactness:** v10b equals the experimental variants to the cent; with its four changes undone it equals v9 to
  the cent (Full 12345, Small 12345).
- **`sbf check mpc_pulse_rec_v10b`:** passes on Small (slowest week 0.16 s of 2 s) and Full (0.78 s of 4 s), the same
  as v9. `--docker` not run.
- **Uploaded on 2026-10-09** at the participant's request: **0.8845 on the public board (submission 971275), no fallback week**, against 0.8827 for v9: +0.0018, in line with the local Small sets (+0.0021 on average).

## Where things are

| Path | What |
| --- | --- |
| `agents/mpc_pulse_rec_v10b/` | the new agent: v9's files with `pulse.py` (two rules) and `agent.py` (four settings) changed |
| `agents/mpc_pulse_rec_v9/` | the teammate's v9, copied from `origin/pulse-agent-v9` |
| `agents/mpc_pulse_rec_v9b/` | session 1's candidate (`pulse_run` 6) |
| `experiments/pulse_agent/versions/mpc_pulse_rec_v10b_x/` | the experimental copy with every switch tried at night (all off by default) |
| `experiments/pulse_agent/versions/mpc_pulse_rec_v9b_x/` | session 1's experimental copy (end-value switches) |
| `experiments/night/` | the night log, variants files, `products.py` (chip gap by product), `holding.py` (holding by stock) |
| `experiments/window_end/` | the day log, variants files, `report.py` (by harm level), `gap.py` (costs against the clairvoyant plan), `lots.py` |

## Leads for later

1. **More of the same mechanism:** every gain at night came from treating a grid's fuels together. Left: crude held
   upstream of a small terminal; a grid whose nuclear is also short (the pulse stands down there).
2. **Small's many small losses** under `pulse_payback` 0: a payback test that counts the second fuel might keep the
   gains and drop them.
3. **A larger Full confirmation** (64 episodes) would pin down v10b's gain, which sits within the spread between
   sets of 16-32 episodes.

## Reproducing

```
uv run python experiments/approaches/tune.py run --variants=experiments/night/final_fresh.json --task=full --entropy=2026 --episodes=32 --n_jobs=2
uv run python experiments/approaches/tune.py report --base=v9 --task=full --entropy=2026 --episodes=32
uv run sbf compare mpc_pulse_rec_v10b mpc_pulse_rec_v9 --task=full
```

Results cache under `outputs/recovery_tune/<task>-e<root>-n<N>/<name>.json` (not committed); `tune.py` looks a
`base` up in `agents/` and then in `experiments/pulse_agent/versions/`.
