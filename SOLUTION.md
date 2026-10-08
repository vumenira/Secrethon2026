# Our solution: the pulse agent

Українською: [SOLUTION.uk.md](SOLUTION.uk.md)

**The agent to submit is `agents/mpc_pulse_rec_v8_on`. It scores 0.8768 on the public board (Small).** It is the
benchmark's own rolling LP planner with a handful of rules added, each found by measuring where the planner loses
money. It has no learned parts, reads no warnings or announcements, and uses at most a quarter of the CPU limit.

This page is the short version. Every measurement is in
[experiments/pulse_agent/README.md](experiments/pulse_agent/README.md). The experiments after v8 (window length, end value,
settings sweeps, the cost gap against the clairvoyant plan) are in
[experiments/window_end/README.md](experiments/window_end/README.md).

## What it scores

Each row adds to the one above. The two middle columns are local scores on episodes that nothing was tuned on (root
2026: 64 Small episodes, 16 Full episodes).

| Step | Small | Full | Public board |
| --- | --- | --- | --- |
| The package's planner, 20-week window (`mpc_det_h20`) | 0.669 | 0.290 | |
| + a value for goods at the window's end | 0.743 | 0.619 | |
| + the pulse, a fuel reserve, a shortfall premium | 0.821 | 0.857 | 0.8289 |
| + recovery forecast, end-of-sales rule | 0.839 | 0.867 | 0.8429 |
| + wafer top-up, run-down before sales end | 0.844 | 0.871 | 0.8498 |
| + a small fuel no longer blocks the pulse | 0.850 | 0.873 | 0.8577 |
| + gas trimmed in slack weeks, unburnable fuel not shipped | 0.865 | 0.882 | 0.8714 |
| + trim for other fuels, opening hold (the final agent) | 0.874 | 0.886 | 0.8768 |

- **The private board plays Full.** Locally the final agent scores 0.84 to 0.90 there depending on the episode set
  (0.840, 0.865, 0.886, 0.896 on four sets), so expect about the same as on Small.
- **No week fell back to the naive rule** in any upload or local play.
- **CPU:** slowest week 0.22 s of 2 s on Small and 1.0 s of 4 s on Full (`sbf check` under Linux).

## Why these rules: three things we measured

1. **Running out is what costs.** In the kit heuristic's episodes, power shed at grids and unserved chip demand are
   about 2,900 of the 2,944 bn USD an episode costs; freight, tariffs, holding and the rest are 43. So the agent
   should spend freely on anything that keeps grids and fabs running.
2. **The planner stops caring at the end of its window.** It prices whatever is left after 20 weeks at the
   end-of-episode salvage price. On Full, where the window is a fifth of the episode, that alone holds it at 0.29.
3. **A grid serves its base load before its fabs, and the planner does not know.** A grid's power barely covers base
   load plus fabs, so a fuel that is a few per cent short takes all the fabs' energy for as long as the shortage
   lasts. The planner, from its second week on, plans to shed base load to power the fabs, which the simulator never
   does.

We also measured what the signals are worth, by feeding the planner the truth: every event known two weeks ahead adds
+0.003, decoy sanctions +0.0004, true demand nothing. So nothing reads the warnings.

## How it works

Each week: the planner solves a linear program for the cheapest flows over the next 20 weeks, assuming the network
stays as observed, and sends the first week's flows. Around that:

- **Window-end value (`reserve.py`).** Goods in stock, in transit or in process when the window ends are credited a
  share of their customs value instead of salvage. The largest single gain: +0.07 on Small, +0.33 on Full.
- **The pulse (`pulse.py`).** When a grid is short of a fuel, the agent holds that fuel at the terminal for some weeks
  and delivers a full week's burn in others. The energy burnt is the same, but in the full weeks the fabs run. It only
  changes terminal-to-grid flows, after the planner has chosen them. Together with the reserve and the premium:
  +0.08 on Small, +0.24 on Full.
- **Fuel trims (`pulse.py`).** In a week with power to spare, the simulator throttles every segment alike, so fuel
  burns where the fuel-free segment could have carried the load. The agent delivers only what the week needs and
  keeps the rest for a week that would otherwise shed. Fuel that can never be burnt before the episode ends is not
  shipped at all. +0.018 on Small, +0.012 on Full.
- **Recovery forecast (`recovery.py`).** A closed strait, a struck port or a grid short of power is planned to end as
  its event type's duration law says, not to last the whole window. +0.010 on Small, +0.007 on Full.
- **Smaller rules,** each worth up to about 0.01: a fab stops counting toward a pulse once its lots could not be sold
  before the episode ends; wafers are topped up at fabs that are about to get power; a fuel whose share is smaller
  than the fabs' draw does not block the pulse; a fuel reserve at grids; scarce fuel priced by the chips it costs; gas
  held in weeks 2 to 4 at grids the first plan already expects to run short (helps on Small only).

Every rule is a setting in `PARAMS` of `agent.py`, computed from the instance and the observation (no grid or fuel is
named in code), and wrapped so that a failure falls back to the planner's own flows.

## How we worked

- **Tune on one set, confirm on another.** Settings were chosen on root 12345 and confirmed once on root 2026 (and on
  a third Full set, root 777). The 20 dev episodes were played last. The board has stayed within about 0.006 of the
  fresh Small set.
- **Paired comparisons.** Every change is played on the same episodes as its predecessor and reported with a 90 %
  paired interval and the count of episodes better and worse (`experiments/approaches/tune.py`).
- **Ceiling before building.** Before building an idea we fed the planner the truth to see what the idea could be
  worth at best (`foresight.py`). That dropped five of the eight ideas we started with.
- **Follow the money.** The pulse and the trims came from comparing our play with the clairvoyant plan cost part by
  cost part, then with an exact model of the grid rule.
- **Merges proven exact.** With a new rule switched off, a version must reproduce its predecessor's cost to the cent
  on every episode before it is measured.

## What did not work

| Idea | Result |
| --- | --- |
| Acting on closure threats, a hazard model, contingency scenarios | ceiling +0.003 at a two-week lead with perfect prediction |
| Discounting decoy sanctions | ceiling +0.0004 |
| A longer window (24 to 40 weeks) | +0.03 without the window-end value, nothing with it, and it costs CPU |
| Pricing "base load first" in every week of the LP | -0.02 to -0.04 |
| Telling the planner which weeks the fabs will run | ceiling +0.001 |
| A premium on a grid's scarcest fuel only | no gain on fresh episodes |
| Not sending chips into full packaging plants | -0.0001 |
| Demand and throughput safety scales | +0.003 at most |
| A longer or shorter window on v8 (12 to 48 weeks) | -0.0004 (24) to -0.016 (48), -0.003 (16); 20 is the optimum |
| Leftovers valued above salvage at the episode's end; a tapered window-end value | -0.0013 to -0.0019; neutral |
| Sweeps of 27 settings (Small) and 13 (Full) on v8 | best `pulse_run` 6, +0.0005 over 8 sets; board 0.8767 (`mpc_pulse_rec_v9b`) |

## Run it

```bash
uv sync
uv run sbf evaluate mpc_pulse_rec_v8_on --task=small        # the 20 dev episodes
uv run sbf compare mpc_pulse_rec_v8_on mpc_det_h20 --task=small
uv run sbf check mpc_pulse_rec_v8_on --task=full            # needs Linux; on Windows use scripts/wsl_check.sh
uv run sbf upload mpc_pulse_rec_v8_on --dry_run             # then without --dry_run: one of the day's 3 submissions
```

On native Windows, `sbf evaluate` and `sbf compare` stop on a missing Unix module unless two stand-in files,
`fcntl.py` and `termios.py`, are put in `.venv\Lib\site-packages` (each only has to raise `OSError` when used), and
`sbf check` does not run at all. Under WSL everything works as it is.

## Where things are

| Path | What |
| --- | --- |
| `agents/mpc_pulse_rec_v8_on/` | the final agent: `agent.py` (settings, glue), `pulse.py`, `reserve.py`, `recovery.py`, `sbflow/` (the vendored planner), `params.json` (switches the opening hold on) |
| `agents/mpc_det_h20/` | the baseline every comparison is against |
| `experiments/pulse_agent/README.md` | the full log: every version, table and dropped idea |
| `experiments/pulse_agent/versions/` | all 22 earlier versions, runnable, with an index |
| `experiments/pulse_agent/analysis/` | the analysis scripts as they were run (a record, not a tool) |
| `experiments/approaches/` | the eight ideas we started from and their outcomes; the harness `tune.py`, `foresight.py`, `diag.py` |
| `scripts/wsl_check.sh` | `sbf check` under WSL for Windows machines |

## What is left

The last analyses agree that what separates the agent from the best achievable plan is now mostly knowledge of events
that have not happened yet. The rules that needed none are used up, and the steps have shrunk from +0.014 to about
+0.005 per version. Known weak spots: an energy shock that starts at a grid being trimmed costs about a week of the
trim, and the opening hold makes about a third of Small episodes slightly worse while helping the rest more.
