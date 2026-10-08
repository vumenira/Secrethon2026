# Earlier versions of the pulse agent

Every agent folder built on the way to `agents/mpc_pulse_rec_v9`, kept as it was played so that any number in
[../README.md](../README.md) can be replayed. None of these is the agent to submit.

- **Each folder is a complete submission folder** (`agent.py`, its modules and its own copy of the vendored planner
  `sbflow/`). Play one by path: `uv run sbf evaluate experiments/pulse_agent/versions/mpc_pulse_rec_v6 --task=small`.
- **`experiments/approaches/tune.py` finds them by name**, so a variants file can still say `"base": "ft_buffer_v4"`.
- **The code is untouched since it was played.** Docstrings inside still say `agents/<name>` for folders that now live
  here.
- **`sbflow/` is the same in every folder** as in `agents/mpc_det_h20/`, except in `fs_probe`, which adds one switch.

Scores are on Small unless marked: "tuning" is root 12345, "fresh" is root 2026 (64 episodes each), "board" is the
public leaderboard.

## The line that was uploaded

| Folder | What it adds to the one before | Score |
| --- | --- | --- |
| `ft_buffer` | window-end value, fuel reserve, pulse, shortfall premium (on `mpc_det_h20`) | board 0.8289 |
| `mpc_pulse_rec_v2` | recovery forecast; a fab stops counting once its lots cannot be sold in time | board 0.8429 |
| `mpc_pulse_rec_v3` | wafer top-up at pulsed fabs; run-down before sales end | board 0.8498 |
| `mpc_pulse_rec_v4` | a small fuel no longer blocks the pulse | board 0.8577 |
| `mpc_pulse_rec_v6` | gas trimmed in slack weeks; unburnable fuel not shipped | board 0.8714 |
| `mpc_pulse_rec_v8_on` | trim for other fuels and a better forecast; opening hold (its `params.json` switches the hold on) | board 0.8768 |
| `agents/mpc_pulse_rec_v9` (the final agent, not in this folder) | runs of full weeks start sooner and end cleanly | board 0.8827 |

## Steps between the uploads

| Folder | What it is | Score |
| --- | --- | --- |
| `mpc_pulse_rec` | `ft_buffer` plus the recovery forecast, before the end-of-sales rule | fresh 0.8318 |
| `mpc_pulse_rec_v7` | `mpc_pulse_rec_v6` plus the two trim rules, before the opening hold | fresh 0.8676 |
| `mpc_pulse_rec_v8` | the code of `mpc_pulse_rec_v8_on` with the opening hold switched off; plays exactly as v7 | fresh 0.8676 |
| `ft_buffer_v2`, `_v3`, `_v4` | the pulse of v2, v3 and v4 without the recovery forecast; each has a `params.json` | tuning 0.8327, 0.8373, 0.8496 |
| `ft_buffer_v5` | research on telling the planner which weeks the fabs run; plays as `ft_buffer_v4` | not a gain |

## Before the pulse

| Folder | What it is | Score |
| --- | --- | --- |
| `mpc_recovery` | first recovery forecast on `mpc_det_h20` | tuning 0.6954 at best |
| `rec_rule` | forecast rules for closures and grid shortages | tuning 0.7001 |
| `rec_capacity` | capacity cuts split into strikes (forecast) and sanctions or wars (kept) | tuning 0.6896 |
| `rec_merged` | `rec_rule` and `rec_capacity` together | tuning 0.7042 |
| `buf_knobs` | window, demand scale and throughput scale as settings | tuning 0.7251 at a 40-week window |
| `buf_lp` | first `reserve.py`: window-end value, fuel reserve, ramp | tuning 0.7562 |
| `mpc_buffer` | `buf_lp` and `buf_knobs` together; has a `params.json` | tuning 0.7562 |
| `mpc_combined` | `mpc_buffer` plus `rec_merged` | fresh 0.7545 (0.7568 with the reserve off) |

## Analysis probes (not agents)

| Folder | What it is |
| --- | --- |
| `fs_probe` | `mpc_det_h20` with a hook through which `experiments/approaches/foresight.py` feeds the scenario's ground truth |
| `fs_probe_ev` | the same hook on `ft_buffer` |
