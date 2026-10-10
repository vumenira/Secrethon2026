# Analysis scripts behind the pulse agent

The scripts that produced the measurements in [../README.md](../README.md), saved as they were run. They are a record,
not a tool:

- **They are not maintained and not linted.** Many hold absolute paths of the laptop they ran on (a temporary scratch
  folder) and expect data files that are not here.
- **The data is not committed.** Recorded trajectories, solver solutions and per-episode costs were several hundred MB.
  The scripts regenerate them, given a few hours.
- **The maintained harness is elsewhere:** `experiments/approaches/tune.py` (play variants, paired comparison),
  `foresight.py` (feed the planner ground truth) and `diag.py`.

| Folder | What is in it | Start with |
| --- | --- | --- |
| top level | work of 2026-10-05 and 10-06: the capacity-cut analysis (`cap*.py`), the trace of the pulse week by week, the first exact model of the grid rule, CPU timing, merge checks | `trace_v2.py`, `analyze_v2.py`, `milp_real.py`, `cpu_time.py`, `pairs.py` |
| `t1b/` | closures and grid shortages: event spells sampled from the generator, calibration of the duration laws | `spells.py`, `calib.py` |
| `gap/` | the exact model with our own fab calendar kept, and the split of the remaining gap by decision group and by time | `common.py` (the model), `replay.py`, `solve.py`, `report.py`, `slack.py`, `trim.py` |
| `gap/v6/`, `gap/v7/` | probes written while building v6 and v7 (the slack split, failure paths, CPU) | `gap/v7/split.py` |
| `left/` | the opening weeks and the choice of fab weeks; the play test of the opening hold; the exact rule that showed the earlier "realizable" plan was not realizable | `playhold.py`, `tight.py`, `opening.py` |
| `v5/` | the hindsight-calendar ceiling and the calendar forecast | `ceiling.py`, `lab.py` |
| `v8/` | probes written while building the opening hold | `crit.py`, `look.py` |
| `v13/` | the round behind v13 (2026-10-10): `stand_down/` (the census of every week the pulse leaves a grid as it is), `runs/` (how often `schedule_ends`, `schedule_runs` and the wafer top-up misfire), `worst_episodes/` (the twelve worst Full episodes against the clairvoyant plan, the bound, the strait reserve), `cpu/` (the first week's CPU, the scoring host's speed, the guard's test) and `probes/` (lost sales at sinks, the buffer that was dropped, grid and fab traces of recorded episodes) | `pooled.py` (several sets as one), `with_lock.py` (plays of several sessions share the machine), `stand_down/census.py`, `worst_episodes/rec.py`, `cpu/t_host.py` |

Two scripts worth knowing by name:

- **`wsl_check.sh`** is superseded by `scripts/wsl_check.sh` in the repository root, which has no fixed paths.
- **`pairs.py`** checks that two cached plays are equal to the cent, episode by episode. Every merge of two versions
  was accepted only on that check.
