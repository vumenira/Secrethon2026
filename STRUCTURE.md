# Repository map

Every folder of the repository and what it holds. Back to the [README](README.md).

**Legend.** ★ the final solution · **ours**: built by the team · **kit**: from the organisers' starter kit ·
*local only*: not tracked by git (it exists on one team member's machine and is not on GitHub).

## At a glance

```
Secrethon2026/
├── README.md              the capstone write-up: the task, our approach, the final agent, the results
├── STRUCTURE.md           this map
├── SOLUTION.md            the solution in short, with every measurement (SOLUTION.uk.md: in Ukrainian)
├── PLAN.md                the team's plan at the start of the hackathon (in Ukrainian)
├── READMEOLD.md           the organisers' original README: installation, the task, every sbf command
├── AGENTS.md, CLAUDE.md   instructions for a coding assistant working in the repository
├── CHANGELOG.md           the starter kit's changelog (kit)
├── agents/                one folder per agent: ★ mpc_pulse_rec_v13 and every earlier one
├── experiments/           one write-up per approach, with its research code
├── src/sbf_starter/       the `sbf` command line (kit), with our episode viewer in viewer/
├── scripts/               small tools: vendoring the planner, the leaderboard, the README's charts
├── examples/              the kit's seven example scripts
├── docs/                  the task's guide (kit), the viewer's guide, the images
├── tests/                 the test suite (`uv run pytest -n 3`)
├── outputs/               run folders, caches, viewer pages, zips (gitignored)
├── pyproject.toml, uv.lock, Makefile, LICENSE
```

## agents/

An agent is a folder with `agent.py` at its root, defining `class Agent`. Name one in any `sbf` command
(`uv run sbf evaluate mpc_pulse_rec_v13`).

### The final agent and its line

| Folder | What it is | Public board |
| --- | --- | --- |
| ★ `mpc_pulse_rec_v13/` | **the final agent** (ours): `agent.py` (settings, glue, CPU guard), `pulse.py`, `reserve.py`, `recovery.py`, `sbflow/` (the package's planner, vendored), `instances/` (the public networks, preloaded) | **0.8972** |
| `mpc_pulse_rec_v12/` | the version before: fuel by pipe, lean hold, tariff cut (ours) | 0.8910 |
| `mpc_pulse_rec_v11/` | scarce fuels held together, shared routes (ours); not uploaded | |
| `mpc_pulse_rec_v10/` | a whole week needs every fuel (ours) | |
| `mpc_pulse_rec_v10_on/` | the uploaded snapshot of v10, switches set by `params.json` (ours) | 0.8854 |
| `mpc_pulse_rec_v10b/` | a parallel branch: v9 with two pulse rules and two settings for grids short of two fuels (ours) | 0.8845 |
| `mpc_pulse_rec_v9b/` | v8 with `pulse_run` 6, from a settings sweep (ours) | 0.8767 |
| `mpc_pulse_rec_v11b/` | *local only*: v10 with two settings of the v10b branch (`experiments/combine/`) | |
| `mpc_det_h20/` | the baseline: the package's rolling LP planner with a 20-week window, vendored | 0.6751 |

Earlier versions (ft_buffer, v2 to v9, the v12 pieces, probes) are runnable under
`experiments/pulse_agent/versions/` (below).

### Earlier approaches

| Folder | What it is | Status |
| --- | --- | --- |
| `3rule_based/` | rule-based agent: what each destination needs, on the best open route (ours) | abandoned; Small dev 0.551 |
| `bc_tiny_v12/` | behaviour cloning of the clairvoyant teacher: the Tiny baseline (ours) | abandoned; ties the heuristic |
| `bc_tiny/`, `bc_tiny_nonoise/`, `bc_tiny_rel/`, `bc_tiny_v4/`, `bc_tiny_v4_ship/`, `bc_tiny_v6/`, `bc_tiny_v8/`, `bc_tiny_v9/`, `bc_tiny_v10/`, `bc_tiny_v11/`, `bc_tiny_dagger/`, `bc_tiny_best/`, `bc_small_v1/` | *local only*: the twelve behaviour-cloning runs and their variants (DAgger, noise, a Small run) | abandoned |
| `1ppo_tiny/` | *local only*: PPO trained on Tiny with `examples/05_train_ppo.py` | abandoned |
| `2search_tiny/` | *local only*: the heuristic tuned by `examples/06_policy_search.py` | abandoned |
| `test0/` | *local only*: a scratch copy of the template, the first test upload | scratch |

### From the kit

| Folder | What it is |
| --- | --- |
| `template/` | sends the maximum on every allowed route; the starting point of a new agent |
| `random/` | random flows |
| `heuristic/` | the kit's heuristic: the maximum scaled by each strait's open fraction (Small dev 0.400) |

## experiments/

One write-up per approach, next to the code it ran.

| Path | What it is |
| --- | --- |
| `pulse_agent/README.md` | **the full log of the final agent**: every version from `ft_buffer` to v13, every table and dropped idea |
| `pulse_agent/versions/` | every earlier version, runnable, with an index (`versions/README.md`): `ft_buffer` to `ft_buffer_v5`, `mpc_pulse_rec` and `_v2` to `_v9`, the v12 pieces `_v12a`, `_v12b`, `_v12c`, side branches (`_v9_safe`, `_v9_tail`, `_v9_war`, `_v10_kr`, `_v9b_x`, `_v10b_x`), probes that feed the planner the truth (`fs_probe`, `fs_probe_ev`), and the first upgrades (`mpc_recovery`, `rec_*`, `buf_*`, `mpc_buffer`, `mpc_combined`). `mpc_pulse_rec_v11b_x/` is *local only* |
| `pulse_agent/analysis/` | about 100 analysis scripts as they were run (cost gaps, grid traces, bounds, CPU timing): a record, not a tool |
| `pulse_agent/V9V10RESULTS.md` | the steps after v8 (v9b, v9, v10b): what was tried, kept, and where each number comes from |
| `approaches/` | the eight upgrades we started from, their measured ceilings and outcomes (`README.md`); the harness `tune.py` (paired comparison of variants, cached), `foresight.py` (feeds the planner the truth to measure a ceiling), `diag.py` (episodes where two variants differ most) |
| `window_end/` | experiments on v8: window length, window-end value, settings sweeps, the gap against the clairvoyant plan |
| `night/` | a night of experiments on v10: the v10b branch |
| `combine/` | combining the two v10 branches (v11b); the agents it played are *local only* |
| `full_night/` | v12 played on 14 Full sets (396 episodes), Full-only settings, and `diag/`: a diagnosis of the worst Full episodes with integer-program bounds (its four test copies of v12, `diag/agents/`, are *local only*; the rule they test is described in `diag/README.md`) |
| `eda/` | exploratory analysis of the environment (where the money is, which signals carry information) |
| `bc/` | behaviour cloning and DAgger: training code (`train_replan.py`, `nets.py`, `replan_lib.py`) and the log of 12 runs |
| `hypotheses.md` | four ideas for learning from the clairvoyant plan, ranked |
| `heuristic.md`, `3rule_based.md` | the first two agents' write-ups |

## src/sbf_starter/: the `sbf` command line

| File | What it does |
| --- | --- |
| `cli.py` | the commands: `evaluate`, `compare`, `view`, `check`, `pack`, `token`, `upload`, `status` (kit; `view` is ours) |
| `scoring.py`, `check.py`, `container.py`, `play.py`, `agents.py` | scoring, the isolated timed run, the Docker run, playing, agent names to folders (kit) |
| `codabench.py` | the Codabench API: token, upload, status (kit) |
| `viewer/` | **our episode viewer** (`sbf view`): `__init__.py` (the command and its cache), `export.py` (a recording to JSON), `page.html` (the single-file page). Guide: [docs/VIEWER.md](docs/VIEWER.md) |

## scripts/

| File | What it does |
| --- | --- |
| `vendor_planner.py` | copies the package's planner into an agent's `sbflow/`, patched to use the HiGHS that SciPy bundles (ours) |
| `leaderboard.py` | shows which of our submissions is on the leaderboard, or puts another one on it (ours) |
| `readme_figures.py` | draws the README's charts into `docs/img/` (ours) |
| `wsl_check.sh` | runs `sbf check` under WSL on a Windows machine (ours) |
| `fields_docs.py` | regenerates `docs/fields/` from the package (kit) |

## examples/ (kit)

`01_quickstart.py` (the gym loop), `02_play_agents.py`, `03_heuristic_agent.py`, `04_evaluate.py`,
`05_train_ppo.py` (PPO), `06_policy_search.py` (evolutionary search), `07_dashboard.py` (network map, dashboards, a
GIF), and `ppo_agent.py` (the PPO submission's `agent.py`).

## docs/

| Path | What it is |
| --- | --- |
| `GUIDE.md` | the task's interface, rules and scoring (kit) |
| `fields/` | every observation and action field of each network (kit, generated) |
| `VIEWER.md` | the viewer's user guide (ours) |
| `img/` | images: `network_small.png`, `dashboard_tiny.png` (kit); `viewer_*.png` (screenshots of our viewer); `progress_board.png`, `progress_steps.png`, `pulse.png` (drawn by `scripts/readme_figures.py`); `weekly_loop.svg` (the agent's week) |

## tests/

`test_cli.py`, `test_codabench.py` (with `mock_codabench.py`), `test_examples.py` (kit) and `test_viewer.py` (ours).

## outputs/ (gitignored)

Written by runs, never committed: `recovery_tune/<task>-e<root>-n<N>/` (cached plays of `tune.py`, one JSON per
variant), `view/<agent>_<task>/` (viewer pages), example run folders `<example>/<date_time>/`, and packed zips. The
reference costs of each episode set (the naive and clairvoyant plans) are cached outside the repository, in
`~/.cache/shockbench-flow/`.
