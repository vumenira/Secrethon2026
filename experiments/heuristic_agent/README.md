# Heuristic agent: baseline run (2026-10-04)

The kit's `agents/heuristic` as shipped, no changes: every route sends its nominal capacity, and a sea lane through a
strait sends that capacity times the strait's open fraction. Local runs only; nothing was uploaded.

## Results

|                                         | Small network                    | Full network                     |
| --------------------------------------- | -------------------------------- | -------------------------------- |
| Score, 20 public dev episodes           | **0.4004** (90 %: 0.29 to 0.50)  | not run                          |
| Score by harm level 1 to 4              | 0.31, 0.40, 0.58, 0.57           | not run                          |
| Mean cost per episode, bn USD           | agent 3,049; naive 3,485; clairvoyant 2,555 | not run               |
| Weeks played by the naive rule          | 0 of 1,040                       | not run                          |
| `sbf check`                             | passed                           | passed                           |
| CPU per week: first / median / max      | 0.016 / 0.0017 / 0.016 s of 2 s  | 0.041 / 0.0035 / 0.041 s of 4 s  |

These are local scores on this machine, not leaderboard scores: the 20 dev episodes are practice scenarios that ship
with the package, while the leaderboard scores an uploaded agent on 200 hidden episodes of Small (and later 400 of
Full). The scale is the same: 0 is the naive rule, 1 the clairvoyant plan. Any agent that replaces this one has to
beat 0.40 on Small, tested with `uv run sbf compare <new> heuristic --task=small`.

## How it was run

```
uv run sbf evaluate heuristic --task=small
uv run sbf check heuristic --task=small
uv run sbf check heuristic --task=full
```

- Machine: Intel i5-1235U, 12 logical cores, 16 GB. Python 3.13.16, numpy 2.4.5, scipy 1.18.1, shockbench-flow 0.1.2.
- `evaluate` ran twice, on Windows 11 and on Ubuntu 26.04 under WSL 2: the score and the costs are identical.
- `check` ran under Ubuntu only (it does not run on Windows) and times one dev episode.
- First `evaluate` on Small: 16 min on Windows, 22 min under Ubuntu, of which 4.5 and 6.3 min were logged as building
  the reference costs. Those are cached, so later runs skip that part.
- Submission zip: 2,256 bytes, sha256 `d002c435d514f741e5f465760bafa420c5700c97d5e03647204fdc0b81df5eec`.

## Limits

- 20 episodes is a noisy sample; the board scores 200 hidden episodes.
- `evaluate` enforces no CPU limit; the timing above comes from `check`, not from the scoring container
  (`sbf check --docker` was not run).
- The Full evaluation was not run: its first run builds the Full reference costs (the guide gives over a minute per
  episode for the clairvoyant plan alone), and the machine had 3.5 GB of free memory at the time.

## Windows notes

- `sbf evaluate` needs two stand-in modules, `fcntl.py` and `termios.py`, in `.venv\Lib\site-packages` (the package
  imports these Unix-only modules without using them for local scoring).
- `sbf check` needs Linux. Under WSL, from the project folder on `/mnt/c/...`, set
  `UV_PROJECT_ENVIRONMENT=~/.venvs/secrethon` first so the Linux environment does not collide with the Windows `.venv`.
