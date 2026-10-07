# Training the Small BC model on the Windows PC

Step-by-step instructions for the training PC (Windows, Ryzen 7 7700: 8 cores / 16 threads, 32 GB, RTX 5070). They
assume no earlier context: a person, or a Claude Code session started on this PC, can follow them as they are.

## What this run does, in short

It trains a behaviour-cloning (BC) agent for **Small**, the public board's network. A *teacher* that knows each
episode's future (a linear program re-solved every week) plays about 800 training episodes; small neural networks
learn to copy its weekly decisions. Two kinds of agent are trained from the same data:

- **plain BC:** the network sees the observation only;
- **BC + LP:** the network also sees the plan of the package's LP planner (`mpc_det`) and learns to correct it. Each
  also gets a `_floor` version that never ships less than the planner.

The best candidate on fresh episodes is kept, compared with `heuristic` and with `mpc_det_h20` (the LP planner alone:
0.712 on Small's dev episodes, **the score to beat**), and published to `agents\bc_small_best\`. Background and every
earlier result: [README.md](README.md), section "The pipeline".

## 1. Set up (once)

In **PowerShell**:

1. Install git (https://git-scm.com/download/win) if it is missing.
2. Install uv:

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Close and reopen PowerShell so `uv` is found.
3. Get the code, on the branch with the pipeline:

   ```powershell
   git clone https://github.com/vumenira/Secrethon2026.git
   cd Secrethon2026
   git checkout bc-pipeline
   uv sync
   ```

   `uv sync` installs Python 3.13 and the exact packages (a few minutes the first time).
4. Optional, saves about 40 minutes: copy the reference-cost cache from the laptop, the folder
   `~/.cache/shockbench-flow/` there, to `%USERPROFILE%\.cache\shockbench-flow\` here. Without it, the run computes
   the reference costs it needs itself.

## 2. Run

From the `Secrethon2026` folder:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1 -Cuda
```

- `-Cuda` installs, once, a PyTorch built for CUDA 12.8 (the RTX 5070 needs it; PyPI's Windows PyTorch has no CUDA)
  and trains on the GPU. Leave it out to train on the CPU (much slower: see the table).
- Keep the PC awake (Settings > System > Power: sleep "Never" while it runs). Close heavy programs: it uses 14 worker
  processes and most of the CPU.

### How long

| Stage | What you see in the log | Time (estimate) |
| ----- | ----------------------- | --------------- |
| collect | `collect: chunk 3/50 ... saved; 60 s per play per worker; about 95 min left` | 2 h (stops itself after 2 h) |
| train | `train: device cuda ...`, then one line per epoch per network | about 15 min with `-Cuda`; 2.5 to 3 h on the CPU |
| select | the scores of every candidate, `heuristic` and `mpc_det_h20` on 20 episodes of root 777 | about 30 min (reference costs once) |
| confirm | the best one on dev and on 20 episodes of root 999, with "minus heuristic" and "minus mpc_det_h20" lines | about 30 min |
| publish | `publish: <best> -> ...agents\bc_small_best` and `report: ...REPORT.md` | seconds |

Total: about 3.5 h with `-Cuda`, 5 to 6 h without.

### If it stops

Run the same command again. Finished chunks of data (`outputs\bc\small_pc\chunks\`) and trained candidates
(`outputs\bc\small_pc\candidates\`) are kept and skipped; only the chunk or variant in progress is redone. To run
only some stages, pass them **in quotes** (PowerShell splits unquoted commas):

```powershell
powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1 -Cuda "--stages=select,confirm,publish"
```

Every option: `uv run --no-sync python experiments\bc\pipeline.py --help`.

### Normal messages that are not errors

- `episode 75 (play 0) skipped: RuntimeError: week 3: the teacher's LP was not solved` — about 0.5 % of plays: the
  linear program fails on rare weeks; the play is skipped.
- `UserWarning` lines from gymnasium about unbounded boxes.

### Problems

| Problem | Fix |
| ------- | --- |
| `uv` is not recognised | reopen PowerShell after installing uv |
| "running scripts is disabled" | use the `powershell -ExecutionPolicy Bypass -File ...` form above |
| the CUDA install fails, or `train: device cpu` despite `-Cuda` | run without `-Cuda`, and halve the training: add `"--variants=plain_u4,lp_u4,lp_u8"` |
| out of memory, or the PC freezes during collect | add `--workers=10` (each worker uses about 0.5 GB) |
| a stage crashes with a traceback | run the same command again; if it crashes at the same place, save the log (see below) |

To keep a log of everything (useful to send back), run it like this:

```powershell
powershell -ExecutionPolicy Bypass -File experiments\bc\run_small_pc.ps1 -Cuda *>&1 | Tee-Object -FilePath small_pc.log
```

## 3. Bring the result back

When it ends:

1. Read `outputs\bc\small_pc\REPORT.md`. It holds every candidate's score, the best one, and its paired differences
   against `heuristic` and `mpc_det_h20` on dev and root 999.
2. Copy the report next to the agent (`outputs\` is not committed), then commit and push:

   ```powershell
   copy outputs\bc\small_pc\REPORT.md agents\bc_small_best\REPORT.md
   copy outputs\bc\small_pc\results.json agents\bc_small_best\results.json
   git add agents\bc_small_best
   git commit -m "BC small pipeline: best candidate and report"
   git push origin bc-pipeline
   ```

3. On the Linux laptop: `git pull`, then `uv run sbf check bc_small_best --task=small` (`sbf check` needs Linux) and
   `uv run sbf compare bc_small_best mpc_det_h20 --task=small`.

**Do not upload to Codabench from the PC.** Each upload spends one of the team's 3 daily submissions; decide on the
laptop after `sbf check`.

## For a Claude Code session on this PC

If you are Claude Code helping on this PC: the task is only to set up and run the steps above, watch the run, fix
environment problems (installs, paths, PowerShell quoting, memory), and report the results of `REPORT.md`. Read
[README.md](README.md) ("The pipeline") and the repository's `AGENTS.md` for context.

- Don't change the training code or its settings unless the user asks. If something must change, explain why first.
- The run is resumable: after any interruption, run the same command again.
- Long stages are normal (see the table above); check progress in the log instead of restarting.
- Never upload to Codabench and never print or commit `.env`.
- When it ends, summarise `outputs\bc\small_pc\REPORT.md`: the best candidate, its scores, and whether it beats
  `mpc_det_h20` (the paired interval above 0) or not.
