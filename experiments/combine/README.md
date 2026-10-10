# Combining the teammate's v10 with our v10b (2026-10-09)

The teammate's line since v8 (on `main` after PR #2 and commit `b0e9e59`) and ours both attacked the same failure: a
grid short of two fuels (gas and crude) whose pulse ran them out of step. The question here: which of the agents is
best, and what of ours still adds to the teammate's v10. Agents built here carry a `b`.

## The agents

| Agent | Whose | What it is | Public board |
| --- | --- | --- | --- |
| `versions/mpc_pulse_rec_v9` | teammate | v8_on + `schedule` (runs of whole weeks start sooner, end cleanly) | 0.8827 |
| `versions/mpc_pulse_rec_v9_safe` | teammate | v9 with every searched setting reset (the overfitting test) | 0.8793 |
| `versions/mpc_pulse_rec_v10_on` | teammate | v10's first form, its four switches on (`params.json`) | 0.8854 |
| `agents/mpc_pulse_rec_v10` | teammate | v9 + `schedule_fuels` (a whole week needs every fuel, looked ahead) + `schedule_hold` | not uploaded |
| `agents/mpc_pulse_rec_v9b` | ours | v8_on + `pulse_run` 6 | 0.8767 |
| `agents/mpc_pulse_rec_v10b` | ours | v9 + `pulse_sync` + `pulse_hold_scarce` + `pulse_hold_weeks` 26 + `pulse_payback` 0 | 0.8845 |
| `versions/mpc_pulse_rec_v11b_x` | ours | the teammate's v10 + `pulse_sync` and `pulse_hold_scarce` as switches (off) | |

`schedule_fuels` is a fuller form of our `pulse_sync` (it looks several weeks ahead with the pulse's arrival forecast,
and also decides whether a run goes on). Our `pulse_hold_scarce`, `pulse_hold_weeks` 26 and `pulse_payback` 0 have
no counterpart in the teammate's code.

Combinations, all on the teammate's v10 (`mpc_pulse_rec_v11b_x`):

- `c1_scarce`: + `pulse_hold_scarce`
- `c2_scarce_h26_p0`: + `pulse_hold_scarce`, `pulse_hold_weeks` 26, `pulse_payback` 0
- `c3_h26_p0`: + `pulse_hold_weeks` 26, `pulse_payback` 0 (no new code)
- `c4_all`: + `pulse_sync` too

## Round 1: Full 777 and 2026 (32 episodes each), Small 2026 (64)

Differences against the teammate's v10 (`t_v10`), 90 % paired intervals, better / worse episodes:

| Agent | Full 777 | Full 2026 | Small 2026 |
| --- | --- | --- | --- |
| `t_v10` (score) | 0.8509 | 0.8951 | 0.8796 |
| v9 | -0.0062 (-0.0098 to -0.0030), 4 / 19 | -0.0029 (-0.0049 to -0.0013), 6 / 12 | -0.0030 (-0.0046 to -0.0016), 16 / 17 |
| `t_v9_safe` | -0.0131 (-0.0182 to -0.0083), 3 / 29 | -0.0105 (-0.0149 to -0.0064), 6 / 26 | -0.0083 (-0.0126 to -0.0045), 22 / 42 |
| `t_v10_on` | -0.0019 (-0.0030 to -0.0010), 2 / 15 | -0.0007 (-0.0014 to +0.0001), 6 / 11 | -0.0004 (-0.0013 to +0.0004), 13 / 13 |
| v10b (ours) | +0.0032 (+0.0010 to +0.0056), 20 / 12 | +0.0031 (+0.0005 to +0.0058), 14 / 18 | -0.0012 (-0.0031 to +0.0007), 21 / 32 |
| `c1_scarce` | -0.0003 (-0.0010 to +0.0005), 9 / 10 | -0.0005 (-0.0014 to +0.0004), 9 / 11 | -0.0010 (-0.0025 to +0.0005), 7 / 14 |
| `c2_scarce_h26_p0` | +0.0046 (+0.0022 to +0.0071), 21 / 11 | +0.0042 (+0.0016 to +0.0069), 15 / 17 | +0.0004 (-0.0012 to +0.0022), 17 / 33 |
| **`c3_h26_p0`** | **+0.0046 (+0.0029 to +0.0066), 19 / 10** | **+0.0043 (+0.0023 to +0.0066), 18 / 10** | **+0.0013 (+0.0005 to +0.0022), 15 / 31** |
| `c4_all` | +0.0042 (+0.0019 to +0.0066), 21 / 11 | +0.0039 (+0.0013 to +0.0068), 15 / 17 | +0.0005 (0.8801) |

- **The teammate's v10 beats v9 everywhere,** and the snapshot that leads the board (`v10_on`) is slightly worse than
  his final v10 on all three sets: its board lead over our v10b (0.8854 against 0.8845) is within noise.
- **`v9_safe` loses 0.008 to 0.013**: the settings searched before v9 are real gains, not noise.
- **Our two rules add nothing to his:** `pulse_hold_scarce` alone is -0.0003 to -0.0010 (his `schedule_fuels`
  already holds the second fuel for the week it can run), and `pulse_sync` on top (`c4`) is a touch below `c2`.
- **Our two settings add a lot:** `pulse_hold_weeks` 26 and `pulse_payback` 0 on his v10 (`c3`) gain +0.0046 and
  +0.0043 on Full and +0.0013 on Small, every interval above 0. It is the best agent on all three sets, and it needs
  no new code: two values in `PARAMS`.
- With our switches off, `mpc_pulse_rec_v11b_x` must equal his v10 to the cent (`c0_off`, Full 12345: below).
