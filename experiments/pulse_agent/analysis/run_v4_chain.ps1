# The eleven plays of agents/mpc_pulse_rec_v4, one at a time with two workers: the regressions on root 12345 (each
# gated: the chain stops if one is not equal to the cent), then both settings on all four episode sets.
# Stopped by the system on 2026-10-06 at 23:43, about 30 s in, for lack of memory; nothing was cached. A play that is
# cached is skipped, so the script can be started again after any interruption.
$env:PATH = "C:\Users\mykha\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe;" + $env:PATH
Set-Location "C:\Users\mykha\Claude code projects\Secrethon2026"
$s = "C:\Users\mykha\AppData\Local\Temp\claude\C--Users-mykha-Claude-code-projects-Secrethon2026\d989db8b-5de3-4e20-b2c1-6278ade0c74d\scratchpad"
$v = "outputs/recovery_tune/variants"

"##### full 12345, regressions"
uv run python experiments/approaches/tune.py run --variants=$v/v_reg_full.json --task=full --entropy=12345 --episodes=16 --n_jobs=2
$same = uv run python "$s\pairs.py" same --pairs="v_reg_v4=ft_v4,v_reg_v3=w_all" --task=full --entropy=12345
$same | ForEach-Object { "GATE $_" }
if (@($same | Where-Object { $_ -match "16 of 16 episodes equal" }).Count -ne 2) {
    "##### a Full regression is not exact: stopped before the other plays"
    exit 3
}

"##### small 12345, regression"
uv run python experiments/approaches/tune.py run --variants=$v/v_reg_small.json --n_jobs=2
$same = uv run python "$s\pairs.py" same --pairs="v_reg_v3=w_all" --task=small --entropy=12345
$same | ForEach-Object { "GATE $_" }
if (@($same | Where-Object { $_ -match "64 of 64 episodes equal" }).Count -ne 1) {
    "##### the Small regression is not exact: stopped before the other plays"
    exit 3
}

"##### small 12345"
uv run python experiments/approaches/tune.py run --variants=$v/v_two.json --n_jobs=2
"##### full 12345"
uv run python experiments/approaches/tune.py run --variants=$v/v_two.json --task=full --entropy=12345 --episodes=16 --n_jobs=2
"##### small 2026"
uv run python experiments/approaches/tune.py run --variants=$v/v_two.json --entropy=2026 --n_jobs=2
"##### full 2026"
uv run python experiments/approaches/tune.py run --variants=$v/v_two.json --task=full --entropy=2026 --episodes=16 --n_jobs=2
"##### all v plays done"
