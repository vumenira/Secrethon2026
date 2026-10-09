#!/usr/bin/env bash
# `sbf check` under Linux, for Windows machines: the isolated timed run needs symlinks and a Unix transport, so it does
# not work natively there. Run it from PowerShell through WSL, from any directory:
#
#   $p = wsl -d Ubuntu -- wslpath -a "scripts/wsl_check.sh"
#   wsl -d Ubuntu -- bash "$p" mpc_pulse_rec_v10 full
#
# usage: wsl_check.sh <agent> <task>     (agent: a folder name of agents/ or a path; task: tiny, small or full)
# It keeps its own environment (~/.venvs/secrethon) so that uv does not fight over the Windows .venv.
set -u
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/secrethon}"
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
echo "== sbf check $1 --task=$2 (started $(date +%H:%M:%S)) =="
uv run sbf check "$1" --task="$2" 2>&1 | tail -n 40
status=${PIPESTATUS[0]}
echo "== exit $status (ended $(date +%H:%M:%S)) =="
exit "$status"
