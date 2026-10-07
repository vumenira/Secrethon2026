#!/usr/bin/env bash
# usage: wsl_check.sh <agent> <task>
export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/secrethon
export PATH=$HOME/.local/bin:$PATH
cd "/mnt/c/Users/mykha/Claude code projects/Secrethon2026" || exit 1
echo "== sbf check $1 --task=$2 (started $(date +%H:%M:%S)) =="
uv run sbf check "$1" --task="$2" 2>&1 | tail -n 40
echo "== exit ${PIPESTATUS[0]} (ended $(date +%H:%M:%S)) =="
