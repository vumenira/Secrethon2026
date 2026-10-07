#!/bin/bash
# usage: mk_dev.sh <name> '<extra json keys or empty>'  -> scratch copy of scratchpad/agents/v4_dev with v3's params plus the extra keys, strict
SP="$(cd "$(dirname "$0")" && pwd)"
SRC="$SP/agents/v4_dev"
BASE='"h_weeks": 20, "end_value": 2.0, "reserve_weeks": 8.0, "reserve_feeders": true, "reserve_price": 0.001, "reserve_fuels": "all", "shed_ramp": 0.002, "pulse": true, "pulse_run": 4, "pulse_hold_weeks": 12, "pulse_partial": 0.5, "pulse_payback": 12, "pulse_tail": 2, "pulse_rundown": true, "pulse_wafers": 1.0, "short_premium": 1.0, "strict": true'
rm -rf "$SP/agents/$1" && mkdir -p "$SP/agents/$1" && cp "$SRC/agent.py" "$SRC/pulse.py" "$SRC/reserve.py" "$SP/agents/$1/" && cp -r "$SRC/sbflow" "$SP/agents/$1/sbflow" && find "$SP/agents/$1" -name __pycache__ -type d -prune -exec rm -rf {} +
if [ -n "$2" ]; then echo "{$BASE, $2}" > "$SP/agents/$1/params.json"; else echo "{$BASE}" > "$SP/agents/$1/params.json"; fi
