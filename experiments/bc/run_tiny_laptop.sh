#!/usr/bin/env bash
# The BC pipeline on Tiny, sized for this laptop (8 threads, 8 GB). Run from anywhere:
#
#     bash experiments/bc/run_tiny_laptop.sh                    # everything, about 1.5 to 2 hours
#     bash experiments/bc/run_tiny_laptop.sh --stages=select,confirm,publish   # any stage again, any option
#
# Close the browser first: the number of worker processes follows the free memory (about 0.35 GB each).
# Run it again after a crash or Ctrl+C: finished chunks and trained candidates are kept (outputs/bc/tiny_laptop/).
# The result: agents/bc_tiny_best/ and outputs/bc/tiny_laptop/REPORT.md. Details: experiments/bc/README.md.
set -euo pipefail
cd "$(dirname "$0")/../.."

uv sync
uv run python experiments/bc/pipeline.py \
    --task=tiny --run=tiny_laptop --publish=bc_tiny_best \
    --episodes=2000 --collect_hours=1.5 \
    "$@"
uv run sbf check bc_tiny_best --task=tiny
