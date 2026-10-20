#!/usr/bin/env bash
# Retrain every NFLELO model on fresh nflverse data and save the outputs into notebooks/train_all_models.ipynb.
#   scripts/train_all.sh          # full (about 75 minutes)
#   scripts/train_all.sh quick    # skips the slow tuning searches and the long simulation backtest (about 25 minutes)
# Needs the project .venv (setup commands are in the notebook's first cell). Nothing tracked is modified except the notebook.
set -euo pipefail
cd "$(dirname "$0")/.."

export TRAIN_MODE="${1:-${TRAIN_MODE:-full}}"
[ -x .venv/bin/jupyter ] || { echo "No .venv found: see the first cell of notebooks/train_all_models.ipynb for setup."; exit 1; }
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 notebooks/train_all_models.ipynb
echo "Done. Outputs saved in notebooks/train_all_models.ipynb (mode: $TRAIN_MODE)."
