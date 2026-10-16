#!/bin/sh
# Regenerate every result in the paper, including the experiments added for the review
set -e
mkdir -p logs
M=label_noise_experiment
for model in lr hgb; do
  uv run python -m $M.run --model $model                       > logs/run_$model.log 2>&1
  uv run python -m $M.oracle_check --model $model --oracle-frac 0.1 > logs/oracle10_$model.log 2>&1
  uv run python -m $M.oracle_check --model $model --oracle-frac 0.3 > logs/oracle30_$model.log 2>&1
  uv run python -m $M.mechanism --model $model                 > logs/mechanism_$model.log 2>&1
  echo "done $model"
done
uv run python -m $M.hgb_lr_check > logs/hgb_lr_check.log 2>&1
echo "ALL DONE"
