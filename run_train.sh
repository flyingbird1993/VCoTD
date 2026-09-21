#!/usr/bin/env bash
set -euo pipefail

# Portable tmux launcher for the paper-aligned GAP model.
# Environment overrides: VCOTD_ENV, VCOTD_SESSION, VCOTD_BATCH_SIZE,
# VCOTD_FEATURE_DIR, VCOTD_OUTPUT_DIR, VCOTD_LOG.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${VCOTD_SESSION:-vcotd_train}"
CONDA_ENV="${VCOTD_ENV:-qwen_vl_vadb}"
FEATURE_DIR="${VCOTD_FEATURE_DIR:-$SCRIPT_DIR/teacher_features}"
OUTPUT_DIR="${VCOTD_OUTPUT_DIR:-$SCRIPT_DIR/saves/vcotd_paper_gap}"
BATCH_SIZE="${VCOTD_BATCH_SIZE:-64}"
LOG_FILE="${VCOTD_LOG:-$OUTPUT_DIR/train.log}"

if [[ "${1:-}" == "--attach" ]]; then
  exec tmux attach -t "$SESSION"
fi

RESUME=()
if [[ "${1:-}" == "--resume" ]]; then
  RESUME=(--resume)
  shift
fi

TRAIN_CMD=(
  conda run -n "$CONDA_ENV" --no-capture-output
  python train_vcotd.py
  --mode distill
  --teacher_feat_dir "$FEATURE_DIR"
  --output_dir "$OUTPUT_DIR"
  --visual_aggregation gap
  --vis_tokens 1
  --epochs 12
  --batch_size "$BATCH_SIZE"
  --lr 1e-4
  --num_workers 8
  --log_interval 50
  --save_every 3
  "${RESUME[@]}"
  "$@"
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists."
  echo "Attach with: tmux attach -t '$SESSION'"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
printf -v COMMAND_STRING '%q ' "${TRAIN_CMD[@]}"
printf -v LOG_STRING '%q' "$LOG_FILE"
tmux new-session -d -s "$SESSION" -c "$SCRIPT_DIR" \
  "$COMMAND_STRING 2>&1 | tee $LOG_STRING"

echo "Started VCoTD training in tmux session '$SESSION'."
echo "Attach: tmux attach -t '$SESSION'"
echo "Log:    $LOG_FILE"
