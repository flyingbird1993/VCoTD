#!/usr/bin/env bash
set -euo pipefail

# Runs the five ablations from the VCoTD paper in sequence. Pass extra
# train_vcotd.py arguments after the optional run name, for example:
#   bash run_vcotd_ablations.sh full_adaptive --epochs 1 --max_train_samples 64

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VCOTD_PYTHON:-python}"
FEATURE_DIR="${VCOTD_FEATURE_DIR:-$SCRIPT_DIR/teacher_features}"
OUTPUT_ROOT="${VCOTD_ABLATION_DIR:-$SCRIPT_DIR/saves/vcotd_ablations}"
RUN_NAME="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
cd "$SCRIPT_DIR"

run_experiment() {
  local name="$1"
  shift
  echo "[VCoTD] Starting $name"
  "$PYTHON_BIN" "$SCRIPT_DIR/train_vcotd.py" \
    --teacher_feat_dir "$FEATURE_DIR" \
    --output_dir "$OUTPUT_ROOT/$name" \
    --visual_aggregation gap \
    --vis_tokens 1 \
    "$@"
}

run_selected() {
  local name="$1"
  shift
  if [[ "$RUN_NAME" == "all" || "$RUN_NAME" == "$name" ]]; then
    run_experiment "$name" "$@" "${EXTRA_ARGS[@]}"
  fi
}

EXTRA_ARGS=("$@")
run_selected no_distillation --mode gt_only
run_selected global --mode distill --lambda1 1 --lambda2 0 --lambda3 0 --disable_adaptive --fixed_alpha 1
run_selected global_spatial --mode distill --lambda1 1 --lambda2 1 --lambda3 0 --disable_adaptive --fixed_alpha 1
run_selected hierarchical_fixed --mode distill --lambda1 1 --lambda2 1 --lambda3 0.5 --disable_adaptive --fixed_alpha 1
run_selected full_adaptive --mode distill --lambda1 1 --lambda2 1 --lambda3 0.5

if [[ "$RUN_NAME" != "all" && "$RUN_NAME" != "no_distillation" && \
      "$RUN_NAME" != "global" && "$RUN_NAME" != "global_spatial" && \
      "$RUN_NAME" != "hierarchical_fixed" && "$RUN_NAME" != "full_adaptive" ]]; then
  echo "Unknown run '$RUN_NAME'." >&2
  exit 2
fi
