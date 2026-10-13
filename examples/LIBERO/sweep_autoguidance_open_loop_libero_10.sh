#!/usr/bin/env bash
# Open-loop evaluation sweep for autoguidance on LIBERO-10 (Long).
#
# Counterpart of sweep_autoguidance_libero_10.sh that needs no simulator: it compares the
# predicted action chunks with the demonstrated ones, for the unguided main model, each
# guide checkpoint alone (w=0), and every (guide checkpoint, guidance scale) pair.
# Leave GUIDE_RUN_DIR unset to evaluate only the unguided model (a baseline run).
# By default every trajectory of the dataset is evaluated; set TRAJ_IDS to pick some.
#
# For every configuration it saves mse / mae for each evaluated chunk (per_step.csv/.npz,
# including mse/mae per action-horizon step and per action key) and reports avg_mse, avg_mae
# and guidance-quality metrics (change vs. the unguided model, win rate, action shift,
# magnitude/jerk ratios, gripper sign accuracy). See gr00t/eval/open_loop_autoguidance_eval.py.
#
# Usage:
#   MODEL_PATH=<main run dir> GUIDE_RUN_DIR=<guide run dir> GUIDE_LABEL=autoguidance-dit16 \
#   GUIDE_STEPS="202 404 808" SCALES="1.1 1.5 2.0 2.5 3.0" CUDA_VISIBLE_DEVICES=0 \
#     bash examples/LIBERO/sweep_autoguidance_open_loop_libero_10.sh
#
# Environment variables:
#   GUIDE_RUN_DIR             guide training output dir (checkpoint-<step> inside); unset = baseline only
#   MODEL_PATH                main model (default: the LIBERO-10 baseline)
#   GUIDE_LABEL               name used in the output dir (default: autoguidance)
#   GUIDE_STEPS / SCALES      guide checkpoints and guidance weights to sweep
#   DATASET_PATH              dataset (default: the one the main model was trained on)
#   TRAJ_IDS                  space-separated trajectory indices (default: all)
#   STRIDE                    step between chunks (default: the action horizon)
#   MAX_STEPS_PER_TRAJ        only evaluate chunks starting before this step (default: all)
#   BATCH_SIZE, SEED          sampling batch size / noise seed (defaults 16 / 0)
#   ACG_SKIP_BLOCKS           also evaluate Action Coherence Guidance (config acg_w<w> for each of
#                             SCALES, no guide model needed), e.g. "15 19 23"; unset = skip
#   INCLUDE_GUIDE_ONLY        1 also evaluates each guide alone (default 1)
#   RUN_NAME, OUTPUT_ROOT     output location (default: eval_results/open-loop_<label>_seed<SEED>)
#   SKIP_EXISTING             1 skips a finished run (default 1)
# Closed-loop settings such as N_ENVS / N_EPISODES are accepted and ignored.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/mnt/data/sftp/data/duydt6/workspace/diffusion/Isaac-GR00T-baseline/examples/LIBERO/logs/libero-baseline/2026-09-19_gr00t-n17_baseline_libero-panda_libero-10}"
# Without GUIDE_RUN_DIR only the unguided model is evaluated (baseline run).
GUIDE_RUN_DIR="${GUIDE_RUN_DIR:-}"
if [ -n "$GUIDE_RUN_DIR" ]; then
  GUIDE_LABEL="${GUIDE_LABEL:-autoguidance}"
  GUIDE_STEPS="${GUIDE_STEPS:-2000 5000 10000}"
else
  GUIDE_LABEL="${GUIDE_LABEL:-$([ -n "${ACG_SKIP_BLOCKS:-}" ] && echo acg || echo baseline)}"
  GUIDE_STEPS=""
fi
SCALES="${SCALES:-1.1 1.5 2.0 2.5 3.0}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
INCLUDE_GUIDE_ONLY="${INCLUDE_GUIDE_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_NAME="${RUN_NAME:-open-loop_${GUIDE_LABEL}_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/LIBERO/logs/eval_results/$RUN_NAME}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

for dir in "$MODEL_PATH" ${GUIDE_RUN_DIR:+"$GUIDE_RUN_DIR"}; do
  if [ ! -d "$dir" ]; then
    echo "ERROR: $dir not found" >&2
    exit 1
  fi
done

# shellcheck disable=SC2206
STEPS=($GUIDE_STEPS)
# shellcheck disable=SC2206
SCALE_LIST=($SCALES)
for STEP in "${STEPS[@]}"; do
  if [ ! -d "$GUIDE_RUN_DIR/checkpoint-$STEP" ]; then
    echo "ERROR: $GUIDE_RUN_DIR/checkpoint-$STEP not found" >&2
    exit 1
  fi
done

ARGS=(
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_ROOT"
  --embodiment-tag "$EMBODIMENT_TAG"
  --seed "$SEED"
  --batch-size "$BATCH_SIZE"
  --scales "${SCALE_LIST[@]}"
)
[ -z "$GUIDE_RUN_DIR" ] || ARGS+=(--guide-run-dir "$GUIDE_RUN_DIR" --guide-steps "${STEPS[@]}")
if [ -n "${ACG_SKIP_BLOCKS:-}" ]; then
  # shellcheck disable=SC2206
  ACG_LIST=($ACG_SKIP_BLOCKS)
  ARGS+=(--acg-skip-blocks "${ACG_LIST[@]}")
fi
[ "$INCLUDE_GUIDE_ONLY" = "1" ] || ARGS+=(--no-include-guide-only)
[ "$SKIP_EXISTING" = "1" ] || ARGS+=(--no-skip-existing)
[ -z "${DATASET_PATH:-}" ] || ARGS+=(--dataset-path "$DATASET_PATH")
[ -z "${STRIDE:-}" ] || ARGS+=(--stride "$STRIDE")
[ -z "${MAX_STEPS_PER_TRAJ:-}" ] || ARGS+=(--max-steps-per-traj "$MAX_STEPS_PER_TRAJ")
if [ -n "${TRAJ_IDS:-}" ]; then
  # shellcheck disable=SC2206
  TRAJ_LIST=($TRAJ_IDS)
  ARGS+=(--traj-ids "${TRAJ_LIST[@]}")
fi

mkdir -p "$OUTPUT_ROOT"
echo "Open-loop autoguidance sweep -> $OUTPUT_ROOT"
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" uv run python gr00t/eval/open_loop_autoguidance_eval.py "${ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/eval.log"
exit "${PIPESTATUS[0]}"
