#!/usr/bin/env bash
# Visualize how autoguidance, ACG and AFIL change the sampled action distribution on LIBERO-10,
# like the 2D toy example of the autoguidance paper (see gr00t/eval/visualize_guidance.py).
#
# Writes samples.png, trajectories.png, arrows.png, summary.png, metrics.csv/json and
# samples.npz to OUTPUT_ROOT. Methods are enabled by their settings; give at least one.
#
# Usage:
#   GUIDE_CKPT=<guide run>/checkpoint-5000 GUIDE_SCALES="1.5 3.0" \
#   ACG_SKIP_BLOCKS="15 19 23" ACG_SCALES="2.0 3.0" \
#   FAILURE_CKPT=<failure run>/checkpoint-5000 AFIL_ALPHAS="1.0 2.0" \
#     bash examples/LIBERO/visualize_guidance_libero_10.sh
#
# Environment variables:
#   MODEL_PATH            main model (default: the LIBERO-10 baseline); a checkpoint dir
#   GUIDE_CKPT, GUIDE_SCALES        autoguidance guide checkpoint and weights (default "1.5 3.0")
#   ACG_SKIP_BLOCKS, ACG_SCALES     ACG blocks (e.g. "15 19 23") and weights (default "2.0 3.0")
#   FAILURE_CKPT, AFIL_ALPHAS       AFIL failure model checkpoint and alphas (default "1.0 2.0")
#   FAILURE_DATASET_PATH  failure dataset (build_failure_dataset.py); drawn as red outlines
#   DATASET_PATH          dataset (default: the one MODEL_PATH was trained on)
#   OBSERVATIONS          space-separated "traj:step" frames (default: NUM_OBSERVATIONS random ones)
#   NUM_OBSERVATIONS, NUM_SAMPLES, NUM_TRAJECTORIES, SEED
#   OUTPUT_ROOT           default examples/LIBERO/logs/guidance_viz/<run name>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:-/mnt/data/sftp/data/duydt6/workspace/diffusion/Isaac-GR00T-baseline/examples/LIBERO/logs/libero-baseline/2026-09-19_gr00t-n17_baseline_libero-panda_libero-10}"
SEED="${SEED:-0}"
RUN_NAME="${RUN_NAME:-guidance-viz_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/LIBERO/logs/guidance_viz/$RUN_NAME}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

ARGS=(
  --model-path "$MODEL_PATH"
  --output-dir "$OUTPUT_ROOT"
  --num-observations "${NUM_OBSERVATIONS:-4}"
  --num-samples "${NUM_SAMPLES:-256}"
  --num-trajectories "${NUM_TRAJECTORIES:-24}"
  --seed "$SEED"
)
if [ -n "${GUIDE_CKPT:-}" ]; then
  # shellcheck disable=SC2206
  L=(${GUIDE_SCALES:-1.5 3.0})
  ARGS+=(--guide-ckpt "$GUIDE_CKPT" --guide-scales "${L[@]}")
fi
if [ -n "${ACG_SKIP_BLOCKS:-}" ]; then
  # shellcheck disable=SC2206
  B=($ACG_SKIP_BLOCKS)
  # shellcheck disable=SC2206
  L=(${ACG_SCALES:-2.0 3.0})
  ARGS+=(--acg-skip-blocks "${B[@]}" --acg-scales "${L[@]}")
fi
if [ -n "${FAILURE_CKPT:-}" ]; then
  # shellcheck disable=SC2206
  L=(${AFIL_ALPHAS:-1.0 2.0})
  ARGS+=(--failure-ckpt "$FAILURE_CKPT" --afil-alphas "${L[@]}")
fi
[ -z "${FAILURE_DATASET_PATH:-}" ] || ARGS+=(--failure-dataset-path "$FAILURE_DATASET_PATH")
[ -z "${DATASET_PATH:-}" ] || ARGS+=(--dataset-path "$DATASET_PATH")
if [ -n "${OBSERVATIONS:-}" ]; then
  # shellcheck disable=SC2206
  O=($OBSERVATIONS)
  ARGS+=(--observations "${O[@]}")
fi

mkdir -p "$OUTPUT_ROOT"
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" uv run python gr00t/eval/visualize_guidance.py "${ARGS[@]}" \
  2>&1 | tee "$OUTPUT_ROOT/viz.log"
exit "${PIPESTATUS[0]}"
