#!/usr/bin/env bash
# Open-loop evaluation sweep for Correction Training on the held-out LIBERO-10 split.
#
# Counterpart of sweep_correction_libero_10.sh that needs no simulator: it compares the
# predicted action chunks with the demonstrated ones. Each checkpoint is evaluated once with
# gr00t/eval/open_loop_eval_correction.py, which sweeps every inference setting on it:
#   K          (NUM_CORRECTIONS, corrector applications per denoising step; K=0 = base denoiser)
#   interval   (INTERVALS, flow-time window [lo, hi) where correction is applied; 0 = noise)
#   denoising  (DENOISING_STEPS, flow-matching Euler steps)
# Checkpoints come from RUN_DIRS x CKPT_STEPS and/or MODEL_PATHS (trained with
# --use-correction), plus optional BASELINE_MODEL_PATHS (trained without it; K=0 only).
#
# For every setting it saves mse / mae for each evaluated chunk (per_step.csv/.npz, including
# mse/mae per action-horizon step and per action key) and reports avg_mse, avg_mae and
# paired metrics against the same checkpoint's base denoiser K0_d<d> (change in MSE, win rate,
# action shift, magnitude/jerk ratios, gripper sign accuracy), as in
# sweep_autoguidance_open_loop_libero_10.sh. Every checkpoint is sampled with the same noise on
# the same chunks, so the combined table also pairs each setting against the first baseline
# checkpoint (columns "vs_base").
#
# Usage:
#   RUN_DIRS="examples/LIBERO/logs/libero-subset-lerobot-10/<correction run>" \
#   CKPT_STEPS="3232 final" NUM_CORRECTIONS="0 1 2 4" INTERVALS="0.0,1.0 0.0,0.5" \
#   BASELINE_MODEL_PATHS="examples/LIBERO/logs/libero-subset-lerobot-10/<baseline run>" \
#   CUDA_VISIBLE_DEVICES=0 bash examples/LIBERO/sweep_correction_open_loop_libero_10.sh
#
# Split the checkpoints across 2 GPUs (re-run with SUMMARY_ONLY=1 afterwards for the table):
#   NUM_SHARDS=2 SHARD_ID=0 CUDA_VISIBLE_DEVICES=0 bash ... &
#   NUM_SHARDS=2 SHARD_ID=1 CUDA_VISIBLE_DEVICES=1 bash ... &
#
# Environment variables:
#   RUN_DIRS / CKPT_STEPS     correction training dirs and their checkpoint steps ("final" = run-dir root)
#   MODEL_PATHS               explicit correction checkpoint dirs
#   BASELINE_MODEL_PATHS      non-correction checkpoints; the first is the "vs_base" reference
#   NUM_CORRECTIONS           K values (default "0 1 2 4")
#   INTERVALS                 "lo,hi" pairs (default "0.0,1.0")
#   DENOISING_STEPS           denoising steps (default "4")
#   DATASET_PATH              held-out dataset (default: LIBERO-10 long_90%)
#   TRAJ_IDS                  space-separated trajectory indices (default: all)
#   STRIDE                    step between chunks (default: the action horizon)
#   MAX_STEPS_PER_TRAJ        only evaluate chunks starting before this step (default: all)
#   BATCH_SIZE, SEED          sampling batch size / noise seed (defaults 16 / 0)
#   SWEEP_NAME, OUTPUT_ROOT   output location (default: logs/open_loop_results/open-loop_<name>_seed<SEED>)
#   SKIP_EXISTING             1 skips a finished checkpoint (default 1)
#   NUM_SHARDS, SHARD_ID      split the checkpoints across processes
#   DRY_RUN=1, SUMMARY_ONLY=1 list checkpoints only / only rebuild the combined table

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC2206
RUN_DIRS=(${RUN_DIRS:-})
# shellcheck disable=SC2206
CKPT_STEPS=(${CKPT_STEPS:-final})
# shellcheck disable=SC2206
MODEL_PATHS=(${MODEL_PATHS:-})
# shellcheck disable=SC2206
BASELINE_MODEL_PATHS=(${BASELINE_MODEL_PATHS:-})
# shellcheck disable=SC2206
NUM_CORRECTIONS=(${NUM_CORRECTIONS:-0 1 2 4})
# shellcheck disable=SC2206
INTERVALS=(${INTERVALS:-0.0,1.0})
# shellcheck disable=SC2206
DENOISING_STEPS=(${DENOISING_STEPS:-4})
DATASET_PATH="${DATASET_PATH:-/mnt/data/sftp/data/duydt6/workspace/data/LIBERO/libero-subset-lerobot-10/long_90%}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
SWEEP_NAME="${SWEEP_NAME:-correction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/LIBERO/logs/open_loop_results/open-loop_${SWEEP_NAME}_seed${SEED}}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

# ---- Collect checkpoints: "<tag>|<path>|<is_correction>" --------------------------
CKPTS=()
for BP in ${BASELINE_MODEL_PATHS[@]+"${BASELINE_MODEL_PATHS[@]}"}; do
  CKPTS+=("baseline-$(basename "$BP")|$BP|0")
done
for RUN_DIR in ${RUN_DIRS[@]+"${RUN_DIRS[@]}"}; do
  for STEP in "${CKPT_STEPS[@]}"; do
    if [ "$STEP" = "final" ]; then
      CKPTS+=("ct-$(basename "$RUN_DIR")-final|$RUN_DIR|1")
    else
      CKPTS+=("ct-$(basename "$RUN_DIR")-step${STEP}|$RUN_DIR/checkpoint-$STEP|1")
    fi
  done
done
for MP in ${MODEL_PATHS[@]+"${MODEL_PATHS[@]}"}; do
  CKPTS+=("ct-$(basename "$(dirname "$MP")")-$(basename "$MP")|$MP|1")
done
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "ERROR: set RUN_DIRS / MODEL_PATHS (correction checkpoints) and/or BASELINE_MODEL_PATHS" >&2
  exit 1
fi
for ENTRY in "${CKPTS[@]}"; do
  IFS='|' read -r _ CKPT_PATH IS_CT <<<"$ENTRY"
  if [ ! -f "$CKPT_PATH/config.json" ]; then
    echo "ERROR: no checkpoint at $CKPT_PATH (missing config.json)" >&2
    exit 1
  fi
  if [ "$IS_CT" = "1" ] && ! python3 -c \
      'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("use_correction") else 1)' \
      "$CKPT_PATH/config.json"; then
    echo "ERROR: $CKPT_PATH was not trained with --use-correction (use BASELINE_MODEL_PATHS for it)" >&2
    exit 1
  fi
done

echo "Open-loop correction sweep on $DATASET_PATH -> $OUTPUT_ROOT"
echo "${#CKPTS[@]} checkpoints (shard $SHARD_ID/$NUM_SHARDS); K=(${NUM_CORRECTIONS[*]}) intervals=(${INTERVALS[*]}) d=(${DENOISING_STEPS[*]})"
for i in "${!CKPTS[@]}"; do
  IFS='|' read -r TAG _ _ <<<"${CKPTS[$i]}"
  mark=" "
  [ $((i % NUM_SHARDS)) -eq "$SHARD_ID" ] && mark="*"
  done_mark=""
  [ -f "$OUTPUT_ROOT/$TAG/sweep_summary.json" ] && done_mark=" (done)"
  echo " $mark [$i] $TAG$done_mark"
done
[ "$DRY_RUN" = "1" ] && exit 0

# ---- Run ----------------------------------------------------------------------
if [ "$SUMMARY_ONLY" != "1" ]; then
  for i in "${!CKPTS[@]}"; do
    [ $((i % NUM_SHARDS)) -eq "$SHARD_ID" ] || continue
    IFS='|' read -r TAG CKPT_PATH _ <<<"${CKPTS[$i]}"
    OUT="$OUTPUT_ROOT/$TAG"
    echo
    echo "##### [$(date +%H:%M:%S)] [$i/${#CKPTS[@]}] $TAG #####"
    ARGS=(
      --model-path "$CKPT_PATH"
      --output-dir "$OUT"
      --dataset-path "$DATASET_PATH"
      --embodiment-tag "$EMBODIMENT_TAG"
      --seed "$SEED"
      --batch-size "$BATCH_SIZE"
      --num-corrections "${NUM_CORRECTIONS[@]}"
      --intervals "${INTERVALS[@]}"
      --denoising-steps "${DENOISING_STEPS[@]}"
    )
    [ "$SKIP_EXISTING" = "1" ] || ARGS+=(--no-skip-existing)
    [ -z "${STRIDE:-}" ] || ARGS+=(--stride "$STRIDE")
    [ -z "${MAX_STEPS_PER_TRAJ:-}" ] || ARGS+=(--max-steps-per-traj "$MAX_STEPS_PER_TRAJ")
    if [ -n "${TRAJ_IDS:-}" ]; then
      # shellcheck disable=SC2206
      TRAJ_LIST=($TRAJ_IDS)
      ARGS+=(--traj-ids "${TRAJ_LIST[@]}")
    fi
    mkdir -p "$OUT"
    HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" uv run python gr00t/eval/open_loop_eval_correction.py \
      "${ARGS[@]}" 2>&1 | tee "$OUT/eval.log"
    status="${PIPESTATUS[0]}"
    [ "$status" -eq 0 ] || exit "$status"
  done
fi

# ---- Combined table ---------------------------------------------------------------
TAGS=()
for C in "${CKPTS[@]}"; do TAGS+=("${C%%|*}"); done
BASE_TAG=""
[ "${#BASELINE_MODEL_PATHS[@]}" -eq 0 ] || BASE_TAG="${TAGS[0]}"
python3 - "$OUTPUT_ROOT" "$BASE_TAG" "${TAGS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

root, base_tag, *tags = sys.argv[1:]
root = Path(root)


def load_chunks(tag, config):
    path = root / tag / config / "per_step.npz"
    if not path.exists():
        return None
    z = np.load(path)
    order = np.lexsort((z["step"], z["traj_id"]))
    return {k: z[k][order] for k in ("traj_id", "step", "mse", "arm_mse", "grip_mse")}


def win_rate(err, ref):
    ok = np.isfinite(err) & np.isfinite(ref)
    return float(((err[ok] < ref[ok]) + 0.5 * (err[ok] == ref[ok])).mean()) if ok.any() else None


rows, missing = [], []
for tag in tags:
    path = root / tag / "sweep_summary.json"
    if not path.exists():
        missing.append(tag)
        continue
    train = json.loads((root / tag / "eval_config.json").read_text()).get("train") or {}
    for r in json.loads(path.read_text()):
        rows.append({"checkpoint": tag, **r, "train": train})
if not rows:
    print("No finished checkpoints yet.")
    sys.exit(0)

# Paired comparison against the baseline checkpoint's K0 at the same number of denoising steps.
for r in rows:
    r["delta_mse_pct_vs_base"] = r["win_rate_vs_base"] = None
    if not base_tag or r["checkpoint"] == base_tag:
        continue
    ref = load_chunks(base_tag, f"K0_d{r['denoising_steps']}")
    cur = load_chunks(r["checkpoint"], r["config"])
    if ref is None or cur is None:
        continue
    if not (np.array_equal(ref["traj_id"], cur["traj_id"]) and np.array_equal(ref["step"], cur["step"])):
        print(f"WARNING: {r['checkpoint']}/{r['config']} evaluated different chunks than the baseline")
        continue
    r["delta_mse_pct_vs_base"] = 100.0 * float((cur["mse"] - ref["mse"]).mean()) / float(ref["mse"].mean())
    r["win_rate_vs_base"] = win_rate(cur["mse"], ref["mse"])

cols = ["checkpoint", "config", "num_corrections", "correction_interval", "denoising_steps", "gate",
        "n_chunks", "n_trajectories", "avg_mse", "avg_mae", "avg_mse_per_traj", "avg_arm_mse",
        "avg_gripper_mse", "gripper_sign_acc", "arm_magnitude_ratio", "arm_jerk_ratio",
        "avg_shift_vs_k0", "delta_mse_pct_vs_k0", "delta_mse_se_by_traj", "win_rate_vs_k0",
        "arm_win_rate_vs_k0", "gripper_win_rate_vs_k0", "trajectory_win_rate_vs_k0",
        "delta_mse_pct_vs_base", "win_rate_vs_base"]
train_cols = ["correction_loss_weight", "correction_train_steps", "correction_freeze_denoiser"]
(root / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
with open(root / "sweep_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols + train_cols)
    for r in rows:
        w.writerow([r.get(c) for c in cols] + [r["train"].get(c) for c in train_cols])


def fmt(x, spec):
    width = int("".join(c for c in spec.split(".")[0] if c.isdigit()) or 0)
    return "-".rjust(width) if x is None else format(x, spec)


n_chunks = rows[0]["n_chunks"]
print(f"\n=== Open-loop correction sweep: {n_chunks} chunks, {rows[0]['n_trajectories']} trajectories ===")
print("d_mse%/win/a_win/g_win/shift: vs the same checkpoint's K0; b_mse%/b_win: vs " + (base_tag or "- (no baseline)"))
width = max(len(r["checkpoint"]) for r in rows)
print(
    f"{'checkpoint':<{width}s} {'config':<20s} {'gate':>8s} {'avg_mse':>9s} {'avg_mae':>9s} "
    f"{'d_mse%':>7s} {'win':>5s} {'a_win':>5s} {'g_win':>5s} {'shift':>7s} {'arm_mse':>9s} "
    f"{'grip_mse':>9s} {'g_acc':>5s} {'mag':>5s} {'jerk':>5s} {'b_mse%':>7s} {'b_win':>5s}"
)
for r in rows:
    print(
        f"{r['checkpoint']:<{width}s} {r['config']:<20s} {fmt(r['gate'], '8.5f')} "
        f"{r['avg_mse']:>9.5f} {r['avg_mae']:>9.5f} {fmt(r.get('delta_mse_pct_vs_k0'), '+7.2f')} "
        f"{fmt(r.get('win_rate_vs_k0'), '5.2f')} {fmt(r.get('arm_win_rate_vs_k0'), '5.2f')} "
        f"{fmt(r.get('gripper_win_rate_vs_k0'), '5.2f')} {r['avg_shift_vs_k0']:>7.4f} "
        f"{r['avg_arm_mse']:>9.5f} {r['avg_gripper_mse']:>9.5f} {r['gripper_sign_acc']:>5.3f} "
        f"{r['arm_magnitude_ratio']:>5.2f} {r['arm_jerk_ratio']:>5.2f} "
        f"{fmt(r['delta_mse_pct_vs_base'], '+7.2f')} {fmt(r['win_rate_vs_base'], '5.2f')}"
    )
best = min(rows, key=lambda r: r["avg_mse"])
print(f"\nBest (lowest avg_mse): {best['checkpoint']} / {best['config']} (avg_mse={best['avg_mse']:.5f})")
if missing:
    print(f"{len(missing)} checkpoint(s) not finished yet (other shards or interrupted).")
print(f"Combined summary written to {root / 'sweep_summary.json'} and {root / 'sweep_summary.csv'}")
PY
