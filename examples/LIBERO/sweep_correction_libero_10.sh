#!/usr/bin/env bash
# Hyperparameter sweep for Correction Training on LIBERO-10 (Long).
#
# Evaluates every combination of
#   checkpoint   (RUN_DIRS x CKPT_STEPS, and/or explicit MODEL_PATHS)
#   K            (NUM_CORRECTIONS, corrector applications per denoising step; K=0 = base denoiser)
#   interval     (INTERVALS, flow-time window [lo, hi) where correction is applied; 0 = noise)
#   denoising    (DENOISING_STEPS, flow-matching Euler steps)
# with examples/LIBERO/eval_libero_10.sh, plus an optional non-correction baseline checkpoint,
# then prints/saves a success-rate table. Training-time hyperparameters (loss weight, K_train,
# frozen denoiser, ...) are swept by passing several training runs in RUN_DIRS; they are read
# from each checkpoint's config.json and shown in the table.
#
# Each configuration writes to its own examples/LIBERO/logs/eval_results/<run-name>, so an
# interrupted sweep resumes where it stopped. K=0 ignores the interval, so it is run once per
# (checkpoint, denoising steps).
#
# Usage (screening, 10 episodes/task):
#   RUN_DIRS="examples/LIBERO/logs/libero-subset-lerobot-10/<correction run>" \
#   CKPT_STEPS="3232 6464" NUM_CORRECTIONS="0 1 2 4" INTERVALS="0.0,1.0 0.0,0.5" \
#     bash examples/LIBERO/sweep_correction_libero_10.sh
#
# Split across 2 GPUs (run both; the last one to finish prints the full table, or re-run with
# SUMMARY_ONLY=1):
#   NUM_SHARDS=2 SHARD_ID=0 CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PORT=5556 bash ... &
#   NUM_SHARDS=2 SHARD_ID=1 CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 PORT=5557 bash ... &
#
# Other knobs: BASELINE_MODEL_PATH (non-correction checkpoint as control), N_EPISODES (10),
# SEED (42), DRY_RUN=1 (list configurations only), SUMMARY_ONLY=1, SWEEP_NAME, and any
# eval_libero_10.sh variable (N_ENVS, PORT, MAX_EPISODE_STEPS, ...).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC2206
RUN_DIRS=(${RUN_DIRS:-})
# "final" = the model saved at the root of the run dir at the end of training.
# shellcheck disable=SC2206
CKPT_STEPS=(${CKPT_STEPS:-final})
# shellcheck disable=SC2206
MODEL_PATHS=(${MODEL_PATHS:-})
# shellcheck disable=SC2206
NUM_CORRECTIONS=(${NUM_CORRECTIONS:-0 1 2 4})
# shellcheck disable=SC2206
INTERVALS=(${INTERVALS:-0.0,1.0})
# shellcheck disable=SC2206
DENOISING_STEPS=(${DENOISING_STEPS:-4})
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-0}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
export N_EPISODES="${N_EPISODES:-10}"
export SEED="${SEED:-42}"
SWEEP_NAME="${SWEEP_NAME:-correction}"
EVAL_ROOT="${EVAL_ROOT:-examples/LIBERO/logs/eval_results}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-$EVAL_ROOT/sweep_${SWEEP_NAME}_n${N_EPISODES}-seed${SEED}}"

# ---- Collect checkpoints -----------------------------------------------------
CKPTS=()   # entries: "<tag>|<path>"
for RUN_DIR in ${RUN_DIRS[@]+"${RUN_DIRS[@]}"}; do
  for STEP in "${CKPT_STEPS[@]}"; do
    if [ "$STEP" = "final" ]; then
      CKPTS+=("$(basename "$RUN_DIR")-final|$RUN_DIR")
    else
      CKPTS+=("$(basename "$RUN_DIR")-step${STEP}|$RUN_DIR/checkpoint-$STEP")
    fi
  done
done
for MP in ${MODEL_PATHS[@]+"${MODEL_PATHS[@]}"}; do
  CKPTS+=("$(basename "$(dirname "$MP")")-$(basename "$MP")|$MP")
done
if [ "${#CKPTS[@]}" -eq 0 ] && [ -z "$BASELINE_MODEL_PATH" ]; then
  echo "ERROR: set RUN_DIRS (training output dirs) and/or MODEL_PATHS (checkpoint dirs)" >&2
  exit 1
fi
for ENTRY in ${CKPTS[@]+"${CKPTS[@]}"}; do
  CKPT_PATH="${ENTRY#*|}"
  if [ ! -f "$CKPT_PATH/config.json" ]; then
    echo "ERROR: no checkpoint at $CKPT_PATH (missing config.json)" >&2
    exit 1
  fi
  if ! python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("use_correction") else 1)' \
      "$CKPT_PATH/config.json"; then
    echo "ERROR: $CKPT_PATH was not trained with --use-correction (use BASELINE_MODEL_PATH for it)" >&2
    exit 1
  fi
done

# ---- Build the configuration list ---------------------------------------------
# entries: "<run_name>|<model_path>|<K>|<interval 'lo hi'>|<denoising steps>"
CONFIGS=()
if [ -n "$BASELINE_MODEL_PATH" ]; then
  for D in "${DENOISING_STEPS[@]}"; do
    CONFIGS+=("baseline-$(basename "$BASELINE_MODEL_PATH")_d${D}_n${N_EPISODES}-seed${SEED}|$BASELINE_MODEL_PATH|||$D")
  done
fi
for ENTRY in ${CKPTS[@]+"${CKPTS[@]}"}; do
  TAG="${ENTRY%%|*}"
  CKPT_PATH="${ENTRY#*|}"
  for D in "${DENOISING_STEPS[@]}"; do
    for K in "${NUM_CORRECTIONS[@]}"; do
      if [ "$K" = "0" ]; then
        CONFIGS+=("ct-${TAG}_K0_d${D}_n${N_EPISODES}-seed${SEED}|$CKPT_PATH|0||$D")
        continue
      fi
      for IV in "${INTERVALS[@]}"; do
        LO="${IV%,*}"
        HI="${IV#*,}"
        CONFIGS+=("ct-${TAG}_K${K}_int${LO}-${HI}_d${D}_n${N_EPISODES}-seed${SEED}|$CKPT_PATH|$K|$LO $HI|$D")
      done
    done
  done
done

echo "Sweep '$SWEEP_NAME': ${#CONFIGS[@]} configurations (shard $SHARD_ID/$NUM_SHARDS)"
for i in "${!CONFIGS[@]}"; do
  IFS='|' read -r NAME _ K IV D <<<"${CONFIGS[$i]}"
  mark=" "
  [ $((i % NUM_SHARDS)) -eq "$SHARD_ID" ] && mark="*"
  echo " $mark [$i] $NAME"
done

# ---- Run ----------------------------------------------------------------------
if [ "$DRY_RUN" != "1" ] && [ "$SUMMARY_ONLY" != "1" ]; then
  for i in "${!CONFIGS[@]}"; do
    [ $((i % NUM_SHARDS)) -eq "$SHARD_ID" ] || continue
    IFS='|' read -r NAME MP K IV D <<<"${CONFIGS[$i]}"
    if [ -f "$EVAL_ROOT/$NAME/summary.json" ]; then
      echo "##### [$i] $NAME: done, skipping #####"
      continue
    fi
    echo
    echo "##### [$(date +%H:%M:%S)] [$i/${#CONFIGS[@]}] $NAME #####"
    RUN_NAME="$NAME" OUTPUT_ROOT="$EVAL_ROOT/$NAME" MODEL_PATH="$MP" NUM_CORRECTIONS="$K" CORRECTION_INTERVAL="$IV" \
      DENOISING_STEPS="$D" bash examples/LIBERO/eval_libero_10.sh
  done
fi
[ "$DRY_RUN" = "1" ] && exit 0

# ---- Summary table ---------------------------------------------------------------
NAMES=()
for C in "${CONFIGS[@]}"; do NAMES+=("${C%%|*}"); done
python3 - "$EVAL_ROOT" "$SWEEP_SUMMARY" "${NAMES[@]}" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root, out_prefix, *names = sys.argv[1:]
root = Path(root)
rows, missing = [], []
for name in names:
    run = root / name
    if not (run / "summary.json").exists():
        missing.append(name)
        continue
    summary = json.loads((run / "summary.json").read_text())
    cfg = json.loads((run / "eval_config.json").read_text())
    gate = None
    server_log = run / "logs" / "server.log"
    if server_log.exists():
        m = re.search(r"gate=([-0-9.eE]+)", server_log.read_text(errors="ignore"))
        gate = float(m.group(1)) if m else None
    train = cfg.get("train") or {}
    rows.append(
        {
            "run_name": name,
            "model_path": cfg["model_path"],
            "use_correction": cfg.get("use_correction"),
            "num_corrections": cfg.get("num_corrections"),
            "correction_interval": cfg.get("correction_interval"),
            "denoising_steps": cfg.get("denoising_steps"),
            "correction_loss_weight": train.get("correction_loss_weight"),
            "correction_train_steps": train.get("correction_train_steps"),
            "correction_freeze_denoiser": train.get("correction_freeze_denoiser"),
            "gate": gate,
            "n_success": summary["overall_n_success"],
            "n_episodes": summary["overall_n_episodes"],
            "success_rate": summary["overall_success_rate"],
            "per_task": {r["task"]: r["n_success"] for r in summary["per_task"]},
        }
    )

if not rows:
    print("No finished configurations yet.")
    sys.exit(0)

Path(out_prefix + ".json").write_text(json.dumps(rows, indent=2))
tasks = list(rows[0]["per_task"])
with open(out_prefix + ".csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["run_name", "K", "interval", "denoising_steps", "loss_weight", "k_train", "frozen",
         "gate", "n_success", "n_episodes", "success_rate"] + [f"T{i}" for i in range(len(tasks))]
    )
    for r in rows:
        writer.writerow(
            [r["run_name"], r["num_corrections"], r["correction_interval"], r["denoising_steps"],
             r["correction_loss_weight"], r["correction_train_steps"],
             r["correction_freeze_denoiser"], r["gate"], r["n_success"], r["n_episodes"],
             f"{r['success_rate']:.4f}"] + [r["per_task"].get(t, "") for t in tasks]
        )

def fmt_interval(iv):
    return "-" if not iv else f"[{iv[0]:g},{iv[1]:g})"

print()
print("=== Correction-training sweep ===")
print("per-task columns follow the LIBERO-10 task order:")
for i, task in enumerate(tasks):
    print(f"  T{i}: {task}")
width = max(len(r["run_name"]) for r in rows)
print(
    f"{'run':<{width}s} {'K':>2s} {'interval':>10s} {'d':>2s} {'gate':>7s} {'success':>9s} "
    f"{'rate':>6s}  " + " ".join(f"T{i:<2d}" for i in range(len(tasks)))
)
for r in rows:
    gate = "-" if r["gate"] is None else f"{r['gate']:.4f}"
    per_task = " ".join(f"{r['per_task'].get(t, 0):<3d}" for t in tasks)
    print(
        f"{r['run_name']:<{width}s} {r['num_corrections'] or 0:>2d} "
        f"{fmt_interval(r['correction_interval']):>10s} {r['denoising_steps'] or 0:>2d} "
        f"{gate:>7s} {r['n_success']:>4d}/{r['n_episodes']:<4d} {r['success_rate']:>6.3f}  {per_task}"
    )
best = max(rows, key=lambda r: r["success_rate"])
print(f"\nBest: {best['run_name']} ({best['n_success']}/{best['n_episodes']} = {best['success_rate']:.3f})")
if missing:
    print(f"{len(missing)} configuration(s) not finished yet (other shards or interrupted).")
print(f"Sweep summary written to {out_prefix}.json and {out_prefix}.csv")
PY
