#!/usr/bin/env bash
# Screening sweep for autoguidance on LIBERO-10 (Long).
#
# Runs examples/LIBERO/eval_libero_10.sh once per (guide checkpoint, guidance
# scale) pair, plus an unguided control (w=1) on the same main model, seed and
# episode count, then prints a success-rate table. Each configuration writes to
# its own examples/LIBERO/logs/eval_results/<run-name>, so an interrupted sweep
# resumes where it stopped (eval_libero_10.sh's SKIP_EXISTING=1 default).
#
# Usage:
#   GUIDE_RUN_DIR=examples/LIBERO/logs/libero-rtc/<guide-run> \
#     bash examples/LIBERO/sweep_autoguidance_libero_10.sh
#
# Set ACG_SKIP_BLOCKS (e.g. "15 19 23") to also sweep Action Coherence Guidance over the same
# SCALES (no guide model needed; GUIDE_RUN_DIR may then be left unset to run only ACG).
#
# Adaptive Failure-Informed Learning (AFIL, see AFIL.md): set AFIL_ALPHAS (alpha of
# lambda = alpha * (1 - cos(v_succ, v_fail))). GUIDE_RUN_DIR is then the training dir of the
# *failure* model and every (GUIDE_STEPS checkpoint, alpha) pair is evaluated, like the
# (GUIDE_STEPS, SCALES) pairs of autoguidance (which is skipped for that dir):
#   GUIDE_RUN_DIR=<failure run> GUIDE_STEPS="1000 3000 5000" AFIL_ALPHAS="0.5 1.0 2.0" \
#     bash examples/LIBERO/sweep_autoguidance_libero_10.sh
# Alternatively give a single failure checkpoint in FAILURE_MODEL_PATH (AFIL_ALPHAS defaults to
# "0.5 1.0 2.0"); GUIDE_RUN_DIR / GUIDE_STEPS / SCALES then keep their autoguidance meaning.
# ACG_SKIP_BLOCKS can be combined with either.
#
# Any eval_libero_10.sh setting (MODEL_PATH, N_ENVS, PORT, ...) can be
# overridden through the environment as well. See AUTOGUIDANCE.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

ACG_SKIP_BLOCKS="${ACG_SKIP_BLOCKS:-}"
FAILURE_MODEL_PATH="${FAILURE_MODEL_PATH:-}"
GUIDE_RUN_DIR="${GUIDE_RUN_DIR:-}"
if [ -n "$FAILURE_MODEL_PATH" ]; then
  # shellcheck disable=SC2206
  AFIL_ALPHAS=(${AFIL_ALPHAS:-0.5 1.0 2.0})
else
  # shellcheck disable=SC2206
  AFIL_ALPHAS=(${AFIL_ALPHAS:-})
fi
# Without FAILURE_MODEL_PATH, AFIL_ALPHAS makes GUIDE_RUN_DIR a failure run.
AFIL_RUN_DIR=""
if [ -z "$FAILURE_MODEL_PATH" ] && [ "${#AFIL_ALPHAS[@]}" -gt 0 ]; then
  AFIL_RUN_DIR="${GUIDE_RUN_DIR:?AFIL_ALPHAS needs GUIDE_RUN_DIR (failure run dir) or FAILURE_MODEL_PATH}"
fi
if [ -z "$ACG_SKIP_BLOCKS" ] && [ -z "$FAILURE_MODEL_PATH" ] && [ -z "$GUIDE_RUN_DIR" ]; then
  echo "ERROR: set GUIDE_RUN_DIR to the training output dir of the guide model" >&2
  exit 1
fi
GUIDE_LABEL="${GUIDE_LABEL:-$(
  if [ -n "$AFIL_RUN_DIR" ]; then echo afil
  elif [ -n "$GUIDE_RUN_DIR" ]; then echo autoguidance-dit8
  elif [ -n "$ACG_SKIP_BLOCKS" ]; then echo acg
  else echo afil; fi
)}"
if [ -n "$GUIDE_RUN_DIR" ]; then
  # shellcheck disable=SC2206
  GUIDE_STEPS=(${GUIDE_STEPS:-2000 5000 10000})
else
  GUIDE_STEPS=()
fi
# shellcheck disable=SC2206
SCALES=(${SCALES:-1.5 2.0 3.0})
INCLUDE_CONTROL="${INCLUDE_CONTROL:-1}"
export N_EPISODES="${N_EPISODES:-10}"
export SEED="${SEED:-42}"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-examples/LIBERO/logs/eval_results/sweep_${GUIDE_LABEL}_n${N_EPISODES}-seed${SEED}.json}"

RUN_NAMES=()
run_eval() {
  local run_name="$1"
  shift
  RUN_NAMES+=("$run_name")
  echo
  echo "##### [$(date +%H:%M:%S)] $run_name #####"
  env "$@" RUN_NAME="$run_name" bash examples/LIBERO/eval_libero_10.sh
}

if [ "$INCLUDE_CONTROL" = "1" ]; then
  run_eval "gr00t-n17_baseline_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
    GUIDANCE_METHOD= GUIDE_MODEL_PATH= GUIDANCE_SCALE=1.0
fi

if [ -n "$ACG_SKIP_BLOCKS" ]; then
  BLOCK_TAG="blocks$(echo "$ACG_SKIP_BLOCKS" | tr ' ' '-')"
  for W in "${SCALES[@]}"; do
    run_eval "gr00t-n17_acg-${BLOCK_TAG}-w${W}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
      GUIDANCE_METHOD=acg GUIDE_MODEL_PATH= GUIDANCE_SCALE="$W" ACG_SKIP_BLOCKS="$ACG_SKIP_BLOCKS"
  done
fi

if [ -n "$FAILURE_MODEL_PATH" ]; then
  if [ ! -d "$FAILURE_MODEL_PATH" ]; then
    echo "ERROR: $FAILURE_MODEL_PATH not found" >&2
    exit 1
  fi
  FAILURE_TAG="${FAILURE_TAG:-$(basename "$FAILURE_MODEL_PATH")}"
  for ALPHA in "${AFIL_ALPHAS[@]}"; do
    run_eval "gr00t-n17_afil-${FAILURE_TAG}-alpha${ALPHA}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
      GUIDANCE_METHOD=afil GUIDE_MODEL_PATH="$FAILURE_MODEL_PATH" GUIDANCE_SCALE="$ALPHA"
  done
fi

# GUIDE_RUN_DIR as a failure run: AFIL over (checkpoint, alpha), no fixed-scale autoguidance.
if [ -n "$AFIL_RUN_DIR" ]; then
  FAILURE_TAG="${FAILURE_TAG:-$(basename "$AFIL_RUN_DIR")}"
  for STEP in "${GUIDE_STEPS[@]}"; do
    FAIL_CKPT="$AFIL_RUN_DIR/checkpoint-$STEP"
    if [ ! -d "$FAIL_CKPT" ]; then
      echo "ERROR: $FAIL_CKPT not found" >&2
      exit 1
    fi
    for ALPHA in "${AFIL_ALPHAS[@]}"; do
      run_eval "gr00t-n17_afil-${FAILURE_TAG}-step${STEP}-alpha${ALPHA}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
        GUIDANCE_METHOD=afil GUIDE_MODEL_PATH="$FAIL_CKPT" GUIDANCE_SCALE="$ALPHA"
    done
  done
  GUIDE_STEPS=()
fi

for STEP in ${GUIDE_STEPS[@]+"${GUIDE_STEPS[@]}"}; do
  GUIDE_CKPT="$GUIDE_RUN_DIR/checkpoint-$STEP"
  if [ ! -d "$GUIDE_CKPT" ]; then
    echo "ERROR: $GUIDE_CKPT not found" >&2
    exit 1
  fi
  for W in "${SCALES[@]}"; do
    run_eval "gr00t-n17_${GUIDE_LABEL}-step${STEP}-w${W}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
      GUIDANCE_METHOD= GUIDE_MODEL_PATH="$GUIDE_CKPT" GUIDANCE_SCALE="$W"
  done
done

python3 - "$SWEEP_SUMMARY" "${RUN_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path, *run_names = sys.argv[1:]
root = Path("examples/LIBERO/logs/eval_results")
rows = []
for name in run_names:
    summary = json.loads((root / name / "summary.json").read_text())
    config = json.loads((root / name / "eval_config.json").read_text())
    rows.append(
        {
            "run_name": name,
            "guidance_method": config.get("guidance_method"),
            "guide_model_path": config["guide_model_path"],
            "acg_skip_blocks": config.get("acg_skip_blocks"),
            "guidance_scale": config["guidance_scale"],
            "n_success": summary["overall_n_success"],
            "n_episodes": summary["overall_n_episodes"],
            "success_rate": summary["overall_success_rate"],
            "per_task": {r["task"]: r["n_success"] for r in summary["per_task"]},
        }
    )
Path(summary_path).write_text(json.dumps(rows, indent=2))

tasks = list(rows[0]["per_task"])
print()
print("=== Autoguidance sweep ===")
print("per-task columns follow the LIBERO-10 task order:")
for i, task in enumerate(tasks):
    print(f"  T{i}: {task}")
header = f"{'run':<75s} {'success':>9s} {'rate':>6s}  " + " ".join(
    f"T{i:<2d}" for i in range(len(tasks))
)
print(header)
for row in rows:
    per_task = " ".join(f"{row['per_task'][t]:<3d}" for t in tasks)
    print(
        f"{row['run_name']:<75s} {row['n_success']:>4d}/{row['n_episodes']:<4d} "
        f"{row['success_rate']:>6.3f}  {per_task}"
    )
print(f"\nSweep summary written to {summary_path}")
PY
