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
# Any eval_libero_10.sh setting (MODEL_PATH, N_ENVS, PORT, ...) can be
# overridden through the environment as well. See AUTOGUIDANCE.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

GUIDE_RUN_DIR="${GUIDE_RUN_DIR:?set GUIDE_RUN_DIR to the training output dir of the guide model}"
GUIDE_LABEL="${GUIDE_LABEL:-autoguidance-dit8}"
# shellcheck disable=SC2206
GUIDE_STEPS=(${GUIDE_STEPS:-2000 5000 10000})
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
    GUIDE_MODEL_PATH= GUIDANCE_SCALE=1.0
fi

for STEP in "${GUIDE_STEPS[@]}"; do
  GUIDE_CKPT="$GUIDE_RUN_DIR/checkpoint-$STEP"
  if [ ! -d "$GUIDE_CKPT" ]; then
    echo "ERROR: $GUIDE_CKPT not found" >&2
    exit 1
  fi
  for W in "${SCALES[@]}"; do
    run_eval "gr00t-n17_${GUIDE_LABEL}-step${STEP}-w${W}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
      GUIDE_MODEL_PATH="$GUIDE_CKPT" GUIDANCE_SCALE="$W"
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
            "guide_model_path": config["guide_model_path"],
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
