#!/usr/bin/env bash
# Sweep Action Coherence Guidance (ACG) over guidance scales on LIBERO-10 (Long).
#
# Runs examples/LIBERO/eval_libero_10.sh once per guidance scale with GUIDANCE_METHOD=acg, plus an
# unguided control (w=1) on the same main model, seed and episode count, then prints a table.
# Each configuration writes to its own examples/LIBERO/logs/eval_results/<run-name>, so an
# interrupted sweep resumes where it stopped.
#
# Usage:
#   N_EPISODES=10 SCALES="1.5 2.0 3.0" ACG_SKIP_BLOCKS="15 19 23" \
#     bash examples/LIBERO/sweep_acg_libero_10.sh
#
# Any eval_libero_10.sh setting (MODEL_PATH, N_ENVS, PORT, ...) can be overridden as well.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC2206
SCALES=(${SCALES:-1.5 2.0 3.0})
ACG_SKIP_BLOCKS="${ACG_SKIP_BLOCKS:-15 19 23}"
INCLUDE_CONTROL="${INCLUDE_CONTROL:-1}"
export N_EPISODES="${N_EPISODES:-10}"
export SEED="${SEED:-42}"
BLOCK_TAG="blocks$(echo "$ACG_SKIP_BLOCKS" | tr ' ' '-')"
SWEEP_SUMMARY="${SWEEP_SUMMARY:-examples/LIBERO/logs/eval_results/sweep_acg-${BLOCK_TAG}_n${N_EPISODES}-seed${SEED}.json}"

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

for W in "${SCALES[@]}"; do
  run_eval "gr00t-n17_acg-${BLOCK_TAG}-w${W}_libero-panda_libero-10_n${N_EPISODES}-seed${SEED}" \
    GUIDANCE_METHOD=acg GUIDE_MODEL_PATH= GUIDANCE_SCALE="$W" ACG_SKIP_BLOCKS="$ACG_SKIP_BLOCKS"
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
print("=== ACG sweep ===")
print("per-task columns follow the LIBERO-10 task order:")
for i, task in enumerate(tasks):
    print(f"  T{i}: {task}")
print(f"{'run':<75s} {'success':>9s} {'rate':>6s}  " + " ".join(f"T{i:<2d}" for i in range(len(tasks))))
for row in rows:
    per_task = " ".join(f"{row['per_task'][t]:<3d}" for t in tasks)
    print(
        f"{row['run_name']:<75s} {row['n_success']:>4d}/{row['n_episodes']:<4d} "
        f"{row['success_rate']:>6.3f}  {per_task}"
    )
print(f"\nSweep summary written to {summary_path}")
PY
