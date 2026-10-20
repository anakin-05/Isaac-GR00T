#!/usr/bin/env bash
# LIBERO-10 (Long) closed-loop benchmark for one checkpoint and one inference configuration,
# including the Correction Training knobs (see getting_started/correction_training.md).
#
# Starts a GR00T inference server for MODEL_PATH, then runs the LIBERO client
# (gr00t/eval/rollout_policy.py) on all 10 LIBERO-10 tasks in sequence. For each task it saves
# the rollout videos, a JSON of the per-episode results and the tail of the client log. A
# summary.json aggregating all tasks and an eval_config.json describing the run are written to
# examples/LIBERO/logs/eval_results/$RUN_NAME.
#
# Usage:
#   MODEL_PATH=<ckpt dir> bash examples/LIBERO/eval_libero_10.sh
#
# Correction training (only for checkpoints trained with --use-correction):
#   MODEL_PATH=<ckpt> NUM_CORRECTIONS=2 CORRECTION_INTERVAL="0.0 0.5" \
#     bash examples/LIBERO/eval_libero_10.sh
#   NUM_CORRECTIONS=0 evaluates the checkpoint's base denoiser without the corrector.
#   Empty NUM_CORRECTIONS / CORRECTION_INTERVAL use the values saved in the checkpoint config.
#
# Everything is overridable through environment variables, e.g.
#   CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=1 PORT=5557 N_EPISODES=10 SEED=0 ...
# An interrupted run resumes: tasks whose result JSON exists are skipped (SKIP_EXISTING=1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- Configuration (override via environment variables) --------------------
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the checkpoint directory to evaluate}"
NUM_CORRECTIONS="${NUM_CORRECTIONS:-}"
CORRECTION_INTERVAL="${CORRECTION_INTERVAL:-}"   # "LO HI" in flow time (0 = noise, 1 = data)
DENOISING_STEPS="${DENOISING_STEPS:-}"           # empty = checkpoint default (4)
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-42}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
PORT="${PORT:-5556}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

METHOD="$(basename "$MODEL_PATH")"
[ -z "$NUM_CORRECTIONS" ] || METHOD="${METHOD}_K${NUM_CORRECTIONS}"
[ -z "$CORRECTION_INTERVAL" ] || METHOD="${METHOD}_int$(echo "$CORRECTION_INTERVAL" | tr ' ' '-')"
[ -z "$DENOISING_STEPS" ] || METHOD="${METHOD}_d${DENOISING_STEPS}"
RUN_NAME="${RUN_NAME:-${METHOD}_n${N_EPISODES}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/LIBERO/logs/eval_results/$RUN_NAME}"
VIDEO_ROOT="${VIDEO_ROOT:-$OUTPUT_ROOT/sim_eval_videos}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-600}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Space-separated subset of task names (quick smoke tests). Default: the full LIBERO-10 suite.
# shellcheck disable=SC2206
TASKS_OVERRIDE=(${TASKS_OVERRIDE:-})

RESULTS_DIR="$OUTPUT_ROOT/results"
LOGS_DIR="$OUTPUT_ROOT/logs"
SUMMARY_JSON="$OUTPUT_ROOT/summary.json"
CLIENT_PYTHON="gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python"

DEFAULT_TASKS=(
  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket"
  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket"
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it"
  "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it"
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy"
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate"
  "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket"
  "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
  "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it"
)
if [ "${#TASKS_OVERRIDE[@]}" -gt 0 ]; then
  TASKS=("${TASKS_OVERRIDE[@]}")
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

# ---- Preflight ---------------------------------------------------------------
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: checkpoint not found at $MODEL_PATH" >&2
  exit 1
fi
if [ ! -x "$CLIENT_PYTHON" ]; then
  echo "ERROR: $CLIENT_PYTHON not found. Run: bash gr00t/eval/sim/LIBERO/setup_libero.sh" >&2
  exit 1
fi
SERVER_ARGS=()
[ -z "$NUM_CORRECTIONS" ] || SERVER_ARGS+=(--num-corrections "$NUM_CORRECTIONS")
if [ -n "$CORRECTION_INTERVAL" ]; then
  # shellcheck disable=SC2206
  INTERVAL_PAIR=($CORRECTION_INTERVAL)
  if [ "${#INTERVAL_PAIR[@]}" -ne 2 ]; then
    echo "ERROR: CORRECTION_INTERVAL must be two numbers \"LO HI\", got '$CORRECTION_INTERVAL'" >&2
    exit 1
  fi
  SERVER_ARGS+=(--correction-interval "${INTERVAL_PAIR[@]}")
fi
[ -z "$DENOISING_STEPS" ] || SERVER_ARGS+=(--denoising-steps "$DENOISING_STEPS")

mkdir -p "$VIDEO_ROOT" "$RESULTS_DIR" "$LOGS_DIR"

# Record what was evaluated (inference knobs + the checkpoint's correction-training config).
python3 - "$OUTPUT_ROOT/eval_config.json" "$MODEL_PATH" "$NUM_CORRECTIONS" "$CORRECTION_INTERVAL" \
  "$DENOISING_STEPS" "$N_EPISODES" "$SEED" "$N_ENVS" "$N_ACTION_STEPS" "$MAX_EPISODE_STEPS" <<'PY'
import json
import sys
from pathlib import Path

(path, model, k, interval, steps, n_episodes, seed, n_envs, n_action_steps,
 max_steps) = sys.argv[1:]
ckpt_cfg = {}
cfg_file = Path(model) / "config.json"
if cfg_file.exists():
    ckpt_cfg = json.loads(cfg_file.read_text())
use_correction = bool(ckpt_cfg.get("use_correction", False))
config = {
    "model_path": model,
    "use_correction": use_correction,
    "num_corrections": int(k) if k else (
        ckpt_cfg.get("num_inference_corrections") if use_correction else 0
    ),
    "correction_interval": [float(x) for x in interval.split()] if interval else (
        ckpt_cfg.get("correction_interval") if use_correction else None
    ),
    "denoising_steps": int(steps) if steps else ckpt_cfg.get("num_inference_timesteps"),
    "train": {
        key: ckpt_cfg.get(key)
        for key in (
            "correction_loss_weight",
            "base_loss_weight",
            "correction_train_steps",
            "correction_freeze_denoiser",
            "correction_gate_shape",
        )
    } if use_correction else None,
    "n_episodes": int(n_episodes),
    "seed": int(seed),
    "n_envs": int(n_envs),
    "n_action_steps": int(n_action_steps),
    "max_episode_steps": int(max_steps),
}
if not config["num_corrections"]:
    config["correction_interval"] = None  # no correction applied
with open(path, "w") as f:
    json.dump(config, f, indent=2)
PY

# ---- Start the inference server ----------------------------------------------
SERVER_LOG="$LOGS_DIR/server.log"
echo "Starting GR00T server (model: $MODEL_PATH, K: ${NUM_CORRECTIONS:-ckpt}, interval: ${CORRECTION_INTERVAL:-ckpt}, denoising steps: ${DENOISING_STEPS:-ckpt}, port: $PORT) -> $SERVER_LOG"
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" setsid uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --use-sim-policy-wrapper \
  --port "$PORT" \
  ${SERVER_ARGS[@]+"${SERVER_ARGS[@]}"} \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Stopping GR00T server (pid $SERVER_PID)..."
    kill -TERM -"$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo -n "Waiting for the server to load the model and start listening on port $PORT"
waited=0
until (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "ERROR: server exited before it started listening. Last 100 lines of $SERVER_LOG:" >&2
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$waited" -ge "$SERVER_STARTUP_TIMEOUT" ]; then
    echo
    echo "ERROR: server not listening within ${SERVER_STARTUP_TIMEOUT}s. Last 100 lines of $SERVER_LOG:" >&2
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
  echo -n "."
done
echo " ready (${waited}s)."

# ---- Run each task sequentially against the one server -----------------------
for TASK in "${TASKS[@]}"; do
  TASK_LOG="$LOGS_DIR/$TASK.log"
  TASK_JSON="$RESULTS_DIR/$TASK.json"

  if [ "$SKIP_EXISTING" = "1" ] && [ -f "$TASK_JSON" ]; then
    echo "=== $TASK: skipping, $TASK_JSON already exists (SKIP_EXISTING=1) ==="
    continue
  fi

  echo "=== [$(date +%H:%M:%S)] Running $TASK (n_episodes=$N_EPISODES, seed=$SEED) ==="
  FULL_LOG="$(mktemp)"
  set +e
  MUJOCO_EGL_DEVICE_ID="$MUJOCO_EGL_DEVICE_ID" "$CLIENT_PYTHON" gr00t/eval/rollout_policy.py \
    --n-episodes "$N_EPISODES" \
    --policy-client-host 127.0.0.1 \
    --policy-client-port "$PORT" \
    --max-episode-steps "$MAX_EPISODE_STEPS" \
    --env-name "libero_sim/$TASK" \
    --n-action-steps "$N_ACTION_STEPS" \
    --n-envs "$N_ENVS" \
    --seed "$SEED" \
    --video-dir "$VIDEO_ROOT/$TASK" \
    >"$FULL_LOG" 2>&1
  CLIENT_EXIT=$?
  set -e

  tail -100 "$FULL_LOG" >"$TASK_LOG"
  rm -f "$FULL_LOG"

  if [ "$CLIENT_EXIT" -ne 0 ]; then
    echo "ERROR: rollout_policy.py exited with code $CLIENT_EXIT for $TASK. See $TASK_LOG" >&2
    exit 1
  fi

  # rollout_policy.py prints `results:  (env_name, successes, episode_infos)` via repr().
  python3 - "$TASK_LOG" "$TASK_JSON" "$TASK" <<'PY'
import ast, json, re, sys

log_path, json_path, task = sys.argv[1:4]
text = open(log_path).read()
m = re.search(r"^results:\s+(\(.*\))\s*$", text, re.MULTILINE)
if not m:
    print(f"ERROR: no 'results:' line found in {log_path}", file=sys.stderr)
    sys.exit(1)
env_name, successes, episode_infos = ast.literal_eval(m.group(1))
n = len(successes)
n_success = sum(successes)
payload = {
    "task": task,
    "env_name": env_name,
    "n_episodes": n,
    "n_success": n_success,
    "success_rate": n_success / n,
    "successes": successes,
    "episode_infos": episode_infos,
}
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"  -> {n_success}/{n} = {n_success / n:.4f}  (saved {json_path})")
PY
done

# ---- Aggregate summary --------------------------------------------------------
python3 - "$RESULTS_DIR" "$SUMMARY_JSON" "${TASKS[@]}" <<'PY'
import json
import sys
from pathlib import Path

results_dir, summary_path, *tasks = sys.argv[1:]
per_task = []
total_success = 0
total_n = 0
for task in tasks:
    data = json.loads((Path(results_dir) / f"{task}.json").read_text())
    per_task.append(
        {
            "task": task,
            "n_episodes": data["n_episodes"],
            "n_success": data["n_success"],
            "success_rate": data["success_rate"],
        }
    )
    total_success += data["n_success"]
    total_n += data["n_episodes"]

summary = {
    "per_task": per_task,
    "overall_n_success": total_success,
    "overall_n_episodes": total_n,
    "overall_success_rate": total_success / total_n if total_n else None,
}
Path(summary_path).write_text(json.dumps(summary, indent=2))

print()
print("=== Summary ===")
for row in per_task:
    print(f"{row['task']:<100s} {row['n_success']:>3d}/{row['n_episodes']:<3d} ({row['success_rate']:.3f})")
print(f"\nOverall: {total_success}/{total_n} = {total_success / total_n:.4f}")
print(f"Summary written to {summary_path}")
PY
