#!/usr/bin/env bash
# Reproduce the libero_10 (Long) closed-loop benchmark, optionally with
# autoguidance (GUIDE_MODEL_PATH + GUIDANCE_SCALE; see AUTOGUIDANCE.md).
#
# Starts a GR00T inference server for the given checkpoint, then runs the
# client (gr00t/eval/rollout_policy.py) against all 10 LIBERO-10 tasks in
# sequence. For each task this saves: the rollout videos, a JSON dump of the
# task's `results` return value (env_name, per-episode successes, and
# episode_infos), and a truncated (tail -100) log of that rollout_policy.py
# run. A summary.json aggregating all 10 tasks is written at the end.
#
# Usage:
#   bash examples/LIBERO/eval_libero_10.sh
#
# All configuration is overridable via environment variables, e.g.:
#   CUDA_VISIBLE_DEVICES=0 N_EPISODES=10 SEED=0 \
#     bash examples/LIBERO/eval_libero_10.sh
#
# Autoguidance: guide the main model with a weaker checkpoint, e.g.
#   GUIDE_MODEL_PATH=examples/LIBERO/logs/libero-rtc/<guide-run>/checkpoint-5000 \
#   GUIDANCE_SCALE=2.0 bash examples/LIBERO/eval_libero_10.sh
#
# Output goes to examples/LIBERO/logs/eval_results/$RUN_NAME, where RUN_NAME
# encodes the method, guidance scale, episode count and seed, so runs with
# different settings never share (or SKIP_EXISTING-skip) each other's results.
# See examples/LIBERO/AUTOGUIDANCE.md for the full workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ---- Configuration (override via environment variables) --------------------
# Main model: the LIBERO-10 baseline (79,273 steps, batch 32; 471/500 at
# N_EPISODES=50, SEED=42).
MODEL_PATH="${MODEL_PATH:-/mnt/data/sftp/data/duydt6/workspace/diffusion/Isaac-GR00T-baseline/examples/LIBERO/logs/libero-baseline/2026-09-19_gr00t-n17_baseline_libero-panda_libero-10}"
# Autoguidance guide checkpoint (empty = no guidance) and weight w.
GUIDE_MODEL_PATH="${GUIDE_MODEL_PATH:-}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-1.0}"
EMBODIMENT_TAG="${EMBODIMENT_TAG:-LIBERO_PANDA}"
N_EPISODES="${N_EPISODES:-50}"
SEED="${SEED:-42}"
N_ENVS="${N_ENVS:-5}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
MAX_EPISODE_STEPS="${MAX_EPISODE_STEPS:-720}"
PORT="${PORT:-5556}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
if [ -n "$GUIDE_MODEL_PATH" ]; then
  METHOD="autoguidance-${GUIDE_TAG:-$(basename "$GUIDE_MODEL_PATH")}-w${GUIDANCE_SCALE}"
else
  METHOD="$(basename "$MODEL_PATH")"
fi
RUN_NAME="${RUN_NAME:-${METHOD}_n${N_EPISODES}-seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-examples/LIBERO/logs/eval_results/$RUN_NAME}"
VIDEO_ROOT="${VIDEO_ROOT:-$OUTPUT_ROOT/sim_eval_videos}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-600}"
# Skip a task whose result JSON already exists (lets a killed/interrupted run
# resume without redoing already-completed tasks). Set to 0 to force a full
# rerun of every task.
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# Space-separated subset of task names to run instead of the full 10 (for
# quick smoke tests). Defaults to the full LIBERO-10 (Long) suite below.
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
GUIDE_ARGS=()
if [ -n "$GUIDE_MODEL_PATH" ]; then
  if [ ! -d "$GUIDE_MODEL_PATH" ]; then
    echo "ERROR: guide checkpoint not found at $GUIDE_MODEL_PATH" >&2
    exit 1
  fi
  GUIDE_ARGS=(--guide-model-path "$GUIDE_MODEL_PATH" --guidance-scale "$GUIDANCE_SCALE")
elif [ "$GUIDANCE_SCALE" != "1.0" ] && [ "$GUIDANCE_SCALE" != "1" ]; then
  echo "ERROR: GUIDANCE_SCALE=$GUIDANCE_SCALE requires GUIDE_MODEL_PATH" >&2
  exit 1
fi
if [ ! -x "$CLIENT_PYTHON" ]; then
  echo "ERROR: $CLIENT_PYTHON not found." >&2
  echo "Run the one-time LIBERO sim environment setup first:" >&2
  echo "  bash gr00t/eval/sim/LIBERO/setup_libero.sh" >&2
  exit 1
fi

mkdir -p "$VIDEO_ROOT" "$RESULTS_DIR" "$LOGS_DIR"

# Record what was evaluated next to the results.
python3 - "$OUTPUT_ROOT/eval_config.json" "$MODEL_PATH" "$GUIDE_MODEL_PATH" "$GUIDANCE_SCALE" \
  "$N_EPISODES" "$SEED" "$N_ENVS" "$N_ACTION_STEPS" "$MAX_EPISODE_STEPS" <<'PY'
import json
import sys

path, model, guide, scale, n_episodes, seed, n_envs, n_action_steps, max_steps = sys.argv[1:]
config = {
    "model_path": model,
    "guide_model_path": guide or None,
    "guidance_scale": float(scale),
    "n_episodes": int(n_episodes),
    "seed": int(seed),
    "n_envs": int(n_envs),
    "n_action_steps": int(n_action_steps),
    "max_episode_steps": int(max_steps),
}
with open(path, "w") as f:
    json.dump(config, f, indent=2)
PY

# ---- Start the inference server ----------------------------------------------
SERVER_LOG="$LOGS_DIR/server.log"
echo "Starting GR00T server (model: $MODEL_PATH, guide: ${GUIDE_MODEL_PATH:-none}, w: $GUIDANCE_SCALE, port: $PORT) -> $SERVER_LOG"
HF_HUB_DISABLE_XET="$HF_HUB_DISABLE_XET" setsid uv run python gr00t/eval/run_gr00t_server.py \
  --model-path "$MODEL_PATH" \
  --embodiment-tag "$EMBODIMENT_TAG" \
  --use-sim-policy-wrapper \
  --port "$PORT" \
  ${GUIDE_ARGS[@]+"${GUIDE_ARGS[@]}"} \
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
  exec 3<&- 2>/dev/null || true
  exec 3>&- 2>/dev/null || true
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo
    echo "ERROR: server process exited before it started listening. Last 100 lines of $SERVER_LOG:" >&2
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$waited" -ge "$SERVER_STARTUP_TIMEOUT" ]; then
    echo
    echo "ERROR: server did not start listening within ${SERVER_STARTUP_TIMEOUT}s. Last 100 lines of $SERVER_LOG:" >&2
    tail -100 "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
  waited=$((waited + 5))
  echo -n "."
done
exec 3<&- 2>/dev/null || true
exec 3>&- 2>/dev/null || true
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

  # rollout_policy.py's __main__ prints `results:  (env_name, successes, episode_infos)`
  # via repr(); extract that line and re-serialize it as JSON. Plain python3 is
  # enough here (stdlib ast/json/re only), no need for either project venv.
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
