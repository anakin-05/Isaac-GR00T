# Reproducing the LIBERO-10 round-trip-consistency (RTC) closed-loop eval

`eval_libero_10_rtc.sh` reproduces the closed-loop LIBERO-10 (Long) benchmark for
the `libero_10-rtc` checkpoint, with both round-trip-consistency (RTC) mechanisms
enabled: it starts a GR00T inference server with `--enable-round-trip-monitor`
(Mechanism 1, server-side), then runs the LIBERO sim client with
`--enable-round-trip-check` (Mechanism 2, client-side) against all 10 LIBERO-10
tasks in sequence. Because Mechanism 2 and `--video-dir` are both set, every kept
episode's video gets a `C_i(action)`/`C_i(latent)` caption per depth and a matching
per-episode C_i-vs-step plot alongside it (see `gr00t/policy/round_trip_monitor.py`'s
"Two C_i metrics" section for what the two metrics are). Also saves a JSON dump of
each task's result (including Mechanism 2's own C_i entries in `episode_infos`), a
log of each run, Mechanism 1's round-trip-consistency JSONL log (named after the
checkpoint - see `CHECKPOINT_NAME`/`RTC_LOG_PATH` below), and a C_i-vs-outcome
correlation summary built from that JSONL log.

This only makes sense against a checkpoint that was fine-tuned with
`--enable-direction-conditioning` — see
[`HANDOFF_ROUND_TRIP_CONSISTENCY.md`](../../HANDOFF_ROUND_TRIP_CONSISTENCY.md) for
what round-trip-consistency (arXiv:2608.00675) is and how it was integrated. In
short: the direction-conditioned model can also run a backward diffusion step: fed
its own predicted action chunk, it should reconstruct the observation it started
from. $\mathcal{C}_i$ measures how well it does that at rollout depth $i$ — a large
$\mathcal{C}_i$ means the model's forward/backward predictions are inconsistent
with each other, a candidate trust signal for when its forward prediction is more
likely to be wrong. Enabling this on a checkpoint that wasn't trained with
direction conditioning does nothing useful — don't enable it on a plain baseline
checkpoint (see the sibling `Isaac-GR00T-baseline` repo's
`eval_libero_10.sh` for that case, no RTC flags there at all).

## Prerequisites

1. The one-time LIBERO simulation environment setup, if you haven't run it in this
   checkout yet:
   ```bash
   bash gr00t/eval/sim/LIBERO/setup_libero.sh
   ```
   This creates `gr00t/eval/sim/LIBERO/libero_uv/.venv` — a separate, minimal venv
   pinned to LIBERO's own dependency versions, used only to run the sim client. The
   GR00T model loads in the repo's normal top-level `.venv` (via `uv run`), in a
   separate server process. Both venvs bake in absolute paths at setup time (an
   editable install for `libero`, a `.pth` file exposing `gr00t`, and
   `~/.libero/config.yaml`) — if you move or rename this repo afterward, those go
   stale; see the equivalent section in `Isaac-GR00T-baseline/examples/LIBERO/EVAL_LIBERO_10.md`
   for the two-line fix (same underlying issue, same fix, no RTC specifics to it).
2. The `libero_10-rtc` checkpoint at
   `examples/LIBERO/logs/libero-rtc/libero_10-rtc` (or point `MODEL_PATH` at a
   different direction-conditioned checkpoint — see Configuration below).

## Usage

From the repo root:
```bash
CUDA_VISIBLE_DEVICES=<your-gpu-or-MIG-uuid> bash examples/LIBERO/eval_libero_10_rtc.sh
```

`CUDA_VISIBLE_DEVICES` is passed through from your shell, not hardcoded in the
script. On a MIG-enabled host, also confirm `MUJOCO_EGL_DEVICE_ID` (default `0`)
points at a GPU index with a working offscreen-rendering pipeline; MuJoCo's EGL
renderer parses it as a plain integer and crashes on a MIG UUID.

Every setting has an environment-variable override, so a full rerun looks like:
```bash
MODEL_PATH=examples/LIBERO/logs/libero-rtc/libero_10-rtc \
N_EPISODES=50 SEED=42 N_ENVS=5 N_ACTION_STEPS=8 MAX_EPISODE_STEPS=720 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/LIBERO/eval_libero_10_rtc.sh
```
which is exactly what running the script with no overrides does — those are the
defaults.

To smoke-test the plumbing before committing to the full ~2 hour run (RTC's extra
backward-direction inference calls make this a bit slower than the non-RTC
baseline eval), run one task with a handful of short episodes:
```bash
TASKS_OVERRIDE="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
N_EPISODES=2 N_ENVS=1 MAX_EPISODE_STEPS=30 \
OUTPUT_ROOT=/tmp/libero_10_rtc_smoke_test \
bash examples/LIBERO/eval_libero_10_rtc.sh
```

If a run gets killed partway through (OOM, preemption, etc.), just rerun the same
command: `SKIP_EXISTING=1` (the default) skips any task whose result JSON already
exists, so only the unfinished tasks re-run — the round-trip JSONL log is opened in
append mode across the whole run, so restarting doesn't lose the earlier tasks'
rows either. Set `SKIP_EXISTING=0` to force everything to rerun from scratch (and
delete or move the existing `round_trip_consistency/*.jsonl` first if you want a
clean log rather than old and new rows mixed together).

## What it does

1. Starts `gr00t/eval/run_gr00t_server.py --model-path <checkpoint>
   --use-sim-policy-wrapper --enable-round-trip-monitor
   --round-trip-check-interval <ROUND_TRIP_CHECK_INTERVAL> --round-trip-depths 1 2 4
   --round-trip-log-path <path>` in the background (top-level `.venv`, via
   `uv run`), and waits for it to actually accept connections on `PORT` before
   running any client — model loading is synchronous and completes before the
   server starts listening, so a successful connection is a reliable "ready"
   signal, not a fixed sleep. `--round-trip-check-interval` defaults to
   `N_ACTION_STEPS` — the server has no way to see the client's execution
   horizon, so a mismatch here means depth `i` no longer corresponds to `i`
   actual chunk executions — but is independently overridable via
   `ROUND_TRIP_CHECK_INTERVAL` for a deliberate ablation (e.g. checking every
   macro-step instead of every chunk execution); set it explicitly only if you
   understand and want that tradeoff.
2. For each of the 10 LIBERO-10 tasks, runs `gr00t/eval/rollout_policy.py --enable-round-trip-check
   --round-trip-check-interval <ROUND_TRIP_CHECK_INTERVAL> --round-trip-depths 1 2 4`
   (Mechanism 2) as a client (in `libero_uv/.venv`) against that server,
   sequentially — one task at a time, since there's only one server/checkpoint
   here. Mechanism 1 (the server-side monitor) attributes every logged row to a
   `task` and per-slot `episode_id` (see "Later addition: per-episode and
   per-task identity" in `HANDOFF_ROUND_TRIP_CONSISTENCY.md`), so all 10 tasks
   can safely share one continuous JSONL log. Mechanism 2 runs independently on
   the client and, since `--video-dir` is also set, burns its own C_i values
   into each kept episode's video caption and a matching per-episode plot -
   see `gr00t/eval/sim/wrapper/video_recording_wrapper.py`.
3. Captures each client run's full output, keeps the last 100 lines as that task's
   log, and parses the `results:  (env_name, successes, episode_infos)` line
   `rollout_policy.py` prints out of it (via `ast.literal_eval`, then
   re-serializes as JSON) to produce that task's result file — including
   `episode_infos["round_trip_episode_id"]`, index-aligned with `successes`, which
   is what step 5 joins against the round-trip log.
4. After all tasks finish, aggregates every task's JSON into one `summary.json`
   with the true overall success rate (sum of successes over sum of episodes, not
   an average of per-task rates — LIBERO's vectorized collector can produce a task
   with a couple more or fewer episodes than requested, so a simple average would
   silently misweight the small tasks).
5. Joins each task's per-episode outcome against the round-trip log's
   `task`/`episode_id`/`action_c_i` fields and writes
   `round_trip_consistency/correlation_summary.json`: mean action-space
   $\mathcal{C}_i$ (averaged over every depth and check logged during that
   episode) for failed vs. succeeded episodes, per task and overall. This is
   descriptive, not a statistical test — treat it as a first read on whether
   $\mathcal{C}_i$ tracks failure on this run, not a significance claim. The
   JSONL log also carries a `latent_c_i` field (pre-decoder DiT-hidden-state
   MSE — see `round_trip_monitor.py`'s "Two C_i metrics" section) that this
   correlation step doesn't join against yet — a real, separate follow-up.
6. Always stops the server on exit (success, failure, or Ctrl-C) via a trap on the
   process group it was started in.

## Output layout

```
examples/LIBERO/logs/eval_results/<CHECKPOINT_NAME>/
├── sim_eval_videos/
│   └── <TASK_NAME>/                    # one subdir per task; rollout_policy.py's --video-dir -
│                                        # each kept video has a C_i(action)/C_i(latent) caption
│                                        # and a matching <video-name>.png plot alongside it
├── results/
│   └── <TASK_NAME>.json                # {task, env_name, n_episodes, n_success, success_rate,
│                                        #  successes, episode_infos} for that task's `results`
├── logs/
│   ├── server.log                      # full server output for the whole run
│   └── <TASK_NAME>.log                 # tail -100 of that task's rollout_policy.py run
├── round_trip_consistency/
│   ├── <CHECKPOINT_NAME>.jsonl          # one row per {env_slot, call_index, depth}; every
│   │                                    # row tagged with task + episode_id + action_c_i +
│   │                                    # latent_c_i + per_key
│   └── correlation_summary.json        # mean action_c_i for failed vs. succeeded episodes
└── summary.json                        # aggregated per-task + overall success rate
```

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `examples/LIBERO/logs/libero-rtc/libero_10-rtc` | Checkpoint directory passed to the server; must be direction-conditioned |
| `CHECKPOINT_NAME` | `$(basename "$MODEL_PATH")` | Not independently settable - derived from `MODEL_PATH`. Used as the default `OUTPUT_ROOT`/`RTC_LOG_PATH` basename so two different checkpoints never collide or silently skip each other's tasks via `SKIP_EXISTING` |
| `EMBODIMENT_TAG` | `LIBERO_PANDA` | Passed to the server |
| `N_EPISODES` | `50` | Episodes per task |
| `SEED` | `42` | Forwarded to `rollout_policy.py --seed`; seeds Python/NumPy/torch/cuda RNGs and gives each parallel env a distinct, deterministic per-env seed (`seed + env_idx`) |
| `N_ENVS` | `5` | Parallel envs per task |
| `N_ACTION_STEPS` | `8` | Receding-horizon execution length; also the default for `ROUND_TRIP_CHECK_INTERVAL` (see below) |
| `MAX_EPISODE_STEPS` | `720` | Per-episode step budget |
| `PORT` | `5555` | Server port |
| `MUJOCO_EGL_DEVICE_ID` | `0` | Offscreen-rendering GPU index (see Prerequisites) |
| `HF_HUB_DISABLE_XET` | `1` | Works around the HF Xet backend hanging on some hosts |
| `ROUND_TRIP_DEPTHS` | `1 2 4` | Rollout depths (in get_action calls) the monitor evaluates |
| `RTC_LOG_PATH` | `$OUTPUT_ROOT/round_trip_consistency/$CHECKPOINT_NAME.jsonl` | Round-trip JSONL sidecar path |
| `OUTPUT_ROOT` | `examples/LIBERO/logs/eval_results/$CHECKPOINT_NAME` | Base output directory |
| `VIDEO_ROOT` | `$OUTPUT_ROOT/sim_eval_videos` | Video base directory |
| `SERVER_STARTUP_TIMEOUT` | `600` (seconds) | How long to wait for the server before giving up |
| `SKIP_EXISTING` | `1` | Skip a task whose result JSON already exists |
| `TASKS_OVERRIDE` | (unset = full 10) | Space-separated task-name subset, for smoke tests |
| `ROUND_TRIP_CHECK_INTERVAL` | `$N_ACTION_STEPS` | Server-side check cadence; independently overridable for a deliberate depth-semantics ablation (see "What it does" above) |

`CUDA_VISIBLE_DEVICES` is read from your shell as-is (not listed above since the
script never sets a default for it) — required, since the server otherwise
defaults to whatever GPU is visible/free on your machine.

## Runtime

Sequential (one server, one client at a time), and slower per task than a plain
baseline eval — every `round_trip_check_interval` calls, the round-trip monitor
does one extra backward-direction inference call per env slot, a real compute
cost. Budget **more than the ~2 hours** the non-RTC baseline eval takes for the
same n_episodes/task count; the two hardest tasks
(`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` and
`KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`) tend
to run longest.
