# Reproducing the LIBERO-10 baseline closed-loop eval

`eval_libero_10.sh` reproduces the closed-loop LIBERO-10 (Long) benchmark for the
`libero_10` baseline checkpoint: it starts a GR00T inference server, then runs the
LIBERO sim client against all 10 LIBERO-10 tasks in sequence, saving videos, a
JSON dump of each task's result, and a log of each run.

## Prerequisites

1. The one-time LIBERO simulation environment setup, if you haven't run it in this
   checkout yet:
   ```bash
   bash gr00t/eval/sim/LIBERO/setup_libero.sh
   ```
   This creates `gr00t/eval/sim/LIBERO/libero_uv/.venv` — a separate, minimal venv
   pinned to LIBERO's own dependency versions (its own torch/numpy/gymnasium), used
   only to run the sim client. The GR00T model itself loads in the repo's normal
   top-level `.venv` (via `uv run`), in a separate server process. **Both venvs bake
   in absolute paths at setup time** (an editable install for `libero`, a `.pth` file
   exposing `gr00t`, and `~/.libero/config.yaml`) — see "If you move or rename this
   repo" below if `setup_libero.sh` was run under a different path than the one
   you're using now.
2. The `libero_10` checkpoint at `examples/LIBERO/logs/libero-baseline/libero_10`
   (or point `MODEL_PATH` at a different one — see Configuration below).

## Usage

From the repo root:
```bash
CUDA_VISIBLE_DEVICES=<your-gpu-or-MIG-uuid> bash examples/LIBERO/eval_libero_10.sh
```

`CUDA_VISIBLE_DEVICES` is passed through from your shell, not hardcoded in the
script — set it to whatever GPU (or MIG slice) you want the server to use. On a
MIG-enabled host, also confirm `MUJOCO_EGL_DEVICE_ID` (default `0`) points at a GPU
index with a working offscreen-rendering pipeline; MuJoCo's EGL renderer parses it
as a plain integer and crashes on a MIG UUID.

Every setting has an environment-variable override, so a full rerun looks like:
```bash
MODEL_PATH=examples/LIBERO/logs/libero-baseline/libero_10 \
N_EPISODES=50 SEED=42 N_ENVS=5 N_ACTION_STEPS=8 MAX_EPISODE_STEPS=720 \
CUDA_VISIBLE_DEVICES=0 \
bash examples/LIBERO/eval_libero_10.sh
```
which is exactly what running the script with no overrides does — those are the
defaults.

To smoke-test the plumbing before committing to the full ~2 hour run (see Runtime
below), run one task with a handful of short episodes:
```bash
TASKS_OVERRIDE="KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it" \
N_EPISODES=2 N_ENVS=1 MAX_EPISODE_STEPS=30 \
OUTPUT_ROOT=/tmp/libero_10_smoke_test \
bash examples/LIBERO/eval_libero_10.sh
```

If a run gets killed partway through (OOM, preemption, etc.), just rerun the same
command: `SKIP_EXISTING=1` (the default) skips any task whose result JSON already
exists, so only the unfinished tasks re-run. Set `SKIP_EXISTING=0` to force
everything to rerun from scratch.

## What it does

1. Starts `gr00t/eval/run_gr00t_server.py --model-path <checkpoint> --use-sim-policy-wrapper`
   in the background (top-level `.venv`, via `uv run`), and waits for it to actually
   accept connections on `PORT` before running any client — model loading is
   synchronous and completes before the server starts listening, so a successful
   connection is a reliable "ready" signal, not a fixed sleep.
2. For each of the 10 LIBERO-10 tasks, runs `gr00t/eval/rollout_policy.py` as a
   client (in `libero_uv/.venv`) against that server, sequentially — one task at a
   time, since there's only one server/checkpoint here.
3. Captures each client run's full output, keeps the last 100 lines as that task's
   log, and parses the `results:  (env_name, successes, episode_infos)` line
   `rollout_policy.py` prints out of it (via `ast.literal_eval`, then re-serializes
   as JSON) to produce that task's result file.
4. After all tasks finish, aggregates every task's JSON into one `summary.json` with
   the true overall success rate (sum of successes over sum of episodes, not an
   average of per-task rates — LIBERO's vectorized collector can produce a
   task with a couple more or fewer episodes than requested, so a simple average
   would silently misweight the small tasks).
5. Always stops the server on exit (success, failure, or Ctrl-C) via a trap on the
   process group it was started in.

## Output layout

```
examples/LIBERO/logs/eval_results/libero_10/
├── sim_eval_videos/
│   └── <TASK_NAME>/               # one subdir per task; rollout_policy.py's --video-dir
├── results/
│   └── <TASK_NAME>.json           # {task, env_name, n_episodes, n_success, success_rate,
│                                   #  successes, episode_infos} for that task's `results`
├── logs/
│   ├── server.log                 # full server output for the whole run
│   └── <TASK_NAME>.log            # tail -100 of that task's rollout_policy.py run
└── summary.json                   # aggregated per-task + overall success rate
```

## Configuration reference

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `examples/LIBERO/logs/libero-baseline/libero_10` | Checkpoint directory passed to the server |
| `EMBODIMENT_TAG` | `LIBERO_PANDA` | Passed to the server |
| `N_EPISODES` | `50` | Episodes per task |
| `SEED` | `42` | Forwarded to `rollout_policy.py --seed`; seeds Python/NumPy/torch/cuda RNGs and gives each parallel env a distinct, deterministic per-env seed (`seed + env_idx`) |
| `N_ENVS` | `5` | Parallel envs per task |
| `N_ACTION_STEPS` | `8` | Receding-horizon execution length |
| `MAX_EPISODE_STEPS` | `720` | Per-episode step budget |
| `PORT` | `5556` | Server port |
| `MUJOCO_EGL_DEVICE_ID` | `0` | Offscreen-rendering GPU index (see Prerequisites) |
| `HF_HUB_DISABLE_XET` | `1` | Works around the HF Xet backend hanging on some hosts |
| `OUTPUT_ROOT` | `examples/LIBERO/logs/eval_results/libero_10` | Base output directory |
| `VIDEO_ROOT` | `$OUTPUT_ROOT/sim_eval_videos` | Video base directory |
| `SERVER_STARTUP_TIMEOUT` | `600` (seconds) | How long to wait for the server before giving up |
| `SKIP_EXISTING` | `1` | Skip a task whose result JSON already exists |
| `TASKS_OVERRIDE` | (unset = full 10) | Space-separated task-name subset, for smoke tests |

`CUDA_VISIBLE_DEVICES` is read from your shell as-is (not listed above since the
script never sets a default for it) — required, since the server otherwise
defaults to whatever GPU is visible/free on your machine.

## Runtime

Sequential (one server, one client at a time): roughly 10-20 minutes per task at
`n_episodes=50`, `n_envs=5`, so budget **~2 hours** for the full 10-task suite. The
two hardest tasks (`KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` and
`KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`) tend
to run longest.

## If you move or rename this repo

`setup_libero.sh` writes a few **absolute paths** into `libero_uv/.venv` and into
`~/.libero/config.yaml` at setup time (an editable install of the vendored LIBERO
repo, a `.pth` file exposing this repo's `gr00t` package, and LIBERO's own asset/
benchmark paths). Moving or renaming the repo directory after running it once
leaves those stale, and you'll see `ModuleNotFoundError: No module named 'libero'`
(or `'gr00t'`) even though the venv still exists. Fix by rewriting the two files to
the new path instead of rerunning the whole setup script:

```bash
NEW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"  # or hardcode it
VENV_SITE="$NEW_ROOT/gr00t/eval/sim/LIBERO/libero_uv/.venv/lib/python3.12/site-packages"
echo "$NEW_ROOT" > "$VENV_SITE/gr00t.pth"
echo "$NEW_ROOT/external_dependencies/LIBERO" > "$VENV_SITE"/__editable__.libero-*.pth
sed -i "s#/old/path/to/old-repo-name#$NEW_ROOT#g" ~/.libero/config.yaml
```
The top-level `.venv` (`uv run ...`) does not have this problem — `uv run`
re-syncs the editable install against the current path automatically on every
invocation.
