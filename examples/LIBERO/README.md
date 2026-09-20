# LIBERO

Benchmark for studying knowledge transfer in lifelong robot learning. Includes multiple suites: **Spatial** (spatial reasoning), **Object** (object generalization), **Goal** (goal-conditioned learning), and **10 Long** (long-horizon multi-step tasks). Provides RGB images, proprioception data, and language task specifications.

For more information, see the [official website](https://libero-project.github.io/main.html).

---

# LIBERO evaluation benchmark result

> **Note:** The full task list is attached at the end of this document.

All four suites were finetuned with the same hyper-parameters, including
`--state-dropout-prob 0.2` (the finetune CLI default from
`gr00t/configs/finetune_config.py`).

| Task      | Success rate       | max_steps | grad_accum_steps | batch_size |
|-----------|--------------------|-----------|------------------|------------|
| Spatial   | 195/200 (97.65%)        | 20K       | 1                | 640        |
| Goal      | 195/200 (97.5%)        | 20K       | 1                | 640        |
| Object    | 197/200 (98.45%)        | 20K       | 1                | 640        |
| 10 (Long) | 189/200 (94.35%)        | 20K       | 1                | 640        |

# Fine-tune LIBERO 10 (long)

To reproduce our finetune results, use the following commands to setup dataset and launch finetune experiments. Please remember to set `WANDB_API_KEY` since W&B logging is on by default (`USE_WANDB=1` in `examples/finetune.sh`). If you don't have a WANDB account, prepend `USE_WANDB=0` to the launch command to disable it:

```bash
uv run hf download \
    --repo-type dataset IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_10 \
    --state-dropout-prob 0.2
```

# Fine-tune LIBERO goal

```bash
uv run hf download \
    --repo-type dataset IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/meta/
## This is a patch for one of the episode where the image seems to be corrupted.
cp examples/LIBERO/patches/episode_000082.mp4 examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/videos/chunk-000/observation.images.wrist_image/
```

Run the shared finetune launcher:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_goal
```

# Fine-tune LIBERO object

```bash
uv run hf download \
    --repo-type dataset IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_object_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_object
```

# Fine-tune LIBERO spatial

```bash
uv run hf download \
    --repo-type dataset IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
    --local-dir examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/

# Copy the patches and run the finetune script
cp -r examples/LIBERO/modality.json examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/meta/
```

Run the shared finetune launcher:
```bash
NUM_GPUS=8 MAX_STEPS=20000 GLOBAL_BATCH_SIZE=640 SAVE_STEPS=1000 uv run bash examples/finetune.sh \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_spatial_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA \
    --output-dir /tmp/libero_spatial
```

# Evaluate checkpoint

First, complete the [one-time simulation environment setup](../../README.md#one-time-simulation-environment-setup), then run this benchmark's setup script (only needed once per benchmark):

```bash
bash gr00t/eval/sim/LIBERO/setup_libero.sh
```

Then, download the finetuned model to a local directory (HuggingFace does not support nested repo paths directly):
```bash
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/config.json" "libero_10/embodiment_id.json" "libero_10/model-*.safetensors" "libero_10/model.safetensors.index.json" "libero_10/processor_config.json" "libero_10/statistics.json" --local-dir checkpoints/GR00T-N1.7-LIBERO
```

Run client server evaluation under the project root directory in separate terminals:

**Terminal 1 - Server:**
```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper
```

> **Note:** Replace `checkpoints/GR00T-N1.7-LIBERO/libero_10` with your own checkpoint path (e.g., `/tmp/libero_10/checkpoint-20000/`) if evaluating a locally finetuned model.

**Terminal 2 - Client:**
```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n-action-steps 8 \
    --n-envs 5
```

## Round-trip consistency monitoring

For a checkpoint fine-tuned with `--enable-direction-conditioning` (see the
[round-trip-consistency integration handoff](../../HANDOFF_ROUND_TRIP_CONSISTENCY.md)), add
these flags to the server command above to also log a round-trip-consistency diagnostic
alongside the rollout, with **no changes to the client command**. Two $\mathcal{C}_i$ metrics
are logged together, both from the same backward-direction query: `action_c_i` (MSE between
the true past state and the *decoded* backward-reconstruction of it — the original metric) and
`latent_c_i` (MSE between the DiT's own last-timestep hidden state — the representation
immediately *before* decoding — from the real forward call and the backward call's equivalent).
See [`round_trip_monitor.py`](../../gr00t/policy/round_trip_monitor.py)'s "Two C_i metrics"
section for the full math.

```bash
uv run python gr00t/eval/run_gr00t_server.py \
    --model-path <checkpoint-dir> \
    --embodiment-tag LIBERO_PANDA \
    --use-sim-policy-wrapper \
    --enable-round-trip-monitor \
    --round-trip-check-interval 8 \
    --round-trip-depths 1 2 4 \
    --round-trip-log-path /tmp/round_trip_consistency/libero_goal.jsonl
```

Keep `--round-trip-check-interval` equal to the client's `--n-action-steps` above (both
default to 8) — the server has no way to see the client's execution horizon, so a mismatch
here means depth `i` no longer corresponds to `i` actual chunk executions. This adds one
extra backward-direction inference call per env slot every `round_trip_check_interval`
calls (a real compute cost, not free), and does nothing for a checkpoint that wasn't trained
with direction conditioning — don't enable it for a baseline (non-RTC) checkpoint.

To run the full LIBERO-10 (Long) suite (all 10 tasks) against an RTC checkpoint in
one command — including a JSON dump of each task's results, the round-trip JSONL
log, and a C_i-vs-outcome correlation summary — instead of one task at a time by
hand, see [`eval_libero_10_rtc.sh`](./eval_libero_10_rtc.sh) and
[`EVAL_LIBERO_10_RTC.md`](./EVAL_LIBERO_10_RTC.md).

### Seeing C_i on the video, and a per-episode C_i plot

The server-side monitor above (Mechanism 1) logs to a JSONL sidecar, not to the
client, so it can't put anything on the recorded rollout video. For that, use
Mechanism 2 instead — a *client-side* round-trip check, added directly to the
`rollout_policy.py` client command (no server flags needed for this):

```bash
gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 \
    --policy-client-host 127.0.0.1 \
    --policy-client-port 5555 \
    --max-episode-steps 720 \
    --env-name libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
    --n-action-steps 8 \
    --n-envs 5 \
    --video-dir /tmp/sim_eval_videos \
    --enable-round-trip-check \
    --round-trip-check-interval 8 \
    --round-trip-depths 1 2 4
```

With `--enable-round-trip-check` and a `--video-dir` both set, every kept
episode's video gets two extra caption lines — `C_i(action): d1=... d2=...`
and `C_i(latent): d1=... d2=...` — showing each depth's latest known value of
both metrics (updated whenever a check fires, holding its value between
checks — reads "(pending)" until the first one lands), and a
`<video-name>.png` plot is saved right alongside the `.mp4` with two stacked
subplots (action_c_i on top, latent_c_i on the bottom — different scales, so
not one shared axes), one line per depth in each. Same
`--round-trip-check-interval` caveat as Mechanism 1 above (keep it equal to
`--n-action-steps`). This is independent of Mechanism 1 — enable either, or
both, or neither; the overlay/plot only appears when
`--enable-round-trip-check` is set, since that's the mechanism whose C_i the
client already computes directly, with no extra server-side plumbing needed.

# Full task list

## Libero 10 (Long)
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- `libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- `libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- `libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- `libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- `libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
- `libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
- `libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`

## Libero Goal
- `libero_sim/open_the_middle_drawer_of_the_cabinet`
- `libero_sim/put_the_bowl_on_the_stove`
- `libero_sim/put_the_wine_bottle_on_top_of_the_cabinet`
- `libero_sim/open_the_top_drawer_and_put_the_bowl_inside`
- `libero_sim/put_the_bowl_on_top_of_the_cabinet`
- `libero_sim/push_the_plate_to_the_front_of_the_stove`
- `libero_sim/put_the_cream_cheese_in_the_bowl`
- `libero_sim/turn_on_the_stove`
- `libero_sim/put_the_bowl_on_the_plate`
- `libero_sim/put_the_wine_bottle_on_the_rack`

## Libero Object
- `libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_butter_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_milk_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket`

## Libero Spatial
- `libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`