# Autoguidance for GR00T N1.7 on LIBERO-10

Implementation of autoguidance from Karras et al., "Guiding a Diffusion Model with a Bad
Version of Itself" (NeurIPS 2024; reference code: NVlabs/edm2), applied to GR00T N1.7's
flow-matching action head.

## Method

Autoguidance samples from a main model `D_main` while extrapolating away from a weaker
*guide* model `D_guide` that has the **same conditioning** but is smaller and trained for
less time:

```
D = D_guide + w * (D_main - D_guide)          # EDM2: ref_Dx.lerp(Dx, guidance)
```

GR00T predicts a flow-matching velocity `v` instead of a denoised sample. For a fixed noisy
action `x_t` and time `t`, the denoised estimate is affine in `v`, so the same rule applies to
the velocity at every Euler step of the 4-step sampler:

```
v = v_guide + w * (v_main - v_guide)
```

`w = 1` recovers the unguided main model.

Design choices for GR00T:

- **Shared backbone.** Finetuning keeps the Cosmos-Reason2 VLM frozen (`tune_llm=False`,
  `tune_visual=False`), so the main model and the guide have identical backbone weights. The
  backbone runs once per inference call; only the action head runs twice per denoising step.
  The server verifies the backbones are bit-identical before sharing.
- **Guide = smaller, shorter-trained action head.** The guide keeps the first 8 of the 32
  pretrained DiT blocks (`--action-head-num-layers 8`). Its DiT has 270.7M parameters versus
  1,082.7M, and the whole action head 808.5M versus 1,620.5M. All other action-head weights
  (state/action encoders, decoder, vlln, VL self-attention) start from the same pretrained
  checkpoint. It is finetuned on the same LIBERO-10 data with the same recipe, for at most
  10k steps against the main model's 79,273.
- **Same normalization.** Guidance mixes velocities in normalized action space, so the server
  also checks that the guide's `statistics.json`, `embodiment_id.json` and normalization
  settings match the main model's.

Code:

| What | Where |
|---|---|
| Guided Euler sampler | `Gr00tN1d7ActionHead.get_action` / `get_action_with_features` in `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` |
| Attach a guide head | `Gr00tN1d7.set_autoguidance` (same file) |
| Load and verify the guide | `Gr00tPolicy(..., guide_model_path=, guidance_scale=)` in `gr00t/policy/gr00t_policy.py` |
| Server flags | `--guide-model-path`, `--guidance-scale` in `gr00t/eval/run_gr00t_server.py` |
| Truncated-DiT finetuning | `--action-head-num-layers` (`gr00t/configs/finetune_config.py`, `gr00t/model/gr00t_n1d7/setup.py`) |
| Tests | `tests/gr00t/model/test_action_head.py::TestActionHeadAutoguidance`, `tests/gr00t/model/test_autoguidance_loading.py` |

## 1. Main model

The main model is the existing LIBERO-10 baseline, trained with:

```bash
uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --embodiment-tag LIBERO_PANDA \
    --num-gpus 1 --output-dir <dir> --max-steps 79273 --save-steps 1000 \
    --global-batch-size 32 --dataloader-num-workers 4 --use_wandb
```

Location:
`/mnt/data/sftp/data/duydt6/workspace/diffusion/Isaac-GR00T-baseline/examples/LIBERO/logs/libero-baseline/2026-09-19_gr00t-n17_baseline_libero-panda_libero-10`.
It scores 471/500 = 94.2% at 50 episodes per task, seed 42. That checkpoint is the default
`MODEL_PATH` of `eval_libero_10.sh`.

## 2. Train the guide

```bash
uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_10_no_noops_1.0.0_lerobot/ --embodiment-tag LIBERO_PANDA \
    --num-gpus 1 \
    --output-dir examples/LIBERO/logs/libero-rtc/$(date +%Y-%m-%d_%H-%M-%S)_gr00t-n17_autoguidance-guide-dit8_libero-panda_libero-10 \
    --max-steps 10000 --save-steps 1000 --save-total-limit 10 \
    --global-batch-size 32 --dataloader-num-workers 4 --use_wandb \
    --action-head-num-layers 8
```

`--save-total-limit 10` keeps every 1k-step checkpoint, so the guide's training length can be
chosen at evaluation time without retraining. The run takes about 45 minutes on one H100
(about 3.8 it/s).

## 3. Evaluate

One-time client environment: `bash gr00t/eval/sim/LIBERO/setup_libero.sh`. That script
deletes and regenerates `~/.libero/config.yaml`, which other checkouts on the same machine
share. If another checkout already set it up with the same LIBERO commit, run the script
without its `rm -rf $HOME/.libero` line.

A single configuration:

```bash
GUIDE_MODEL_PATH=examples/LIBERO/logs/libero-rtc/<guide-run>/checkpoint-5000 GUIDANCE_SCALE=2.0 \
N_EPISODES=50 CUDA_VISIBLE_DEVICES=0 bash examples/LIBERO/eval_libero_10.sh
```

The screening sweep runs guide checkpoints {2k, 5k, 10k} × w ∈ {1.5, 2, 3} at 10 episodes
per task, plus a w=1 control:

```bash
GUIDE_RUN_DIR=examples/LIBERO/logs/libero-rtc/<guide-run> CUDA_VISIBLE_DEVICES=0 \
    bash examples/LIBERO/sweep_autoguidance_libero_10.sh
```

Results go to `examples/LIBERO/logs/eval_results/<run-name>/{summary.json,eval_config.json}`.
The sweep table goes to `examples/LIBERO/logs/eval_results/sweep_*.json`.

### Open-loop sweep (no simulator)

`sweep_autoguidance_open_loop_libero_10.sh` compares predicted action chunks with the
demonstrated ones, so a sweep over many guide checkpoints and weights takes minutes instead of
hours. It uses the same environment variables as the closed-loop sweep (closed-loop-only ones
such as `N_ENVS` / `N_EPISODES` are ignored). By default it evaluates every full action chunk of
every trajectory, in the dataset the main model was trained on; set `TRAJ_IDS="0 5 12"` to pick
trajectories, or `DATASET_PATH` to evaluate elsewhere.

```bash
GUIDE_RUN_DIR=<guide run dir> MODEL_PATH=<main run dir> GUIDE_LABEL=autoguidance-dit16 \
GUIDE_STEPS="202 404 808" SCALES="1.1 1.5 2.0 2.5 3.0" CUDA_VISIBLE_DEVICES=0 \
    bash examples/LIBERO/sweep_autoguidance_open_loop_libero_10.sh
```

The backbone runs once per batch and every configuration is sampled with the same noise, so
configurations are compared on identical chunks and identical noise. Per configuration
(`main`, `guide<step>_w0` = the guide alone, `guide<step>_w<w>`) it writes `per_step.csv` /
`per_step.npz` (mse and mae for every chunk, plus mse/mae per action-horizon step and per action
key) and `summary.json`; `sweep_summary.json/.csv` has one row per configuration:

| Metric | Meaning |
|---|---|
| `avg_mse`, `avg_mae` | mean over all chunks (physical action units); `*_per_traj` averages per trajectory first |
| `avg_arm_*`, `avg_gripper_*`, `gripper_sign_acc` | error split into arm and gripper dimensions |
| `delta_mse_pct_vs_main`, `delta_mse_se_by_traj` | change relative to the unguided model, with a standard error across trajectories |
| `win_rate_vs_main`, `trajectory_win_rate_vs_main` | fraction of chunks / trajectories where guidance lowers the error |
| `arm_win_rate_vs_main`, `gripper_win_rate_vs_main` | the same win rate computed on arm-only / gripper-only error (exact ties, common for the gripper, count as half a win) |
| `avg_shift_vs_main` | mean absolute change of the predicted actions caused by guidance |
| `arm_magnitude_ratio`, `arm_jerk_ratio` | predicted vs demonstrated action size and smoothness (1.0 = same) |

In the closed-loop runs above, open-loop error was neutral up to w≈1.25 and rose sharply at
w ≥ 1.5, matching the drop in success rate.

Caveat on screening: 100 episodes per configuration gives a standard error of about
2.4 points at a 94% success rate. The screen only separates clearly good settings from clearly
bad ones. Confirm candidates at 50 episodes per task (500 episodes, standard error about
1 point). All runs use the same seed, so the initial states are paired across configurations.

## Alternative: Action Coherence Guidance (ACG)

ACG (Park & Kim et al., ICRA 2026, arXiv:2510.22201) uses the same sampler update as autoguidance,
`v = v_bad + w * (v_main - v_bad)` at each Euler step, but needs no guide model: `v_bad` comes from
the main model itself, with the attention map of a few DiT **self-attention** blocks replaced by the
identity (`out = to_out(to_v(x))`), so action tokens stop attending to each other and the prediction
loses temporal coherence. Implementation: `gr00t/model/modules/acg.py` (identity processor, swapped in
only for the bad forward pass and restored afterwards), `Gr00tN1d7.set_acg`, and the `acg_blocks`
argument of `Gr00tN1d7ActionHead.get_action`. ACG and autoguidance are mutually exclusive.

Blocks must be odd-indexed (self-attention; even blocks are cross-attention). The default
`15 19 23` is the reference's blocks 7, 9, 11 of 16 at the same relative depth in the 32-block DiT.
The reference samples with 16 Euler steps; GR00T N1.7 uses 4 here.

```bash
# Closed loop, switch from autoguidance to ACG
GUIDANCE_METHOD=acg GUIDANCE_SCALE=3.0 ACG_SKIP_BLOCKS="15 19 23" \
  bash examples/LIBERO/eval_libero_10.sh
# Scale sweep with a w=1 control
N_EPISODES=10 SCALES="1.5 2.0 3.0" bash examples/LIBERO/sweep_acg_libero_10.sh
# Server
uv run python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag LIBERO_PANDA \
  --guidance-method acg --guidance-scale 3.0 --acg-skip-blocks 15 19 23
# Open loop (adds configs acg_w<w> next to any autoguidance configs; GUIDE_RUN_DIR is optional)
ACG_SKIP_BLOCKS="15 19 23" SCALES="1.5 2.0 3.0" bash examples/LIBERO/sweep_autoguidance_open_loop_libero_10.sh
```

In Python: `Gr00tPolicy(..., acg_skip_blocks=[15, 19, 23], guidance_scale=3.0)`.
No ACG results are recorded here yet.

## Results

Main model: the LIBERO-10 baseline above. Guide: 8-block DiT run
`examples/LIBERO/logs/libero-rtc/2026-10-08_16-37-44_gr00t-n17_autoguidance-guide-dit8_libero-panda_libero-10`
(flow-matching loss ≈0.160 at 2k steps, 0.128 at 5k and 0.099 at 10k).

### Screening (10 episodes per task, 100 total, seed 42)

| guide \ w | 1.1 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|
| 2k steps | – | – | 74/101 (73.3%) | 58/101 (57.4%) | 26/103 (25.2%) |
| 5k steps | 93/100 (93.0%) | 95/100 (95.0%) | 90/102 (88.2%) | 61/100 (61.0%) | 36/102 (35.3%) |
| 10k steps | 94/100 (94.0%) | 88/100 (88.0%) | 90/100 (90.0%) | 74/102 (72.5%) | 45/100 (45.0%) |

The unguided control (w=1) on the same seeds scored **97/100 (97.0%)**.

Success falls steadily as w grows, and falls faster with a weaker (less-trained) guide. The
failures are timeouts at the 720-step limit, concentrated in the two-stage tasks: the policy
slows down or stalls rather than crashing. At w ≤ 1.25 every configuration lands between 88%
and 95%, within the ±2–3 point noise of a 100-episode screen.

### Open-loop action error

Mean squared error against the demonstrated 16-step action chunks, in physical action units,
on 61 training frames from 10 episodes:

| guide \ w | 0 (guide only) | 1 (main) | 1.1 | 1.25 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|---|
| 2k | 0.0373 | 0.0088 | 0.0089 | 0.0087 | 0.0094 | 0.0136 | 0.0200 |
| 5k | 0.0268 | 0.0088 | 0.0086 | 0.0085 | 0.0092 | 0.0112 | 0.0190 |
| 10k | 0.0179 | 0.0088 | 0.0087 | 0.0085 | 0.0086 | 0.0115 | 0.0161 |

The guide is 2–4× worse than the main model on its own, which is the intended "bad version".
Guidance is about neutral up to w≈1.25 and moves actions away from the demonstrations at
w ≥ 1.5. Mean arm-action magnitude barely changes (0.142 at w=1, 0.145–0.154 at w=3; ground
truth 0.140). So the harm comes from changing the direction and shape of the action chunks,
not from scaling them up.

### Full evaluation (50 episodes per task, seed 42)

These used the two best low-weight configurations from the screen, with the same script, seed
and episode count as the baseline.

| Task | Baseline (w=1) | 10k guide, w=1.1 | 5k guide, w=1.25 |
|---|---|---|---|
| `LIVING_ROOM_SCENE2_..._alphabet_soup_and_the_tomato_sauce...` | 47/50 | 48/50 | 46/50 |
| `LIVING_ROOM_SCENE2_..._cream_cheese_box_and_the_butter...` | 48/50 | 48/51 | 49/50 |
| `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it` | 47/50 | 47/50 | 46/50 |
| `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer...` | 50/50 | 50/50 | 50/50 |
| `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate...` | 48/50 | 47/51 | 46/50 |
| `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment...` | 48/50 | 45/50 | 46/50 |
| `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate...` | 45/50 | 43/50 | 45/50 |
| `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box...` | 48/50 | 49/51 | 48/50 |
| `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove` | 45/50 | 46/50 | **33/50** |
| `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave...` | 45/50 | 44/50 | 43/50 |
| **Overall** | **471/500 (94.2%)** | **467/503 (92.8%)** | **452/500 (90.4%)** |
| Difference from baseline (z) | – | −1.4 points (z = −0.87, not significant) | −3.8 points (z = −2.26, p ≈ 0.02) |

Results are in `examples/LIBERO/logs/eval_results/<run-name>/summary.json`. The baseline row
comes from `.../2026-09-19_gr00t-n17_baseline_libero-panda_libero-10/eval_results/summary.json`.

### Conclusion

On LIBERO-10, autoguidance did **not** improve GR00T N1.7 with this guide.

- **At w=1.1 it is statistically indistinguishable from no guidance.** The score was 92.8%
  against 94.2%.
- **At w=1.25 it is significantly worse**, driven by `KITCHEN_SCENE8` (33/50 against 45/50).
- **At w ≥ 1.5 it is clearly harmful**, more so the weaker the guide.

This differs from the image-generation results in the paper. Likely reasons:

1. **The main model is already near ceiling** on its training-task distribution: 94% success,
   and its open-loop error is 2–4× lower than the guide's. That leaves little of the
   "outlier / under-fit" error that autoguidance corrects.
2. **Closed-loop control compounds small action errors.** A change in action-chunk shape that
   would be harmless in a single image causes stalls over a 720-step episode.
3. **The sampler is very short.** GR00T uses only 4 Euler steps, versus EDM2's 32-step Heun
   sampler, so each guided step makes a large, uncorrected move.

Directions not tried here:
- a guide degraded only in capacity (fewer DiT blocks trained for the full 79k steps), or only
  in training time (a full-size head trained briefly);
- applying guidance only on part of the noise schedule (a guidance interval);
- more denoising steps when guiding;
- evaluating on held-out or perturbed scenes, where the main model is not at ceiling.
