# Correction Training for the GR00T N1.7 action head

This is an implementation of *Correction Training for Diffusion Models* (arXiv 2610.14136) for
the flow-matching action head. The reference implementation is in `../correction-training`.

## Method

GR00T's action head is trained with flow matching. Time t=0 is noise and t=1 is data:

```
z = (1 - t) * noise + t * a          target velocity v = a - noise
```

Correction training adds a **corrector** C. It is a full copy of the action head's state
encoder, action encoder, position embedding, DiT and action decoder, initialized from the
pretrained head. C refines the denoiser's own prediction, with stop-gradient, and is
supervised with the same target. This setup matches the reference repo's SiT recipe
(`data_time=1`, v-prediction):

```
v0     = D(z, t, cond)                       # denoiser velocity
x_hat  = z + (1 - t) * sg(v0)                # clean-action proposal
v1     = sg(v0) + g * (C(sg(x_hat), t, cond) - sg(v0))   # gate g initialized to 0
loss   = base_loss_weight * ||v0 - v||^2 + correction_loss_weight * ||v1 - v||^2
```

- **Gate.** Because g starts at 0, the corrected policy is identical to the base policy at
  the start of fine-tuning.
- **Stop-gradient.** The correction loss sends no gradient into the denoiser, the VLM
  features or the `vlln`.
- **Conditioning.** The corrector sees the same VL features, a state embedding from its own
  state encoder with the same state-dropout mask, and the same timestep.

At inference, C is applied K times inside every Euler step, with z and t held fixed:

```
for each denoising step:
    v = D(z, t)
    repeat K times:  v = sg(v) + g * (C(z + (1 - t) * v, t) - v)
    z = z + dt * v
```

The paper trains with K_train = 1 and samples with larger K.

## Training

```bash
bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <dataset> --embodiment-tag <tag> --output-dir <out> \
  -- --use-correction
```

When the base checkpoint has no corrector weights, the corrector is initialized from the
loaded action head automatically. Training logs `base_loss`, `correction_loss` and
`correction_gate`.

| Flag (`launch_finetune.py`) | Default | Meaning |
|---|---|---|
| `--use-correction` | off | Enable correction training |
| `--correction-loss-weight` | 1.0 | λ_corr. The paper uses D:C = 1:1 |
| `--correction-train-steps` | 1 | K_train, the corrections unrolled in training |
| `--correction-freeze-denoiser` | off | Freeze the action-head denoiser and train only C. The backbone still follows `--tune-llm` and `--tune-visual` |
| `--num-inference-corrections` | 2 | Default K saved into the checkpoint config |
| `--correction-interval LO HI` | 0 1 | Correct only at steps with flow time in [LO, HI). Time 0 is noise, so the paper's "early, high-noise" steps are near 0 |

The model config also has `base_loss_weight`, `correction_gate_init` and
`correction_gate_shape` (`scalar` or `channel`). `launch_train.py` exposes them as
`--model.<field>`.

## Inference

K can be overridden without re-saving the checkpoint:

```bash
python gr00t/eval/run_gr00t_server.py --model-path <ckpt> --embodiment-tag <tag> --num-corrections 4
python gr00t/eval/open_loop_eval.py   --model-path <ckpt> --embodiment-tag <tag> --num-corrections 4
```

In Python, use `Gr00tPolicy(..., num_corrections=K)` or
`policy.get_action(obs, options={"num_corrections": K})`. `K=0` recovers the base denoiser.

The server also accepts `--correction-interval LO HI` and `--denoising-steps N`.

## LIBERO-10 hyperparameter sweep

`examples/LIBERO/sweep_correction_libero_10.sh` evaluates every combination of checkpoint, K,
correction interval and denoising steps. Each combination runs through
`examples/LIBERO/eval_libero_10.sh`, and the sweep writes a success-rate table as JSON and CSV.
Training-time settings such as the loss weight, K_train or a frozen denoiser are compared by
listing several training runs in `RUN_DIRS`; each checkpoint's settings appear in the table.

```bash
RUN_DIRS="examples/LIBERO/logs/libero-subset-lerobot-10/<correction run>" \
CKPT_STEPS="3232 6464" NUM_CORRECTIONS="0 1 2 4" INTERVALS="0.0,1.0 0.0,0.5" \
BASELINE_MODEL_PATH=<non-correction checkpoint> N_EPISODES=10 \
  bash examples/LIBERO/sweep_correction_libero_10.sh
```

- **Control.** K=0 evaluates the same checkpoint's base denoiser, so it is the built-in control.
- **GPUs.** To split a sweep across GPUs, run one shard per GPU with `NUM_SHARDS`, `SHARD_ID`,
  `CUDA_VISIBLE_DEVICES`, `MUJOCO_EGL_DEVICE_ID` and `PORT`. Then run with `SUMMARY_ONLY=1` for
  the combined table.
- **Preview and resume.** `DRY_RUN=1` lists the configurations without running them. Finished
  configurations are skipped, so an interrupted sweep resumes where it stopped.

### Open-loop sweep on held-out demonstrations

`examples/LIBERO/sweep_correction_open_loop_libero_10.sh` sweeps the same grid without a
simulator, against a held-out split (default: the LIBERO-10 `long_90%` split). For each
checkpoint it runs `gr00t/eval/open_loop_eval_correction.py` once, and that run evaluates every
setting of K, interval and denoising steps.
- **Shared backbone.** The backbone runs once per batch and is shared by every setting.
- **Same noise.** Every setting and every checkpoint is sampled with the same noise on the same
  chunks, so all comparisons are paired.
- **Metrics.** The metrics and outputs match `sweep_autoguidance_open_loop_libero_10.sh`:
  - `avg_mse` and `avg_mae` over every full action chunk.
  - Arm and gripper MSE, and gripper sign accuracy.
  - Magnitude and jerk ratios against the demonstrations.
  - MSE and MAE per horizon step and per action key.
  - Per-chunk `per_step.csv` / `.npz`.
- **Paired comparisons.** These are made against the same checkpoint's base denoiser `K0_d<d>`:
  - `delta_mse_pct_vs_k0`, with its standard error over trajectories.
  - Win rates (all, arm, gripper and per trajectory), where exact ties count as half.
  - `avg_shift_vs_k0`.
- **Baseline columns.** The combined table also compares each setting against the first
  `BASELINE_MODEL_PATHS` checkpoint (`b_mse%`, `b_win`).

```bash
RUN_DIRS="examples/LIBERO/logs/libero-subset-lerobot-10/<correction run>" \
CKPT_STEPS="3232 final" NUM_CORRECTIONS="0 1 2 4" INTERVALS="0.0,1.0 0.0,0.5" \
BASELINE_MODEL_PATHS=<non-correction checkpoint> \
  bash examples/LIBERO/sweep_correction_open_loop_libero_10.sh
```

Results go to `examples/LIBERO/logs/open_loop_results/open-loop_<SWEEP_NAME>_seed<SEED>/`, with
one directory per checkpoint plus a combined `sweep_summary.json` / `.csv`. All trajectories are
evaluated unless `TRAJ_IDS` or `MAX_STEPS_PER_TRAJ` is set. `NUM_SHARDS` / `SHARD_ID` split the
checkpoints across GPUs, `SUMMARY_ONLY=1` rebuilds the table and finished checkpoints are skipped.

Each corrected denoising step costs `1 + K` DiT evaluations. The TensorRT `dit_only`
deployment path replaces `get_action_with_features` and does not run the corrector.
