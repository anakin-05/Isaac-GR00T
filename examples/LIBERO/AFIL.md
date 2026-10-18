# Adaptive Failure-Informed Learning (AFIL) for GR00T N1.7 on LIBERO-10

Implements "Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action
Models" (Zheng et al., 2026, arXiv:2605.08434) for GR00T N1.7.

## Method

AFIL keeps a **success** action generator (here: the existing LIBERO-10 baseline) and adds a
**failure** action generator trained only on failed rollouts of the policy. Both share the frozen
VLM backbone (one backbone forward pass; the failure generator is an extra action head). At
sampling time, at every Euler step, the success velocity is pushed away from the failure
velocity, more strongly where the two disagree:

```
lambda = alpha * (1 - cos(v_succ, v_fail))        (cosine per sample over the action chunk)
v      = v_succ + lambda * (v_succ - v_fail)
```

This is Eq. 6 + 8 of the paper. `alpha` is the guidance scale (`--guidance-scale`; paper default
1.0). Where success and failure agree (early reaching) `lambda ~ 0` and sampling is unchanged;
where they diverge the failure direction is repelled. A static scale (the paper's Static-FIL,
lambda = 0.05) is plain autoguidance with `w = 1 + lambda`: use `--guidance-method autoguidance
--guidance-scale 1.05` with the failure model as the guide.

Notes on the implementation:

* The cosine is taken over the real action chunk only (`action horizon x action dim`, 16 x 7 for
  LIBERO), not the padded 40 x 132 model output, so padding does not dilute it.
* The paper's Eq. 7 is written loosely (`eps_succ - lambda_hat * eps_fail`); Eq. 6 is the repulsion
  `eps_succ - lambda * (eps_fail - eps_succ)` with the sign taken so that sampling moves *away*
  from the failure prediction, which is what is implemented: `v_succ + lambda * (v_succ - v_fail)`.
* Normalization must match the main model, because velocities of the two heads are mixed. The
  dataset builder copies the success dataset's `stats.json` / `relative_stats.json`, and the
  policy refuses to load a failure model whose processor statistics differ (same check as
  autoguidance).
* Differences from the paper: failures are used as they are (no motion-planner "failure
  correction" data); the success generator is the already trained baseline instead of being
  trained jointly with the failure one (equivalent at inference since the backbone is frozen).

Code: `Gr00tN1d7ActionHead.get_action_with_features(..., adaptive_region=...)` in
`gr00t/model/gr00t_n1d7/gr00t_n1d7.py`, `Gr00tPolicy(..., guide_model_path=..., guide_is_failure_model=True)`,
server `--guidance-method afil`.

## 1. Collect failures

Roll out the success policy and record every failed episode. An episode is recorded only if it
did not succeed and ran to the step limit. Use a fresh `SEED` (different initial states than your
evaluation) and as many episodes as you need; the baseline succeeds ~94%, so 500 episodes give
only ~30 failures. A weaker checkpoint (e.g. an early one) gives more failures per episode.

```bash
N_EPISODES=50 SEED=1000 RECORD_FAILURES_DIR=examples/LIBERO/logs/failures/baseline_seed1000 \
  OUTPUT_ROOT=examples/LIBERO/logs/eval_results/collect_baseline_seed1000 \
  bash examples/LIBERO/eval_libero_10.sh
```

Each failure is saved as `<dir>/<task>/<id>/{image,wrist_image}.mp4, state.npy, action.npy,
meta.json`; actions are the executed policy actions in the dataset's convention.

## 2. Build the failure dataset

```bash
uv run python examples/LIBERO/build_failure_dataset.py \
  --recordings-dir examples/LIBERO/logs/failures/baseline_seed1000 \
  --template-dataset examples/LIBERO/libero_10_no_noops_1.0.0_lerobot \
  --output-dir examples/LIBERO/libero_10_failures_lerobot
```

Recordings from several runs can be merged by pointing `--recordings-dir` at a directory that
contains them all (it looks for `<task>/<id>/meta.json`).

## 3. Train the failure generator

Same launcher as the main model, on the failure dataset, from the base model (the backbone is
frozen, so it stays identical to the main model's). `--action-head-num-layers N` keeps only the
first N DiT blocks for a lighter failure head (the paper calls it lightweight); omit it for a
full-size head.

```bash
uv run python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path examples/LIBERO/libero_10_failures_lerobot/ --embodiment-tag LIBERO_PANDA \
  --num-gpus 1 --output-dir examples/LIBERO/logs/libero-afil/$(date +%Y-%m-%d_%H-%M-%S)_gr00t-n17_afil-failure-head \
  --max-steps 5000 --save-steps 1000 --global-batch-size 32 --dataloader-num-workers 4 \
  --action-head-num-layers 16
```

## 4. Evaluate

```bash
# one configuration
GUIDANCE_METHOD=afil GUIDE_MODEL_PATH=<failure run>/checkpoint-5000 GUIDANCE_SCALE=1.0 \
  bash examples/LIBERO/eval_libero_10.sh

# sweep failure checkpoints x alphas, with a w=1 control (10 episodes per task); GUIDE_RUN_DIR
# is the failure run, exactly like GUIDE_STEPS x SCALES for autoguidance
GUIDE_RUN_DIR=<failure run> GUIDE_STEPS="1000 3000 5000" AFIL_ALPHAS="0.5 1.0 2.0" \
  N_EPISODES=10 bash examples/LIBERO/sweep_autoguidance_libero_10.sh
# or a single failure checkpoint (AFIL_ALPHAS defaults to "0.5 1.0 2.0")
FAILURE_MODEL_PATH=<failure run>/checkpoint-5000 N_EPISODES=10 \
  bash examples/LIBERO/sweep_autoguidance_libero_10.sh

# open loop (no simulator); adds guide<step>_afil<alpha> configs
MODEL_PATH=<main run> GUIDE_RUN_DIR=<failure run> GUIDE_STEPS="2000 5000" AFIL_ALPHAS="0.5 1.0 2.0" \
  SCALES="1.1" bash examples/LIBERO/sweep_autoguidance_open_loop_libero_10.sh
```

Server: `run_gr00t_server.py --model-path <main> --guidance-method afil --guide-model-path
<failure ckpt> --guidance-scale 1.0`. Python: `Gr00tPolicy(embodiment_tag, <main>, device=0,
guide_model_path=<failure ckpt>, guidance_scale=1.0, guide_is_failure_model=True)`.

Open-loop MSE is measured against successful demonstrations, so it is a rough proxy for AFIL: a
failure-informed push is expected to help closed-loop recovery rather than imitate the demos more
closely.

## Results

None recorded yet.
