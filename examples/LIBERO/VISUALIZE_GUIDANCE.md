# Visualizing guidance on LIBERO

`gr00t/eval/visualize_guidance.py` (wrapper: `examples/LIBERO/visualize_guidance_libero_10.sh`)
reproduces the idea of the 2D toy example of "Guiding a Diffusion Model with a Bad Version of
Itself" on real GR00T N1.7 samples and the LIBERO dataset, for autoguidance, ACG and AFIL.

Instead of training 2D toy models, it takes a few dataset observations, draws many action chunks
per method with identical noise, and projects the flattened (16 x 7) chunks to 2D with a z-scored
PCA fitted on the dataset's action chunks of the same task. The data distribution plays the role of
the toy's ground-truth density.

```bash
MODEL_PATH=<main ckpt> \
GUIDE_CKPT=<guide run>/checkpoint-5000 GUIDE_SCALES="1.5 3.0" \
ACG_SKIP_BLOCKS="15 19 23" ACG_SCALES="2.0 3.0" \
FAILURE_CKPT=<failure run>/checkpoint-5000 AFIL_ALPHAS="1.0 2.0" \
FAILURE_DATASET_PATH=examples/LIBERO/libero_10_failures_lerobot \
  bash examples/LIBERO/visualize_guidance_libero_10.sh
```

Every method is optional; give at least one. The guide and failure checkpoints must share the main
model's normalization (same checks as in the evaluation), so use the dataset the main model was
trained on. `OBSERVATIONS="12:80 33:278"` picks specific `traj:step` frames.

Outputs in `OUTPUT_ROOT` (default `examples/LIBERO/logs/guidance_viz/<run name>`), one row per
observation and one column per method:

* `samples.png` -- guided samples (dots) over the task data (orange highest-density regions, line =
  90% region), the demonstrated chunk (star), the unguided 90% region (dashed grey) and, with
  `FAILURE_DATASET_PATH`, the failure data (red). Each panel lists the metrics below.
* `trajectories.png` -- denoising paths from noise to the final chunk (the toy's trajectories).
* `arrows.png` -- how each method shifts the initial noise at step 0 (the toy's score arrows;
  lengths are scaled per row).
* `summary.png`, `metrics.csv/json` -- metrics per observation and method:
  `dist_to_demo` (RMS distance to the demonstrated chunk, z-scored units), `dist_to_manifold`
  (distance to the nearest dataset chunk of the task), `in_data_hdr90` (share of samples inside the
  data's 2D 90% region), `in_failure_hdr90`, `spread` (diversity) and `shift_vs_main`.
* `samples.npz` -- the decoded samples for further analysis; `config.json`.

Notes: the conditional distribution given one observation is narrow, so the plots are zoomed onto
the samples (`--no-zoom` frames the whole task data). The 2D projection discards most of the
112 dimensions, so read it as an illustration; the metrics are computed in the full chunk space,
except `in_*_hdr90`, which uses the 2D projection.
