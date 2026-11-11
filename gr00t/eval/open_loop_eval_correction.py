# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open-loop evaluation sweep for correction training.

Compares predicted action chunks against the demonstrated ones for one checkpoint under every
inference setting: K corrector applications per denoising step (``--num-corrections``), the
flow-time interval where they are applied (``--intervals``) and the number of denoising steps
(``--denoising-steps``). ``K0_d<d>`` is the checkpoint's own base denoiser and is the reference
for every setting with the same number of denoising steps. A checkpoint trained without
correction only gets the ``K0_d<d>`` settings, so it can serve as a baseline.

The backbone runs once per batch and is shared by every setting, and every setting is sampled
with the same noise (seed + batch index), so the comparison between settings is paired. Two
checkpoints evaluated with the same seed, dataset and chunks also see the same noise, so their
per-chunk results can be paired too (examples/LIBERO/sweep_correction_open_loop_libero_10.sh does
this against a baseline). By default every full action chunk of every trajectory in the dataset
is evaluated; ``--traj-ids`` restricts the trajectories.

Outputs (under ``--output-dir``), in the same format as open_loop_autoguidance_eval.py:
    eval_config.json                 what was evaluated
    sweep_summary.json / .csv        one row of aggregate metrics per setting
    <setting>/per_step.csv           mse / mae and related metrics for every evaluated chunk
    <setting>/per_step.npz           the same, plus mse/mae per action-horizon step and per key
    <setting>/summary.json           aggregate metrics for that setting

Example:
    uv run python gr00t/eval/open_loop_eval_correction.py \\
        --model-path <correction checkpoint> --dataset-path <held-out dataset> \\
        --num-corrections 0 1 2 4 --intervals 0.0,1.0 0.0,0.5 \\
        --output-dir examples/LIBERO/logs/open_loop_results/demo
"""

import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
import tyro


TRAIN_KEYS = (
    "use_correction",
    "correction_loss_weight",
    "base_loss_weight",
    "correction_train_steps",
    "correction_freeze_denoiser",
    "correction_gate_init",
    "correction_gate_shape",
)


@dataclass
class ArgsConfig:
    model_path: str
    """Checkpoint directory (trained with or without --use-correction)."""

    output_dir: str
    """Where results are written."""

    num_corrections: list[int] = field(default_factory=lambda: [0, 1, 2, 4])
    """K values to evaluate. K=0 (the base denoiser) is always evaluated and ignores the interval.
    Ignored for checkpoints without a corrector."""

    intervals: list[str] = field(default_factory=lambda: ["0.0,1.0"])
    """Correction intervals "lo,hi": correct only at denoising steps with flow time
    (0 = noise, 1 = data) in [lo, hi)."""

    denoising_steps: list[int] = field(default_factory=lambda: [4])
    """Numbers of flow-matching denoising steps."""

    dataset_path: str | None = None
    """LeRobot dataset to evaluate on. Defaults to the dataset the model was trained on
    (read from ``<model_path>/experiment_cfg/conf.yaml``)."""

    embodiment_tag: str = "LIBERO_PANDA"

    traj_ids: list[int] | None = None
    """Trajectory (episode) indices to evaluate. Default: all trajectories."""

    stride: int | None = None
    """Step between evaluated chunks. Default: the action horizon (non-overlapping chunks)."""

    max_steps_per_traj: int | None = None
    """Only evaluate chunks starting before this step of each trajectory. Default: whole trajectory."""

    batch_size: int = 16
    seed: int = 0
    """Sampling-noise seed; the same noise is used for every setting."""

    device: str = "cuda"

    skip_existing: bool = True
    """Do nothing if ``<output_dir>/sweep_summary.json`` already exists."""


def infer_dataset_path(model_path: str) -> str:
    conf = Path(model_path) / "experiment_cfg" / "conf.yaml"
    if not conf.exists():
        raise FileNotFoundError(f"{conf} not found; pass --dataset-path explicitly")
    return OmegaConf.load(conf).data.datasets[0].dataset_paths[0]


def reference_name(denoising_steps: int) -> str:
    return f"K0_d{denoising_steps}"


def config_name(k: int, interval: tuple[float, float] | None, denoising_steps: int) -> str:
    if k == 0:
        return reference_name(denoising_steps)
    return f"K{k}_int{interval[0]:g}-{interval[1]:g}_d{denoising_steps}"


def chunk_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    ref: np.ndarray | None,
    key_slices: list[slice],
    arm_idx: np.ndarray,
    grip_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-chunk metrics. pred/gt/ref: (B, H, D) actions in physical units."""
    h = min(pred.shape[1], gt.shape[1])
    pred, gt = pred[:, :h], gt[:, :h]
    err = pred - gt
    sq, ab = err**2, np.abs(err)
    nan = np.full(len(pred), np.nan, dtype=np.float32)
    has_grip = len(grip_idx) > 0
    return {
        "mse": sq.mean((1, 2)),
        "mae": ab.mean((1, 2)),
        "mse_h": sq.mean(2),
        "mae_h": ab.mean(2),
        "mse_key": np.stack([sq[..., s].mean((1, 2)) for s in key_slices], axis=1),
        "mae_key": np.stack([ab[..., s].mean((1, 2)) for s in key_slices], axis=1),
        "arm_mse": sq[..., arm_idx].mean((1, 2)),
        "arm_mae": ab[..., arm_idx].mean((1, 2)),
        "grip_mse": sq[..., grip_idx].mean((1, 2)) if has_grip else nan,
        "grip_mae": ab[..., grip_idx].mean((1, 2)) if has_grip else nan,
        "grip_sign_acc": (
            (np.sign(pred[..., grip_idx]) == np.sign(gt[..., grip_idx])).mean((1, 2))
            if has_grip
            else nan
        ),
        "pred_abs_arm": np.abs(pred[..., arm_idx]).mean((1, 2)),
        "gt_abs_arm": np.abs(gt[..., arm_idx]).mean((1, 2)),
        "pred_jerk": np.abs(np.diff(pred[..., arm_idx], axis=1)).mean((1, 2)),
        "gt_jerk": np.abs(np.diff(gt[..., arm_idx], axis=1)).mean((1, 2)),
        # Mean absolute change of the predicted chunk relative to the base denoiser (K=0).
        "shift": (
            np.abs(pred - ref[:, :h]).mean((1, 2)) if ref is not None else np.zeros(len(pred))
        ),
    }


def _mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _win_rate(err: np.ndarray, ref: np.ndarray) -> float:
    """Fraction of chunks with lower error than the reference; exact ties count as half."""
    ok = np.isfinite(err) & np.isfinite(ref)
    if not ok.any():
        return float("nan")
    return float(((err[ok] < ref[ok]) + 0.5 * (err[ok] == ref[ok])).mean())


def summarize(
    name: str,
    m: dict[str, np.ndarray],
    ref_m: dict[str, np.ndarray] | None,
    traj_id: np.ndarray,
    key_names: list[str],
) -> dict:
    """Aggregate metrics of one setting over all evaluated chunks."""
    trajs = np.unique(traj_id)
    per_traj = lambda x: np.array([x[traj_id == t].mean() for t in trajs])  # noqa: E731
    out = {
        "config": name,
        "n_chunks": int(len(traj_id)),
        "n_trajectories": int(len(trajs)),
        "avg_mse": _mean(m["mse"]),
        "avg_mae": _mean(m["mae"]),
        "avg_mse_per_traj": float(per_traj(m["mse"]).mean()),
        "avg_mae_per_traj": float(per_traj(m["mae"]).mean()),
        "avg_arm_mse": _mean(m["arm_mse"]),
        "avg_arm_mae": _mean(m["arm_mae"]),
        "avg_gripper_mse": _mean(m["grip_mse"]),
        "avg_gripper_mae": _mean(m["grip_mae"]),
        "gripper_sign_acc": _mean(m["grip_sign_acc"]),
        "mse_per_horizon_step": m["mse_h"].mean(0).tolist(),
        "mae_per_horizon_step": m["mae_h"].mean(0).tolist(),
        "mse_per_key": dict(zip(key_names, m["mse_key"].mean(0).tolist())),
        "mae_per_key": dict(zip(key_names, m["mae_key"].mean(0).tolist())),
        # Action scale and smoothness relative to the demonstrations (1.0 = same as ground truth).
        "arm_magnitude_ratio": _mean(m["pred_abs_arm"]) / _mean(m["gt_abs_arm"]),
        "arm_jerk_ratio": _mean(m["pred_jerk"]) / _mean(m["gt_jerk"]),
        "avg_shift_vs_k0": _mean(m["shift"]),
    }
    if ref_m is not None:
        # Paired comparison against the base denoiser (same chunks, same noise).
        d = m["mse"] - ref_m["mse"]
        d_traj = per_traj(d)
        out.update(
            {
                "delta_mse_vs_k0": float(d.mean()),
                "delta_mse_pct_vs_k0": 100.0 * float(d.mean()) / _mean(ref_m["mse"]),
                "delta_mae_vs_k0": float((m["mae"] - ref_m["mae"]).mean()),
                "delta_mse_se_by_traj": float(d_traj.std(ddof=1) / math.sqrt(len(d_traj)))
                if len(d_traj) > 1
                else float("nan"),
                # Exact ties (e.g. a corrector whose gate is still ~0) count as half a win.
                "win_rate_vs_k0": _win_rate(m["mse"], ref_m["mse"]),
                "arm_win_rate_vs_k0": _win_rate(m["arm_mse"], ref_m["arm_mse"]),
                "gripper_win_rate_vs_k0": _win_rate(m["grip_mse"], ref_m["grip_mse"]),
                "trajectory_win_rate_vs_k0": float((d_traj < 0).mean()),
            }
        )
    return out


def save_config_results(
    out_dir: Path, name: str, m: dict[str, np.ndarray], meta: dict, summary: dict
) -> None:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "per_step.npz", **meta, **m)
    scalar_cols = [k for k, v in m.items() if v.ndim == 1]
    with open(d / "per_step.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["traj_id", "step", *scalar_cols])
        for i in range(len(meta["traj_id"])):
            w.writerow(
                [int(meta["traj_id"][i]), int(meta["step"][i])] + [m[k][i] for k in scalar_cols]
            )
    (d / "summary.json").write_text(json.dumps(summary, indent=2))


def main(args: ArgsConfig) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (out_dir / "sweep_summary.json").exists():
        print(
            f"{out_dir / 'sweep_summary.json'} exists; nothing to do (--no-skip-existing to rerun)"
        )
        return

    tag = EmbodimentTag.resolve(args.embodiment_tag)
    dataset_path = args.dataset_path or infer_dataset_path(args.model_path)
    print(f"Model: {args.model_path}\nDataset: {dataset_path}", flush=True)

    train_cfg = json.loads((Path(args.model_path) / "config.json").read_text())
    policy = Gr00tPolicy(tag, args.model_path, device=args.device)
    model = policy.model
    action_head = model.action_head
    use_correction = getattr(action_head, "corrector", None) is not None
    gate = action_head.corrector.gate.float().mean().item() if use_correction else None
    print(f"use_correction={use_correction} gate={gate}", flush=True)

    # Settings: (name, K, interval, denoising steps). K0_d<d> first, as the reference.
    intervals = [tuple(float(x) for x in iv.split(",")) for iv in args.intervals]
    configs: list[tuple[str, int, tuple[float, float] | None, int]] = []
    for d in args.denoising_steps:
        configs.append((reference_name(d), 0, None, d))
        if not use_correction:
            continue
        for k in args.num_corrections:
            if k == 0:
                continue
            for iv in intervals:
                configs.append((config_name(k, iv, d), k, iv, d))

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path, modality_configs=policy.get_modality_config()
    )
    action_cfg = loader.modality_configs["action"]
    action_keys = action_cfg.modality_keys
    horizon = len(action_cfg.delta_indices)
    stride = args.stride or horizon
    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(loader)))
    bad = [t for t in traj_ids if not 0 <= t < len(loader)]
    if bad:
        raise ValueError(f"traj ids {bad} out of range for a dataset with {len(loader)} episodes")

    (out_dir / "eval_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "dataset_path": dataset_path,
                "use_correction": use_correction,
                "gate": gate,
                "train": {key: train_cfg.get(key) for key in TRAIN_KEYS},
                "configs": [c[0] for c in configs],
            },
            indent=2,
        )
    )

    obs_keys = loader.modality_configs["language"].modality_keys
    key_slices: list[slice] = []
    arm_idx: list[int] = []
    grip_idx: list[int] = []
    results: dict[str, list[dict[str, np.ndarray]]] = {c[0]: [] for c in configs}
    meta_traj: list[np.ndarray] = []
    meta_step: list[np.ndarray] = []
    batch: list[dict] = []
    batch_index = 0
    default_interval = tuple(action_head.config.correction_interval)

    def flush() -> None:
        nonlocal batch, batch_index
        if not batch:
            return
        collated = _rec_to_dtype(
            policy.collate_fn([b["processed"] for b in batch]), dtype=torch.bfloat16
        )
        batched_states = {
            k: np.stack([b["states"][k] for b in batch], axis=0)
            for k in policy.modality_configs["state"].modality_keys
        }
        gt = np.stack([b["gt"] for b in batch], axis=0)

        def decode(normalized: torch.Tensor) -> np.ndarray:
            decoded = policy.processor.decode_action(
                normalized.float().cpu().numpy(), tag, batched_states
            )
            return np.concatenate([decoded[k] for k in action_keys], axis=-1).astype(np.float32)

        with torch.inference_mode():
            backbone_inputs, action_inputs = model.prepare_input(collated["inputs"])
            backbone_out = model.backbone(backbone_inputs)  # shared by every setting
            ref_pred = {}
            for name, k, iv, d in configs:
                torch.manual_seed(args.seed + batch_index)  # same noise for every setting
                action_head.num_inference_timesteps = d
                action_head.config.correction_interval = iv or default_interval
                # get_action overwrites backbone_features in place (vlln), so pass a copy.
                out = action_head.get_action(
                    BatchFeature(data=dict(backbone_out)),
                    action_inputs,
                    {"num_corrections": k} if use_correction else None,
                )
                pred = decode(out["action_pred"])
                if k == 0:
                    ref_pred[d] = pred
                results[name].append(
                    chunk_metrics(
                        pred,
                        gt,
                        None if k == 0 else ref_pred[d],
                        key_slices,
                        np.array(arm_idx),
                        np.array(grip_idx),
                    )
                )
        meta_traj.append(np.array([b["traj_id"] for b in batch]))
        meta_step.append(np.array([b["step"] for b in batch]))
        batch, batch_index = [], batch_index + 1

    t0 = time.time()
    n_chunks = 0
    for traj_id in tqdm(traj_ids, desc="trajectories"):
        traj = loader[traj_id]
        last_start = len(traj) - horizon  # only full chunks have a complete ground truth
        if args.max_steps_per_traj is not None:
            last_start = min(last_start, args.max_steps_per_traj - 1)
        for step in range(0, last_start + 1, stride):
            dp = extract_step_data(traj, step, loader.modality_configs, tag)
            if not key_slices:
                start = 0
                for k in action_keys:
                    dim = dp.actions[k].shape[-1]
                    key_slices.append(slice(start, start + dim))
                    (grip_idx if "gripper" in k else arm_idx).extend(range(start, start + dim))
                    start += dim
            obs = {f"state.{k}": v for k, v in dp.states.items()}
            obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
            for key in obs_keys:
                obs[key] = dp.text
            parsed = parse_observation_gr00t(obs, loader.modality_configs)
            vla = policy._to_vla_step_data(policy._unbatch_observation(parsed)[0])
            processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}])
            batch.append(
                {
                    "processed": processed,
                    "states": vla.states,
                    "gt": np.concatenate([dp.actions[k] for k in action_keys], axis=-1),
                    "traj_id": traj_id,
                    "step": step,
                }
            )
            n_chunks += 1
            if len(batch) == args.batch_size:
                flush()
    flush()
    if n_chunks == 0:
        raise RuntimeError(
            "No chunks were evaluated (trajectories shorter than the action horizon?)"
        )
    print(f"Evaluated {n_chunks} chunks x {len(configs)} settings in {time.time() - t0:.0f}s")

    meta = {"traj_id": np.concatenate(meta_traj), "step": np.concatenate(meta_step)}
    merged = {
        name: {k: np.concatenate([b[k] for b in parts]) for k in parts[0]}
        for name, parts in results.items()
    }
    rows = []
    for name, k, iv, d in configs:
        ref_m = None if k == 0 else merged[reference_name(d)]
        summary = summarize(name, merged[name], ref_m, meta["traj_id"], action_keys)
        summary.update(
            {
                "num_corrections": k if use_correction else None,
                "correction_interval": list(iv) if iv else None,
                "denoising_steps": d,
                "gate": gate,
            }
        )
        save_config_results(out_dir, name, merged[name], meta, summary)
        rows.append(summary)
    (out_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
    flat_cols = list(
        dict.fromkeys(k for r in rows for k, v in r.items() if not isinstance(v, (list, dict)))
    )
    with open(out_dir / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n=== Open-loop correction sweep: {n_chunks} chunks, "
        f"{len(set(meta['traj_id']))} trajectories, gate={gate} ==="
    )
    print(
        f"{'config':<20s} {'avg_mse':>9s} {'avg_mae':>9s} {'d_mse%':>8s} {'win':>6s} {'a_win':>6s} {'g_win':>6s} {'shift':>8s} {'arm_mse':>9s} {'grip_mse':>9s} {'mag':>6s} {'jerk':>6s}"
    )
    for r in rows:
        print(
            f"{r['config']:<20s} {r['avg_mse']:>9.5f} {r['avg_mae']:>9.5f} "
            f"{r.get('delta_mse_pct_vs_k0', 0.0):>+8.1f} {r.get('win_rate_vs_k0', float('nan')):>6.2f} "
            f"{r.get('arm_win_rate_vs_k0', float('nan')):>6.2f} "
            f"{r.get('gripper_win_rate_vs_k0', float('nan')):>6.2f} "
            f"{r['avg_shift_vs_k0']:>8.4f} {r['avg_arm_mse']:>9.5f} {r['avg_gripper_mse']:>9.5f} "
            f"{r['arm_magnitude_ratio']:>6.2f} {r['arm_jerk_ratio']:>6.2f}"
        )
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
