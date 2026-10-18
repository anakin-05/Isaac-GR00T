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

"""Open-loop evaluation sweep for autoguidance.

Compares predicted action chunks against the demonstrated ones for the unguided main model,
each guide model on its own (w=0), and every (guide checkpoint, guidance scale w) pair.

The backbone runs once per batch and is shared by every configuration, and every
configuration is sampled with the same noise, so the comparison between configurations is
paired. By default every full action chunk of every trajectory in the dataset is evaluated;
``--traj-ids`` restricts the trajectories.

Outputs (under ``--output-dir``):
    eval_config.json                 what was evaluated
    sweep_summary.json / .csv        one row of aggregate metrics per configuration
    <config>/per_step.csv            mse / mae and related metrics for every evaluated chunk
    <config>/per_step.npz            the same, plus mse/mae per action-horizon step and per key
    <config>/summary.json            aggregate metrics for that configuration

Example:
    uv run python gr00t/eval/open_loop_autoguidance_eval.py \\
        --model-path <main run dir> --guide-run-dir <guide run dir> --guide-steps 202 404 808 \\
        --scales 1.1 1.5 2.0 --output-dir examples/LIBERO/logs/eval_results/open-loop_demo
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
from gr00t.model.modules.acg import validate_acg_blocks
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype, _resolve_processor_dir
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature
import tyro


MAIN = "main"


@dataclass
class ArgsConfig:
    model_path: str
    """Main model checkpoint directory."""

    output_dir: str
    """Where results are written."""

    guide_run_dir: str | None = None
    """Training output directory of the guide model; checkpoints are read from
    ``<guide_run_dir>/checkpoint-<step>``. Omit (with no --guide-steps) to evaluate only the
    unguided main model."""

    guide_steps: list[int] = field(default_factory=list)
    """Guide checkpoint steps to evaluate. Empty: only the unguided main model."""

    scales: list[float] = field(default_factory=lambda: [1.1, 1.5, 2.0, 2.5, 3.0])
    """Guidance weights w. w=1 is the unguided main model and is always evaluated."""

    acg_skip_blocks: list[int] | None = None
    """Also evaluate Action Coherence Guidance (config ``acg_w<w>`` for every w in --scales):
    self-attention (odd) DiT blocks whose attention map is replaced by identity for the bad
    prediction. Needs no guide model and can be combined with --guide-steps."""

    afil_alphas: list[float] = field(default_factory=list)
    """Treat each --guide-steps checkpoint as a *failure* model (Adaptive Failure-Informed Learning)
    and evaluate configs ``guide<step>_afil<alpha>`` with lambda = alpha * (1 - cos(v_succ, v_fail))
    for every alpha listed. The usual fixed-scale configs still follow --scales."""

    include_guide_only: bool = True
    """Also evaluate each guide alone (w=0), to show how much worse the 'bad version' is."""

    dataset_path: str | None = None
    """LeRobot dataset to evaluate on. Defaults to the dataset the main model was trained on
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
    """Sampling-noise seed; the same noise is used for every configuration."""

    device: str = "cuda"

    skip_existing: bool = True
    """Do nothing if ``<output_dir>/sweep_summary.json`` already exists."""


def infer_dataset_path(model_path: str) -> str:
    conf = Path(model_path) / "experiment_cfg" / "conf.yaml"
    if not conf.exists():
        raise FileNotFoundError(f"{conf} not found; pass --dataset-path explicitly")
    return OmegaConf.load(conf).data.datasets[0].dataset_paths[0]


def config_name(guide_step: int | None, scale: float, acg: bool = False, afil: bool = False) -> str:
    if afil:
        return f"guide{guide_step}_afil{scale:g}"
    if acg:
        return f"acg_w{scale:g}"
    return MAIN if guide_step is None else f"guide{guide_step}_w{scale:g}"


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
        # Mean absolute change of the predicted chunk relative to the unguided main model.
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
    main_m: dict[str, np.ndarray],
    traj_id: np.ndarray,
    key_names: list[str],
) -> dict:
    """Aggregate metrics of one configuration over all evaluated chunks."""
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
        "avg_shift_vs_main": _mean(m["shift"]),
    }
    if name != MAIN:
        # Paired comparison against the unguided main model (same chunks, same noise).
        d = m["mse"] - main_m["mse"]
        d_traj = per_traj(d)
        out.update(
            {
                "delta_mse_vs_main": float(d.mean()),
                "delta_mse_pct_vs_main": 100.0 * float(d.mean()) / _mean(main_m["mse"]),
                "delta_mae_vs_main": float((m["mae"] - main_m["mae"]).mean()),
                "delta_mse_se_by_traj": float(d_traj.std(ddof=1) / math.sqrt(len(d_traj)))
                if len(d_traj) > 1
                else float("nan"),
                "win_rate_vs_main": float((m["mse"] < main_m["mse"]).mean()),
                "arm_win_rate_vs_main": float((m["arm_mse"] < main_m["arm_mse"]).mean()),
                # Gripper error is often exactly 0 for both models; count those chunks as ties
                # (half a win) so the rate is not biased toward 0.
                "gripper_win_rate_vs_main": _win_rate(m["grip_mse"], main_m["grip_mse"]),
                "trajectory_win_rate_vs_main": float((d_traj < 0).mean()),
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

    if args.guide_steps and args.guide_run_dir is None:
        raise ValueError("--guide-steps requires --guide-run-dir")
    if args.afil_alphas and not args.guide_steps:
        raise ValueError("--afil-alphas needs the failure model checkpoints in --guide-steps")
    tag = EmbodimentTag.resolve(args.embodiment_tag)
    dataset_path = args.dataset_path or infer_dataset_path(args.model_path)
    print(f"Main model: {args.model_path}\nDataset: {dataset_path}", flush=True)

    policy = Gr00tPolicy(tag, args.model_path, device=args.device)
    model = policy.model
    processor_dir = _resolve_processor_dir(Path(args.model_path))
    guides = {}
    for step in args.guide_steps:
        # Reuses the policy's backbone / normalization compatibility checks.
        policy._attach_autoguidance(
            Path(args.guide_run_dir) / f"checkpoint-{step}",
            1.0,
            device=args.device,
            processor_dir=processor_dir,
        )
        guides[step] = model.autoguidance_head
    model.set_autoguidance(None, 1.0)
    acg_blocks = None
    if args.acg_skip_blocks is not None:
        acg_blocks = validate_acg_blocks(
            args.acg_skip_blocks, model.action_head.model.config.num_layers
        )

    loader = LeRobotEpisodeLoader(
        dataset_path=dataset_path, modality_configs=policy.get_modality_config()
    )
    action_cfg = loader.modality_configs["action"]
    action_keys = action_cfg.modality_keys
    horizon = len(action_cfg.delta_indices)
    # AFIL takes its cosine over the real (unpadded) action chunk.
    afil_region = (horizon, policy.processor.state_action_processor.get_action_dim(tag.value))
    stride = args.stride or horizon
    traj_ids = args.traj_ids if args.traj_ids is not None else list(range(len(loader)))
    bad = [t for t in traj_ids if not 0 <= t < len(loader)]
    if bad:
        raise ValueError(f"traj ids {bad} out of range for a dataset with {len(loader)} episodes")

    # Configurations: unguided main, then per guide the guide alone and every w != 1.
    # (name, guide step, w, ACG blocks, AFIL); ACG configs perturb the main model itself.
    configs: list[tuple[str, int | None, float, list[int] | None, bool]] = [
        (MAIN, None, 1.0, None, False)
    ]
    for step in args.guide_steps:
        if args.include_guide_only:
            configs.append((config_name(step, 0.0), step, 0.0, None, False))
        for w in args.scales:
            if w != 1.0:  # identical to the main model
                configs.append((config_name(step, w), step, w, None, False))
        for alpha in args.afil_alphas:
            if alpha != 0.0:  # identical to the main model
                configs.append((config_name(step, alpha, afil=True), step, alpha, None, True))
    if acg_blocks is not None:
        for w in args.scales:
            if w != 1.0:
                configs.append((config_name(None, w, acg=True), None, w, acg_blocks, False))

    (out_dir / "eval_config.json").write_text(
        json.dumps(
            {**vars(args), "dataset_path": dataset_path, "configs": [c[0] for c in configs]},
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
            backbone_out = model.backbone(backbone_inputs)  # shared by every configuration
            main_pred = None
            for name, step, w, blocks, afil in configs:
                torch.manual_seed(args.seed + batch_index)  # same noise for every configuration
                # get_action overwrites backbone_features in place (vlln), so pass a copy.
                out = model.action_head.get_action(
                    BatchFeature(data=dict(backbone_out)),
                    action_inputs,
                    guide=None if step is None else guides[step],
                    guidance_scale=w,
                    acg_blocks=blocks,
                    adaptive_region=afil_region if afil else None,
                )
                pred = decode(out["action_pred"])
                if name == MAIN:
                    main_pred = pred
                results[name].append(
                    chunk_metrics(
                        pred,
                        gt,
                        None if name == MAIN else main_pred,
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
    print(f"Evaluated {n_chunks} chunks x {len(configs)} configs in {time.time() - t0:.0f}s")

    meta = {"traj_id": np.concatenate(meta_traj), "step": np.concatenate(meta_step)}
    merged = {
        name: {k: np.concatenate([b[k] for b in parts]) for k in parts[0]}
        for name, parts in results.items()
    }
    rows = []
    for name, *_ in configs:
        summary = summarize(name, merged[name], merged[MAIN], meta["traj_id"], action_keys)
        save_config_results(out_dir, name, merged[name], meta, summary)
        rows.append(summary)
    (out_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2))
    flat_cols = [k for k, v in rows[-1].items() if not isinstance(v, (list, dict))]
    with open(out_dir / "sweep_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flat_cols, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)

    print(
        f"\n=== Open-loop autoguidance sweep: {n_chunks} chunks, {len(set(meta['traj_id']))} trajectories ==="
    )
    print(
        f"{'config':<18s} {'avg_mse':>9s} {'avg_mae':>9s} {'d_mse%':>8s} {'win':>6s} {'a_win':>6s} {'g_win':>6s} {'shift':>8s} {'arm_mse':>9s} {'grip_mse':>9s} {'mag':>6s} {'jerk':>6s}"
    )
    for r in rows:
        print(
            f"{r['config']:<18s} {r['avg_mse']:>9.5f} {r['avg_mae']:>9.5f} "
            f"{r.get('delta_mse_pct_vs_main', 0.0):>+8.1f} {r.get('win_rate_vs_main', float('nan')):>6.2f} "
            f"{r.get('arm_win_rate_vs_main', float('nan')):>6.2f} "
            f"{r.get('gripper_win_rate_vs_main', float('nan')):>6.2f} "
            f"{r['avg_shift_vs_main']:>8.4f} {r['avg_arm_mse']:>9.5f} {r['avg_gripper_mse']:>9.5f} "
            f"{r['arm_magnitude_ratio']:>6.2f} {r['arm_jerk_ratio']:>6.2f}"
        )
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
