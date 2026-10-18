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

"""Visualize what guidance does to the sampled action distribution, like the 2D toy example of
"Guiding a Diffusion Model with a Bad Version of Itself" (Karras et al., 2024), but on real
GR00T N1.7 samples and a real dataset (LIBERO).

For a few dataset observations the script draws many action chunks per method with the same
noise (paired) and projects the flattened chunks to 2D with a PCA fitted on the dataset's action
chunks of the same task. Per observation and method it shows

* the data distribution of that task (orange highest-density regions), the demonstrated chunk
  at that observation (star), optionally the failure distribution (red outline), and the
  unguided samples (grey outline) for reference -- the toy example's ``gt`` / ``samples``;
* the guided samples (``samples.png``), their denoising paths (``trajectories.png``) and the
  initial guidance shift as arrows (``arrows.png``, the toy example's ``scores_ratio``);
* metrics (``metrics.csv/json``, ``summary.png``): distance to the demonstration, distance to
  the data manifold, share of samples inside the data's 2D HDR, sample spread, and the shift
  from the unguided samples.

Methods (each is a column): the unguided main model, autoguidance (``--guide-ckpt``, also the
guide alone), ACG (``--acg-skip-blocks``) and AFIL (``--failure-ckpt``).

Example:
    uv run python gr00t/eval/visualize_guidance.py --model-path <main run> \
        --output-dir examples/LIBERO/logs/guidance_viz/demo \
        --guide-ckpt <guide run>/checkpoint-5000 --guide-scales 1.5 3.0 \
        --acg-skip-blocks 15 19 23 --acg-scales 2.0 3.0 \
        --failure-ckpt <failure run>/checkpoint-5000 --afil-alphas 1.0 2.0
"""

import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
import textwrap

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType
from gr00t.data.utils import parse_observation_gr00t
from gr00t.model.modules.acg import validate_acg_blocks
from gr00t.policy.gr00t_policy import Gr00tPolicy, _rec_to_dtype, _resolve_processor_dir
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers.feature_extraction_utils import BatchFeature  # noqa: E402
import tyro  # noqa: E402


@dataclass
class ArgsConfig:
    model_path: str
    """Main model checkpoint directory."""

    output_dir: str

    dataset_path: str | None = None
    """LeRobot dataset (default: the one the main model was trained on)."""

    embodiment_tag: str = "LIBERO_PANDA"

    guide_ckpt: str | None = None
    """Autoguidance guide checkpoint (a weaker model, e.g. a short-trained smaller DiT)."""
    guide_scales: list[float] = field(default_factory=lambda: [1.5, 3.0])
    """Autoguidance weights w; the guide alone (w=0) is drawn too (--no-show-guide-only to skip)."""
    show_guide_only: bool = True

    acg_skip_blocks: list[int] | None = None
    """Enable ACG with identity attention in these self-attention (odd) DiT blocks."""
    acg_scales: list[float] = field(default_factory=lambda: [2.0, 3.0])

    failure_ckpt: str | None = None
    """AFIL failure-model checkpoint (see examples/LIBERO/AFIL.md)."""
    afil_alphas: list[float] = field(default_factory=lambda: [1.0, 2.0])

    failure_dataset_path: str | None = None
    """Optional failure dataset (build_failure_dataset.py); its chunks are drawn as red outlines."""

    observations: list[str] | None = None
    """Observations as ``traj:step`` (e.g. ``12:80``). Default: --num-observations frames, one per
    task, at a random mid-episode step."""
    num_observations: int = 4

    num_samples: int = 256
    """Action chunks sampled per observation and method."""
    num_trajectories: int = 24
    """Denoising paths drawn per panel in trajectories.png."""
    ref_stride: int = 2
    """Stride of the dataset action chunks forming the reference distribution."""

    zoom: bool = True
    """Frame the plots on the samples/paths/demonstration (the conditional distribution given the
    observation is narrow); --no-zoom frames the whole task data."""

    seed: int = 0
    device: str = "cuda"
    dpi: int = 110


# ---------------------------------------------------------------------------------------------
# Methods


@dataclass
class Method:
    name: str
    kind: str  # main | guide_only | autoguidance | acg | afil
    scale: float = 1.0
    head: object = None
    blocks: list[int] | None = None


def build_methods(args: ArgsConfig, policy: Gr00tPolicy) -> list[Method]:
    model = policy.model
    processor_dir = _resolve_processor_dir(Path(args.model_path))

    def load_head(ckpt: str):
        policy._attach_autoguidance(
            Path(ckpt), 1.0, device=args.device, processor_dir=processor_dir
        )
        head = model.autoguidance_head
        model.set_autoguidance(None, 1.0)
        return head

    methods = [Method("main", "main")]
    if args.guide_ckpt:
        head = load_head(args.guide_ckpt)
        if args.show_guide_only:
            methods.append(Method("guide only", "guide_only", 0.0, head))
        methods += [
            Method(f"autoguidance w={w:g}", "autoguidance", w, head)
            for w in args.guide_scales
            if w != 1.0
        ]
    if args.acg_skip_blocks:
        blocks = validate_acg_blocks(
            args.acg_skip_blocks, model.action_head.model.config.num_layers
        )
        methods += [
            Method(f"ACG w={w:g}", "acg", w, blocks=blocks) for w in args.acg_scales if w != 1.0
        ]
    if args.failure_ckpt:
        head = load_head(args.failure_ckpt)
        methods += [Method(f"AFIL a={a:g}", "afil", a, head) for a in args.afil_alphas if a != 0.0]
    if len(methods) == 1:
        raise ValueError(
            "Nothing to compare: give --guide-ckpt, --acg-skip-blocks or --failure-ckpt"
        )
    return methods


# ---------------------------------------------------------------------------------------------
# Data


def infer_dataset_path(model_path: str) -> str:
    conf = Path(model_path) / "experiment_cfg" / "conf.yaml"
    if not conf.exists():
        raise FileNotFoundError(f"{conf} not found; pass --dataset-path explicitly")
    return OmegaConf.load(conf).data.datasets[0].dataset_paths[0]


def episode_tasks(dataset: Path) -> list[tuple[int, str, int]]:
    """(episode_index, task text, length) per episode, in loader order."""
    rows = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines()]
    return [(r["episode_index"], r["tasks"][0], r["length"]) for r in rows]


def reference_chunks(
    dataset: Path, task: str, horizon: int, stride: int, allow_empty: bool = False
) -> np.ndarray | None:
    """All action chunks [N, horizon * dim] of the episodes of ``task`` (read from parquet)."""
    chunks = []
    for ep_idx, text, _ in episode_tasks(dataset):
        if text != task:
            continue
        path = next(dataset.glob(f"data/chunk-*/episode_{ep_idx:06d}.parquet"))
        actions = np.stack(pd.read_parquet(path, columns=["action"])["action"].to_numpy())
        for s in range(0, len(actions) - horizon + 1, stride):
            chunks.append(actions[s : s + horizon].reshape(-1))
    if not chunks and allow_empty:
        return None
    if not chunks:
        raise ValueError(f"No chunks for task {task!r} in {dataset}")
    return np.asarray(chunks, dtype=np.float32)


def choose_observations(args: ArgsConfig, tasks: list[tuple[int, str, int]], horizon: int):
    if args.observations:
        obs = []
        for item in args.observations:
            traj, step = (int(x) for x in item.split(":"))
            obs.append((traj, step))
        return obs
    rng = np.random.RandomState(args.seed)
    by_task: dict[str, list[int]] = {}
    for i, (_, text, _) in enumerate(tasks):
        by_task.setdefault(text, []).append(i)
    names = sorted(by_task)
    rng.shuffle(names)
    obs = []
    for text in names[: args.num_observations]:
        traj = int(rng.choice(by_task[text]))
        length = tasks[traj][2]
        lo, hi = int(0.15 * length), max(int(0.7 * length), int(0.15 * length) + 1)
        obs.append((traj, int(rng.randint(lo, min(hi, length - horizon)))))
    return obs


# ---------------------------------------------------------------------------------------------
# Sampling


def expand_batch(bf: BatchFeature, n: int) -> BatchFeature:
    return BatchFeature(
        data={
            k: (
                v.repeat_interleave(n, 0)
                if torch.is_tensor(v) and v.ndim > 0 and v.shape[0] == 1
                else v
            )
            for k, v in bf.items()
        }
    )


class Sampler:
    """Draws guided action chunks for one observation with identical noise for all methods."""

    def __init__(self, policy: Gr00tPolicy, tag: EmbodimentTag, action_keys: list[str], args):
        self.policy, self.tag, self.action_keys, self.args = policy, tag, action_keys, args
        horizon = len(policy.modality_configs["action"].delta_indices)
        self.region = (
            horizon,
            policy.processor.state_action_processor.get_action_dim(tag.value),
        )

    def decode(self, normalized: torch.Tensor, states: dict[str, np.ndarray]) -> np.ndarray:
        """[..., H_model, D_model] normalized -> [..., H, 7] physical, with states for every row."""
        lead = normalized.shape[:-2]
        flat = normalized.reshape(-1, *normalized.shape[-2:]).float().cpu().numpy()
        reps = flat.shape[0]
        batched = {k: np.repeat(v[None], reps, axis=0) for k, v in states.items()}
        decoded = self.policy.processor.decode_action(flat, self.tag, batched)
        out = np.concatenate([decoded[k] for k in self.action_keys], axis=-1).astype(np.float32)
        return out.reshape(*lead, *out.shape[-2:])

    @torch.inference_mode()
    def run(self, processed, states, methods: list[Method], n: int, seed: int):
        model = self.policy.model
        collated = _rec_to_dtype(self.policy.collate_fn([processed]), dtype=torch.bfloat16)
        backbone_inputs, action_inputs = model.prepare_input(collated["inputs"])
        backbone_out = expand_batch(model.backbone(backbone_inputs), n)
        action_inputs = expand_batch(action_inputs, n)
        results = {}
        for m in methods:
            torch.manual_seed(seed)  # same noise for every method
            out = model.action_head.get_action(
                BatchFeature(data=dict(backbone_out)),
                action_inputs,
                guide=m.head,
                guidance_scale=m.scale,
                acg_blocks=m.blocks,
                adaptive_region=self.region if m.kind == "afil" else None,
                return_trajectory=True,
            )
            traj = out["action_trajectory"]  # [T+1, n, H, D]
            shift = out["guidance_shift"]  # [T, n, H, D]
            dt = 1.0 / shift.shape[0]
            results[m.name] = {
                "samples": self.decode(out["action_pred"], states),
                "paths": self.decode(traj[:, : self.args.num_trajectories], states),
                # Where the step-0 guidance shift moves the initial noise (decoded affinely).
                "arrow_from": self.decode(traj[0], states),
                "arrow_to": self.decode(traj[0] + dt * shift[0], states),
            }
        return results


# ---------------------------------------------------------------------------------------------
# Geometry and metrics


class Projection:
    """z-scored PCA to 2D, fitted on the reference chunks."""

    def __init__(self, ref: np.ndarray):
        self.mean = ref.mean(0)
        self.std = ref.std(0) + 1e-3
        z = (ref - self.mean) / self.std
        _, _, vt = np.linalg.svd(z[:: max(1, len(z) // 4000)] - z.mean(0), full_matrices=False)
        self.components = vt[:2]

    def zscore(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.zscore(x) @ self.components.T


class DensityRegion:
    """KDE of 2D points with highest-density-region levels (the toy example's contours)."""

    def __init__(self, pts: np.ndarray, max_points: int = 3000, seed: int = 0):
        rng = np.random.RandomState(seed)
        if len(pts) > max_points:
            pts = pts[rng.choice(len(pts), max_points, replace=False)]
        self.kde = gaussian_kde(pts.T, bw_method=0.25)
        self.at_data = np.sort(self.kde(pts.T))

    def density(self, pts: np.ndarray) -> np.ndarray:
        return self.kde(pts.T)

    def level(self, mass: float) -> float:
        """Density threshold of the region containing ``mass`` of the data."""
        return float(self.at_data[int((1.0 - mass) * (len(self.at_data) - 1))])

    def inside(self, pts: np.ndarray, mass: float = 0.9) -> np.ndarray:
        return self.density(pts) >= self.level(mass)


def chunk_metrics(samples, demo, main, ref, proj) -> dict:
    """Metrics for samples [n, H*7]; distances are RMS per element in z-scored units."""
    z = proj.zscore(samples)
    zref = proj.zscore(ref[:: max(1, len(ref) // 600)])
    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    nearest = np.sqrt(((z[:, None] - zref[None]) ** 2).mean(-1)).min(1)
    return {
        "dist_to_demo": rms(z - proj.zscore(demo)[None]),
        "dist_to_manifold": float(nearest.mean()),
        "spread": float(z.std(0).mean()),
        "shift_vs_main": rms(z - proj.zscore(main)) if main is not None else 0.0,
    }


# ---------------------------------------------------------------------------------------------
# Plotting


COLORS = {
    "main": "#555555",
    "guide_only": "#c0392b",
    "autoguidance": "#1b7a3d",
    "acg": "#1f5fa8",
    "afil": "#8e44ad",
}


def view_limits(*arrays, margin=0.12):
    pts = np.concatenate([a.reshape(-1, 2) for a in arrays])
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    pad = (hi - lo) * margin
    return lo - pad, hi + pad


def grid_for(lims, res=160):
    gx = np.linspace(lims[0][0], lims[1][0], res)
    gy = np.linspace(lims[0][1], lims[1][1], res)
    xx, yy = np.meshgrid(gx, gy)
    return xx, yy, np.stack([xx.ravel(), yy.ravel()])


def draw_reference(ax, data_region, fail_region, main_region, lims):
    xx, yy, g = grid_for(lims)
    shape = xx.shape
    d = data_region.kde(g).reshape(shape)
    levels = sorted({data_region.level(m) for m in (0.99, 0.9, 0.7, 0.5, 0.3, 0.1)})
    ax.contourf(xx, yy, d, levels=[levels[0], *levels[1:], d.max() + 1], cmap="Oranges", alpha=0.45)
    ax.contour(xx, yy, d, levels=[data_region.level(0.9)], colors=["#c27c3e"], linewidths=1.2)
    if fail_region is not None:
        f = fail_region.kde(g).reshape(shape)
        ax.contour(xx, yy, f, levels=[fail_region.level(0.9)], colors=["#d62728"], linewidths=1.4)
    if main_region is not None:
        m = main_region.kde(g).reshape(shape)
        ax.contour(
            xx,
            yy,
            m,
            levels=[main_region.level(0.9)],
            colors=["#888888"],
            linewidths=1.2,
            linestyles="dashed",
        )


def style_axis(ax, lims):
    ax.set_xlim(lims[0][0], lims[1][0])
    ax.set_ylim(lims[0][1], lims[1][1])
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_figure(n_rows, methods, title):
    fig, axes = plt.subplots(
        n_rows, len(methods), figsize=(3.1 * len(methods), 3.0 * n_rows + 0.6), squeeze=False
    )
    fig.suptitle(title, fontsize=13)
    return fig, axes


def label_for(method, row_meta=None):
    return method.name


def plot_all(out_dir: Path, observations, methods, per_obs, args):
    legend_note = (
        "orange: task data (HDR 10-99%); orange line: data HDR 90%; dashed grey: unguided HDR 90%; "
        "red line: failure data HDR 90%; star: demonstration"
    )
    for kind, fname, title in (
        ("samples", "samples.png", "Guided samples (dots) in the 2D PCA of the action chunk"),
        ("trajectories", "trajectories.png", "Denoising paths (noise -> action chunk)"),
        (
            "arrows",
            "arrows.png",
            "Step-0 guidance shift of the initial noise (arrows scaled per row, longer = stronger)",
        ),
    ):
        show = [m for m in methods if kind != "arrows" or m.kind != "main"]
        note = textwrap.fill(legend_note, width=max(45, 17 * len(show)))
        fig, axes = make_figure(len(observations), show, f"{title}\n{note}")
        for r, rec in enumerate(per_obs):
            for c, m in enumerate(show):
                ax = axes[r][c]
                draw_reference(
                    ax, rec["data_region"], rec["fail_region"], rec["main_region"], rec["lims"]
                )
                res = rec["results"][m.name]
                color = COLORS[m.kind]
                if kind == "samples":
                    pts = res["proj_samples"]
                    ax.plot(*pts.T, ".", color=color, alpha=0.45, markersize=5, markeredgewidth=0)
                elif kind == "trajectories":
                    paths = res["proj_paths"]  # [T+1, k, 2]
                    ax.plot(
                        paths[:, :, 0], paths[:, :, 1], "-", color=color, alpha=0.35, linewidth=1
                    )
                    ax.plot(*paths[-1].T, ".", color=color, markersize=5)
                else:
                    a, b = res["proj_arrow_from"], res["proj_arrow_to"]
                    # Arrows start at the initial noise; draw the ones nearest the view centre,
                    # with one length scale per row so that methods stay comparable.
                    center = (rec["lims"][0] + rec["lims"][1]) / 2
                    keep = np.argsort(np.linalg.norm(a - center, axis=1))[:50]
                    ax.quiver(
                        a[keep, 0],
                        a[keep, 1],
                        (b - a)[keep, 0],
                        (b - a)[keep, 1],
                        angles="xy",
                        scale_units="xy",
                        scale=rec["arrow_scale"],
                        color=color,
                        width=0.005,
                    )
                ax.plot(
                    *rec["demo_proj"], "*", color="black", markersize=13, markeredgecolor="white"
                )
                style_axis(ax, rec["lims"])
                if r == 0:
                    ax.set_title(label_for(m), fontsize=11, color=color)
                if c == 0:
                    ax.set_ylabel(rec["label"], fontsize=8)
                if kind == "samples":
                    mt = res["metrics"]
                    ax.text(
                        0.02,
                        0.02,
                        f"demo {mt['dist_to_demo']:.2f}  manifold {mt['dist_to_manifold']:.2f}\n"
                        f"in-HDR {mt['in_data_hdr90']:.0%}  spread {mt['spread']:.2f}",
                        transform=ax.transAxes,
                        fontsize=7,
                        va="bottom",
                        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7),
                    )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out_dir / fname, dpi=args.dpi)
        plt.close(fig)


def plot_summary(out_dir: Path, methods, metrics_rows, args):
    keys = [
        ("dist_to_demo", "distance to demonstration (lower)"),
        ("dist_to_manifold", "distance to data manifold (lower)"),
        ("in_data_hdr90", "share inside data HDR 90% (higher)"),
        ("spread", "sample spread (diversity)"),
        ("shift_vs_main", "shift from unguided samples"),
    ]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.8))
    names = [m.name for m in methods]
    for ax, (key, title) in zip(axes, keys):
        vals = [np.mean([r[key] for r in metrics_rows if r["method"] == n]) for n in names]
        errs = [np.std([r[key] for r in metrics_rows if r["method"] == n]) for n in names]
        ax.bar(
            range(len(names)), vals, yerr=errs, color=[COLORS[m.kind] for m in methods], alpha=0.85
        )
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Mean over observations (error bars: std over observations)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "summary.png", dpi=args.dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------


def main(args: ArgsConfig) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = EmbodimentTag.resolve(args.embodiment_tag)
    dataset = Path(args.dataset_path or infer_dataset_path(args.model_path))
    print(f"Main model: {args.model_path}\nDataset: {dataset}", flush=True)

    policy = Gr00tPolicy(tag, args.model_path, device=args.device)
    methods = build_methods(args, policy)
    loader = LeRobotEpisodeLoader(
        dataset_path=str(dataset), modality_configs=policy.get_modality_config()
    )
    action_cfg = loader.modality_configs["action"]
    action_keys = action_cfg.modality_keys
    horizon = len(action_cfg.delta_indices)
    obs_keys = loader.modality_configs["language"].modality_keys
    tasks = episode_tasks(dataset)
    fail_dataset = Path(args.failure_dataset_path) if args.failure_dataset_path else None
    observations = choose_observations(args, tasks, horizon)
    print(f"Observations (traj:step): {observations}", flush=True)

    sampler = Sampler(policy, tag, action_keys, args)
    per_obs, metrics_rows, saved = [], [], {}
    for obs_i, (traj_id, step) in enumerate(tqdm(observations, desc="observations")):
        traj = loader[traj_id]
        if step + horizon > len(traj):
            raise ValueError(
                f"traj {traj_id} has {len(traj)} steps; step {step} leaves no full chunk"
            )
        dp = extract_step_data(traj, step, loader.modality_configs, tag)
        obs = {f"state.{k}": v for k, v in dp.states.items()}
        obs.update({f"video.{k}": np.array(v) for k, v in dp.images.items()})
        for key in obs_keys:
            obs[key] = dp.text
        parsed = parse_observation_gr00t(obs, loader.modality_configs)
        vla = policy._to_vla_step_data(policy._unbatch_observation(parsed)[0])
        processed = policy.processor([{"type": MessageType.EPISODE_STEP.value, "content": vla}])
        demo = np.concatenate([dp.actions[k] for k in action_keys], axis=-1).astype(np.float32)

        results = sampler.run(processed, vla.states, methods, args.num_samples, args.seed + obs_i)

        task = tasks[traj_id][1]
        ref = reference_chunks(dataset, task, horizon, args.ref_stride)
        proj = Projection(ref)
        data_region = DensityRegion(proj(ref))
        fail_ref = (
            reference_chunks(fail_dataset, task, horizon, 1, allow_empty=True)
            if fail_dataset
            else None
        )
        fail_region = None
        if fail_ref is not None and len(fail_ref) >= 20:
            fail_region = DensityRegion(proj(fail_ref))
        main_flat = results["main"]["samples"].reshape(args.num_samples, -1)
        main_region = DensityRegion(proj(main_flat))

        for m in methods:
            res = results[m.name]
            flat = res["samples"].reshape(args.num_samples, -1)
            res["proj_samples"] = proj(flat)
            paths = res["paths"]
            res["proj_paths"] = proj(paths.reshape(*paths.shape[:2], -1))
            res["proj_arrow_from"] = proj(res["arrow_from"].reshape(args.num_samples, -1))
            res["proj_arrow_to"] = proj(res["arrow_to"].reshape(args.num_samples, -1))
            mt = chunk_metrics(flat, demo.reshape(-1), main_flat, ref, proj)
            mt["in_data_hdr90"] = float(data_region.inside(res["proj_samples"], 0.9).mean())
            if fail_region is not None:
                mt["in_failure_hdr90"] = float(fail_region.inside(res["proj_samples"], 0.9).mean())
            res["metrics"] = mt
            metrics_rows.append(
                {"obs": obs_i, "traj": traj_id, "step": step, "task": task, "method": m.name, **mt}
            )
            saved[f"obs{obs_i}/{m.name}/samples"] = res["samples"]
        saved[f"obs{obs_i}/demo"] = demo
        lims = view_limits(
            *([] if args.zoom else [proj(ref)]),
            *[results[m.name]["proj_samples"] for m in methods],
            *[results[m.name]["proj_paths"] for m in methods],
            proj(demo.reshape(1, -1)),
            margin=0.3 if args.zoom else 0.12,
        )
        shifts = np.concatenate(
            [
                np.linalg.norm(
                    results[m.name]["proj_arrow_to"] - results[m.name]["proj_arrow_from"], axis=1
                )
                for m in methods
                if m.kind != "main"
            ]
        )
        per_obs.append(
            {
                "arrow_scale": float(np.median(shifts) / (0.12 * (lims[1][0] - lims[0][0]) + 1e-9)),
                "label": f"traj {traj_id} step {step}\n{task[:38]}",
                "results": results,
                "data_region": data_region,
                "fail_region": fail_region,
                "main_region": main_region,
                "demo_proj": proj(demo.reshape(1, -1))[0],
                "lims": lims,
            }
        )

    plot_all(out_dir, observations, methods, per_obs, args)
    plot_summary(out_dir, methods, metrics_rows, args)
    np.savez_compressed(out_dir / "samples.npz", **saved)
    cols = list(dict.fromkeys(k for r in metrics_rows for k in r))
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(metrics_rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics_rows, indent=2))
    (out_dir / "config.json").write_text(
        json.dumps(
            {**vars(args), "observations": observations, "methods": [m.name for m in methods]},
            indent=2,
        )
    )

    num_cols = [c for c in cols if c not in ("obs", "traj", "step", "task", "method")]
    print("\nMean over observations:")
    print(f"{'method':<20s} " + " ".join(f"{c:>17s}" for c in num_cols))
    for m in methods:
        vals = [
            np.mean([r[c] for r in metrics_rows if r["method"] == m.name and c in r])
            for c in num_cols
        ]
        print(f"{m.name:<20s} " + " ".join(f"{v:>17.3f}" for v in vals))
    print(f"\nFigures and metrics written to {out_dir}")


if __name__ == "__main__":
    main(tyro.cli(ArgsConfig))
