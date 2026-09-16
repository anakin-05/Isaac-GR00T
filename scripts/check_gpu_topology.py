#!/usr/bin/env python3

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

"""Preflight check: refuse to launch NCCL training across MIG instances that
share one physical GPU.

MIG (Multi-Instance GPU) partitions all report the *same* PCI bus ID as their
parent card, so when a SLURM MIG partition hands a job two or more instances
carved from the same physical GPU, NCCL either aborts with
``Duplicate GPU detected`` or hangs until its collective timeout once
``torchrun`` spawns one rank per visible device -- NCCL does not support
collectives between MIG instances of the same physical GPU. Run this before
``torchrun`` so a bad allocation fails in seconds instead of burning a full
NCCL timeout's worth of GPU-hours.

Usage: ``python scripts/check_gpu_topology.py --num-gpus <N>``, matching
``--nproc_per_node``. Exits non-zero (with a report of the conflicting ranks
and a working ``CUDA_VISIBLE_DEVICES`` subset) when two or more of the first
``N`` visible devices share a physical GPU. Skips (exit 0) whenever the
topology can't be determined -- e.g. no GPUs, ``nvidia-smi`` unavailable --
so it never blocks a legitimate launch on an unrelated environment quirk.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
import subprocess
import sys


_GPU_RE = re.compile(r"^GPU (\d+): .*\(UUID: (GPU-[0-9a-fA-F-]+)\)")
_MIG_RE = re.compile(r"^\s+MIG .*\(UUID: (MIG-[0-9a-fA-F-]+)\)")


@dataclass(frozen=True)
class PhysicalGPU:
    index: int
    uuid: str
    mig_uuids: tuple[str, ...] = ()

    @property
    def has_mig(self) -> bool:
        return len(self.mig_uuids) > 0


def parse_nvidia_smi_l(text: str) -> list[PhysicalGPU]:
    """Parse ``nvidia-smi -L`` output into one :class:`PhysicalGPU` per card,
    each carrying the MIG instance UUIDs (if any) sliced from it, in order."""
    gpus: list[PhysicalGPU] = []
    index: int | None = None
    uuid: str | None = None
    mig_uuids: list[str] = []

    def flush() -> None:
        if index is not None and uuid is not None:
            gpus.append(PhysicalGPU(index, uuid, tuple(mig_uuids)))

    for line in text.splitlines():
        gpu_match = _GPU_RE.match(line)
        if gpu_match:
            flush()
            index = int(gpu_match.group(1))
            uuid = gpu_match.group(2)
            mig_uuids = []
            continue
        mig_match = _MIG_RE.match(line)
        if mig_match and index is not None:
            mig_uuids.append(mig_match.group(1))
    flush()
    return gpus


def flatten_devices(gpus: list[PhysicalGPU]) -> list[tuple[str, int]]:
    """CUDA's default enumeration order: ``(device_id, physical_gpu_index)``
    pairs, expanding each MIG-enabled GPU into its instance UUIDs."""
    flattened: list[tuple[str, int]] = []
    for gpu in gpus:
        if gpu.has_mig:
            flattened.extend((mig_uuid, gpu.index) for mig_uuid in gpu.mig_uuids)
        else:
            flattened.append((gpu.uuid, gpu.index))
    return flattened


def resolve_physical_indices(tokens: list[str], gpus: list[PhysicalGPU]) -> list[int]:
    """Map each ``CUDA_VISIBLE_DEVICES`` token (a UUID or a positional index)
    to the physical GPU index it resolves to."""
    flattened = flatten_devices(gpus)
    device_to_physical = dict(flattened)
    resolved: list[int] = []
    for raw in tokens:
        token = raw.strip()
        if token in device_to_physical:
            resolved.append(device_to_physical[token])
        elif token.isdigit():
            position = int(token)
            if position >= len(flattened):
                raise ValueError(
                    f"CUDA_VISIBLE_DEVICES index {position} is out of range for the "
                    f"{len(flattened)} device(s) reported by `nvidia-smi -L`"
                )
            resolved.append(flattened[position][1])
        else:
            raise ValueError(
                f"could not resolve device {token!r} against `nvidia-smi -L` output "
                "(no matching UUID or valid index)"
            )
    return resolved


def find_conflicting_physical_gpus(physical_indices: list[int]) -> set[int]:
    """Physical GPU indices claimed by more than one local rank."""
    seen: set[int] = set()
    conflicting: set[int] = set()
    for phys in physical_indices:
        if phys in seen:
            conflicting.add(phys)
        seen.add(phys)
    return conflicting


def safe_subset(tokens: list[str], physical_indices: list[int]) -> list[str]:
    """First token seen per distinct physical GPU, in original order --
    a ``CUDA_VISIBLE_DEVICES`` value with no same-GPU conflicts."""
    seen: set[int] = set()
    subset: list[str] = []
    for token, phys in zip(tokens, physical_indices):
        if phys not in seen:
            seen.add(phys)
            subset.append(token)
    return subset


def get_visible_tokens(num_gpus: int, gpus: list[PhysicalGPU]) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw:
        tokens = [t for t in raw.split(",") if t]
    else:
        tokens = [device_id for device_id, _ in flatten_devices(gpus)]
    return tokens[:num_gpus]


def build_conflict_report(tokens: list[str], physical_indices: list[int], num_gpus: int) -> str:
    conflicting = find_conflicting_physical_gpus(physical_indices)
    lines = [
        "check_gpu_topology: refusing to launch -- NCCL cannot run collectives between MIG",
        "instances that share one physical GPU (they report the same PCI bus ID, so NCCL",
        'aborts with "Duplicate GPU detected", or hangs until its collective timeout).',
        "",
        "Local rank -> device -> physical GPU:",
    ]
    for local_rank, (token, phys) in enumerate(zip(tokens, physical_indices)):
        flag = "  <-- CONFLICT" if phys in conflicting else ""
        lines.append(f"  rank {local_rank}: {token} (physical GPU {phys}){flag}")

    subset = safe_subset(tokens, physical_indices)
    distinct = len(set(physical_indices))
    lines += [
        "",
        f"CUDA_VISIBLE_DEVICES spans {distinct} distinct physical GPU(s), but "
        f"--num-gpus={num_gpus} was requested.",
        "",
        "Fix one of:",
        "  1. Ask SLURM for a MIG allocation with one instance per distinct physical GPU",
        "     (or request whole, non-MIG GPUs) so num-gpus distinct-GPU ranks are available.",
        "  2. Train with just the distinct physical GPUs already in this allocation, and",
        "     recover the intended global batch size via gradient accumulation instead of",
        "     extra data-parallel ranks:",
        f"       NUM_GPUS={len(subset)} CUDA_VISIBLE_DEVICES={','.join(subset)}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of NCCL ranks about to be launched (torchrun's --nproc_per_node).",
    )
    args = parser.parse_args(argv)

    if args.num_gpus <= 1:
        return 0

    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(
            f"check_gpu_topology: could not run `nvidia-smi -L` ({exc}); "
            "skipping MIG topology check.",
            file=sys.stderr,
        )
        return 0

    gpus = parse_nvidia_smi_l(result.stdout)
    if not gpus:
        print(
            "check_gpu_topology: `nvidia-smi -L` reported no GPUs; skipping MIG topology check.",
            file=sys.stderr,
        )
        return 0

    tokens = get_visible_tokens(args.num_gpus, gpus)
    if len(tokens) < args.num_gpus:
        print(
            f"check_gpu_topology: only {len(tokens)} device(s) visible via "
            f"CUDA_VISIBLE_DEVICES, but --num-gpus={args.num_gpus} was requested; "
            "skipping MIG topology check (torchrun will fail on its own with a clearer error).",
            file=sys.stderr,
        )
        return 0

    try:
        physical_indices = resolve_physical_indices(tokens, gpus)
    except ValueError as exc:
        print(
            f"check_gpu_topology: {exc}; skipping MIG topology check.",
            file=sys.stderr,
        )
        return 0

    if not find_conflicting_physical_gpus(physical_indices):
        print(
            f"check_gpu_topology: OK -- {args.num_gpus} NCCL rank(s) map to "
            f"{len(set(physical_indices))} distinct physical GPU(s)."
        )
        return 0

    print(build_conflict_report(tokens, physical_indices, args.num_gpus), file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
