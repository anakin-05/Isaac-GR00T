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

from __future__ import annotations

import subprocess

import pytest
from scripts.check_gpu_topology import (
    find_conflicting_physical_gpus,
    flatten_devices,
    main,
    parse_nvidia_smi_l,
    resolve_physical_indices,
    safe_subset,
)


# Mirrors a real MIG-partitioned SLURM node: GPUs 0-2 are split into two
# 3g.40gb instances each (a same-physical-GPU conflict once >5 ranks are
# requested), GPUs 3-4 are split into one instance each.
MIG_TOPOLOGY = """\
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-e7c0528f-9bf2-576b-bd71-27a5e4dce44f)
  MIG 3g.40gb     Device  0: (UUID: MIG-9bd72ea7-e38a-5047-ae31-7d8ca051a2d8)
  MIG 3g.40gb     Device  1: (UUID: MIG-fba82d3a-a822-5930-86c5-e91f3cdb1f43)
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-54a94d05-e0cd-5501-eb8d-aa3a80c0aa15)
  MIG 3g.40gb     Device  0: (UUID: MIG-cd55dc2d-bc3b-59c0-9bc5-e156d0ef8b74)
  MIG 3g.40gb     Device  1: (UUID: MIG-2d8f02b5-2b8c-511d-bb71-67fe7277450e)
GPU 2: NVIDIA H100 80GB HBM3 (UUID: GPU-8be121e8-9496-33e0-751c-bce5b0c5ed75)
  MIG 3g.40gb     Device  0: (UUID: MIG-08fb55c7-4894-552e-ade4-ed755fd2754d)
  MIG 3g.40gb     Device  1: (UUID: MIG-e28bf578-ad62-5d32-b4be-6796bb60e2d8)
GPU 3: NVIDIA H100 80GB HBM3 (UUID: GPU-f17c29ac-0a4e-c803-db39-06f84ff0add8)
  MIG 3g.40gb     Device  0: (UUID: MIG-ba31337f-3b19-5369-b893-5ceb62a5de29)
GPU 4: NVIDIA H100 80GB HBM3 (UUID: GPU-54e6f669-b83d-db6c-cbc3-28b22964b9b3)
  MIG 3g.40gb     Device  0: (UUID: MIG-bf76cd56-4a14-57e4-8f8a-940f42bca9f6)
"""

NON_MIG_TOPOLOGY = """\
GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-aaaaaaaa-0000-0000-0000-000000000000)
GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-bbbbbbbb-0000-0000-0000-000000000000)
"""


def test_parse_nvidia_smi_l_groups_mig_instances_under_their_physical_gpu():
    gpus = parse_nvidia_smi_l(MIG_TOPOLOGY)

    assert [gpu.index for gpu in gpus] == [0, 1, 2, 3, 4]
    assert gpus[0].mig_uuids == (
        "MIG-9bd72ea7-e38a-5047-ae31-7d8ca051a2d8",
        "MIG-fba82d3a-a822-5930-86c5-e91f3cdb1f43",
    )
    assert gpus[4].mig_uuids == ("MIG-bf76cd56-4a14-57e4-8f8a-940f42bca9f6",)


def test_parse_nvidia_smi_l_without_mig():
    gpus = parse_nvidia_smi_l(NON_MIG_TOPOLOGY)

    assert [gpu.has_mig for gpu in gpus] == [False, False]
    assert flatten_devices(gpus) == [
        ("GPU-aaaaaaaa-0000-0000-0000-000000000000", 0),
        ("GPU-bbbbbbbb-0000-0000-0000-000000000000", 1),
    ]


def test_resolve_physical_indices_by_uuid_detects_same_gpu_pairs():
    gpus = parse_nvidia_smi_l(MIG_TOPOLOGY)
    tokens = [mig_uuid for gpu in gpus for mig_uuid in gpu.mig_uuids]  # all 8 instances

    physical_indices = resolve_physical_indices(tokens, gpus)

    assert physical_indices == [0, 0, 1, 1, 2, 2, 3, 4]
    assert find_conflicting_physical_gpus(physical_indices) == {0, 1, 2}


def test_resolve_physical_indices_by_positional_index():
    gpus = parse_nvidia_smi_l(MIG_TOPOLOGY)

    # Positional indices follow nvidia-smi's flattened enumeration order.
    assert resolve_physical_indices(["0", "1", "6", "7"], gpus) == [0, 0, 3, 4]


def test_resolve_physical_indices_rejects_unknown_token():
    gpus = parse_nvidia_smi_l(MIG_TOPOLOGY)

    with pytest.raises(ValueError, match="could not resolve device"):
        resolve_physical_indices(["MIG-does-not-exist"], gpus)


def test_safe_subset_keeps_first_instance_per_physical_gpu():
    gpus = parse_nvidia_smi_l(MIG_TOPOLOGY)
    tokens = [mig_uuid for gpu in gpus for mig_uuid in gpu.mig_uuids]
    physical_indices = resolve_physical_indices(tokens, gpus)

    subset = safe_subset(tokens, physical_indices)

    assert subset == [
        "MIG-9bd72ea7-e38a-5047-ae31-7d8ca051a2d8",
        "MIG-cd55dc2d-bc3b-59c0-9bc5-e156d0ef8b74",
        "MIG-08fb55c7-4894-552e-ade4-ed755fd2754d",
        "MIG-ba31337f-3b19-5369-b893-5ceb62a5de29",
        "MIG-bf76cd56-4a14-57e4-8f8a-940f42bca9f6",
    ]


def _stub_nvidia_smi(monkeypatch: pytest.MonkeyPatch, stdout: str) -> None:
    def fake_run(cmd, capture_output, text, check):  # noqa: ANN001 - test stub
        assert cmd == ["nvidia-smi", "-L"]
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_main_fails_fast_on_same_gpu_conflict(monkeypatch, capsys):
    _stub_nvidia_smi(monkeypatch, MIG_TOPOLOGY)
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        ",".join(mig for gpu in parse_nvidia_smi_l(MIG_TOPOLOGY) for mig in gpu.mig_uuids),
    )

    exit_code = main(["--num-gpus", "8"])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "CONFLICT" in err
    assert "NUM_GPUS=5" in err


def test_main_passes_when_devices_map_to_distinct_physical_gpus(monkeypatch, capsys):
    _stub_nvidia_smi(monkeypatch, MIG_TOPOLOGY)
    safe_devices = ",".join(gpu.mig_uuids[0] for gpu in parse_nvidia_smi_l(MIG_TOPOLOGY))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", safe_devices)

    exit_code = main(["--num-gpus", "5"])

    assert exit_code == 0
    assert "OK" in capsys.readouterr().out


def test_main_skips_check_for_single_gpu(monkeypatch):
    # Must not even try to call nvidia-smi.
    def fail_run(*args, **kwargs):
        raise AssertionError("nvidia-smi should not be invoked for --num-gpus=1")

    monkeypatch.setattr(subprocess, "run", fail_run)

    assert main(["--num-gpus", "1"]) == 0


def test_main_skips_when_nvidia_smi_missing(monkeypatch, capsys):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert main(["--num-gpus", "4"]) == 0
    assert "skipping MIG topology check" in capsys.readouterr().err
