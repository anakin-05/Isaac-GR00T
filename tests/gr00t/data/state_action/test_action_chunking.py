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

"""
Tests for ActionChunk.to_reversed_chunking (round-trip consistency's chunk
inversion utility - see the integration plan).

Covers:
- Round-trip identity for EndEffectorActionChunk (SE(3) composition) and
  JointActionChunk (vector composition)
- The H=1 edge case
"""

from gr00t.data.state_action.action_chunking import EndEffectorActionChunk, JointActionChunk
from gr00t.data.state_action.pose import EndEffectorPose, JointPose
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def _random_eef_pose() -> EndEffectorPose:
    return EndEffectorPose(
        translation=np.random.randn(3),
        rotation=Rotation.random().as_quat(),
        rotation_type="quat",
        rotation_order="xyzw",
    )


def _random_joint_pose(dim: int = 7) -> JointPose:
    return JointPose(np.random.randn(dim))


class TestToReversedChunkingEEF:
    """Round-trip identity for EndEffectorActionChunk (SE(3) composition)."""

    def test_roundtrip_identity(self):
        np.random.seed(0)
        reference = _random_eef_pose()
        absolute_poses = [_random_eef_pose() for _ in range(5)]
        absolute_chunk = EndEffectorActionChunk(absolute_poses)

        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)
        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference)

        # new_reference must equal this chunk's own composed absolute endpoint.
        expected_endpoint = relative_chunk.to_absolute_chunking(reference_frame=reference).poses[
            -1
        ]
        np.testing.assert_allclose(
            new_reference.translation, expected_endpoint.translation, atol=1e-6
        )
        np.testing.assert_allclose(
            new_reference.rotation_matrix, expected_endpoint.rotation_matrix, atol=1e-6
        )

        # Composing reversed_chunk back onto new_reference must recover the
        # original reference frame at the last timestep.
        recovered = reversed_chunk.to_absolute_chunking(reference_frame=new_reference).poses[-1]
        np.testing.assert_allclose(recovered.translation, reference.translation, atol=1e-5)
        np.testing.assert_allclose(
            recovered.rotation_matrix, reference.rotation_matrix, atol=1e-5
        )

    def test_reversed_chunk_length_matches_original(self):
        reference = _random_eef_pose()
        absolute_chunk = EndEffectorActionChunk([_random_eef_pose() for _ in range(6)])
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)
        reversed_chunk, _ = relative_chunk.to_reversed_chunking(reference)
        assert len(reversed_chunk) == len(relative_chunk) == 6

    def test_single_step_edge_case(self):
        """H=1: the reversed chunk's sole pose must recover the reference exactly."""
        reference = _random_eef_pose()
        absolute_chunk = EndEffectorActionChunk([_random_eef_pose()])
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)

        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference)
        assert len(reversed_chunk) == 1

        recovered = reversed_chunk.to_absolute_chunking(reference_frame=new_reference).poses[0]
        np.testing.assert_allclose(recovered.translation, reference.translation, atol=1e-6)
        np.testing.assert_allclose(
            recovered.rotation_matrix, reference.rotation_matrix, atol=1e-6
        )


class TestToReversedChunkingJoint:
    """Round-trip identity for JointActionChunk (pure vector algebra)."""

    def test_roundtrip_identity(self):
        np.random.seed(1)
        reference = _random_joint_pose()
        absolute_chunk = JointActionChunk([_random_joint_pose() for _ in range(5)])

        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)
        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference)

        expected_endpoint = relative_chunk.to_absolute_chunking(reference_frame=reference).poses[
            -1
        ]
        np.testing.assert_allclose(new_reference.joints, expected_endpoint.joints, atol=1e-8)

        recovered = reversed_chunk.to_absolute_chunking(reference_frame=new_reference).poses[-1]
        np.testing.assert_allclose(recovered.joints, reference.joints, atol=1e-8)

    def test_single_step_edge_case(self):
        reference = _random_joint_pose()
        absolute_chunk = JointActionChunk([_random_joint_pose()])
        relative_chunk = absolute_chunk.relative_chunking(reference_frame=reference)

        reversed_chunk, new_reference = relative_chunk.to_reversed_chunking(reference)
        assert len(reversed_chunk) == 1

        recovered = reversed_chunk.to_absolute_chunking(reference_frame=new_reference).poses[0]
        np.testing.assert_allclose(recovered.joints, reference.joints, atol=1e-8)


class TestToReversedChunkingEmptyRaises:
    def test_empty_chunk_raises(self):
        # ActionChunk itself forbids empty construction (see __init__), so a
        # reversal of an empty chunk can never be constructed either.
        with pytest.raises(ValueError):
            JointActionChunk([])
