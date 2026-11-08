"""Tests for EssentialMatrix and FundamentalMatrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from projective import EssentialMatrix, FundamentalMatrix
from projective.types import FloatArray

_EPIPOLAR_TOLERANCE: float = 1e-3


@dataclass(frozen=True)
class StereoScene:
    intrinsic_matrix: FloatArray
    points1: FloatArray
    points2: FloatArray


def _project(intrinsic_matrix: FloatArray, points_3d: FloatArray) -> FloatArray:
    projected = points_3d @ intrinsic_matrix.T
    return projected[:, :2] / projected[:, 2:3]


@pytest.fixture
def stereo_scene() -> StereoScene:
    rng = np.random.default_rng(0)
    intrinsic_matrix = np.array(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points_3d = rng.uniform([-2.0, -2.0, 4.0], [2.0, 2.0, 8.0], size=(50, 3))
    angle = 0.1
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    translation = np.array([0.5, 0.0, 0.1])
    points_3d_second = points_3d @ rotation.T + translation
    return StereoScene(
        intrinsic_matrix=intrinsic_matrix,
        points1=_project(intrinsic_matrix, points_3d),
        points2=_project(intrinsic_matrix, points_3d_second),
    )


def _homogeneous(points: FloatArray) -> FloatArray:
    return np.hstack([points, np.ones((points.shape[0], 1))])


def test_essential_matrix_satisfies_epipolar_constraint(
    stereo_scene: StereoScene,
) -> None:
    essential, mask = EssentialMatrix.from_points(
        stereo_scene.points1, stereo_scene.points2, stereo_scene.intrinsic_matrix
    )
    intrinsic_inverse = np.linalg.inv(stereo_scene.intrinsic_matrix)
    normalized1 = _homogeneous(stereo_scene.points1) @ intrinsic_inverse.T
    normalized2 = _homogeneous(stereo_scene.points2) @ intrinsic_inverse.T
    residuals = np.sum((normalized2 @ essential.value) * normalized1, axis=1)
    assert mask.shape == (stereo_scene.points1.shape[0], 1)
    assert np.max(np.abs(residuals)) < _EPIPOLAR_TOLERANCE


def test_fundamental_matrix_satisfies_epipolar_constraint(
    stereo_scene: StereoScene,
) -> None:
    fundamental, mask = FundamentalMatrix.from_points(
        stereo_scene.points1, stereo_scene.points2
    )
    homogeneous1 = _homogeneous(stereo_scene.points1)
    homogeneous2 = _homogeneous(stereo_scene.points2)
    residuals = np.sum((homogeneous2 @ fundamental.value) * homogeneous1, axis=1)
    assert mask.shape == (stereo_scene.points1.shape[0], 1)
    assert np.max(np.abs(residuals)) < _EPIPOLAR_TOLERANCE


def test_essential_matrix_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="3x3"):
        _ = EssentialMatrix(value=np.eye(2))


def test_fundamental_matrix_rejects_too_few_points(stereo_scene: StereoScene) -> None:
    with pytest.raises(ValueError, match="At least 8"):
        _ = FundamentalMatrix.from_points(
            stereo_scene.points1[:7], stereo_scene.points2[:7]
        )
