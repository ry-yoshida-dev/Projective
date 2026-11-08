"""Tests for PerspectiveMatrix.projective_transformation on affine and homography matrices."""

from __future__ import annotations

import numpy as np
import pytest

from projective import AffineMatrix, HomographyMatrix, PerspectiveMatrix
from projective.types import FloatArray

_POINT_TOLERANCE: float = 1e-10

_AFFINE = AffineMatrix(value=np.array([[2.0, 0.1, 1.0], [-0.2, 1.5, 2.0]]))
_HOMOGRAPHY = HomographyMatrix(
    value=np.array([[2.0, 0.1, 1.0], [-0.2, 1.5, 2.0], [1e-3, 2e-3, 1.0]])
)
_MATRICES = pytest.mark.parametrize(
    "matrix", [_AFFINE, _HOMOGRAPHY], ids=["affine", "homography"]
)


@pytest.fixture
def sample_points() -> FloatArray:
    return np.array([[1.0, 2.0], [3.0, 4.0], [-0.5, 1.5]], dtype=np.float64)


def _manual_map(matrix: PerspectiveMatrix, points: FloatArray) -> FloatArray:
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    mapped = homogeneous @ matrix.homogeneous.T
    return mapped[:, :2] / mapped[:, 2:3]


@pytest.mark.parametrize("matrix_class", [AffineMatrix, HomographyMatrix])
def test_identity_preserves_points(
    matrix_class: type[PerspectiveMatrix], sample_points: FloatArray
) -> None:
    transformed = matrix_class.create_identity_matrix().projective_transformation(
        sample_points
    )
    np.testing.assert_allclose(transformed, sample_points, atol=_POINT_TOLERANCE)


@_MATRICES
def test_forward_matches_manual_homogeneous_map(
    matrix: PerspectiveMatrix, sample_points: FloatArray
) -> None:
    np.testing.assert_allclose(
        matrix.projective_transformation(sample_points),
        _manual_map(matrix, sample_points),
        atol=_POINT_TOLERANCE,
    )


@_MATRICES
def test_inverse_recovers_original_points(
    matrix: PerspectiveMatrix, sample_points: FloatArray
) -> None:
    forward = matrix.projective_transformation(sample_points)
    recovered = matrix.projective_transformation(forward, is_inverse=True)
    np.testing.assert_allclose(recovered, sample_points, atol=_POINT_TOLERANCE)


@_MATRICES
def test_homogeneous_input_is_dehomogenized(
    matrix: PerspectiveMatrix, sample_points: FloatArray
) -> None:
    weights = np.array([[2.0], [0.5], [4.0]])
    homogeneous = np.hstack([sample_points, np.ones((3, 1))]) * weights
    np.testing.assert_allclose(
        matrix.projective_transformation(homogeneous),
        _manual_map(matrix, sample_points),
        atol=_POINT_TOLERANCE,
    )


@pytest.mark.parametrize("up_axis_index", [0, 1, 2])
def test_identity_keeps_xy_for_every_up_axis(up_axis_index: int) -> None:
    points = np.array([[2.0, 4.0]], dtype=np.float64)
    transformed = HomographyMatrix.create_identity_matrix().projective_transformation(
        points, up_axis_index=up_axis_index
    )
    np.testing.assert_allclose(transformed, points, atol=_POINT_TOLERANCE)


def test_up_axis_index_zero_divides_by_first_coordinate() -> None:
    points = np.array([[2.0, 4.0, 6.0]], dtype=np.float64)
    transformed = HomographyMatrix.create_identity_matrix().projective_transformation(
        points, up_axis_index=0
    )
    np.testing.assert_allclose(transformed, [[2.0, 3.0]], atol=_POINT_TOLERANCE)


def test_invalid_up_axis_index_raises(sample_points: FloatArray) -> None:
    with pytest.raises(ValueError, match="Up axis index"):
        _ = _HOMOGRAPHY.projective_transformation(sample_points, up_axis_index=3)


@_MATRICES
def test_invalid_point_shape_raises(matrix: PerspectiveMatrix) -> None:
    with pytest.raises(ValueError, match="shape"):
        _ = matrix.projective_transformation(np.array([1.0, 2.0, 3.0]))
