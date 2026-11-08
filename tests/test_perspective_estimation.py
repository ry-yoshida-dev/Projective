"""Tests for estimating perspective matrices from point correspondences."""

from __future__ import annotations

import numpy as np
import pytest

from projective import (
    AffineMatrix,
    EstimationError,
    HomographyMatrix,
    PerspectiveMatrix,
    PerspectiveTransformationMethod,
)
from projective.types import FloatArray

_MATRIX_TOLERANCE: float = 1e-4


@pytest.fixture
def grid_points() -> FloatArray:
    xs, ys = np.meshgrid(np.linspace(0.0, 100.0, 5), np.linspace(0.0, 80.0, 4))
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)


def test_homography_recovers_known_matrix(grid_points: FloatArray) -> None:
    expected = HomographyMatrix(
        value=np.array(
            [[1.1, 0.05, 5.0], [-0.03, 0.95, -3.0], [1e-4, 2e-4, 1.0]],
            dtype=np.float64,
        )
    )
    destination = expected.projective_transformation(grid_points)
    matrix, mask = HomographyMatrix.create_from_points(grid_points, destination)
    assert isinstance(matrix, HomographyMatrix)
    assert mask.shape == (grid_points.shape[0], 1)
    np.testing.assert_allclose(
        HomographyMatrix.from_unnormalized_value(matrix.value).value,
        expected.value,
        atol=_MATRIX_TOLERANCE,
    )


def test_affine_recovers_similarity(grid_points: FloatArray) -> None:
    angle = 0.2
    scale = 1.5
    expected = np.array(
        [
            [scale * np.cos(angle), -scale * np.sin(angle), 4.0],
            [scale * np.sin(angle), scale * np.cos(angle), -2.0],
        ]
    )
    destination = AffineMatrix(value=expected).projective_transformation(grid_points)
    matrix, mask = (
        PerspectiveTransformationMethod.AFFINE.perspective_class.create_from_points(
            grid_points, destination
        )
    )
    assert isinstance(matrix, AffineMatrix)
    assert mask.shape == (grid_points.shape[0], 1)
    np.testing.assert_allclose(matrix.value, expected, atol=_MATRIX_TOLERANCE)


def test_affine_accepts_two_points() -> None:
    origin = np.array([[0.0, 0.0], [1.0, 0.0]])
    destination = np.array([[1.0, 1.0], [3.0, 1.0]])
    matrix, _ = AffineMatrix.create_from_points(origin, destination)
    np.testing.assert_allclose(
        matrix.value, [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0]], atol=_MATRIX_TOLERANCE
    )


def test_homography_rejects_three_points() -> None:
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="At least 4"):
        _ = HomographyMatrix.create_from_points(points, points)


def test_collinear_points_raise_estimation_error() -> None:
    points = np.column_stack([np.arange(6.0), np.arange(6.0)])
    with pytest.raises(EstimationError):
        _ = HomographyMatrix.create_from_points(points, points * 2)


def test_one_dimensional_points_raise_value_error() -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        _ = HomographyMatrix.create_from_points(np.zeros(8), np.zeros(8))


def test_mismatched_shapes_raise_value_error(grid_points: FloatArray) -> None:
    with pytest.raises(ValueError, match="same shape"):
        _ = HomographyMatrix.create_from_points(grid_points, grid_points[:-1])


@pytest.mark.parametrize(
    ("transform_type", "expected_class"),
    [
        (PerspectiveTransformationMethod.AFFINE, AffineMatrix),
        (PerspectiveTransformationMethod.HOMOGRAPHY, HomographyMatrix),
    ],
)
def test_from_points_dispatches_on_transform_type(
    grid_points: FloatArray,
    transform_type: PerspectiveTransformationMethod,
    expected_class: type[PerspectiveMatrix],
) -> None:
    matrix, _ = PerspectiveMatrix.from_points(
        grid_points, grid_points * 2, transform_type=transform_type
    )
    assert type(matrix) is expected_class
