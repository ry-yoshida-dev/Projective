"""Tests for HomographyMatrix.projective_transformation."""

from __future__ import annotations

import numpy as np
import pytest

from projective import HomographyMatrix
from projective.types import FloatArray

_POINT_TOLERANCE: float = 1e-10


@pytest.fixture
def sample_points() -> FloatArray:
    return np.array([[1.0, 2.0], [3.0, 4.0], [-0.5, 1.5]], dtype=np.float64)


@pytest.fixture
def scaled_translation_homography() -> HomographyMatrix:
    return HomographyMatrix(
        value=np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    )


def test_identity_preserves_points(sample_points: FloatArray) -> None:
    homography = HomographyMatrix.create_identity_matrix()
    transformed = homography.projective_transformation(sample_points)
    np.testing.assert_allclose(transformed, sample_points, atol=_POINT_TOLERANCE)


def test_forward_matches_manual_homogeneous_map(
    scaled_translation_homography: HomographyMatrix,
    sample_points: FloatArray,
) -> None:
    homogeneous = np.hstack([sample_points, np.ones((sample_points.shape[0], 1))])
    mapped = (scaled_translation_homography.value @ homogeneous.T).T
    expected = mapped[:, :2] / mapped[:, 2:3]
    actual = scaled_translation_homography.projective_transformation(sample_points)
    np.testing.assert_allclose(actual, expected, atol=_POINT_TOLERANCE)


def test_inverse_recover_original_points(
    scaled_translation_homography: HomographyMatrix,
    sample_points: FloatArray,
) -> None:
    forward = scaled_translation_homography.projective_transformation(sample_points)
    recovered = scaled_translation_homography.projective_transformation(
        forward,
        is_inverse=True,
    )
    np.testing.assert_allclose(recovered, sample_points, atol=_POINT_TOLERANCE)


def test_is_inverse_uses_inverse_matrix(
    scaled_translation_homography: HomographyMatrix,
    sample_points: FloatArray,
) -> None:
    homogeneous = np.hstack([sample_points, np.ones((sample_points.shape[0], 1))])
    expected = (scaled_translation_homography.inverse @ homogeneous.T).T
    expected_xy = expected[:, :2] / expected[:, 2:3]
    actual = scaled_translation_homography.projective_transformation(
        sample_points,
        is_inverse=True,
    )
    np.testing.assert_allclose(actual, expected_xy, atol=_POINT_TOLERANCE)


def test_accepts_homogeneous_input(
    scaled_translation_homography: HomographyMatrix,
) -> None:
    points = np.array([[1.0, 2.0, 1.0], [3.0, 4.0, 1.0]], dtype=np.float64)
    transformed = scaled_translation_homography.projective_transformation(points)
    assert transformed.shape == (2, 2)


def test_up_axis_index_two_is_default_cartesian(
    sample_points: FloatArray,
) -> None:
    homography = HomographyMatrix.create_identity_matrix()
    transformed = homography.projective_transformation(
        sample_points,
        up_axis_index=2,
    )
    np.testing.assert_allclose(transformed, sample_points, atol=_POINT_TOLERANCE)


def test_up_axis_index_zero_divides_by_first_coordinate() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    points = np.array([[2.0, 4.0]], dtype=np.float64)
    transformed = homography.projective_transformation(points, up_axis_index=0)
    np.testing.assert_allclose(
        transformed, np.array([[2.0, 0.5]]), atol=_POINT_TOLERANCE
    )


def test_up_axis_index_one_keeps_xy_for_identity() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    points = np.array([[3.0, 6.0]], dtype=np.float64)
    transformed = homography.projective_transformation(points, up_axis_index=1)
    np.testing.assert_allclose(transformed, points, atol=_POINT_TOLERANCE)


def test_invalid_up_axis_index_raises(sample_points: FloatArray) -> None:
    homography = HomographyMatrix.create_identity_matrix()
    with pytest.raises(ValueError, match="Up axis index"):
        _ = homography.projective_transformation(sample_points, up_axis_index=3)


def test_invalid_point_shape_raises() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    with pytest.raises(ValueError, match="shape"):
        _ = homography.projective_transformation(np.array([1.0, 2.0, 3.0]))
