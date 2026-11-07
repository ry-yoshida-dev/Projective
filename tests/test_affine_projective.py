"""Tests for AffineMatrix.projective_transformation."""

from __future__ import annotations

import numpy as np
import pytest

from projective import AffineMatrix
from projective.types import FloatArray

_POINT_TOLERANCE: float = 1e-10


@pytest.fixture
def sample_points() -> FloatArray:
    return np.array([[0.0, 0.0], [100.0, 50.0]], dtype=np.float64)


def test_create_identity_matrix_shape() -> None:
    affine = AffineMatrix.create_identity_matrix()
    np.testing.assert_array_equal(
        affine.value,
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
    )


def test_translation_affine(sample_points: FloatArray) -> None:
    affine = AffineMatrix(
        value=np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]], dtype=np.float64)
    )
    transformed = affine.projective_transformation(sample_points)
    expected = sample_points + np.array([10.0, 20.0])
    np.testing.assert_allclose(transformed, expected, atol=_POINT_TOLERANCE)


def test_is_inverse_raises(sample_points: FloatArray) -> None:
    affine = AffineMatrix.create_identity_matrix()
    with pytest.raises(ValueError, match="Inverse transformation"):
        _ = affine.projective_transformation(sample_points, is_inverse=True)


def test_invalid_point_shape_raises() -> None:
    affine = AffineMatrix.create_identity_matrix()
    with pytest.raises(ValueError, match="shape"):
        _ = affine.projective_transformation(np.array([1.0, 2.0, 3.0]))
