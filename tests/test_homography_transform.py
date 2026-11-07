"""Tests for HomographyMatrix transform views and inverse."""

from __future__ import annotations

import numpy as np
import pytest

from projective import HomographyMatrix

_IDENTITY_TOLERANCE: float = 1e-10


def test_create_identity_matrix_is_3x3() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    np.testing.assert_array_equal(homography.value, np.eye(3))


def test_invalid_shape_raises() -> None:
    with pytest.raises(ValueError, match="3x3"):
        _ = HomographyMatrix(value=np.eye(2))


def test_inverse_matches_numpy_linalg_inv() -> None:
    matrix_value = np.array(
        [
            [1.2, 0.1, 3.0],
            [0.0, 0.9, -1.0],
            [0.001, 0.002, 1.0],
        ],
        dtype=np.float64,
    )
    homography = HomographyMatrix(value=matrix_value)
    np.testing.assert_allclose(homography.inverse, np.linalg.inv(matrix_value))


def test_inverse_product_is_identity() -> None:
    homography = HomographyMatrix(
        value=np.array(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 2.0],
                [0.01, 0.02, 1.0],
            ],
            dtype=np.float64,
        )
    )
    product = homography.inverse @ homography.value
    np.testing.assert_allclose(product, np.eye(3), atol=_IDENTITY_TOLERANCE)


def test_inverse_is_cached() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    first_inverse = homography.inverse
    second_inverse = homography.inverse
    assert first_inverse is second_inverse
