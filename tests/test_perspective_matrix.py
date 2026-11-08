"""Tests for construction, homogeneous conversion, inverse, normalization, and scale correction."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projective import (
    AffineMatrix,
    HomographyMatrix,
    PerspectiveMatrix,
    PerspectiveTransformationMethod,
)

_TOLERANCE: float = 1e-10

_AFFINE = AffineMatrix(value=np.array([[1.1, 0.1, 5.0], [0.0, 0.9, -2.0]]))
_HOMOGRAPHY = HomographyMatrix(
    value=np.array([[1.1, 0.1, 5.0], [0.0, 0.9, -2.0], [1e-3, 0.0, 1.0]])
)
_MATRICES = pytest.mark.parametrize(
    "matrix", [_AFFINE, _HOMOGRAPHY], ids=["affine", "homography"]
)


@pytest.mark.parametrize(
    ("matrix_class", "expected"),
    [(AffineMatrix, np.eye(2, 3)), (HomographyMatrix, np.eye(3))],
)
def test_create_identity_matrix(
    matrix_class: type[PerspectiveMatrix], expected: np.ndarray
) -> None:
    np.testing.assert_array_equal(matrix_class.create_identity_matrix().value, expected)


@pytest.mark.parametrize(
    ("matrix_class", "value"),
    [(AffineMatrix, np.eye(3)), (HomographyMatrix, np.eye(2, 3))],
)
def test_invalid_shape_raises(
    matrix_class: type[PerspectiveMatrix], value: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="must have shape"):
        _ = matrix_class(value=value)


@_MATRICES
def test_array_views(matrix: PerspectiveMatrix) -> None:
    size = matrix.value.size
    np.testing.assert_array_equal(matrix.T, matrix.value.T)
    np.testing.assert_array_equal(matrix.flatten, matrix.value.ravel())
    np.testing.assert_array_equal(matrix.column_vector, matrix.value.reshape(size, 1))
    np.testing.assert_array_equal(matrix.row_vector, matrix.value.reshape(1, size))
    assert matrix.shape == matrix.value.shape


@_MATRICES
def test_homogeneous_round_trip(matrix: PerspectiveMatrix) -> None:
    restored = type(matrix).from_homogeneous(matrix.homogeneous)
    np.testing.assert_array_equal(restored.value, matrix.value)


def test_affine_rejects_projective_homogeneous() -> None:
    with pytest.raises(ValueError, match="not affine"):
        _ = AffineMatrix.from_homogeneous(_HOMOGRAPHY.value)


@_MATRICES
def test_inverse_inverts_homogeneous_matrix(matrix: PerspectiveMatrix) -> None:
    np.testing.assert_allclose(
        matrix.inverse @ matrix.homogeneous, np.eye(3), atol=_TOLERANCE
    )


def test_inverse_matches_numpy_linalg_inv() -> None:
    np.testing.assert_allclose(_HOMOGRAPHY.inverse, np.linalg.inv(_HOMOGRAPHY.value))


def test_inverse_is_cached() -> None:
    assert _HOMOGRAPHY.inverse is _HOMOGRAPHY.inverse


def test_from_unnormalized_value_scales_last_entry_to_one() -> None:
    restored = HomographyMatrix.from_unnormalized_value(_HOMOGRAPHY.value * 3.0)
    np.testing.assert_allclose(restored.value, _HOMOGRAPHY.value)


def test_from_unnormalized_value_rejects_zero_normalizer() -> None:
    value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="H\\[2, 2\\]"):
        _ = HomographyMatrix.from_unnormalized_value(value)


@pytest.mark.parametrize(
    ("matrix", "expected"),
    [
        (_AFFINE, PerspectiveTransformationMethod.AFFINE),
        (_HOMOGRAPHY, PerspectiveTransformationMethod.HOMOGRAPHY),
    ],
)
def test_transform_type(
    matrix: PerspectiveMatrix, expected: PerspectiveTransformationMethod
) -> None:
    assert matrix.transform_type is expected


@_MATRICES
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_scale_correction_maps_rescaled_points(
    matrix: PerspectiveMatrix, scale: float
) -> None:
    points = np.array([[10.0, 20.0], [30.0, 5.0]])
    corrected = matrix.scale_correction(scale)
    assert type(corrected) is type(matrix)
    np.testing.assert_allclose(
        corrected.projective_transformation(points / scale),
        matrix.projective_transformation(points) / scale,
    )


@_MATRICES
@pytest.mark.parametrize("scale", [0.0, math.inf, math.nan])
def test_scale_correction_rejects_invalid_scale(
    matrix: PerspectiveMatrix, scale: float
) -> None:
    with pytest.raises(ValueError, match="Scale"):
        _ = matrix.scale_correction(scale)
