"""Tests for decomposition properties and repr of perspective matrices."""

from __future__ import annotations

import math

import numpy as np
import pytest

from projective import AffineMatrix, HomographyMatrix
from projective.types import FloatArray

_TOLERANCE: float = 1e-12


def _affine_and_homography(
    linear: FloatArray, translation: FloatArray
) -> tuple[AffineMatrix, HomographyMatrix]:
    affine_value = np.column_stack([linear, translation])
    homography_value = np.vstack([affine_value, [0.0, 0.0, 1.0]])
    return AffineMatrix(value=affine_value), HomographyMatrix(value=homography_value)


def test_decomposition_matches_between_affine_and_homography() -> None:
    linear = np.array([[1.0, 0.5], [0.2, 1.3]])
    affine, homography = _affine_and_homography(linear, np.array([3.0, -1.0]))
    np.testing.assert_allclose(affine.shear, homography.shear, atol=_TOLERANCE)
    np.testing.assert_allclose(affine.scale, homography.scale, atol=_TOLERANCE)
    np.testing.assert_allclose(
        affine.translation, homography.translation, atol=_TOLERANCE
    )
    assert math.isclose(affine.rotation, homography.rotation)


def test_pure_shear() -> None:
    affine, _ = _affine_and_homography(np.array([[1.0, 0.5], [0.0, 1.0]]), np.zeros(2))
    np.testing.assert_allclose(affine.shear, [0.5, 0.4], atol=_TOLERANCE)


def test_rotation_and_scale_of_similarity() -> None:
    angle = 0.3
    scale = 2.0
    linear = scale * np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    affine, _ = _affine_and_homography(linear, np.zeros(2))
    assert math.isclose(affine.rotation, angle)
    np.testing.assert_allclose(affine.scale, [scale, scale], atol=_TOLERANCE)
    np.testing.assert_allclose(affine.shear, [0.0, 0.0], atol=_TOLERANCE)


def test_homography_decomposition_is_scale_invariant() -> None:
    _, homography = _affine_and_homography(
        np.array([[1.2, 0.1], [0.0, 0.9]]), np.array([4.0, 5.0])
    )
    scaled = HomographyMatrix(value=homography.value * 3.0)
    np.testing.assert_allclose(scaled.scale, homography.scale, atol=_TOLERANCE)
    np.testing.assert_allclose(
        scaled.translation, homography.translation, atol=_TOLERANCE
    )
    np.testing.assert_allclose(scaled.shear, homography.shear, atol=_TOLERANCE)


def test_homography_with_zero_normalizer_raises() -> None:
    value = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="H\\[2, 2\\]"):
        _ = HomographyMatrix(value=value).scale


def test_zero_column_has_zero_rotation_and_shear() -> None:
    affine, _ = _affine_and_homography(np.zeros((2, 2)), np.zeros(2))
    assert affine.rotation == 0.0
    np.testing.assert_array_equal(affine.shear, [0.0, 0.0])


def test_has_perspective() -> None:
    assert not HomographyMatrix.create_identity_matrix().has_perspective
    value = np.eye(3)
    value[2, 0] = 1e-3
    assert HomographyMatrix(value=value).has_perspective
    assert not AffineMatrix.create_identity_matrix().has_perspective


def test_repr_shows_shape_and_type() -> None:
    expected = "HomographyMatrix(value.shape=(3, 3), transform_type=PerspectiveTransformationMethod.HOMOGRAPHY)"
    assert repr(HomographyMatrix.create_identity_matrix()) == expected
