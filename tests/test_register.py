"""Tests for register_perspective_matrix."""

from __future__ import annotations

import numpy as np
import pytest

from projective import (
    AffineMatrix,
    HomographyMatrix,
    PerspectiveTransformationMethod,
    register_perspective_matrix,
)


@pytest.mark.parametrize(
    ("transform_type", "expected_class"),
    [
        (PerspectiveTransformationMethod.AFFINE, AffineMatrix),
        (PerspectiveTransformationMethod.HOMOGRAPHY, HomographyMatrix),
    ],
)
def test_none_returns_identity(
    transform_type: PerspectiveTransformationMethod,
    expected_class: type[AffineMatrix] | type[HomographyMatrix],
) -> None:
    matrix = register_perspective_matrix(None, transform_type)
    assert isinstance(matrix, expected_class)
    np.testing.assert_array_equal(
        matrix.value, expected_class.create_identity_matrix().value
    )


def test_array_is_wrapped() -> None:
    value = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])
    matrix = register_perspective_matrix(value, PerspectiveTransformationMethod.AFFINE)
    assert isinstance(matrix, AffineMatrix)
    np.testing.assert_array_equal(matrix.value, value)


def test_array_with_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        _ = register_perspective_matrix(
            np.eye(2, 3), PerspectiveTransformationMethod.HOMOGRAPHY
        )


def test_matching_typed_matrix_passes_through() -> None:
    homography = HomographyMatrix.create_identity_matrix()
    assert register_perspective_matrix(homography) is homography


def test_mismatched_typed_matrix_raises() -> None:
    affine = AffineMatrix.create_identity_matrix()
    with pytest.raises(ValueError, match="does not match"):
        _ = register_perspective_matrix(
            affine, PerspectiveTransformationMethod.HOMOGRAPHY
        )
