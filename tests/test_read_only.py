"""Tests that matrix containers hold an independent, read-only copy of their value."""

from __future__ import annotations

import numpy as np
import pytest

from projective import (
    AffineMatrix,
    EssentialMatrix,
    FundamentalMatrix,
    HomographyMatrix,
)
from projective.types import FloatArray

_MatrixClass = (
    type[AffineMatrix]
    | type[HomographyMatrix]
    | type[EssentialMatrix]
    | type[FundamentalMatrix]
)

_CASES = pytest.mark.parametrize(
    ("matrix_class", "value"),
    [
        (AffineMatrix, np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]])),
        (HomographyMatrix, np.eye(3)),
        (EssentialMatrix, np.eye(3)),
        (FundamentalMatrix, np.eye(3)),
    ],
    ids=["affine", "homography", "essential", "fundamental"],
)


@_CASES
def test_value_is_independent_of_input(
    matrix_class: _MatrixClass, value: FloatArray
) -> None:
    original = value.copy()
    matrix = matrix_class(value=value)
    value[0, 0] = 42.0
    np.testing.assert_array_equal(matrix.value, original)
    assert not np.shares_memory(matrix.value, value)


@_CASES
def test_value_is_read_only(matrix_class: _MatrixClass, value: FloatArray) -> None:
    matrix = matrix_class(value=value)
    with pytest.raises(ValueError, match="read-only"):
        matrix.value[0, 0] = 42.0


def test_value_keeps_dtype() -> None:
    matrix = HomographyMatrix(value=np.eye(3, dtype=np.float32))
    assert matrix.value.dtype == np.float32


def test_cached_inverse_stays_valid_after_input_changes() -> None:
    value = np.eye(3)
    homography = HomographyMatrix(value=value)
    _ = homography.inverse
    value[0, 2] = 5.0
    np.testing.assert_array_equal(homography.inverse, np.eye(3))
