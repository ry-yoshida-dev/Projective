from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from ....types import FloatArray
from ...mixin.transform import PerspectiveTransformMixin

if TYPE_CHECKING:
    from ..matrix import HomographyMatrix


class HomographyTransformMixin(PerspectiveTransformMixin):
    """Transform the homography matrix container itself."""

    @property
    @override
    def column_vector(self) -> FloatArray:
        """
        Return the column vector of a 3×3 homography matrix.

        Returns
        -------
        FloatArray
            Column vector with shape (9, 1).
        """
        return super().column_vector

    @property
    @override
    def row_vector(self) -> FloatArray:
        """
        Return the row vector of a 3×3 homography matrix.

        Returns
        -------
        FloatArray
            Row vector with shape (1, 9).
        """
        return super().row_vector

    @property
    @override
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the homography matrix.

        Returns
        -------
        tuple[int, int]
            Matrix shape (3, 3).
        """
        return super().shape

    @property
    @override
    def flatten(self) -> FloatArray:
        """
        Return the flattened homography matrix.

        Returns
        -------
        FloatArray
            One-dimensional array with shape (9,).
        """
        return super().flatten

    @cached_property
    def inverse(self) -> FloatArray:
        """
        Return the inverse of the homography matrix.

        Returns
        -------
        FloatArray
            Inverse matrix with shape (3, 3).
        """
        return np.linalg.inv(self.value).astype(np.float64)

    @property
    def T(self) -> FloatArray:
        """
        Return the transpose of the homography matrix.

        Returns
        -------
        FloatArray
            Transposed matrix with shape (3, 3).
        """
        return self.value.T

    @override
    def scale_correction(self, scale: float) -> HomographyMatrix:
        """
        Correct the scale of the homography matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        HomographyMatrix
            Homography matrix with scale corrected under similarity conjugation.
        """
        S = np.diag([scale, scale, 1])
        S_inv = np.linalg.inv(S)
        scaled_matrix = S_inv @ self.value @ S
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=scaled_matrix)
