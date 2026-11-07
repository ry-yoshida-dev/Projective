from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import override

from ....types import FloatArray
from ...mixin.transform import PerspectiveTransformMixin

if TYPE_CHECKING:
    from ..matrix import AffineMatrix


class AffineTransformMixin(PerspectiveTransformMixin):
    """Transform the affine matrix container itself."""

    @property
    @override
    def column_vector(self) -> FloatArray:
        """
        Return the column vector of a 2×3 affine matrix.

        Returns
        -------
        FloatArray
            Column vector with shape (6, 1).
        """
        return super().column_vector

    @property
    @override
    def row_vector(self) -> FloatArray:
        """
        Return the row vector of a 2×3 affine matrix.

        Returns
        -------
        FloatArray
            Row vector with shape (1, 6).
        """
        return super().row_vector

    @property
    @override
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the affine matrix.

        Returns
        -------
        tuple[int, int]
            Matrix shape (2, 3).
        """
        return super().shape

    @property
    @override
    def flatten(self) -> FloatArray:
        """
        Return the flattened affine matrix.

        Returns
        -------
        FloatArray
            One-dimensional array with shape (6,).
        """
        return super().flatten

    @override
    def scale_correction(
        self,
        scale: float,
    ) -> AffineMatrix:
        """
        Correct the scale of the affine matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        AffineMatrix
            Affine matrix with scale corrected under similarity conjugation.
        """
        extended = np.vstack([self.value, [0.0, 0.0, 1.0]])

        S = np.diag([scale, scale, 1.0])
        S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])

        corrected = S_inv @ extended @ S
        from ..matrix import AffineMatrix as _AffineMatrix

        return _AffineMatrix(value=corrected[:2])
