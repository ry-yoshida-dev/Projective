from __future__ import annotations

import math
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Self, cast

import numpy as np

from ...types import FloatArray


class PerspectiveTransformMixin(ABC):
    """
    Views and transformations of the matrix container itself (not external point coordinates).

    Concrete matrices provide the conversion to and from the 3×3 homogeneous form;
    the inverse and scale correction are derived from it.
    """

    value: FloatArray

    @property
    def T(self) -> FloatArray:
        """
        Return the transpose of the stored matrix.

        Returns
        -------
        FloatArray
            Transposed matrix with the reversed shape.
        """
        return self.value.T

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the stored matrix.

        Returns
        -------
        tuple[int, int]
            Two-dimensional shape of value.
        """
        # Concrete matrices validate that value is two-dimensional.
        return cast(tuple[int, int], self.value.shape)

    @property
    def flatten(self) -> FloatArray:
        """
        Return the stored matrix flattened in row-major order.

        Returns
        -------
        FloatArray
            One-dimensional copy with shape (value.size,).
        """
        return self.value.flatten()

    @property
    def column_vector(self) -> FloatArray:
        """
        Return the stored matrix as a column vector in row-major order.

        Returns
        -------
        FloatArray
            Column vector with shape (value.size, 1).
        """
        return self.value.reshape(-1, 1)

    @property
    def row_vector(self) -> FloatArray:
        """
        Return the stored matrix as a row vector in row-major order.

        Returns
        -------
        FloatArray
            Row vector with shape (1, value.size).
        """
        return self.value.reshape(1, -1)

    @property
    @abstractmethod
    def homogeneous(self) -> FloatArray:
        """
        Return the matrix as a 3×3 homogeneous transformation.

        Returns
        -------
        FloatArray
            The homogeneous matrix with shape (3, 3).
        """

    @classmethod
    @abstractmethod
    def from_homogeneous(cls, matrix: FloatArray) -> Self:
        """
        Create the matrix from a 3×3 homogeneous transformation.

        Parameters
        ----------
        matrix : FloatArray
            Homogeneous matrix with shape (3, 3).

        Returns
        -------
        Self
            The matrix of this type.

        Raises
        ------
        ValueError
            If the homogeneous matrix cannot be represented by this type.
        """

    @cached_property
    def inverse(self) -> FloatArray:
        """
        Return the inverse of the homogeneous matrix.

        Returns
        -------
        FloatArray
            Inverse matrix with shape (3, 3).
        """
        return np.linalg.inv(self.homogeneous).astype(np.float64)

    @property
    def _normalized_homogeneous(self) -> FloatArray:
        normalizer = self.homogeneous.item(2, 2)
        if normalizer == 0:
            raise ValueError(
                "Matrix with H[2, 2] == 0 cannot be normalized or decomposed"
            )
        return self.homogeneous / normalizer

    def scale_correction(self, scale: float) -> Self:
        """
        Conjugate the transformation by an isotropic scaling.

        The result maps points given in a coordinate system scaled by ``1 / scale``,
        e.g. an image resized by ``1 / scale``.

        Parameters
        ----------
        scale : float
            Non-zero, finite scale factor.

        Returns
        -------
        Self
            The scale-corrected matrix of this type.

        Raises
        ------
        ValueError
            If ``scale`` is zero or not finite.
        """
        if scale == 0 or not math.isfinite(scale):
            raise ValueError(f"Scale must be non-zero and finite, got {scale}")
        S = np.diag([scale, scale, 1.0])
        S_inv = np.diag([1.0 / scale, 1.0 / scale, 1.0])
        return self.from_homogeneous(S_inv @ self.homogeneous @ S)
