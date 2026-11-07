from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from ...types import FloatArray

if TYPE_CHECKING:
    from ..matrix import PerspectiveMatrix


class PerspectiveTransformMixin(ABC):
    """Transform the matrix container itself (not external point coordinates)."""

    value: FloatArray

    @property
    def column_vector(self) -> FloatArray:
        """
        Return the column vector of the matrix.

        Returns
        -------
        FloatArray
            Column vector with shape (value.size, 1).
        """
        return self.value.reshape(-1, 1)

    @property
    def row_vector(self) -> FloatArray:
        """
        Return the row vector of the matrix.

        Returns
        -------
        FloatArray
            Row vector with shape (1, value.size).
        """
        return self.value.reshape(1, -1)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the matrix.

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
        Return the flattened matrix.

        Returns
        -------
        FloatArray
            One-dimensional view with shape (value.size,).
        """
        return self.value.flatten()

    @abstractmethod
    def scale_correction(
        self,
        scale: float,
    ) -> PerspectiveMatrix:
        """
        Correct the scale of the perspective matrix.

        Parameters
        ----------
        scale : float
            Scale factor.

        Returns
        -------
        PerspectiveMatrix
            The perspective matrix with corrected scale.
        """
