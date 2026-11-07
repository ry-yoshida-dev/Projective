from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ...types import FloatArray

if TYPE_CHECKING:
    pass


class PerspectiveProjectiveMixin(ABC):
    """Project external 2D points with the matrix (homogeneous map + dehomogenize)."""

    value: FloatArray

    @abstractmethod
    def projective_transformation(
        self,
        points: FloatArray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> FloatArray:
        """
        Apply the perspective matrix to 2D points.

        Parameters
        ----------
        points : FloatArray
            Input points of shape (N, 2) or (N, 3).
        is_inverse : bool
            If True, apply the inverse matrix.
        up_axis_index : int
            Homography only: index of the homogeneous coordinate used as denominator.

        Returns
        -------
        FloatArray
            Projected points of shape (N, 2).
        """
