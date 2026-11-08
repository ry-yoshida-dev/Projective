from __future__ import annotations

from abc import ABC

import numpy as np

from ...types import FloatArray
from .transform import PerspectiveTransformMixin


class PerspectiveProjectiveMixin(PerspectiveTransformMixin, ABC):
    """Project external 2D points with the matrix (homogeneous map + dehomogenize)."""

    def projective_transformation(
        self,
        points: FloatArray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> FloatArray:
        """
        Apply the transformation to 2D points.

        Parameters
        ----------
        points : FloatArray
            Input points of shape (N, 2) or (N, 3).
            (N, 2) rows are promoted to homogeneous coordinates by inserting 1
            at ``up_axis_index``; (N, 3) rows are used as is.
        is_inverse : bool
            If True, apply the inverse transformation.
        up_axis_index : int
            Index of the homogeneous coordinate used as denominator (0, 1, or 2).

        Returns
        -------
        FloatArray
            Projected points of shape (N, 2): the remaining two coordinates
            divided by the coordinate at ``up_axis_index``.

        Raises
        ------
        ValueError
            If ``up_axis_index`` is not 0, 1, or 2, or the points are not (N, 2) or (N, 3).
        """
        if up_axis_index not in (0, 1, 2):
            raise ValueError(f"Up axis index must be 0, 1, or 2, got {up_axis_index}")
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError(
                f"Input points must have shape (N, 2) or (N, 3), got {points.shape}"
            )

        homogeneous_points = (
            np.insert(points, up_axis_index, 1.0, axis=1)
            if points.shape[1] == 2
            else points
        )
        matrix = self.inverse if is_inverse else self.homogeneous
        transformed = homogeneous_points @ matrix.T

        numerator = np.delete(transformed, up_axis_index, axis=1)
        denominators = transformed[:, up_axis_index : up_axis_index + 1]
        return np.asarray(numerator / denominators, dtype=np.float64)
