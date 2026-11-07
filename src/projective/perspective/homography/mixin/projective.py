from __future__ import annotations

from collections.abc import Callable

import numpy as np
from typing_extensions import override

from ....types import FloatArray
from ...mixin.projective import PerspectiveProjectiveMixin
from ..protocols.inverse import HomographyInverseProtocol


class HomographyProjectiveMixin(PerspectiveProjectiveMixin):
    """Project 2D points with a homography matrix."""

    value: FloatArray

    @override
    def projective_transformation(
        self: HomographyInverseProtocol,
        points: FloatArray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> FloatArray:
        """
        Apply the homography to 2D points.

        Parameters
        ----------
        points : FloatArray
            Input points of shape (N, 2) or (N, 3).
            (x, y) rows are promoted to homogeneous (x, y, 1) unless already (N, 3).
        is_inverse : bool
            If True, apply the inverse homography.
        up_axis_index : int
            Index of the homogeneous coordinate used as denominator (0, 1, or 2).

        Returns
        -------
        FloatArray
            Projected points of shape (N, 2).
        """
        if up_axis_index not in (0, 1, 2):
            raise ValueError("Up axis index must be 0, 1, or 2")
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Input points must have shape (N, 2) or (N, 3)")

        if points.shape[1] == 2:
            ones = np.ones((points.shape[0], 1), dtype=np.float64)
            if up_axis_index == 0:
                points = np.hstack([points, ones])
            elif up_axis_index == 1:
                points = np.hstack([points[:, 0:1], ones, points[:, 1:2]])
            elif up_axis_index == 2:
                points = np.hstack([points, ones])

        matrix = self.inverse if is_inverse else self.value
        transformed = (matrix @ points.T).T

        numerator_getters: dict[int, Callable[[FloatArray], FloatArray]] = {
            0: lambda arr: arr[:, 1:],
            1: lambda arr: arr[:, (0, 2)],
            2: lambda arr: arr[:, :2],
        }
        numerator = numerator_getters[up_axis_index](transformed)
        denominators = transformed[:, up_axis_index][:, None]
        return np.asarray(numerator / denominators, dtype=np.float64)
