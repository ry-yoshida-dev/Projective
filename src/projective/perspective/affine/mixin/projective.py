from __future__ import annotations

import numpy as np
from typing_extensions import override

from ....types import FloatArray
from ...mixin.projective import PerspectiveProjectiveMixin


class AffineProjectiveMixin(PerspectiveProjectiveMixin):
    """Project 2D points with a partial affine matrix."""

    value: FloatArray

    @override
    def projective_transformation(
        self,
        points: FloatArray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> FloatArray:
        """
        Apply the affine matrix to 2D points.

        Parameters
        ----------
        points : FloatArray
            Input points of shape (N, 2) or (N, 3).
        is_inverse : bool
            Not supported for affine matrices.
        up_axis_index : int
            Ignored for affine matrices.

        Returns
        -------
        FloatArray
            Projected points of shape (N, 2).
        """
        del up_axis_index
        if is_inverse:
            raise ValueError(
                "Inverse transformation is not supported for affine matrix"
            )
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Input points must have shape (N, 2) or (N, 3)")

        if points.shape[1] == 2:
            ones = np.ones((points.shape[0], 1))
            points_3d = np.hstack([points, ones])
        else:
            points_3d = points

        transformed = (self.value @ points_3d.T).T

        if points.shape[1] == 3:
            w = points_3d[:, 2]
            transformed = transformed / w[:, np.newaxis]

        return np.asarray(transformed, dtype=np.float64)
