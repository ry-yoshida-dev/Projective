from __future__ import annotations

from typing import ClassVar, Self, cast

import cv2
import numpy as np
from opencv_utility import OpenCVOutlierFilteringFlag
from typing_extensions import override

from ...types import FloatArray, MaskArray
from ..matrix import PerspectiveMatrix
from ..method import PerspectiveTransformationMethod


class AffineMatrix(PerspectiveMatrix):
    """
    Affine transformation matrix [[a, b, tx], [c, d, ty]].

    Estimation fits a partial affine (similarity) transform with
    ``cv2.estimateAffinePartial2D``, which has 4 degrees of freedom.

    Attributes
    ----------
    value: FloatArray
        The affine matrix with shape (2, 3).
    """

    expected_shape: ClassVar[tuple[int, int]] = (2, 3)
    minimum_point_count: ClassVar[int] = 2

    @property
    @override
    def homogeneous(self) -> FloatArray:
        """
        Return the matrix with the row [0, 0, 1] appended.

        Returns
        -------
        FloatArray
            The homogeneous matrix with shape (3, 3).
        """
        return np.vstack([self.value, [0.0, 0.0, 1.0]])

    @classmethod
    @override
    def from_homogeneous(cls, matrix: FloatArray) -> Self:
        """
        Create an affine matrix from a 3×3 homogeneous matrix whose last row is [0, 0, 1].

        Parameters
        ----------
        matrix : FloatArray
            Homogeneous matrix with shape (3, 3).

        Returns
        -------
        Self
            The affine matrix.

        Raises
        ------
        ValueError
            If the matrix is not 3×3 or its last row is not [0, 0, 1].
        """
        if matrix.shape != (3, 3):
            raise ValueError(
                f"Homogeneous matrix must have shape (3, 3), got {matrix.shape}"
            )
        if not np.allclose(matrix[2:, :], [[0.0, 0.0, 1.0]]):
            raise ValueError(
                f"Homogeneous matrix with last row {matrix[2:, :]} is not affine"
            )
        return cls(value=matrix[:2].copy())

    @classmethod
    @override
    def _estimate(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag,
        ransac_th: float,
    ) -> tuple[FloatArray | None, MaskArray | None]:
        return cast(
            tuple[FloatArray | None, MaskArray | None],
            cv2.estimateAffinePartial2D(
                from_=origin_points,
                to=destination_points,
                method=outlier_filtering_flag.cv2_flag,
                ransacReprojThreshold=ransac_th,
            ),
        )

    @property
    @override
    def transform_type(self) -> PerspectiveTransformationMethod:
        """
        Return the affine transformation type.

        Returns
        -------
        PerspectiveTransformationMethod
            Always ``PerspectiveTransformationMethod.AFFINE``.
        """
        return PerspectiveTransformationMethod.AFFINE
