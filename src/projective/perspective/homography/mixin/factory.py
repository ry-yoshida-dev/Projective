from __future__ import annotations

from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from opencv_utility import OpenCVOutlierFilteringFlag
from typing_extensions import override

from ....types import FloatArray, MaskArray
from ...mixin.factory import PerspectiveFactoryMixin

if TYPE_CHECKING:
    from ..matrix import HomographyMatrix


class HomographyFactoryMixin(PerspectiveFactoryMixin):
    """Factory methods for homography matrix estimation."""

    @classmethod
    @override
    def create_identity_matrix(cls) -> HomographyMatrix:
        """
        Create an identity homography matrix.

        Returns
        -------
        HomographyMatrix
            Identity homography matrix.
        """
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=np.eye(3, 3))

    @classmethod
    def from_unnormalized_value(
        cls,
        value: FloatArray,
    ) -> HomographyMatrix:
        """
        Create a homography matrix from an unnormalized value.

        Parameters
        ----------
        value: FloatArray
            Unnormalized value of the homography matrix (shape: (3, 3)).

        Returns
        -------
        HomographyMatrix:
        """
        if value.shape != (3, 3):
            raise ValueError("Unnormalized value must be a 3x3 matrix")
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=value / value.item(2, 2))

    @classmethod
    @override
    def create_from_points(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[HomographyMatrix, MaskArray]:
        """
        Create a homography matrix from a set of origin and destination points.

        Parameters
        ----------
        origin_points : FloatArray
            Origin points (n, 2).
        destination_points : FloatArray
            Destination points (n, 2).
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Outlier filtering flag.
        ransac_th : float
            RANSAC threshold.

        Returns
        -------
        tuple[HomographyMatrix, MaskArray]
            Homography matrix and inlier mask of shape (N, 1).
        """
        if not cls._validate_points(origin_points, destination_points):
            raise ValueError("Invalid points for homography matrix creation")

        origin_points = np.asarray(origin_points, dtype=np.float32)
        destination_points = np.asarray(destination_points, dtype=np.float32)

        matrix, mask = cast(
            tuple[FloatArray, MaskArray],
            cv2.findHomography(
                srcPoints=origin_points,
                dstPoints=destination_points,
                method=outlier_filtering_flag.cv2_flag,
                ransacReprojThreshold=ransac_th,
            ),
        )
        from ..matrix import HomographyMatrix as _HomographyMatrix

        return _HomographyMatrix(value=matrix), mask
