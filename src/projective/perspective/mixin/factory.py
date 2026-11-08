from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
from opencv_utility import OpenCVOutlierFilteringFlag

from ...estimation_error import EstimationError
from ...types import FloatArray, MaskArray
from ..method import PerspectiveTransformationMethod
from .transform import PerspectiveTransformMixin

if TYPE_CHECKING:
    from ..matrix import PerspectiveMatrix


class PerspectiveFactoryMixin(PerspectiveTransformMixin, ABC):
    """Create matrices as the identity or by estimating them from point pairs."""

    minimum_point_count: ClassVar[int]

    @classmethod
    @abstractmethod
    def _from_value(cls, value: FloatArray) -> Self:
        """Create the matrix from its stored value."""

    @classmethod
    @abstractmethod
    def _estimate(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag,
        ransac_th: float,
    ) -> tuple[FloatArray | None, MaskArray | None]:
        """Run the OpenCV estimator on validated float32 points."""

    @classmethod
    def from_points(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[PerspectiveMatrix, MaskArray]:
        """
        Estimate a perspective matrix of ``transform_type`` from origin and destination point pairs.

        Parameters
        ----------
        origin_points : FloatArray
            Origin points (n, 2).
        destination_points : FloatArray
            Destination points (n, 2).
        transform_type : PerspectiveTransformationMethod
            Affine or homography estimation.
        outlier_filtering_flag : OpenCVOutlierFilteringFlag
            Outlier filtering method passed to OpenCV.
        ransac_th : float
            RANSAC reprojection threshold.

        Returns
        -------
        tuple[PerspectiveMatrix, MaskArray]
            The estimated perspective matrix and inlier mask of shape (N, 1).

        Raises
        ------
        ValueError
            If the points are invalid for the requested estimator.
        EstimationError
            If OpenCV cannot estimate a matrix from the points.
        """
        return transform_type.perspective_class.create_from_points(
            origin_points,
            destination_points,
            outlier_filtering_flag,
            ransac_th,
        )

    @classmethod
    def create_identity_matrix(cls) -> Self:
        """
        Create the identity matrix.

        Returns
        -------
        Self
            The identity matrix of this type.
        """
        return cls.from_homogeneous(np.eye(3))

    @classmethod
    def create_from_points(
        cls,
        origin_points: FloatArray,
        destination_points: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[Self, MaskArray]:
        """
        Estimate the matrix from origin and destination point pairs.

        Parameters
        ----------
        origin_points : FloatArray
            Origin points (n, 2), n >= ``minimum_point_count``.
        destination_points : FloatArray
            Destination points (n, 2).
        outlier_filtering_flag : OpenCVOutlierFilteringFlag
            Outlier filtering method passed to OpenCV.
        ransac_th : float
            RANSAC reprojection threshold.

        Returns
        -------
        tuple[Self, MaskArray]
            The estimated matrix and inlier mask of shape (N, 1).

        Raises
        ------
        ValueError
            If the points are not (n, 2), differ in shape, or are too few.
        EstimationError
            If OpenCV cannot estimate a matrix from the points, e.g. collinear points.
        """
        if origin_points.ndim != 2 or origin_points.shape[1] != 2:
            raise ValueError(
                f"Origin points must have shape (N, 2), got {origin_points.shape}"
            )
        if destination_points.shape != origin_points.shape:
            raise ValueError(
                f"Destination points must have the same shape as origin points, got {destination_points.shape} and {origin_points.shape}"
            )
        if origin_points.shape[0] < cls.minimum_point_count:
            raise ValueError(
                f"At least {cls.minimum_point_count} point pairs are required, got {origin_points.shape[0]}"
            )

        matrix, mask = cls._estimate(
            np.asarray(origin_points, dtype=np.float32),
            np.asarray(destination_points, dtype=np.float32),
            outlier_filtering_flag,
            ransac_th,
        )
        if matrix is None or mask is None:
            raise EstimationError(
                f"OpenCV could not estimate {cls.__name__} from the points"
            )
        return cls._from_value(matrix), mask
