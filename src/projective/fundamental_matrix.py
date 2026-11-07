from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
from opencv_utility import OpenCVOutlierFilteringFlag

from .types import FloatArray, MaskArray


@dataclass(frozen=True)
class FundamentalMatrix:
    """
    FundamentalMatrix is a class that represents a fundamental matrix.

    Attributes
    ----------
    value: FloatArray
        The fundamental matrix with shape (3, 3).
    """

    value: FloatArray

    def __post_init__(self) -> None:
        if self.value.shape != (3, 3):
            raise ValueError("Fundamental matrix must be a 3x3 matrix")

    @classmethod
    def from_points(
        cls,
        points1: FloatArray,
        points2: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        ransac_th: float = 3.0,
    ) -> tuple[FundamentalMatrix, MaskArray]:
        """
        Create a FundamentalMatrix from two sets of points.

        Parameters
        ----------
        points1: FloatArray
            Array of shape (N, 2) containing the coordinates of points in the first image.
        points2: FloatArray
            Array of shape (N, 2) containing the coordinates of points in the second image.
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Estimation method passed to ``cv2.findFundamentalMat``.
        ransac_th: float
            RANSAC reprojection threshold (pixels).

        Returns
        -------
        tuple[FundamentalMatrix, MaskArray]: The fundamental matrix and the mask.
        """
        if points1.shape != points2.shape:
            raise ValueError(
                f"Points arrays must have the same shape, got {points1.shape} and {points2.shape}"
            )
        if points1.ndim != 2:
            raise ValueError(f"Points arrays must be 2D, got {points1.ndim}D")
        if points1.shape[0] < 8:
            raise ValueError(
                f"At least 8 point pairs are required to estimate a fundamental matrix, got {points1.shape[0]}"
            )
        if points1.shape[1] != 2:
            raise ValueError(
                f"Points arrays must have 2 columns, got {points1.shape[1]}"
            )

        F, mask = cast(
            tuple[FloatArray, MaskArray],
            cv2.findFundamentalMat(
                points1,
                points2,
                outlier_filtering_flag.fundamental_matrix_flag,
                ransac_th,
            ),
        )
        return cls(value=F), mask
