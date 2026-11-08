from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
from opencv_utility import OpenCVOutlierFilteringFlag

from .estimation_error import EstimationError
from .read_only import read_only_copy
from .types import FloatArray, MaskArray


@dataclass(frozen=True)
class EssentialMatrix:
    """
    EssentialMatrix is a class that represents an essential matrix.

    Attributes
    ----------
    value: FloatArray
        The essential matrix with shape (3, 3).
    """

    value: FloatArray

    def __post_init__(self) -> None:
        if self.value.shape != (3, 3):
            raise ValueError("Essential matrix must be a 3x3 matrix")
        # Store an independent, read-only copy so that the frozen matrix cannot change.
        object.__setattr__(self, "value", read_only_copy(self.value))

    @classmethod
    def from_points(
        cls,
        points1: FloatArray,
        points2: FloatArray,
        intrinsic_matrix: FloatArray,
        outlier_filtering_flag: OpenCVOutlierFilteringFlag = OpenCVOutlierFilteringFlag.RANSAC,
        prob: float = 0.999,
        threshold: float = 1.0,
    ) -> tuple[EssentialMatrix, MaskArray]:
        """
        Create an essential matrix from a set of points and an intrinsic matrix.

        Parameters
        ----------
        points1: FloatArray
            Points in the first image.
        points2: FloatArray
            Points in the second image.
        intrinsic_matrix: FloatArray
            Intrinsic matrix.
        outlier_filtering_flag: OpenCVOutlierFilteringFlag
            Robust estimation method passed to ``cv2.findEssentialMat``.
        prob: float
            Confidence level for RANSAC / LMedS methods.
        threshold: float
            Maximum distance from a point to an epipolar line (pixels) for RANSAC.

        Returns
        -------
        tuple[EssentialMatrix, MaskArray]:
            - EssentialMatrix: The estimated essential matrix.
            - MaskArray: The mask of the estimated essential matrix.

        Raises
        ------
        ValueError
            If the points or the intrinsic matrix have invalid shapes.
        EstimationError
            If OpenCV cannot estimate an essential matrix from the points.
        """
        if points1.shape != points2.shape:
            raise ValueError("Points1 and points2 must have the same shape")
        if points1.shape[0] < 5:
            raise ValueError(
                "At least 5 points are required to estimate an essential matrix"
            )
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError("Intrinsic matrix must be a 3x3 matrix")

        E, e_mask = cast(
            tuple[FloatArray | None, MaskArray | None],
            cv2.findEssentialMat(
                points1=points1,
                points2=points2,
                cameraMatrix=intrinsic_matrix,
                method=outlier_filtering_flag.cv2_flag,
                prob=prob,
                threshold=threshold,
            ),
        )
        if E is None or e_mask is None:
            raise EstimationError("cv2.findEssentialMat could not estimate a matrix")
        return cls(value=E), e_mask
