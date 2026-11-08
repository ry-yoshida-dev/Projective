from __future__ import annotations

from typing import ClassVar, Self, cast

import cv2
from opencv_utility import OpenCVOutlierFilteringFlag
from typing_extensions import override

from ...types import FloatArray, MaskArray
from ..matrix import PerspectiveMatrix
from ..method import PerspectiveTransformationMethod


class HomographyMatrix(PerspectiveMatrix):
    """
    Homography matrix H representing a 2D projective transformation.

    The 3x3 homography matrix H has the form:
        [a  b  tx]
    H = [c  d  ty]
        [p1 p2 h ]

    where a, b, c, d are the linear components, tx, ty the translation,
    p1, p2 the perspective components, and h the overall scale.
    Estimation uses ``cv2.findHomography``, which has 8 degrees of freedom.

    Attributes
    ----------
    value: FloatArray
        The homography matrix with shape (3, 3).
    """

    expected_shape: ClassVar[tuple[int, int]] = (3, 3)
    minimum_point_count: ClassVar[int] = 4

    @property
    @override
    def homogeneous(self) -> FloatArray:
        """
        Return the homography itself.

        Returns
        -------
        FloatArray
            The homogeneous matrix with shape (3, 3).
        """
        return self.value

    @classmethod
    @override
    def from_homogeneous(cls, matrix: FloatArray) -> Self:
        """
        Create a homography from a 3×3 homogeneous matrix.

        Parameters
        ----------
        matrix : FloatArray
            Homogeneous matrix with shape (3, 3).

        Returns
        -------
        Self
            The homography matrix.

        Raises
        ------
        ValueError
            If the matrix is not 3×3.
        """
        return cls(value=matrix)

    @classmethod
    def from_unnormalized_value(cls, value: FloatArray) -> Self:
        """
        Create a homography matrix from an unnormalized value, scaled so that H[2, 2] == 1.

        Parameters
        ----------
        value: FloatArray
            Unnormalized value of the homography matrix (shape: (3, 3)).

        Returns
        -------
        Self
            The normalized homography matrix.

        Raises
        ------
        ValueError
            If the value is not 3x3 or H[2, 2] is zero.
        """
        if value.shape != (3, 3):
            raise ValueError(
                f"Unnormalized value must be a 3x3 matrix, got shape {value.shape}"
            )
        if value.item(2, 2) == 0:
            raise ValueError(
                "Unnormalized value with H[2, 2] == 0 cannot be normalized"
            )
        return cls(value=value / value.item(2, 2))

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
            cv2.findHomography(
                srcPoints=origin_points,
                dstPoints=destination_points,
                method=outlier_filtering_flag.cv2_flag,
                ransacReprojThreshold=ransac_th,
            ),
        )

    @property
    @override
    def transform_type(self) -> PerspectiveTransformationMethod:
        """
        Return the homography transformation type.

        Returns
        -------
        PerspectiveTransformationMethod
            Always ``PerspectiveTransformationMethod.HOMOGRAPHY``.
        """
        return PerspectiveTransformationMethod.HOMOGRAPHY
