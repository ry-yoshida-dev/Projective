from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from .matrix import PerspectiveMatrix


class PerspectiveTransformationMethod(Enum):
    """
    Type of perspective transformation matrix.

    Attributes
    ----------
    AFFINE: Uses the affine transformation matrix.
    HOMOGRAPHY: Uses the homography transformation matrix.
    """

    AFFINE = "Affine"
    HOMOGRAPHY = "Homography"

    @property
    def to_cv2_motion_type(self) -> int:
        """
        Return the OpenCV motion type for the transformation method.

        Returns
        -------
        int
            ``cv2.MOTION_AFFINE`` or ``cv2.MOTION_HOMOGRAPHY``.
        """
        match self:
            case self.AFFINE:
                return cv2.MOTION_AFFINE
            case self.HOMOGRAPHY:
                return cv2.MOTION_HOMOGRAPHY

    @property
    def perspective_class(self) -> type[PerspectiveMatrix]:
        """
        Return the perspective class for the transformation method.

        Returns
        -------
        type[PerspectiveMatrix]
            The perspective class for the transformation method.
        """
        match self:
            case self.AFFINE:
                from .matrices.affine import AffineMatrix

                return AffineMatrix
            case self.HOMOGRAPHY:
                from .matrices.homography import HomographyMatrix

                return HomographyMatrix
