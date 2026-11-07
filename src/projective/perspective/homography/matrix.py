from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ...types import FloatArray
from ..matrix import PerspectiveMatrix
from ..method import PerspectiveTransformationMethod
from .mixin import (
    HomographyFactoryMixin,
    HomographyProjectiveMixin,
    HomographyTransformMixin,
)


@dataclass(frozen=True)
class HomographyMatrix(
    HomographyTransformMixin,
    HomographyProjectiveMixin,
    HomographyFactoryMixin,
    PerspectiveMatrix,
):
    """
    Homography matrix H representing a 2D projective transformation.

    The 3x3 homography matrix H has the form:
        [a  b  tx]
    H = [c  d  ty]
        [p1 p2 1 ]

    Where:
    - a, b, c, d: linear transformation components (rotation, scale, shear)
    - tx, ty: translation components
    - p1, p2: perspective transformation components

    Attributes
    ----------
    value: FloatArray (inherited from base)
        Value of the homography matrix with shape (3, 3).
    """

    def __post_init__(self) -> None:
        if self.value.shape != (3, 3):
            raise ValueError("Homography matrix must be a 3x3 matrix")

    @property
    @override
    def translation(self) -> FloatArray:
        """
        Approximate translation vector (tx, ty) from homography.

        Extracts the translation components from the third column of H:
        H[0,2] = tx, H[1,2] = ty
        """
        return self.value[:2, 2]

    @property
    @override
    def shear(self) -> FloatArray:
        """
        Return the shear components of the homography matrix.

        Returns
        -------
        FloatArray:
            The shear of the homography matrix.
        """
        A = self.value[:2, :2] / self.value.item(2, 2)
        # QR: Q = rotation/reflection, R = upper-triangular (scale on diag., shear off-diag.)
        _, R = np.linalg.qr(A)

        sx = float(np.linalg.norm(R[:, 0]))
        sy = float(np.linalg.norm(R[:, 1]))

        shear_x = R.item(0, 1) / sy if sy != 0 else 0
        shear_y = R.item(1, 0) / sx if sx != 0 else 0
        return np.array([shear_x, shear_y])

    @property
    @override
    def has_perspective(self) -> bool:
        """
        Return True if the homography has a significant perspective component.

        Checks if the perspective components p1 = H[2,0] and p2 = H[2,1]
        are non-zero, indicating a perspective transformation rather than
        just an affine transformation.
        """
        return not np.allclose(self.value[2, :2], 0)

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
