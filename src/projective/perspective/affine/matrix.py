from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ...types import FloatArray
from ..matrix import PerspectiveMatrix
from ..method import PerspectiveTransformationMethod
from .mixin import (
    AffineFactoryMixin,
    AffineProjectiveMixin,
    AffineTransformMixin,
)


@dataclass(frozen=True)
class AffineMatrix(
    AffineTransformMixin,
    AffineProjectiveMixin,
    AffineFactoryMixin,
    PerspectiveMatrix,
):
    """
    AffineMatrix is a class that represents an affine transformation matrix.

    Attributes
    ----------
    value: FloatArray
        The affine transformation matrix with shape (2, 3).
    """

    def __post_init__(self) -> None:
        """
        Post-init validation.

        Raises
        ------
        ValueError: If the affine transformation matrix is not a 2x3 matrix.
        """
        if self.value.shape != (2, 3):
            raise ValueError(
                f"Affine matrix must be a 2x3 matrix, got shape {self.value.shape}"
            )

    @property
    @override
    def translation(self) -> FloatArray:
        """
        Return the translation vector (tx, ty).

        Returns
        -------
        FloatArray:
            The translation vector (tx, ty) with shape (2,).
        """
        return self.value[:, 2]

    @property
    @override
    def shear(self) -> FloatArray:
        """
        Return shear factors along x and y axes.

        Calculated from the linear part of the affine matrix:
        [a b tx]
        [c d ty]

        Returns
        -------
        FloatArray:
            The shear factors (sx, sy) with shape (2,).
        """
        a = self.value.item(0, 0)
        b = self.value.item(0, 1)
        c = self.value.item(1, 0)
        d = self.value.item(1, 1)
        sx2 = a**2 + c**2
        sy2 = b**2 + d**2

        shear_x = (a * b + c * d) / sx2 if not sx2 == 0 else 0
        shear_y = (a * b + c * d) / sy2 if not sy2 == 0 else 0
        return np.array([shear_x, shear_y])

    @property
    @override
    def has_perspective(self) -> bool:
        """
        Return True if the affine matrix has a perspective component.

        Returns
        -------
        bool:
            True if the affine matrix has a perspective component, False otherwise.
        """
        return False

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
