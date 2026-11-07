from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ..types import FloatArray
from .method import PerspectiveTransformationMethod
from .mixin import (
    PerspectiveFactoryMixin,
    PerspectiveProjectiveMixin,
    PerspectiveTransformMixin,
)


@dataclass(frozen=True)
class PerspectiveMatrix(
    PerspectiveFactoryMixin,
    PerspectiveTransformMixin,
    PerspectiveProjectiveMixin,
    ABC,
):
    """
    Container class for perspective transformation matrices.

    This class provides a unified interface for both affine and homography transformations,
    automatically selecting the appropriate matrix type based on the input.

    Attributes
    ----------
    value: FloatArray
        The perspective transformation matrix with shape (N, M).
    """

    value: FloatArray

    @property
    @abstractmethod
    def translation(self) -> FloatArray:
        """
        Get the translation component of the transformation.

        Returns
        -------
        FloatArray:
            The translation component of the transformation with shape (2,).
        """

    @property
    def rotation(self) -> float:
        """
        Return the rotation angle in radians, normalized by scale.

        Calculated from the linear transformation components:
        a, b = H[0,0], H[0,1] (or A[0,0], A[0,1])
        c, d = H[1,0], H[1,1] (or A[1,0], A[1,1])
        """
        a = self.value.item(0, 0)
        c = self.value.item(1, 0)
        sx = math.hypot(a, c)
        if sx == 0:
            return 0.0
        return math.atan2(c / sx, a / sx)

    @property
    def scale(self) -> FloatArray:
        """
        Return the scale factors (sx, sy).

        Calculated from the linear transformation components:
        sx = sqrt(a² + c²), sy = sqrt(b² + d²)

        Returns
        -------
        FloatArray:
            The scale factors with shape (2,).
        """
        a = self.value.item(0, 0)
        b = self.value.item(0, 1)
        c = self.value.item(1, 0)
        d = self.value.item(1, 1)
        return np.array([math.hypot(a, c), math.hypot(b, d)])

    @property
    @abstractmethod
    def shear(self) -> FloatArray:
        """
        Get the shear factors.

        Returns
        -------
        FloatArray:
            The shear factors with shape (2,).
        """

    @property
    @abstractmethod
    def has_perspective(self) -> bool:
        """
        Check if the transformation has perspective components.

        Returns
        -------
        bool:
            True if the transformation has perspective components, False otherwise.
        """

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value.shape={self.value.shape}, transform_type={self.transform_type})"

    @property
    @abstractmethod
    def transform_type(self) -> PerspectiveTransformationMethod:
        """
        Return the type of the transformation.

        Returns
        -------
        PerspectiveTransformationMethod:
            The type of the transformation.
        """
