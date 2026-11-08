from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
from typing_extensions import override

from ..read_only import read_only_copy
from ..types import FloatArray
from .method import PerspectiveTransformationMethod
from .mixin import PerspectiveFactoryMixin, PerspectiveProjectiveMixin


@dataclass(frozen=True)
class PerspectiveMatrix(PerspectiveFactoryMixin, PerspectiveProjectiveMixin, ABC):
    """
    2D perspective transformation matrix.

    Every operation works on the 3×3 ``homogeneous`` form of the matrix,
    so concrete classes only define their stored shape, how they convert to and from
    the homogeneous form, and which OpenCV estimator fits them to points.
    Container operations, point mapping, and estimation live in the mixins;
    this class adds shape validation and decomposition into rotation, scale, and shear.

    Attributes
    ----------
    value: FloatArray
        The stored matrix with shape ``expected_shape``.
    """

    expected_shape: ClassVar[tuple[int, int]]

    value: FloatArray

    def __post_init__(self) -> None:
        if self.value.shape != self.expected_shape:
            raise ValueError(
                f"{type(self).__name__} must have shape {self.expected_shape}, got {self.value.shape}"
            )
        # Store an independent, read-only copy so that the frozen matrix cannot change.
        object.__setattr__(self, "value", read_only_copy(self.value))

    @classmethod
    @override
    def _from_value(cls, value: FloatArray) -> Self:
        return cls(value=value)

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

    @property
    def linear_part(self) -> FloatArray:
        """
        Return the linear part [[a, b], [c, d]] normalized by H[2, 2].

        Returns
        -------
        FloatArray
            The linear part with shape (2, 2).

        Raises
        ------
        ValueError
            If H[2, 2] is zero.
        """
        return self._normalized_homogeneous[:2, :2]

    @property
    def translation(self) -> FloatArray:
        """
        Return the translation (tx, ty) normalized by H[2, 2].

        This is exact only when the matrix has no perspective component.

        Returns
        -------
        FloatArray
            The translation with shape (2,).

        Raises
        ------
        ValueError
            If H[2, 2] is zero.
        """
        return self._normalized_homogeneous[:2, 2]

    @property
    def rotation(self) -> float:
        """
        Return the rotation angle in radians from the first column (a, c) of the linear part.

        Returns
        -------
        float
            The rotation angle, or 0.0 when the first column is zero.
        """
        a = float(self.linear_part.item(0, 0))
        c = float(self.linear_part.item(1, 0))
        if a == 0 and c == 0:
            return 0.0
        return math.atan2(c, a)

    @property
    def scale(self) -> FloatArray:
        """
        Return the scale factors sx = sqrt(a² + c²) and sy = sqrt(b² + d²).

        Returns
        -------
        FloatArray
            The scale factors with shape (2,).
        """
        return np.asarray(np.linalg.norm(self.linear_part, axis=0), dtype=np.float64)

    @property
    def shear(self) -> FloatArray:
        """
        Return the shear factors (ab + cd) / (a² + c²) and (ab + cd) / (b² + d²).

        A factor is 0 when its column of the linear part is zero.

        Returns
        -------
        FloatArray
            The shear factors with shape (2,).
        """
        linear = self.linear_part
        a = float(linear.item(0, 0))
        b = float(linear.item(0, 1))
        c = float(linear.item(1, 0))
        d = float(linear.item(1, 1))
        column_dot = a * b + c * d
        squared_norm_x = a**2 + c**2
        squared_norm_y = b**2 + d**2
        shear_x = column_dot / squared_norm_x if squared_norm_x != 0 else 0.0
        shear_y = column_dot / squared_norm_y if squared_norm_y != 0 else 0.0
        return np.array([shear_x, shear_y])

    @property
    def has_perspective(self) -> bool:
        """
        Return True if the perspective components H[2, 0] or H[2, 1] are non-zero.

        Returns
        -------
        bool
            True if the transformation is not affine.
        """
        return not np.allclose(self.homogeneous[2, :2], 0)

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(value.shape={self.value.shape}, transform_type={self.transform_type})"
