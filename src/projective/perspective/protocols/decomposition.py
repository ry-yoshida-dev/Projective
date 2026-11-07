from __future__ import annotations

from typing import Protocol

from ...types import FloatArray
from ..method import PerspectiveTransformationMethod


class PerspectiveMatrixDecompositionProtocol(Protocol):
    """Protocol for decomposition properties of a perspective matrix."""

    value: FloatArray

    @property
    def translation(self) -> FloatArray:
        """Return the translation vector (tx, ty)."""
        ...

    @property
    def shear(self) -> FloatArray:
        """Return the shear factors (sx, sy)."""
        ...

    @property
    def has_perspective(self) -> bool:
        """Return True if the matrix has a perspective component."""
        ...

    @property
    def transform_type(self) -> PerspectiveTransformationMethod:
        """Return the type of the transformation."""
        ...
