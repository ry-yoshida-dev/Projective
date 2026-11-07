from __future__ import annotations

from typing import Protocol

from ....types import FloatArray
from .matrix_value import AffineMatrixValueProtocol


class AffineMatrixClassProtocol(Protocol):
    """Protocol for the ``AffineMatrix`` class object used by factory mixins."""

    @staticmethod
    def _validate_points(
        origin_points: FloatArray,
        destination_points: FloatArray,
    ) -> bool: ...

    def __call__(self, *, value: FloatArray) -> AffineMatrixValueProtocol:
        """Create an affine matrix from its value."""
        ...
