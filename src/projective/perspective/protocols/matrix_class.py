from __future__ import annotations

from typing import Protocol

from ...types import FloatArray
from .matrix_value import MatrixValueProtocol


class PerspectiveMatrixClassProtocol(Protocol):
    """Protocol for perspective matrix class objects used by factory mixins."""

    @staticmethod
    def _validate_points(
        origin_points: FloatArray,
        destination_points: FloatArray,
    ) -> bool: ...

    def __call__(self, *, value: FloatArray) -> MatrixValueProtocol:
        """Create a perspective matrix from its value."""
        ...
