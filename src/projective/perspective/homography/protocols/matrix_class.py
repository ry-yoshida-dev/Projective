from __future__ import annotations

from typing import Protocol

from ....types import FloatArray
from .matrix_value import HomographyMatrixValueProtocol


class HomographyMatrixClassProtocol(Protocol):
    """Protocol for the ``HomographyMatrix`` class object used by factory mixins."""

    @staticmethod
    def _validate_points(
        origin_points: FloatArray,
        destination_points: FloatArray,
    ) -> bool: ...

    def __call__(self, *, value: FloatArray) -> HomographyMatrixValueProtocol:
        """Create a homography matrix from its value."""
        ...
