from __future__ import annotations

from typing import Protocol

from ...types import FloatArray


class PerspectiveProjectiveProtocol(Protocol):
    """Protocol for projecting 2D points with a perspective matrix."""

    value: FloatArray

    def projective_transformation(
        self,
        points: FloatArray,
        is_inverse: bool = False,
        up_axis_index: int = 2,
    ) -> FloatArray:
        """Project 2D points of shape (N, 2) or (N, 3) to shape (N, 2)."""
        ...
