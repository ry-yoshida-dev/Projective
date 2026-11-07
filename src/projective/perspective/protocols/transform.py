from __future__ import annotations

from typing import Protocol

from .matrix_value import MatrixValueProtocol


class PerspectiveTransformProtocol(Protocol):
    """Protocol for transforming the matrix container itself."""

    def scale_correction(self, scale: float) -> MatrixValueProtocol:
        """Return the matrix corrected for the given scale factor."""
        ...
