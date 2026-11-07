from __future__ import annotations

from functools import cached_property
from typing import Protocol

from ....types import FloatArray


class HomographyInverseProtocol(Protocol):
    """Homography container that exposes a cached matrix inverse."""

    @property
    def value(self) -> FloatArray:
        """Return the homography matrix with shape (3, 3)."""
        ...

    @cached_property
    def inverse(self) -> FloatArray:
        """Return the inverse homography matrix with shape (3, 3)."""
        ...
