from __future__ import annotations

import numpy as np

from .types import FloatArray


def read_only_copy(array: FloatArray) -> FloatArray:
    """
    Return a read-only copy of an array.

    Matrix containers store this copy so that neither the caller's array
    nor in-place writes through ``value`` can change a frozen matrix.

    Parameters
    ----------
    array : FloatArray
        The array to copy.

    Returns
    -------
    FloatArray
        A copy with the same dtype and shape whose ``writeable`` flag is False.
    """
    copied = np.array(array, copy=True)
    copied.flags.writeable = False
    return copied
