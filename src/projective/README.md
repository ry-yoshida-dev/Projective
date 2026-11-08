# projective

## Overview

Utilities for **projective geometry**: epipolar constraints at the package root (`EssentialMatrix`, `FundamentalMatrix`), and 2D **perspective transforms** (affine / homography) in the [`perspective/`](./perspective/README.md) subpackage.

## Components

| Component | Description |
|-----------|-------------|
| [estimation_error.py](./estimation_error.py) | `EstimationError` raised when OpenCV cannot estimate a matrix |
| [types.py](./types.py) | Shared NumPy type aliases (`FloatArray` for float32/float64, `MaskArray` for uint8) |
| [perspective/](./perspective/README.md) | Affine and homography matrices; estimation returns `(matrix, mask)` |
| [essential_matrix.py](./essential_matrix.py) | Immutable `EssentialMatrix` (3×3) |
| [fundamental_matrix.py](./fundamental_matrix.py) | Immutable `FundamentalMatrix` (3×3) |
