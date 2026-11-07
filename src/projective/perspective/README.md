# perspective

## Overview

2D perspective transformation using **partial affine** (2×3, similarity-like) and **homography** (3×3) matrices: shared container API, optional estimation from corresponding points (with inlier **mask**), and registration helpers that normalize raw arrays into the right matrix type.

## Components

| Component | Description |
|-----------|-------------|
| [method.py](./method.py) | `PerspectiveTransformationMethod` enum and OpenCV motion-type mapping |
| [matrix.py](./matrix.py) | Abstract `PerspectiveMatrix` dataclass (decomposition properties, shared API) |
| [register.py](./register.py) | `register_perspective_matrix` — wrap ndarray or pass through typed matrices |
| [protocols/](./protocols/README.md) | Shared `Protocol` types for matrix value, class, transform, projective, and decomposition |
| [mixin/](./mixin/README.md) | Shared factory, transform (matrix self), and projective (point map) mixins for `PerspectiveMatrix` |
| [affine/](./affine/README.md) | Partial affine (`AffineMatrix`, 2×3) implementation package |
| [homography/](./homography/README.md) | Homography (`HomographyMatrix`, 3×3) implementation package |

## Examples

Wrap a matrix with `register_perspective_matrix` (or estimate from point pairs with `PerspectiveMatrix.from_points`), then map 2D points with `projective_transformation`. Estimators return `(matrix, mask)` like `EssentialMatrix.from_points` / `FundamentalMatrix.from_points`.

```python
import numpy as np
from projective import (
    PerspectiveMatrix,
    PerspectiveTransformationMethod,
    register_perspective_matrix,
)

# --- Homography: wrap an existing 3×3 ndarray ---
H = np.array(
    [[1.0, 0.1, 10.0], [0.0, 1.0, 5.0], [0.0002, 0.0, 1.0]],
    dtype=np.float64,
)
T = register_perspective_matrix(H, PerspectiveTransformationMethod.HOMOGRAPHY)
src = np.array([[0.0, 0.0], [100.0, 50.0]], dtype=np.float64)
dst = T.projective_transformation(src)  # shape (N, 2)

# --- Estimate from origin ↔ destination points (≥4 pairs each) ---
origin = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
destination = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=np.float64)
T_fit, mask = PerspectiveMatrix.from_points(
    origin,
    destination,
    transform_type=PerspectiveTransformationMethod.HOMOGRAPHY,
)
mapped = T_fit.projective_transformation(src)
inliers = origin[mask.ravel() == 1]

# --- Partial affine: pass a 2×3 matrix or estimate with AFFINE ---
A = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float64)
T_affine = register_perspective_matrix(A, PerspectiveTransformationMethod.AFFINE)
dst_affine = T_affine.projective_transformation(src)

T_affine_fit, affine_mask = PerspectiveMatrix.from_points(
    origin,
    destination,
    transform_type=PerspectiveTransformationMethod.AFFINE,
)
```
