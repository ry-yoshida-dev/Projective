# perspective

## Overview

2D perspective transforms (affine 2×3 and homography 3×3). All operations — point mapping, inverse, scale correction, and decomposition into rotation, scale, and shear — are defined once on the 3×3 homogeneous form in `PerspectiveMatrix` and its mixins.

## Components

| Component | Description |
|-----------|-------------|
| [matrix.py](./matrix.py) | Abstract `PerspectiveMatrix`: shape validation and decomposition into rotation, scale, and shear |
| [mixin/](./mixin/README.md) | Shared transform, point projection, and estimation mixins |
| [matrices/](./matrices/README.md) | Concrete `AffineMatrix` and `HomographyMatrix` |
| [method.py](./method.py) | `PerspectiveTransformationMethod` enum mapping to matrix classes and OpenCV motion types |
| [register.py](./register.py) | `register_perspective_matrix` — wrap an array, return identity for None, or pass through a matching typed matrix |

## Examples

```python
import numpy as np
from projective import (
    HomographyMatrix,
    PerspectiveTransformationMethod,
    register_perspective_matrix,
)

origin = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
destination = origin * 2

# Estimate a concrete type directly, or pick it from a method
homography, mask = HomographyMatrix.create_from_points(origin, destination)
affine, _ = PerspectiveTransformationMethod.AFFINE.perspective_class.create_from_points(
    origin, destination
)

mapped = homography.projective_transformation(origin)
restored = homography.projective_transformation(mapped, is_inverse=True)
inliers = origin[mask.ravel() == 1]

# Wrap a raw array, or get the identity with None
T = register_perspective_matrix(np.eye(3), PerspectiveTransformationMethod.HOMOGRAPHY)
identity = register_perspective_matrix(None, PerspectiveTransformationMethod.AFFINE)
```
