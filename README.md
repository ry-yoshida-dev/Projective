# Projective

## Overview

`projective` is a Python package for projective geometry: epipolar constraints (`EssentialMatrix`, `FundamentalMatrix`) and 2D perspective transforms that can be wrapped from arrays or estimated from point correspondences with an inlier mask.

| Transform | Class | Shape | OpenCV estimator | Degrees of freedom |
|---|---|---|---|---|
| Partial affine | `AffineMatrix` | 2×3 | `estimateAffinePartial2D` | 4 |
| Homography | `HomographyMatrix` | 3×3 | `findHomography` | 8 |

## Installation

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from projective import (
    AffineMatrix,
    EssentialMatrix,
    HomographyMatrix,
    PerspectiveMatrix,
    PerspectiveTransformationMethod,
)

E = EssentialMatrix(value=np.eye(3))

points = np.array([[0.0, 0.0], [100.0, 50.0]])
affine = AffineMatrix(value=np.array([[1, 0, 10], [0, 1, 20]], dtype=np.float64))
warped = affine.projective_transformation(points)

origin = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
destination = origin * 2
homography, mask = PerspectiveMatrix.from_points(
    origin,
    destination,
    transform_type=PerspectiveTransformationMethod.HOMOGRAPHY,
)
mapped = homography.projective_transformation(points)
```
