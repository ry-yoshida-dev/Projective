# tests

## Overview

Pytest suite for `projective`: perspective transforms (homography and affine) and epipolar matrices.

## Components

| Component | Description |
|-----------|-------------|
| [test_perspective_matrix.py](./test_perspective_matrix.py) | Construction, array views, homogeneous conversion, inverse, normalization, and `scale_correction` |
| [test_perspective_projection.py](./test_perspective_projection.py) | `projective_transformation` for affine and homography |
| [test_perspective_estimation.py](./test_perspective_estimation.py) | `create_from_points`, input validation, and `EstimationError` |
| [test_perspective_decomposition.py](./test_perspective_decomposition.py) | Rotation, scale, shear, translation, `has_perspective`, and `repr` |
| [test_register.py](./test_register.py) | `register_perspective_matrix` |
| [test_epipolar.py](./test_epipolar.py) | `EssentialMatrix` and `FundamentalMatrix` |

## Examples

```bash
uv run pytest
```
