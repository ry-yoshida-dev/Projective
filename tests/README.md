# tests

## Overview

Pytest suite for `projective` perspective transforms (homography and affine).

## Components

| Component | Description |
|-----------|-------------|
| [test_homography_transform.py](./test_homography_transform.py) | `HomographyMatrix.inverse` and matrix validation |
| [test_homography_projective.py](./test_homography_projective.py) | `HomographyMatrix.projective_transformation` |
| [test_affine_projective.py](./test_affine_projective.py) | `AffineMatrix.projective_transformation` |

## Examples

```bash
uv run pytest
```
