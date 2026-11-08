# perspective.matrices

## Overview

Concrete `PerspectiveMatrix` types. Each defines only its stored shape, its conversion to and from the 3×3 homogeneous form, and the OpenCV estimator that fits it to point pairs.

## Components

| Component | Description |
|-----------|-------------|
| [affine.py](./affine.py) | `AffineMatrix` (2×3), estimated with `cv2.estimateAffinePartial2D` from ≥2 point pairs |
| [homography.py](./homography.py) | `HomographyMatrix` (3×3), estimated with `cv2.findHomography` from ≥4 point pairs |
