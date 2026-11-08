# perspective.mixin

## Overview

Mixins that `PerspectiveMatrix` combines, split by concern. Each is written once for every matrix type on top of the 3×3 homogeneous form.

## Components

| Component | Description |
|-----------|-------------|
| [transform.py](./transform.py) | `PerspectiveTransformMixin` — array views (`T`, `flatten`, `column_vector`, `row_vector`), homogeneous conversion, `inverse`, `scale_correction` |
| [projective.py](./projective.py) | `PerspectiveProjectiveMixin` — `projective_transformation` for 2D points |
| [factory.py](./factory.py) | `PerspectiveFactoryMixin` — `create_identity_matrix` and `create_from_points` |
