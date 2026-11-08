class EstimationError(RuntimeError):
    """
    Raised when OpenCV cannot estimate a matrix from the given correspondences.

    Typical causes are degenerate point configurations, such as collinear points,
    or too few inliers for the requested outlier filtering method.
    """
