from ..types import FloatArray
from .affine.matrix import AffineMatrix
from .homography.matrix import HomographyMatrix
from .matrix import PerspectiveMatrix
from .method import PerspectiveTransformationMethod


def register_perspective_matrix(
    matrix: FloatArray | AffineMatrix | HomographyMatrix | None,
    transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY,
) -> PerspectiveMatrix:
    """
    Register a perspective matrix.

    Parameters
    ----------
    matrix: FloatArray | AffineMatrix | HomographyMatrix | None
        The matrix to register.
        If the matrix is None, the identity matrix will be returned.
    transform_type: PerspectiveTransformationMethod
        The type of the transformation.

    Returns
    -------
    PerspectiveMatrix:
        The registered perspective matrix.
    """
    if isinstance(matrix, (AffineMatrix, HomographyMatrix)):
        return matrix

    target_class = (
        HomographyMatrix
        if transform_type == PerspectiveTransformationMethod.HOMOGRAPHY
        else AffineMatrix
    )

    if matrix is None:
        return target_class.create_identity_matrix()
    return target_class(value=matrix)
