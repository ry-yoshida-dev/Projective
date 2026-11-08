from ..types import FloatArray
from .matrix import PerspectiveMatrix
from .method import PerspectiveTransformationMethod


def register_perspective_matrix(
    matrix: FloatArray | PerspectiveMatrix | None,
    transform_type: PerspectiveTransformationMethod = PerspectiveTransformationMethod.HOMOGRAPHY,
) -> PerspectiveMatrix:
    """
    Normalize a raw array, a typed matrix, or None into a perspective matrix of ``transform_type``.

    Parameters
    ----------
    matrix: FloatArray | PerspectiveMatrix | None
        The matrix to register.
        An array is wrapped in the class of ``transform_type``,
        a typed matrix is returned as is,
        and None returns the identity matrix.
    transform_type: PerspectiveTransformationMethod
        The type of the transformation.

    Returns
    -------
    PerspectiveMatrix:
        The registered perspective matrix.

    Raises
    ------
    ValueError
        If a typed matrix does not match ``transform_type``,
        or an array has the wrong shape for it.
    """
    target_class = transform_type.perspective_class
    if isinstance(matrix, PerspectiveMatrix):
        if matrix.transform_type is not transform_type:
            raise ValueError(
                f"Matrix of type {matrix.transform_type} does not match transform type {transform_type}"
            )
        return matrix

    if matrix is None:
        return target_class.create_identity_matrix()
    return target_class(value=matrix)
