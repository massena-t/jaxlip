import jax.numpy as jnp


def orthogonalize(M):
    # by @YouJiacheng (with stability loss idea from @leloykun)
    # https://twitter.com/YouJiacheng/status/1893704552689303901
    # https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae

    """Orthogonalize a matrix using Newton-Schulz iteration.

    This function computes an orthogonal matrix from the input matrix using
    an optimized Newton-Schulz iteration scheme. The algorithm is particularly
    useful for enforcing orthogonality constraints in neural network layers.

    Args:
        M (jax.Array): Input matrix of shape (m, n) to be orthogonalized.

    Returns:
        jax.Array: Orthogonalized matrix of the same shape as input.
                  If m > n, returns an orthogonal matrix with orthonormal columns.
                  If n > m, returns an orthogonal matrix with orthonormal rows.

    Notes:
        - The function automatically handles rectangular matrices by transposing
          when necessary to ensure numerical stability.
        - The matrix is first normalized by its Frobenius norm before iteration.
        - Uses 6 iterations with pre-computed coefficients for optimal convergence.
        - Implementation based on work by @YouJiacheng with stability improvements.

    Examples:
        >>> import jax.numpy as jnp
        >>> M = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        >>> M_orth = orthogonalize(M)
        >>> # M_orth will have orthonormal columns
    """

    abc_list = [
        (3955 / 1024, -8306 / 1024, 5008 / 1024),
        (3735 / 1024, -6681 / 1024, 3463 / 1024),
        (3799 / 1024, -6499 / 1024, 3211 / 1024),
        (4019 / 1024, -6385 / 1024, 2906 / 1024),
        (2677 / 1024, -3029 / 1024, 1162 / 1024),
        (2172 / 1024, -1833 / 1024, 682 / 1024),
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T

    M = M / jnp.linalg.norm(M)

    for a, b, c in abc_list:
        A = M.T @ M
        I = jnp.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M
