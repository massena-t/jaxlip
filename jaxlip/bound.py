import jax
import jax.numpy as jnp
from jax import random
import time
from math import ceil


def reshape_strided_kernel(kernel: jnp.ndarray, s: int = 1) -> jnp.ndarray:
    cout, cin, h, w = kernel.shape
    if isinstance(s, tuple):
        assert s[0] == s[1]
        s = s[0]
    if s != 1:
        pad_h = (-h) % s
        pad_w = (-w) % s
        if pad_h or pad_w:
            kernel = jnp.pad(
                kernel, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode="constant"
            )
            h += pad_h
            w += pad_w

        h_s, w_s = ceil(h / s), ceil(w / s)
        kernel = kernel.reshape(cout, cin, h_s, s, w_s, s)
        kernel = kernel.transpose(0, 1, 3, 5, 2, 4)
        kernel = kernel.reshape(cout, cin * s * s, h_s, w_s)
    return kernel


def compute_tensor_norm(
    key,  # ← no more `KeyArray` annotation
    kernel: jnp.ndarray,
    num_iters: int = 100,
    s: int = 1,
    return_time: bool = False,
):
    K = reshape_strided_kernel(kernel, s).astype(jnp.complex64)
    cout, cin, h, w = K.shape

    # split the PRNG key into three for u1,u2,u3
    key, k1, k2, k3 = random.split(key, 4)
    u1 = random.normal(k1, (cout, 1, 1, 1), dtype=jnp.complex64)
    u2 = random.normal(k2, (1, cin, 1, 1), dtype=jnp.complex64)
    u3 = random.normal(k3, (1, 1, h, 1), dtype=jnp.complex64)

    start = time.time()
    for _ in range(num_iters):
        # update u4
        tmp = K * u1
        tmp = tmp * u3
        tmp = jnp.sum(tmp, axis=(0, 2), keepdims=True)
        u4 = jnp.sum(tmp * u2, axis=1, keepdims=True).conj()
        u4 /= jnp.linalg.norm(u4)

        # update u2
        u2 = jnp.sum(tmp * u4, axis=3, keepdims=True).conj()
        u2 /= jnp.linalg.norm(u2)

        # update u3
        tmp2 = K * u2
        tmp2 = tmp2 * u4
        tmp2 = jnp.sum(tmp2, axis=(1, 3), keepdims=True)
        u3 = jnp.sum(tmp2 * u1, axis=0, keepdims=True).conj()
        u3 /= jnp.linalg.norm(u3)

        # update u1
        u1 = jnp.sum(tmp2 * u3, axis=2, keepdims=True).conj()

    sigma = jnp.linalg.norm(u1) * jnp.sqrt(h * w)
    elapsed = time.time() - start

    return (sigma, elapsed) if return_time else sigma


if __name__ == "__main__":
    from jax import random
    import jax.numpy as jnp

    # Seed PRNG
    key = random.PRNGKey(0)

    # Example: small random kernel [cout=8, cin=3, h=5, w=5]
    key, subkey = random.split(key)
    kernel = random.normal(subkey, (8, 3, 5, 5))

    # Compute without stride
    sigma0, t0 = compute_tensor_norm(key, kernel, num_iters=50, s=1, return_time=True)
    print(f"Spectral‐norm estimate (stride=1): {sigma0:.4f}  (computed in {t0:.3f}s)")

    # Compute with stride=2
    sigma2, t2 = compute_tensor_norm(key, kernel, num_iters=50, s=2, return_time=True)
    print(f"Spectral‐norm estimate (stride=2): {sigma2:.4f}  (computed in {t2:.3f}s)")


def tensor_norm(
    kernel: jnp.ndarray,
    u1: jnp.ndarray,
    u2: jnp.ndarray,
    u3: jnp.ndarray,
    num_iters: int = 100,
    detach_iter: int = 10,
    s: int = 1,
):
    K = reshape_strided_kernel(kernel, s).astype(jnp.complex64)
    cout, cin, h, w = K.shape
    for i in range(num_iters):
        if i == detach_iter - 1:
            u1 = jax.lax.stop_gradient(u1)
            u2 = jax.lax.stop_gradient(u2)
            u3 = jax.lax.stop_gradient(u3)
            u4 = jax.lax.stop_gradient(u4)

        # update u4
        tmp = K * u1
        tmp = tmp * u3
        tmp = jnp.sum(tmp, axis=(0, 2), keepdims=True)
        u4 = jnp.sum(tmp * u2, axis=1, keepdims=True).conj()
        u4 /= jnp.linalg.norm(u4)

        # update u2
        u2 = jnp.sum(tmp * u4, axis=3, keepdims=True).conj()
        u2 /= jnp.linalg.norm(u2)

        # update u3
        tmp2 = K * u2
        tmp2 = tmp2 * u4
        tmp2 = jnp.sum(tmp2, axis=(1, 3), keepdims=True)
        u3 = jnp.sum(tmp2 * u1, axis=0, keepdims=True).conj()
        u3 /= jnp.linalg.norm(u3)

        # update u1
        u1 = jnp.sum(tmp2 * u3, axis=2, keepdims=True).conj()

    sigma = jnp.linalg.norm(u1) * jnp.sqrt(h * w)
    return sigma, u1, u2, u3
