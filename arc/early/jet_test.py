"""Verify jax.experimental.jet computes arc coefficients, and pin down the convention."""
import math
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental.jet import jet
jax.config.update("jax_enable_x64", True)

# ---- ground truth by exact polynomial arithmetic -------------------------
def poly_coeffs(x0, u1, u2, K=6):
    """Coefficients of ((x0 + u1 t + u2 t^2))^4 truncated at t^K."""
    g = np.zeros(K + 1); g[0], g[1], g[2] = x0, u1, u2
    r = np.zeros(K + 1); r[0] = 1.0
    for _ in range(4):
        new = np.zeros(K + 1)
        for i in range(K + 1):
            for j in range(K + 1 - i):
                new[i + j] += r[i] * g[j]
        r = new
    return r

x0, u1, u2 = 0.3, 1.0, -0.5
truth = poly_coeffs(x0, u1, u2)

# ---- jet -----------------------------------------------------------------
f = lambda x: x ** 4
K = 4
series = [jnp.array(u1), jnp.array(u2)] + [jnp.array(0.0)] * (K - 2)

f0, fs = jet(f, (jnp.array(x0),), (series,))
raw = np.array([f0] + list(fs))

print("arc:  gamma(t) = 0.3 + 1.0 t - 0.5 t^2 ,   K(x) = x^4")
print()
print("exact Taylor coefficients c_j of K(gamma(t)):")
print("   ", np.round(truth[:K + 1], 6))
print()
print("jet output (factorial_scaled=True, the default):")
print("   ", np.round(raw, 6))
print()
print("jet output / j!:")
print("   ", np.round(raw / np.array([math.factorial(j) for j in range(K + 1)]), 6))
print()
print("jet output * j!:")
print("   ", np.round(raw * np.array([math.factorial(j) for j in range(K + 1)]), 6))
