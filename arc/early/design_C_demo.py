"""
Design C via jax.experimental.jet: given a 1-jet, does it LIFT to a higher-contact arc?

Test case:  K(a,b) = (b - a^2)^2 ,  W_0 = {b = a^2} , base point = origin.

The linear arc  gamma(t) = (t, 0)  has  K = t^4  -> order 4.
The arc         gamma(t) = (t, t^2) has  K = 0    -> infinite order.

So the direction u1 = (1,0) is NOT a dead direction of order 2 in any final sense:
it lifts. Design C finds the lift by optimizing the higher jet coefficients.
This is exactly the "does this jet lift to a curve in the variety" question.
"""
import numpy as np, math
import jax, jax.numpy as jnp
from jax.experimental.jet import jet
jax.config.update("jax_enable_x64", True)

M = 6                                   # truncation order
K = lambda w: (w[1] - w[0] ** 2) ** 2

def arc_coeffs(u):
    """u has shape (m, 2): coefficients u_1..u_m of gamma(t) = 0 + sum u_j t^j.
    Returns Taylor coefficients c_0..c_M of K(gamma(t))."""
    m = u.shape[0]
    series = [u[j] if j < m else jnp.zeros(2) for j in range(M)]
    c0, cs = jet(K, (jnp.zeros(2),), (series,), factorial_scaled=False)
    return jnp.concatenate([jnp.atleast_1d(c0), jnp.stack(cs)])

def contact_loss(u_free, u1):
    u = jnp.concatenate([u1[None, :], u_free], axis=0)
    return jnp.sum(arc_coeffs(u) ** 2)

u1 = jnp.array([1.0, 0.0])

print("K(a,b) = (b - a^2)^2 ,  base point = origin,  u1 = (1, 0)")
print()
print("linear arc  gamma(t) = t*(1,0):")
print("   coeffs c_0..c_6 =", np.round(np.array(arc_coeffs(u1[None, :])), 8))

# deepen: optimise u_2 (and u_3) to kill as many coefficients as possible
u_free = jnp.zeros((2, 2))
g = jax.jit(jax.grad(contact_loss))
val = jax.jit(contact_loss)
lr = 0.05
for step in range(4000):
    u_free = u_free - lr * g(u_free, u1)

u_star = jnp.concatenate([u1[None, :], u_free], axis=0)
print()
print("after deepening (optimising u_2, u_3):")
print("   u_2 =", np.round(np.array(u_free[0]), 6), "   (true lift: [0, 1])")
print("   u_3 =", np.round(np.array(u_free[1]), 6))
print("   coeffs c_0..c_6 =", np.round(np.array(arc_coeffs(u_star)), 10))
print("   residual sum c_j^2 =", float(val(u_free, u1)))
print()
print("=> the 1-jet (1,0) lifts: contact order goes 4 -> beyond truncation.")
print("   A purely directional (linear-arc / spectral) reading reports order 4 and stops.")
