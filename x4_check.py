"""
Why the tube-volume contact hierarchy fails, on the cleanest case: K = x^4 in one variable.

lambda = rlct(x^4) = 1/4.

p = 2:  coefficient map C_2(gamma) = (a_0, a_1) = (x^4, 4 x^3 c_1),
        gamma = x + c_1 t,  x, c_1 ~ U[-1,1].

V_2(eps) = |{ |x|^4 <= eps  and  |4 x^3 c_1| <= eps }|
         = int_0^{eps^{1/4}} min(1, eps / (4 r^3)) dr

The integrand is 1 (constraint vacuous) for r <= (eps/4)^{1/3}.
That vacuous region alone has measure ~ eps^{1/3}, which DOMINATES
the eps^{1/2} contribution from the region where the constraint bites.

So V_2(eps) ~ eps^{1/3}, i.e. chat_2 = 1/3 and chat_2 / 2 = 1/6 < 1/4 = lambda.

The tube-volume exponent is NOT the codimension of Cont^{>=2}; it is dominated
by the sub-region of jet space where every coefficient is small simply because
the base point sits deep in the zero locus, not because the arc has high contact.
"""
import numpy as np
from scipy import integrate

rng = np.random.default_rng(1)

def V2_exact(eps):
    f = lambda r: min(1.0, eps / (4 * r**3))
    val, _ = integrate.quad(f, 0, eps**0.25, limit=200)
    return val

def V2_mc(eps, N=4_000_000):
    x  = rng.uniform(-1, 1, N)
    c1 = rng.uniform(-1, 1, N)
    return np.mean((np.abs(x)**4 <= eps) & (np.abs(4 * x**3 * c1) <= eps))

eps = np.logspace(-10, -3, 15)

ex = np.array([V2_exact(e) for e in eps])
mc = np.array([V2_mc(e, 2_000_000) for e in eps])

s_ex = np.polyfit(np.log(eps), np.log(ex), 1)[0]
m = mc > 0
s_mc = np.polyfit(np.log(eps[m]), np.log(mc[m]), 1)[0]

print("K = x^4,  p = 2")
print(f"  quadrature   chat_2 = {s_ex:.4f}   chat_2/2 = {s_ex/2:.4f}")
print(f"  monte carlo  chat_2 = {s_mc:.4f}   chat_2/2 = {s_mc/2:.4f}")
print(f"  analytic     chat_2 = {1/3:.4f}   chat_2/2 = {1/6:.4f}")
print(f"  true rlct    lambda = {0.25:.4f}")
print()
print("  => min_p chat_p/p undershoots lambda.  The hierarchy does not compute the rlct.")
