"""
Contact hierarchy via Design C (witness finding by jet-deepening) + local PCA,
seeded Design-D style. Estimates

    dim Cont^{>=p}(K)  locally,  then  codim/p,  then  min_p codim/p

and tests the conjecture that the REAL contact locus codimension recovers the rlct
(as opposed to lct_C, which Mustata's theorem gives over an algebraically closed field).

Method
------
1. WITNESS: minimise  S(gamma) = sum_{j<p} c_j(gamma)^2  over gamma in J_{p-1},
   where c_j are the Taylor coefficients of K(gamma(t)).  Coefficients from
   jax.experimental.jet with factorial_scaled=False.
2. PROJECT: from gamma* + s*delta for random unit delta, re-run the optimiser to
   land back on the variety at gamma'.  The displacement gamma' - gamma* lies
   (to O(s^2)) in the tangent space of Cont^{>=p} at gamma*.
3. PCA the displacements -> dim Cont^{>=p}.  codim = d*p - dim.

Jacobian rank is deliberately NOT used: over R it undercounts badly
(e.g. |x|^2 = 0 is one equation but cuts out a codimension-n real point).
"""
import numpy as np
import jax, jax.numpy as jnp
from jax.experimental.jet import jet

jax.config.update("jax_enable_x64", True)


def make_coeff_map(K, d, p):
    """gamma flat (d*p,) -> Taylor coefficients c_0..c_{p-1} of K(gamma(t))."""
    def coeffs(flat):
        g = flat.reshape(p, d)                    # rows = c_0 (base), c_1, ...
        series = [g[j] for j in range(1, p)] + [jnp.zeros(d)]
        c0, cs = jet(K, (g[0],), (series,), factorial_scaled=False)
        allc = jnp.concatenate([jnp.atleast_1d(c0), jnp.stack(cs)])
        return allc[:p]
    return coeffs


def make_descend(S):
    """Jitted Adam, whole loop inside lax.scan."""
    g = jax.grad(S)
    def run(x0, steps, lr):
        def body(c, t):
            x, m, v = c
            gr = g(x)
            m = 0.9 * m + 0.1 * gr
            v = 0.999 * v + 0.001 * gr ** 2
            mh = m / (1 - 0.9 ** (t + 1)); vh = v / (1 - 0.999 ** (t + 1))
            x = x - lr * mh / (jnp.sqrt(vh) + 1e-12)
            return (x, m, v), None
        (x, _, _), _ = jax.lax.scan(body, (x0, jnp.zeros_like(x0), jnp.zeros_like(x0)),
                                    jnp.arange(steps))
        return x
    return jax.jit(lambda x0, lr: run(x0, 3000, lr)), jax.jit(lambda x0, lr: run(x0, 1500, lr))


def contact_dim(K, d, p, seed=0, n_probe=60, s=1e-2, tol=1e-2, verbose=False):
    key = np.random.default_rng(seed)
    coeffs = make_coeff_map(K, d, p)
    S = jax.jit(lambda x: jnp.sum(coeffs(x) ** 2))
    long_run, short_run = make_descend(lambda x: jnp.sum(coeffs(x) ** 2))

    # 1. witness
    best, bestS = None, np.inf
    for _ in range(6):
        x0 = jnp.array(key.standard_normal(d * p) * 0.5)
        xw = long_run(x0, 2e-2)
        sv = float(S(xw))
        if sv < bestS:
            best, bestS = xw, sv
    if bestS > 1e-14:
        return np.nan, np.nan, bestS
    gamma = best

    # 2. project perturbations back onto the variety
    disp = []
    for _ in range(n_probe):
        delta = key.standard_normal(d * p)
        delta /= np.linalg.norm(delta)
        xp = short_run(gamma + s * jnp.array(delta), 5e-3)
        if float(S(xp)) < 1e-14:
            disp.append(np.array(xp - gamma))
    if len(disp) < 5:
        return np.nan, np.nan, bestS
    D = np.array(disp)

    # 3. local PCA
    sv = np.linalg.svd(D, compute_uv=False)
    dim = int(np.sum(sv > tol * sv[0]))
    if verbose:
        print("   singular values:", np.round(sv / sv[0], 4)[:8])
    return dim, d * p - dim, bestS


CASES = [
    ("x^4",              1, 0.25,     lambda w: w[0] ** 4,                       [1, 2, 3, 4]),
    ("x^2+y^2",          2, 1.0,      lambda w: w[0] ** 2 + w[1] ** 2,           [1, 2, 3]),
    ("x^2+y^2+z^2",      3, 1.5,      lambda w: w[0]**2 + w[1]**2 + w[2]**2,     [1, 2, 3]),
    ("a^2 b^2",          2, 0.5,      lambda w: w[0] ** 2 * w[1] ** 2,           [1, 2, 3, 4]),
    ("a^2 + b^6",        2, 2/3,      lambda w: w[0] ** 2 + w[1] ** 6,           [1, 2, 3, 4, 6]),
]

if __name__ == "__main__":
    print(f"{'case':>12s} {'rlct':>7s} {'p':>3s} {'dim':>4s} {'codim':>6s} {'codim/p':>8s}")
    print("-" * 46)
    for name, d, lam, K, ps in CASES:
        best = np.inf
        for p in ps:
            dim, cod, res = contact_dim(K, d, p)
            if np.isnan(dim):
                print(f"{name:>12s} {lam:7.4f} {p:3d}    -      -        -   (no witness, S={res:.1e})")
                continue
            r = cod / p
            best = min(best, r)
            print(f"{name:>12s} {lam:7.4f} {p:3d} {dim:4d} {cod:6d} {r:8.4f}")
        print(f"{'':>12s} {'':>7s}  ->  min_p codim/p = {best:.4f}   (rlct = {lam:.4f})")
        print("-" * 46)
