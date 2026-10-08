"""Step 4: numerical estimator of the RLCT from REAL contact loci (Design E).

    2 * rlct(K) = min_p  codim_R Cont^{>=p}(f) / p,     K = |f|^2,

codim taken in J_{p-1}(R^d) (dimension d p), with base points free near w0.

Replaces contact_dim.py (4e), which accepted witnesses by a K threshold and failed.
Here:
  * the coefficient map c -> (Taylor coeffs 0..p-1 of f(gamma(t))) comes from
    jax.experimental.jet (factorial_scaled=False, x64), verified in jet_test.py;
  * landing on the contact locus uses Gauss-Newton (min-norm least squares) on
    that map. It converges onto the locus, linearly at worst (non-reduced equations
    like |c1|^2 = 0), so acceptance is by convergence, not by a K threshold;
  * local dimension at a landing jet comes from PCA of re-projected eps-perturbations.
    That is the real dimension, so it counts real arcs. Jacobian rank would
    undercount over R, e.g. for |c1|^2 = 0.
p = 1 is the step-0 tangent-excess estimator (codim of the true set).

Validation: exact table of the context file, (b - a^2)^2, the cusp (a^2 - b^3)^2
(generic-point methods give 1/2, the truth is 5/12), the real-vs-complex case
f = x^2+y^2+z^2, symmetric factorization, and small RRR cells.
"""
import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.jet import jet

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rrr_orbit import aoyagi_lambda_general  # noqa: E402


def make_coeff_map(f, d, p):
    """c (p*d,) -> Taylor coefficients 0..p-1 of f(c0 + c1 t + ... ), flattened."""
    def C(c):
        c = c.reshape(p, d)
        if p == 1:
            return jnp.atleast_1d(f(c[0]))
        y0, ys = jet(f, (c[0],), ([c[j] for j in range(1, p)],), factorial_scaled=False)
        return jnp.concatenate([jnp.atleast_1d(y0)] + [jnp.atleast_1d(y) for y in ys])
    return jax.jit(C), jax.jit(jax.jacfwd(C))


def gn_project(C, JC, c, iters=500, mu0=1.0, q=0.85, tol=1e-30):
    """Proximal Gauss-Newton: approximately argmin_c ||C(c)||^2 + mu ||c - anchor||^2
    with mu -> 0, anchor = the starting point. The limit is the nearest point of
    {C = 0}. Plain Gauss-Newton is not a projection on non-reduced systems: on
    (c0^2, 2 c0 c1) a Newton step halves c1 along with c0, collapsing the solution
    set's tangent directions."""
    anchor = c.copy()
    D = c.size
    mu = mu0

    def phi(x, m):
        rr = np.asarray(C(x))
        return rr @ rr + m * (x - anchor) @ (x - anchor)

    for _ in range(iters):
        r = np.asarray(C(c))
        if r @ r < tol and mu < 1e-24:
            break
        J = np.asarray(JC(c))
        s = np.sqrt(mu)
        A = np.vstack([J, s * np.eye(D)])
        b = -np.concatenate([r, s * (c - anchor)])
        step = np.linalg.lstsq(A, b, rcond=None)[0]
        # predictor-corrector with backtracking. Backtracking: GN drops r * Hess(r),
        # and near a non-reduced zero the anchor pull overshoots into a c -> -c
        # two-cycle. Corrector: along a high-contact near-solution direction (e.g.
        # c1 on x^2 at p=3) the predictor's second-order error dwarfs the residual;
        # one plain GN step back to the locus re-solves the dependent coordinates.
        f0, t = phi(c, mu), 1.0
        while True:
            trial = c + t * step
            rt = np.asarray(C(trial))
            trial = trial - np.linalg.lstsq(np.asarray(JC(trial)), rt, rcond=None)[0]
            if phi(trial, mu) <= f0 or t < 1e-6:
                break
            t *= 0.5
        if phi(trial, mu) <= f0:
            c = trial
        mu *= q
    r = np.asarray(C(c))
    return c, bool(r @ r < 1e-20)


def local_dim(C, JC, c_star, rng, eps=1e-4, n=None, rel=0.1):
    """Dimension of the real solution set of C = 0 near c_star, by PCA of
    eps-perturbations projected back with Gauss-Newton."""
    D = c_star.size
    n = n or 2 * D + 4
    pts = []
    for _ in range(n):
        c, ok = gn_project(C, JC, c_star + eps * rng.standard_normal(D))
        if ok:
            pts.append(c - c_star)
    X = np.array(pts)
    X = X - X.mean(axis=0)
    sv = np.linalg.svd(X, compute_uv=False)
    # a genuine direction spreads ~ eps * sqrt(n); slow-converging GN residue does
    # not, so threshold against the perturbation scale, not against sv[0]
    return int((sv > rel * eps * np.sqrt(len(pts))).sum())


def estimate(f, w0, p_max=6, starts=8, base_sigma=0.05, jet_sigma=1.0, seed=0,
             walk_steps=3, walk_eps=1e-2, verbose=False):
    """Return (2*lambda_hat estimate as min ratio, per-p minimum ratios)."""
    rng = np.random.default_rng(seed)
    d = w0.size
    per_p = {}
    for p in range(1, p_max + 1):
        C, JC = make_coeff_map(f, d, p)
        best = np.inf
        for _ in range(starts):
            c = np.concatenate([w0 + base_sigma * rng.standard_normal(d),
                                jet_sigma * rng.standard_normal(d * (p - 1))])
            c, ok = gn_project(C, JC, c)
            # walk a few eps_walk steps on the locus: landings bunch up where
            # components meet, and PCA there reads the span of their union
            for _ in range(walk_steps):
                if not ok:
                    break
                c, ok = gn_project(C, JC, c + walk_eps * rng.standard_normal(c.size))
            if not ok or np.linalg.norm(c[:d] - w0) > 0.5:
                continue
            dim = local_dim(C, JC, c, rng)
            best = min(best, (d * p - dim) / p)
        per_p[p] = best
        if verbose:
            print(f"      p={p}: min codim/p = {best:.3f}", flush=True)
    return min(per_p.values()), per_p


# ---- test cases ---------------------------------------------------------------

def rrr_case(M, N, H, r0):
    Mstar = jnp.zeros((N, M)).at[jnp.arange(r0), jnp.arange(r0)].set(1.0)

    def f(w):
        W1 = w[: H * M].reshape(H, M)
        W2 = w[H * M:].reshape(N, H)
        return (W2 @ W1 - Mstar).ravel()
    W1 = np.zeros((H, M)); W2 = np.zeros((N, H))
    W1[np.arange(r0), np.arange(r0)] = 1; W2[np.arange(r0), np.arange(r0)] = 1
    return f, np.concatenate([W1.ravel(), W2.ravel()]), float(aoyagi_lambda_general(M, N, H, r0))


def sym_case(n, k, r0):
    Mstar = jnp.zeros((n, n)).at[jnp.arange(r0), jnp.arange(r0)].set(1.0)
    iu = np.triu_indices(n)

    def f(w):
        W = w.reshape(n, k)
        return (W @ W.T - Mstar)[iu]
    W = np.zeros((n, k)); W[np.arange(r0), np.arange(r0)] = 1
    lam = (r0 * n - r0 * (r0 - 1) / 2) / 2 + (n - r0) * (k - r0) / 4
    return f, W.ravel(), lam


CASES = [
    # name, f, w0, true rlct, p_max
    ("x^4 = (x^2)^2", lambda w: w[0] * w[0], np.zeros(1), 0.25, 4),
    ("x^2+y^2", lambda w: w, np.zeros(2), 1.0, 3),
    ("x^2+y^2+z^2", lambda w: w, np.zeros(3), 1.5, 3),
    ("(ab)^2", lambda w: w[0] * w[1], np.zeros(2), 0.5, 4),
    ("a^2 + b^6", lambda w: jnp.stack([w[0], w[1] * w[1] * w[1]]), np.zeros(2), 2 / 3, 6),
    ("(b - a^2)^2", lambda w: w[1] - w[0] * w[0], np.zeros(2), 0.5, 4),
    ("cusp (a^2 - b^3)^2", lambda w: w[0] * w[0] - w[1] * w[1] * w[1], np.zeros(2), 5 / 12, 6),
    ("(x^2+y^2+z^2)^2, f = x^2+y^2+z^2", lambda w: (w * w).sum(), np.zeros(3), 0.75, 4),
]


def main():
    print("Design E: real contact loci by Gauss-Newton on jets + local PCA")
    print(f"{'case':38s} | true  | est   | best p | per-p min codim/p")
    rows = list(CASES)
    for (n, k, r0) in [(1, 3, 0), (2, 2, 1), (2, 4, 1)]:
        f, w0, lam = sym_case(n, k, r0)
        rows.append((f"sym WW^T n={n} k={k} r0={r0}", f, w0, lam, 3))
    for (M, N, H, r0) in [(3, 2, 2, 0), (4, 2, 2, 1), (5, 2, 3, 1), (3, 3, 3, 1)]:
        f, w0, lam = rrr_case(M, N, H, r0)
        rows.append((f"RRR M={M} N={N} H={H} r0={r0}", f, w0, lam, 2))
    for name, f, w0, lam, pmax in rows:
        t0 = time.time()
        two_lam, per_p = estimate(f, w0, p_max=pmax)
        best_p = min(per_p, key=per_p.get)
        pp = " ".join(f"{v:.2f}" for v in per_p.values())
        flag = "" if abs(two_lam / 2 - lam) < 1e-6 else "   <-- off"
        print(f"{name:38s} | {lam:.3f} | {two_lam / 2:.3f} | {best_p:6d} | {pp}"
              f"  ({time.time() - t0:.0f}s){flag}", flush=True)


if __name__ == "__main__":
    main()
