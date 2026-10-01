"""Rank-r matrix factorization model with true rank 0.

Parameters w = (A, B), A in R^{n x r}, B in R^{r x m}, flattened as
w = [vec(A); vec(B)] in R^d, d = r(n+m).

K(w) = ||AB||_F^2 is the population KL divergence to the true
distribution (the zero matrix). Its zero set W0 = {AB = 0} is a
singular variety (union of the linear subspaces {A=0} and {B=0}
when r=1).

Ground truth (r=1): zeta(z) = int K(w)^z dw factors into independent radial
integrals over a in R^n and b in R^m,
    zeta(z) = [int ||a||^{2z} da] * [int ||b||^{2z} db],
with simple poles at z=-n/2 and z=-m/2. The RLCT is the rightmost pole:
    lambda = min(n, m) / 2,  multiplicity 2 iff n == m, else 1.
Every estimator in this project recovers this value, including the exact
analytic Hessian computation. (The r(n+m-r)/2 formula sometimes quoted for
this family is the reduced-rank-regression RLCT, a different model with an
input distribution over x; r > 1 is not used here.)
"""
import numpy as np
import torch


def true_lambda(r, n, m):
    if r != 1:
        raise NotImplementedError("ground truth derived for r=1 only; see rrr_model.py for r0>0 teachers")
    return min(n, m) / 2.0


def dim_w(r, n, m):
    return r * (n + m)


def make_K(r=1, n=2, m=2):
    """Returns K(w) = ||AB||_F^2 as a vectorized numpy callable.

    Accepts w of shape (d,) (a single point, returns a python float) or
    (N, d) (a batch, returns an (N,) array) -- both via the same batched
    einsum, so callers doing Monte Carlo over many samples don't pay for
    a Python-level loop.
    """

    def K(w):
        w = np.asarray(w)
        single = w.ndim == 1
        W = w[None, :] if single else w
        A = W[:, : r * n].reshape(-1, n, r)
        B = W[:, r * n :].reshape(-1, r, m)
        AB = np.einsum("bnr,brm->bnm", A, B)
        vals = (AB**2).sum(axis=(1, 2))
        return float(vals[0]) if single else vals

    return K


def make_K_torch(r=1, n=2, m=2):
    """Returns K(w) = ||AB||_F^2 as a torch callable (w a 1-D tensor)."""

    def K_t(w_t):
        A = w_t[: r * n].reshape(n, r)
        B = w_t[r * n :].reshape(r, m)
        AB = A @ B
        return (AB**2).sum()

    return K_t


def make_K_regular(d=4):
    """K(w) = ||w||^2, the 'regular' comparison model (Gaussian true dist).

    Here lambda = d/2 exactly (RLCT of a nondegenerate quadratic form),
    so ratio lambda/(d/2) = 1.
    """

    def K(w):
        w = np.asarray(w)
        return float((w**2).sum()) if w.ndim == 1 else (w**2).sum(axis=1)

    def K_t(w_t):
        return (w_t**2).sum()

    return K, K_t


def K_grad_hess(w, K_t_fn):
    """Gradient and Hessian of a torch-scalar function K_t_fn at w (numpy array),
    computed in float64 via torch.autograd.functional.
    """
    w_t = torch.tensor(np.asarray(w), dtype=torch.float64)
    grad = torch.autograd.functional.jacobian(K_t_fn, w_t)
    hess = torch.autograd.functional.hessian(K_t_fn, w_t)
    K_val = K_t_fn(w_t).item()
    return K_val, grad.numpy(), hess.numpy()


def sample_on_branch(r, n, m, branch="A", scale=1.0, rng=None):
    """Return a point w on one branch of W0 = {A=0} u {B=0} (r=1 only),
    away from the singular origin, for use as w* in the Hessian estimator.
    """
    if rng is None:
        rng = np.random.default_rng()
    assert r == 1, "sample_on_branch implemented for r=1"
    A = np.zeros((n, r))
    B = np.zeros((r, m))
    if branch == "A":
        B[:] = 0.0
        A[:] = scale * rng.standard_normal((n, r))
    else:
        A[:] = 0.0
        B[:] = scale * rng.standard_normal((r, m))
    return np.concatenate([A.ravel(), B.ravel()])
