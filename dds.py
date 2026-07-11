"""Dead-Direction Signatures (DDS), per Shirodkar & Narayanan 2606.21158.

Our K(w) = ||AB||_F^2 model is already the network DDS was validated on in
its own closed-form-RLCT anchor (Aoyagi & Watanabe reduced-rank regression):
x -> hidden = Bx (the bottleneck, "h1") -> output = A(Bx) (the boundary
layer, "h2"), with a zero teacher, so the per-sample loss is L = ||output||^2
and K(w) = E_x[L] = ||AB||_F^2 exactly for isotropic Gaussian x. No new
model is needed -- DDS is read off empirical activations and per-sample
gradients of this same map.

Three observables, evaluated at a layer ell in {h1, h2}:
  - sigma_min(X_ell): smallest singular value of the (N, width) activation
    matrix -- the activation-side dual.
  - lambda_plus_min(G_ell): smallest strictly-positive eigenvalue of the
    per-sample-gradient Fisher-Gram G_ell = (1/N) sum_i delta_i delta_i^T,
    delta_i = grad of the per-sample loss w.r.t. the layer-ell pre-activation
    -- the rate observable.
  - log_det_plus(G_ell): sum of log(eigenvalue) over strictly-positive
    eigenvalues of G_ell -- the volume observable.

For our r=1 toy cases the bottleneck is 1-dimensional, so G_h1 has a single
eigenvalue and log_det_plus(G_h1) == log(lambda_plus_min(G_h1)) trivially;
h2 (width n, up to 5 in our cells) is where log_det_plus is a genuinely
distinct reading from lambda_plus_min. This mirrors the paper's own
"dimension-fixed boundary layer" preference for cross-cell reads.
"""
import numpy as np


def _smallest_positive_eig_and_logdet(M, rel_tol=1e-10):
    """Eigendecompose a PSD matrix M, discarding eigenvalues below
    rel_tol * (largest eigenvalue) as numerical zero (mirrors the paper's
    fp64 + no-Tikhonov + smallest-positive-eigenvalue recipe, App. B.4).
    """
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, 0, None)  # PSD by construction; guard fp round-off
    top = eigvals.max()
    if top <= 0:
        return 0.0, -np.inf, eigvals
    positive = eigvals[eigvals > rel_tol * top]
    if positive.size == 0:
        return 0.0, -np.inf, eigvals
    lambda_plus_min = positive.min()
    log_det_plus = np.log(positive).sum()
    return lambda_plus_min, log_det_plus, eigvals


def dds_observables(A, B, N=20_000, rel_tol=1e-10, rng=None, target=None):
    """Compute DDS observables at the bottleneck (h1) and output (h2) layers
    of the two-layer linear map x -> A(Bx), x ~ N(0, I_m).

    A: (n, r) array: layer-2 (output) weights.
    B: (r, m) array: layer-1 (bottleneck) weights.
    target: optional (n, m) teacher matrix M* (defaults to the zero teacher,
        matching experiments 1-7). When given, the per-sample loss is
        ||output - M*x||^2 instead of ||output||^2, matching the general
        (truth-rank r0 > 0) Aoyagi-Watanabe reduced-rank-regression anchor.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, r = A.shape
    r2, m = B.shape
    assert r == r2

    x = rng.standard_normal((N, m))
    hidden = x @ B.T          # (N, r): X_h1
    output = hidden @ A.T     # (N, n): X_h2
    target_output = x @ target.T if target is not None else 0.0

    delta_h2 = 2.0 * (output - target_output)   # (N, n): dL/d(output)
    delta_h1 = delta_h2 @ A                      # (N, r): dL/d(hidden), backprop through A

    G_h1 = (delta_h1.T @ delta_h1) / N   # (r, r)
    G_h2 = (delta_h2.T @ delta_h2) / N   # (n, n)

    lam_h1, logdet_h1, eig_h1 = _smallest_positive_eig_and_logdet(G_h1, rel_tol)
    lam_h2, logdet_h2, eig_h2 = _smallest_positive_eig_and_logdet(G_h2, rel_tol)

    sigma_min_h1 = float(np.linalg.svd(hidden, compute_uv=False).min())
    sigma_min_h2 = float(np.linalg.svd(output, compute_uv=False).min())

    return {
        "lambda_plus_min_h1": lam_h1,
        "log_det_plus_h1": logdet_h1,
        "sigma_min_h1": sigma_min_h1,
        "lambda_plus_min_h2": lam_h2,
        "log_det_plus_h2": logdet_h2,
        "sigma_min_h2": sigma_min_h2,
        "eigenvalues_h1": eig_h1,
        "eigenvalues_h2": eig_h2,
    }


def dds_observables_from_w(w, r, n, m, N=20_000, rel_tol=1e-10, rng=None):
    """Convenience wrapper taking a flat w = [vec(A); vec(B)], matching
    model.py's convention (A: (n,r), B: (r,m))."""
    w = np.asarray(w)
    A = w[: r * n].reshape(n, r)
    B = w[r * n :].reshape(r, m)
    return dds_observables(A, B, N=N, rel_tol=rel_tol, rng=rng)
