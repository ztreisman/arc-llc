"""L-layer deep linear network with a rank-deficient teacher: the DDS
paper's own "noisy bridge" testbed (Sec. 4.2 / App. B.8.2), used for their
most discriminating claim -- the rank-multiplicative volume identity
(log det+(G) slope scales with the number of simultaneously-dead
directions r, while lambda_plus_min(G) stays rank-invariant). Our r=1
bottleneck models (experiments 1-8) structurally cannot test this: it
needs bottleneck/layer width >= 2 so several directions can die at once.

Model: y = W_L (... (W_2 (W_1 x)) ...), each W_i a D x D matrix, teacher
M* = diag(1,...,1,0,...,0) with the last r entries zero (r simultaneous
dead directions). Population loss (isotropic Gaussian x, up to an additive
noise-variance constant that doesn't affect gradients):
    K(w) = ||W_L...W_1 - M*||_F^2

Canonical-aligned construction (matching the paper's own depth-controlled
init, and directly generalizing the L=2 approach validated in
experiment_7): every layer is diagonal, diag(1,...,1, tau,...,tau) with r
copies of a SHARED value tau. As tau -> 0, the product's dead-coordinate
entries -> tau^L -> 0, exactly realizing M*. Because every layer
contributes the same factor tau to the dead coordinates, this gives a
direct, training-free sweep over "distance to the singular locus" (no
gradient descent, no convergence-criterion tuning -- avoiding exactly the
protocol sensitivity documented in rrr_model.py's train_rrr_cell).

Per-sample gradients (Fisher-Gram) at each internal layer are computed in
closed form via downstream weight-matrix products, not autograd: for
delta_L = dL/da_L = 2*(y_hat - target), backprop through a stack of linear
layers gives delta_ell = delta_L @ P_{ell+1:L}, where
P_{ell+1:L} = W_L @ ... @ W_{ell+1} is the downstream product. This is the
exact same formula as dds.py's `delta_h1 = delta_h2 @ A` for L=2
(P_{2:2} = W_2 = A), just iterated across more layers.
"""
import numpy as np

from dds import _smallest_positive_eig_and_logdet


def make_teacher_diag(D, r):
    """Rank-(D-r) teacher M* = diag(1,...,1,0,...,0), last r entries zero."""
    diag = np.ones(D)
    diag[D - r:] = 0.0
    return np.diag(diag)


def canonical_layers(D, L, r, tau):
    """L identical diagonal D x D layers, diag(1,...,1,tau,...,tau) with r
    copies of tau on the "dead" coordinates (the last r). As tau -> 0 the
    product converges to make_teacher_diag(D, r) exactly.
    """
    diag = np.ones(D)
    diag[D - r:] = tau
    W = np.diag(diag)
    return [W.copy() for _ in range(L)]


def dds_observables_deep(Ws, N=20_000, target=None, rel_tol=1e-10, rng=None):
    """DDS observables at every internal layer ell = 1, ..., L-1 of the
    L-layer linear map a_0=x -> a_ell = a_{ell-1} @ W_ell.T -> ... -> a_L.

    Ws: list of L (D, D) arrays [W_1, ..., W_L].
    target: optional (D, D) teacher M* (defaults to zero teacher).
    Returns {ell: {"lambda_plus_min", "log_det_plus", "sigma_min"}}.
    """
    if rng is None:
        rng = np.random.default_rng()
    L = len(Ws)
    D = Ws[0].shape[0]

    x = rng.standard_normal((N, D))
    target_output = x @ target.T if target is not None else np.zeros((N, D))

    activations = [x]
    a = x
    for W in Ws:
        a = a @ W.T
        activations.append(a)
    y_hat = activations[-1]
    delta_L = 2.0 * (y_hat - target_output)

    results = {}
    P = Ws[L - 1].copy()  # P_{ell+1:L} for ell=L-1, i.e. P_{L:L} = W_L
    for ell in range(L - 1, 0, -1):
        delta_ell = delta_L @ P
        G_ell = (delta_ell.T @ delta_ell) / N
        lam, logdet, eigs = _smallest_positive_eig_and_logdet(G_ell, rel_tol)
        X_ell = activations[ell]
        sigma_min = float(np.linalg.svd(X_ell, compute_uv=False).min())
        results[ell] = {"lambda_plus_min": lam, "log_det_plus": logdet,
                         "sigma_min": sigma_min, "eigenvalues": eigs}
        P = P @ Ws[ell - 1]  # prepare P_{ell:L} for the next (ell-1) iteration

    return results
