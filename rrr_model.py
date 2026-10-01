"""Aoyagi-Watanabe reduced-rank regression (RRR): the closed-form RLCT
testbed used as DDS's own anchor. Teacher M* in R^{N x M} of rank r0, model
W2 W1 with W1 in R^{H x M}, W2 in R^{N x H}. Population loss (isotropic
Gaussian x):
    K(w) = E_x[||W2 W1 x - M* x||^2] = ||W2 W1 - M*||_F^2

This is a strict generalization of the r0=0 model in model.py (set M*=0
to recover K(w) = ||AB||_F^2 exactly); we keep it as a separate module
since the closed-form ground truth (Aoyagi and Watanabe, 2005) is an
external, independently-published result we are not re-deriving, unlike
model.py::true_lambda for r=1 which we verified from first principles.

Case 3 closed form (the one used by the DDS paper's own Aoyagi 2005
anchor, and independently cross-checked here against our own r0=0 result
within its validity region -- see RESULTS.md):
    lambda = (N*H + M*r0 - H*r0) / 2,   valid when N + H < M + r0
"""
import numpy as np
import torch


def aoyagi_lambda(M, N, H, r0):
    return (N * H + M * r0 - H * r0) / 2.0


def case3_valid(M, N, H, r0):
    return N + H < M + r0


def make_teacher(N, M, r0):
    """Rank-r0 teacher M* in R^{N x M}: diag(1,...,1,0,...,0)."""
    T = np.zeros((N, M))
    for i in range(r0):
        T[i, i] = 1.0
    return T


def make_K_torch_rrr(M_star_t, H):
    """K(w) = ||W2 W1 - M*||_F^2 as a torch callable, w = [vec(W1); vec(W2)],
    W1 in R^{H x M}, W2 in R^{N x H}.
    """
    N, M = M_star_t.shape

    def K_t(w_t):
        W1 = w_t[: H * M].reshape(H, M)
        W2 = w_t[H * M :].reshape(N, H)
        return ((W2 @ W1 - M_star_t) ** 2).sum()

    return K_t


def dim_w_rrr(M, N, H):
    return H * (M + N)


def exact_branch_point(M, N, H, r0, scale=0.03, rng=None):
    """A controlled reference point on the singular locus: the exact
    minimum-norm factorization of M* (W2_star @ W1_star = M* exactly, using
    only the first r0 of the H bottleneck directions, with any H-r0
    "excess" directions exactly zero), perturbed transversally by a small
    amount `scale`.

    This is the RRR analogue of model.py::sample_on_branch, constructing
    the reference point directly rather than gradient-descending to some
    convergence criterion (which we found highly sensitive to the
    training/ridge protocol: when H > r0 the solution set is a manifold and
    GD without ridge lands anywhere on it). Used in experiment 8 only as an
    ablation; the main protocol reads observables from sgld_rrr_chains.

    When H > r0, the perturbation is applied only to the H-r0 excess
    bottleneck directions (the mechanism the paper's own rank-deficit
    story is about). When H == r0 (no excess capacity), there is still a
    nonzero RLCT from a different mechanism: the reparametrization
    (W2,W1) -> (W2 Q, Q^{-1} W1) for invertible Q leaves W2 W1 unchanged,
    so the solution set is a positive-dimensional GL(r0) orbit even
    without excess capacity; there we perturb the r0 block itself instead.

    The perturbation is normalized to a FIXED total Frobenius norm (scale)
    within whichever block it's applied to, not a fixed per-element scale:
    a fixed per-element scale pumps more aggregate perturbation "energy"
    into larger blocks, confounding the comparison with cell size rather
    than isolating the singularity structure (found empirically -- an
    earlier per-element-scale version produced a spurious cell-size-driven
    correlation; see RESULTS.md).
    """
    if rng is None:
        rng = np.random.default_rng()
    W1 = np.zeros((H, M))
    W2 = np.zeros((N, H))
    for i in range(r0):
        W1[i, i] = 1.0
        W2[i, i] = 1.0

    if H > r0:
        dW1 = rng.standard_normal((H - r0, M))
        dW2 = rng.standard_normal((N, H - r0))
        total_norm = np.sqrt((dW1 ** 2).sum() + (dW2 ** 2).sum())
        W1[r0:, :] += scale * dW1 / total_norm
        W2[:, r0:] += scale * dW2 / total_norm
    else:
        dW1 = rng.standard_normal((H, M))
        dW2 = rng.standard_normal((N, H))
        total_norm = np.sqrt((dW1 ** 2).sum() + (dW2 ** 2).sum())
        W1 += scale * dW1 / total_norm
        W2 += scale * dW2 / total_norm
    return W1, W2


def sgld_rrr_chains(M, N, H, r0, n=10_000, chains=8, burn=60_000, steps=60_000,
                    thin=2_000, lr=5e-5, gamma=1.0, seed=0):
    """Localized SGLD posterior around the exact minimum-norm solution of the
    RRR model, with `chains` independent chains run in parallel (analytic
    gradients, batched numpy).

    Target: exp(-n*beta*K(w) - gamma*||w - w*||^2), beta = 1/log(n).

    Step size matters more than it looks: the stiffest direction has curvature
    ~2*n*beta, and an Euler step with lr*n*beta not << 1 inflates the energy
    in exactly the stiff directions (lr=2e-4 overestimated lambda by ~40%;
    lr=5e-5 brings the bias to 5-9%, the remainder being finite-n/localization).

    Returns the per-chain LLC estimate lam_hat = n*beta*E[K] (K(w*)=0 here),
    and the thinned snapshots (list of (W1, W2) arrays with a leading chain axis).
    """
    rng = np.random.default_rng(seed)
    T = make_teacher(N, M, r0)
    W1s = np.zeros((H, M))
    W2s = np.zeros((N, H))
    for i in range(r0):
        W1s[i, i] = 1.0
        W2s[i, i] = 1.0
    nb = n / np.log(n)
    W1 = np.repeat(W1s[None], chains, 0)
    W2 = np.repeat(W2s[None], chains, 0)
    Ks, snaps = [], []
    for t in range(burn + steps):
        R = W2 @ W1 - T
        g1 = 2 * nb * np.swapaxes(W2, 1, 2) @ R + 2 * gamma * (W1 - W1s)
        g2 = 2 * nb * R @ np.swapaxes(W1, 1, 2) + 2 * gamma * (W2 - W2s)
        W1 = W1 - lr / 2 * g1 + np.sqrt(lr) * rng.standard_normal(W1.shape)
        W2 = W2 - lr / 2 * g2 + np.sqrt(lr) * rng.standard_normal(W2.shape)
        if t >= burn:
            Ks.append((R ** 2).sum((1, 2)))
            if (t - burn) % thin == 0:
                snaps.append((W1.copy(), W2.copy()))
    lam_hat = nb * np.mean(Ks, axis=0)
    return lam_hat, snaps, T


def aoyagi_2005_anchor_cells(M=10, N=5):
    """The DDS paper's own 14-cell grid: H in {2,3,4,5}, r0 in {1,...,min(N,H)},
    restricted to cells where the Case-3 closed form is valid.
    """
    cells = []
    for H in [2, 3, 4, 5]:
        for r0 in range(1, min(N, H) + 1):
            if case3_valid(M, N, H, r0):
                cells.append({"M": M, "N": N, "H": H, "r0": r0,
                              "lambda_true": aoyagi_lambda(M, N, H, r0)})
    return cells
