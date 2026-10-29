"""Step 0 of the contact-hierarchy plan: test the "tangent excess" reading of the
RRR gap (NOTEBOOK.md 4f, 4g).

Claim: 2*lambda - rank H(w) = dim T_w Z - dim_w Z, where Z = {W2 W1 = M*}. At the
min-norm point Z's components meet and every Hessian observable undercounts. At a
generic point of the largest component, rank H = 2*lambda.

With zero residual, rank H = rank J, J = Jacobian of (W1, W2) -> W2 W1. Off the
singular point, rank J depends only on the ranks of W1 and W2:
rank J = N rk(W1) + M rk(W2) - rk(W1) rk(W2).

After splitting off the r0 block, the components of Z are labelled by the excess
ranks (alpha, beta) of (W1, W2) with alpha + beta = H' := H - r0, and

    rank J = r0(N+M-r0) + N' alpha + M' beta - alpha beta    (primes = minus r0).

Its minimum over components is 2*lambda; alpha is the k of rrr_orbit.py.

Parts:
  A. exact: generic points on every component, rank J vs the formula.
  B. unregularized GD from random inits at several scales, polished onto Z by
     Gauss-Newton, then rank J; min over restarts vs 2*lambda.
  C. tangent-excess estimator at the min-norm point: perturb by eps, project onto Z,
     rank J at the landing points; min = 2*lambda-hat. Local PCA of the landing
     cloud is shown for contrast.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rrr_model import aoyagi_2005_anchor_cells, make_teacher  # noqa: E402

M_IN, N_OUT = 10, 5
REL_TOL = 1e-8


# ---- linear algebra ---------------------------------------------------------

def jac(W1, W2):
    """J of vec_r(W2 W1) w.r.t. [vec_r(W1), vec_r(W2)]; W1 H x M, W2 N x H."""
    N, H = W2.shape
    M = W1.shape[1]
    return np.hstack([np.kron(W2, np.eye(M)), np.kron(np.eye(N), W1.T)])


def num_rank(X, rel_tol=REL_TOL):
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[0] == 0:
        return 0, np.inf
    r = int((sv > rel_tol * sv[0]).sum())
    gap = sv[r - 1] / sv[r] if r < len(sv) and sv[r] > 0 else np.inf
    return r, gap


def excess_ranks(W1, W2, r0, rel_tol=1e-6):
    """Ranks of W1, W2 minus r0 (the excess ranks alpha, beta)."""
    return num_rank(W1, rel_tol)[0] - r0, num_rank(W2, rel_tol)[0] - r0


def gauss_newton_project(W1, W2, Mstar, iters=60, tol=1e-28):
    """Min-norm Gauss-Newton onto Z = {W2 W1 = Mstar}. Lands at a nearby point."""
    H, M = W1.shape
    for _ in range(iters):
        R = W2 @ W1 - Mstar
        if (R ** 2).sum() < tol:
            break
        step = np.linalg.lstsq(jac(W1, W2), R.ravel(), rcond=None)[0]
        W1 = W1 - step[: H * M].reshape(W1.shape)
        W2 = W2 - step[H * M:].reshape(W2.shape)
    return W1, W2, float(((W2 @ W1 - Mstar) ** 2).sum())


# ---- predictions --------------------------------------------------------------

def components(M, N, H, r0):
    Mp, Np, Hp = M - r0, N - r0, H - r0
    out = []
    for a in range(0, Hp + 1):
        b = Hp - a
        if a <= Mp and b <= Np:
            out.append((a, b, r0 * (N + M - r0) + Np * a + Mp * b - a * b))
    return out


def generic_point(M, N, H, r0, a, b, rng):
    """Generic point of component (a, b): W2 W1 = M*, excess ranks a, b."""
    Mstar = make_teacher(N, M, r0)
    Hp = H - r0
    # excess block: W1' = Q Y (image in first a coords), W2' = X P (kills them)
    W1 = np.zeros((H, M)); W2 = np.zeros((N, H))
    W1[:r0, :r0] = np.eye(r0); W2[:r0, :r0] = np.eye(r0)
    W1[r0:r0 + a, :] = rng.standard_normal((a, M))
    W2[:, r0 + a:r0 + a + b] = rng.standard_normal((N, b))
    # the excess must not leak into M*: W2[:, :r0] W1[:r0] = M*, W2' W1' = 0, and the
    # cross terms W2[:, r0:] W1[r0:] are zero by construction. Mix by GL(H).
    G = rng.standard_normal((H, H))
    W1, W2 = np.linalg.solve(G, W1), W2 @ G
    assert np.allclose(W2 @ W1, Mstar)
    return W1, W2


# ---- part B: batched GD -------------------------------------------------------

def gd_batch(M, N, H, r0, n_init, init_scale, rng, max_steps=60_000, target=1e-10):
    Mstar = make_teacher(N, M, r0)
    W1 = rng.standard_normal((n_init, H, M)) * init_scale
    W2 = rng.standard_normal((n_init, N, H)) * init_scale
    for step in range(max_steps):
        R = W2 @ W1 - Mstar
        K = (R ** 2).sum(axis=(1, 2))
        if K.max() < target:
            break
        L = (np.linalg.norm(W1, 2, axis=(1, 2)) ** 2 + np.linalg.norm(W2, 2, axis=(1, 2)) ** 2)
        lr = (0.1 / np.maximum(L, 1e-3))[:, None, None]
        g1 = 2 * np.transpose(W2, (0, 2, 1)) @ R
        g2 = 2 * R @ np.transpose(W1, (0, 2, 1))
        W1, W2 = W1 - lr * g1, W2 - lr * g2
    return W1, W2, K, step


def main():
    rng = np.random.default_rng(0)
    cells = aoyagi_2005_anchor_cells(M=M_IN, N=N_OUT)

    print("A. Generic points on each component: rank J vs formula")
    nA = bad = 0
    for c in cells:
        M, N, H, r0 = c["M"], c["N"], c["H"], c["r0"]
        for a, b, pred in components(M, N, H, r0):
            for _ in range(3):
                W1, W2 = generic_point(M, N, H, r0, a, b, rng)
                r, _ = num_rank(jac(W1, W2))
                nA += 1
                bad += r != pred
    print(f"   {nA} points, {bad} mismatches")
    print("   min over components of predicted rank J == 2*lambda on every cell:",
          all(min(p for *_, p in components(c['M'], c['N'], c['H'], c['r0']))
              == 2 * c["lambda_true"] for c in cells))

    print("\nB. Unregularized GD + Gauss-Newton polish, 20 restarts per (cell, init scale)")
    print("   H r0 | 2lam | rkH@min-norm | scale: min rkJ  [(alpha,beta): count]")
    scales = [0.05, 0.5, 1.5]
    summary = {s: 0 for s in scales}
    for c in cells:
        M, N, H, r0 = c["M"], c["N"], c["H"], c["r0"]
        two_lam = int(2 * c["lambda_true"])
        hess_minnorm = r0 * (N + M - r0)
        parts = []
        for s in scales:
            W1s, W2s, K, steps = gd_batch(M, N, H, r0, 20, s, rng)
            ranks, comps = [], {}
            for W1, W2 in zip(W1s, W2s):
                W1, W2, Kf = gauss_newton_project(W1, W2, make_teacher(N, M, r0))
                if Kf > 1e-20:
                    continue
                ranks.append(num_rank(jac(W1, W2))[0])
                ab = excess_ranks(W1, W2, r0)
                comps[ab] = comps.get(ab, 0) + 1
            mn = min(ranks) if ranks else None
            summary[s] += mn == two_lam
            comp_str = " ".join(f"{k}:{v}" for k, v in sorted(comps.items()))
            parts.append(f"{s}: {mn} [{comp_str}]")
        print(f"   {H} {r0:2d} | {two_lam:4d} | {hess_minnorm:12d} | " + " | ".join(parts))
    print("   cells with min rkJ == 2*lambda, by init scale:",
          {s: f"{v}/{len(cells)}" for s, v in summary.items()})

    print("\nC. Tangent-excess estimator at the min-norm point (200 projected samples)")
    print("   H r0 | 2lam | rkH(w0) | eps: 2lam-hat (min rkJ), PCA dim of cloud vs dim Z")
    hits = {e: 0 for e in [1e-3, 1e-2, 1e-1]}
    for c in cells:
        M, N, H, r0 = c["M"], c["N"], c["H"], c["r0"]
        d = H * (M + N)
        Mstar = make_teacher(N, M, r0)
        W10 = np.zeros((H, M)); W20 = np.zeros((N, H))
        W10[:r0, :r0] = np.eye(r0); W20[:r0, :r0] = np.eye(r0)
        rk0 = num_rank(jac(W10, W20))[0]
        two_lam = int(2 * c["lambda_true"])
        parts = []
        for eps in hits:
            rks, cloud = [], []
            for _ in range(200):
                W1 = W10 + eps * rng.standard_normal(W10.shape)
                W2 = W20 + eps * rng.standard_normal(W20.shape)
                W1, W2, Kf = gauss_newton_project(W1, W2, Mstar)
                if Kf > 1e-20:
                    continue
                rks.append(num_rank(jac(W1, W2))[0])
                cloud.append(np.concatenate([(W1 - W10).ravel(), (W2 - W20).ravel()]))
            sv = np.linalg.svd(np.array(cloud), compute_uv=False)
            pca = int((sv > 1e-3 * sv[0]).sum())
            hits[eps] += min(rks) == two_lam
            parts.append(f"{eps:g}: {min(rks)}, PCA {pca} vs {d - two_lam}")
        print(f"   {H} {r0:2d} | {two_lam:4d} | {rk0:7d} | " + " | ".join(parts))
    print("   cells with 2lam-hat == 2*lambda, by eps:",
          {e: f"{v}/{len(cells)}" for e, v in hits.items()})


if __name__ == "__main__":
    main()
