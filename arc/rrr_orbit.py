"""RRR RLCT from the fiber ideal by orbit counting (NOTEBOOK.md 4f).

K(A, B) = ||BA - M*||_F^2, B in R^{N x H}, A in R^{H x M}, rank M* = r0.
Fiber ideal a = (entries of BA - M*). Claim: rlct = lct(a) / 2, and lct(a) is
computable exactly from GL(C[[t]]) orbits of matrix arcs.

Derivation
----------
1. Reduction. Near the min-norm point (B1 A1 = M*, excess block zero), a change of
   coordinates splits off r0(N+M-r0) smooth coordinates, leaving the ideal of
   entries of B'A' with B' in Mat(N' x H'), A' in Mat(H' x M'), primes = minus r0.
   So lct(a) = r0(N+M-r0) + lct(B'A').  The first term is rank of the Hessian.

2. Orbits. Stratify arcs B'(t) by Smith form t^{l_1}, ..., t^{l_s}, s = min(N', H'),
   l_1 >= ... >= l_s.  Docampo: the stratum has codim sum_i l_i (|N'-H'| + 2i - 1)
   in J_{p-1} (exponents capped at p).  Given B' in Smith form, B'A' = 0 mod t^p
   iff row i of A' vanishes mod t^{p - l_i}: codim M' * sum_i (p - l_i).
   Total codim is linear in l, so the minimum has l_i in {0, p}: k factors killed,
       codim / p = g(k) = k^2 - k (M' - |N'-H'|) + M' s,    k in {0, ..., s}.
   Independent of p, so lct(B'A') = min_k g(k) = codim of {B'A' = 0} (variety of
   complexes).  Integer minimization of a quadratic is where Aoyagi's parity
   split comes from.

3. Gap.  2*lambda - rank(H) = min_k g(k); in Aoyagi case 3 the minimizer is
   k = s and the gap is (H - r0)(N - r0).

This script checks (2)+(1) against the published formula over a grid, and checks
Docampo's codim for the minimizing strata by Jacobian rank of the jet scheme.
"""
from fractions import Fraction
import itertools
import numpy as np


# ---- closed forms -----------------------------------------------------------

def aoyagi_lambda_general(M, N, H, r):
    """Aoyagi & Watanabe (2005) main theorem, all four cases. M = input dim,
    N = output dim, H = hidden, r = true rank (r <= H)."""
    if M + H < N + r:
        return Fraction(H * M - H * r + N * r, 2)
    if N + H < M + r:
        return Fraction(H * N - H * r + M * r, 2)
    if M + N < H + r:
        return Fraction(M * N, 2)
    num = -(H + r) ** 2 - M ** 2 - N ** 2 + 2 * (H + r) * (M + N) + 2 * M * N
    if (M + H + N + r) % 2:
        num += 1
    return Fraction(num, 8)


def g(k, Mp, Np, Hp):
    s = min(Np, Hp)
    return k * k - k * (Mp - abs(Np - Hp)) + Mp * s


def lct_reduced(Mp, Np, Hp):
    """lct of the ideal of entries of B'A' at the origin, and the minimizing k's."""
    s = min(Np, Hp)
    vals = [g(k, Mp, Np, Hp) for k in range(s + 1)]
    best = min(vals)
    return best, [k for k, v in enumerate(vals) if v == best]


def orbit_lambda(M, N, H, r):
    rank_hess = r * (N + M - r)
    gap, ks = lct_reduced(M - r, N - r, H - r)
    return Fraction(rank_hess + gap, 2), rank_hess, gap, ks


# ---- Jacobian check of the stratum codimension ------------------------------

def poly_mul(X, Y, p):
    """Product of matrix jets X[j], Y[j] (lists of length p), truncated at t^p."""
    out = [np.zeros((X[0].shape[0], Y[0].shape[1])) for _ in range(p)]
    for a in range(p):
        for b in range(p - a):
            out[a + b] += X[a] @ Y[b]
    return out


def random_unit_jet(n, p, rng):
    """A random invertible matrix jet and its inverse mod t^p."""
    G = [rng.standard_normal((n, n)) for _ in range(p)]
    G0inv = np.linalg.inv(G[0])
    Ginv = [G0inv]
    for j in range(1, p):  # solve sum_{a+b=j} G[a] Ginv[b] = 0
        acc = sum(G[a] @ Ginv[j - a] for a in range(1, j + 1))
        Ginv.append(-G0inv @ acc)
    return G, Ginv


def stratum_point(Mp, Np, Hp, k, p, rng):
    """A generic jet (B', A') in the stratum with k invariant factors of B'
    killed mod t^p and the rest units; A' as generic as allowed."""
    s = min(Np, Hp)
    g_, _ = random_unit_jet(Np, p, rng)
    h, hinv = random_unit_jet(Hp, p, rng)
    D = np.zeros((Np, Hp))
    for i in range(s - k):  # unit invariant factors in the first s-k slots
        D[i, i] = 1.0
    Dj = [D] + [np.zeros_like(D) for _ in range(p - 1)]
    B = poly_mul(poly_mul(g_, Dj, p), h, p)
    # B = g D h; need (D h A) = 0 mod t^p, i.e. rows 0..s-k-1 of h A vanish
    C = [rng.standard_normal((Hp, Mp)) for _ in range(p)]
    for c in C:
        c[: s - k, :] = 0.0
    A = poly_mul(hinv, C, p)
    return B, A


def jacobian_codim(B, A, p, tol=1e-8):
    """Rank of the Jacobian of (B, A) -> coefficients t^0..t^{p-1} of BA."""
    Np, Hp = B[0].shape
    Mp = A[0].shape[1]
    nB, nA = Np * Hp, Hp * Mp
    rows = []
    # d(BA)_j = sum_{a+b=j} dB_a A_b + B_a dA_b; build column by column
    cols = []
    for a in range(p):
        for idx in range(nB):
            E = np.zeros(nB); E[idx] = 1.0; E = E.reshape(Np, Hp)
            col = [np.zeros((Np, Mp)) if j < a else E @ A[j - a] for j in range(p)]
            cols.append(np.concatenate([c.ravel() for c in col]))
    for b in range(p):
        for idx in range(nA):
            E = np.zeros(nA); E[idx] = 1.0; E = E.reshape(Hp, Mp)
            col = [np.zeros((Np, Mp)) if j < b else B[j - b] @ E for j in range(p)]
            cols.append(np.concatenate([c.ravel() for c in col]))
    J = np.stack(cols, axis=1)
    sv = np.linalg.svd(J, compute_uv=False)
    return int((sv > tol * sv[0]).sum())


# ---- main -------------------------------------------------------------------

def main():
    print("1. Orbit formula vs Aoyagi-Watanabe, all cases, 1 <= M,N <= 9, r <= H <= 9")
    n, bad = 0, []
    for M, N, H in itertools.product(range(1, 10), repeat=3):
        for r in range(0, min(M, N, H) + 1):
            lam_orbit = orbit_lambda(M, N, H, r)[0]
            lam_aw = aoyagi_lambda_general(M, N, H, r)
            n += 1
            if lam_orbit != lam_aw:
                bad.append((M, N, H, r, lam_orbit, lam_aw))
    print(f"   {n} cells, {len(bad)} mismatches", bad[:5])

    print("\n2. Repo's 14-cell anchor grid (M=10, N=5): gap = 2*lambda - rank(H)")
    print("   H r0 | lambda | rank(H) | gap | (H-r0)(N-r0) | minimizing k")
    for H in [2, 3, 4, 5]:
        for r0 in range(1, min(5, H) + 1):
            if not 5 + H < 10 + r0:
                continue
            lam, rk, gap, ks = orbit_lambda(10, 5, H, r0)
            print(f"   {H} {r0:2d} | {float(lam):6.1f} | {rk:7d} | {gap:3d} | "
                  f"{(H - r0) * (5 - r0):12d} | {ks}")

    print("\n3. Jacobian rank of the jet scheme at generic points of each stratum")
    print("   (Mp,Np,Hp) k p | jac rank | p*g(k) ")
    rng = np.random.default_rng(0)
    ok = True
    for (Mp, Np, Hp) in [(5, 2, 3), (4, 3, 2), (3, 3, 3), (4, 2, 4), (6, 3, 2)]:
        for p in [1, 2, 3]:
            for k in range(min(Np, Hp) + 1):
                B, A = stratum_point(Mp, Np, Hp, k, p, rng)
                rk = jacobian_codim(B, A, p)
                pred = p * g(k, Mp, Np, Hp)
                ok &= rk == pred
                print(f"   ({Mp},{Np},{Hp}) {k} {p} | {rk:8d} | {pred:6d}"
                      + ("" if rk == pred else "   <-- MISMATCH"))
    print("   all strata match" if ok else "   MISMATCHES ABOVE")


if __name__ == "__main__":
    main()
