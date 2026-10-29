"""Deep linear networks: arc-space lct of the zero fiber of the multiplication map,
compared with Pepin Lehalleur & Rimanyi (PLR), arXiv:2411.19920.

PLR compute C = codim {A_N ... A_1 = 0} (and theta = number of top components) by
quiver representations: a quadratic integer program (their Thm 6.1) and a Poincare
series (Thm 5.5). Their rlct = C/2 (Thm 8.6) is obtained by comparing C with Aoyagi's
formula, not proved directly. Here:

1. L = 2 (RRR, N = 2 matrices). Check PLR's QIP equals min_k g(k) from rrr_orbit.py,
   in value and in number of minimizers.

2. L = 3. Arc argument. A_1 : V0 -> V1, A_2 : V1 -> V2, A_3 : V2 -> V3. A_3 and A_1 share
   no vertex, so put both in Smith form independently (exponents lambda_i, nu_j). The
   basis changes act on A_2 by A_2 -> h A_2 g^{-1}, which is volume-preserving on jets,
   so A_2 stays a generic arc. Contact ord(A_3 A_2 A_1) >= p becomes entrywise:
   ord (A_2)_{ij} >= p - lambda_i - nu_j for i <= s3 = min(d3,d2), j <= s1 = min(d1,d0).
   With Docampo's stratum codims and x = lambda/p, y = nu/p in [0,1]:
       codim/p = F(x,y) = sum_i x_i (|d3-d2| + 2i - 1) + sum_j y_j (|d1-d0| + 2j - 1)
                          + sum_{i,j} max(0, 1 - x_i - y_j).
   lct = min over strata = min over rational (x, y) of F (decreasing order is automatic
   by rearrangement, so drop it). F is convex piecewise-linear; as an LP with slacks
   z_ij >= 1 - x_i - y_j the constraint matrix is [bipartite incidence | I], which is
   totally unimodular, so the LP optimum is at an integral vertex: x, y in {0,1}, i.e.
   p = 1. Hence lct = min over (a, b) of
       F(a, b) = a(|d3-d2| + a) + b(|d1-d0| + b) + (s3 - a)(s1 - b) = C.
   That is a direct proof of rlct = C/2 at depth 3 (Smith form works over R[[t]],
   minimizers are real), independent of Aoyagi. It assumes Docampo's stratum
   codimensions (checked numerically below), the reduction from teacher rank r to 0
   for the lct, and RLCT_R >= lct_C.

Checks: QIP vs Poincare series vs min_k g (L=2) / min F (L=3), values and counts;
LP relaxation vs integer min; Jacobian rank of the L=3 jet scheme at stratum points.
"""
import itertools
from functools import lru_cache
import numpy as np
from scipy.optimize import linprog

from rrr_orbit import g, poly_mul, random_unit_jet


# ---- PLR -----------------------------------------------------------------------

def plr_qip(d):
    """Thm 6.1: min and argmin count of G over e in N^N, sum e = d'_0."""
    dp = sorted(d)
    N = len(d) - 1
    best, count = None, 0
    for e in compositions(dp[0], N):
        G = sum(e[i - 1] * (e[j - 1] + dp[j] - dp[j - 1])
                for i in range(1, N + 1) for j in range(1, i + 1))
        if best is None or G < best:
            best, count = G, 1
        elif G == best:
            count += 1
    return best, count


def compositions(n, k):
    if k == 1:
        yield (n,)
        return
    for first in range(n + 1):
        for rest in compositions(n - first, k - 1):
            yield (first,) + rest


def plr_series(d, order=None):
    """Thm 5.5 (r = 0): Q = sum_s (-1)^s q^{s(s-1)/2} P_s prod_i P_{d_i - s},
    P_s = 1/((1-q)...(1-q^s)). Lowest term theta q^C."""
    order = order or sum(d) * max(d) + 5

    @lru_cache(None)
    def P(s):
        c = [0] * (order + 1); c[0] = 1
        for k in range(1, s + 1):  # multiply by 1/(1-q^k)
            for n in range(k, order + 1):
                c[n] += c[n - k]
        return tuple(c)

    def mul(a, b):
        out = [0] * (order + 1)
        for i, ai in enumerate(a):
            if ai:
                for j in range(order + 1 - i):
                    out[i + j] += ai * b[j]
        return out

    total = [0] * (order + 1)
    for s in range(min(d) + 1):
        term = list(P(s))
        for di in d:
            term = mul(term, P(di - s))
        shift = s * (s - 1) // 2
        for n in range(order + 1 - shift):
            total[n + shift] += (-1) ** s * term[n]
    for n, c in enumerate(total):
        if c:
            return n, c
    return None


# ---- ours ------------------------------------------------------------------------

def ours_L2(d):
    """d = (d0, d1, d2) = (M', H', N'): min_k g(k) and number of minimizers."""
    Mp, Hp, Np = d
    vals = [g(k, Mp, Np, Hp) for k in range(min(Np, Hp) + 1)]
    m = min(vals)
    return m, vals.count(m)


def F_int(a, b, d):
    d0, d1, d2, d3 = d
    s3, s1 = min(d3, d2), min(d1, d0)
    return a * (abs(d3 - d2) + a) + b * (abs(d1 - d0) + b) + (s3 - a) * (s1 - b)


def ours_L3(d):
    d0, d1, d2, d3 = d
    vals = [F_int(a, b, d) for a in range(min(d3, d2) + 1) for b in range(min(d1, d0) + 1)]
    m = min(vals)
    return m, vals.count(m)


def lp_L3(d):
    """LP relaxation of min F over x in [0,1]^s3, y in [0,1]^s1."""
    d0, d1, d2, d3 = d
    s3, s1 = min(d3, d2), min(d1, d0)
    nx, ny, nz = s3, s1, s3 * s1
    c = np.concatenate([[abs(d3 - d2) + 2 * i - 1 for i in range(1, s3 + 1)],
                        [abs(d1 - d0) + 2 * j - 1 for j in range(1, s1 + 1)],
                        np.ones(nz)])
    A, b = [], []
    for i in range(s3):
        for j in range(s1):  # -x_i - y_j - z_ij <= -1
            row = np.zeros(nx + ny + nz)
            row[i] = row[nx + j] = row[nx + ny + i * s1 + j] = -1
            A.append(row); b.append(-1)
    bounds = [(0, 1)] * (nx + ny) + [(0, None)] * nz
    if not A:
        return 0.0
    res = linprog(c, A_ub=np.array(A), b_ub=b, bounds=bounds, method="highs")
    return res.fun


# ---- Jacobian check of L=3 stratum codims -----------------------------------------

def L3_stratum_point(d, a, b, p, rng):
    """A3 with a killed Smith factors, A1 with b killed, A2 generic subject to the
    entrywise conditions (X_ij = 0 mod t^p where both factors are units)."""
    d0, d1, d2, d3 = d
    s3, s1 = min(d3, d2), min(d1, d0)
    G3, _ = random_unit_jet(d3, p, rng); H3, H3i = random_unit_jet(d2, p, rng)
    G1, G1i = random_unit_jet(d1, p, rng); H1, _ = random_unit_jet(d0, p, rng)
    D3 = np.zeros((d3, d2)); D1 = np.zeros((d1, d0))
    for i in range(a, s3):
        D3[i, i] = 1.0
    for j in range(b, s1):
        D1[j, j] = 1.0
    z = lambda X: [X] + [np.zeros_like(X) for _ in range(p - 1)]
    A3 = poly_mul(poly_mul(G3, z(D3), p), H3, p)
    A1 = poly_mul(poly_mul(G1, z(D1), p), H1, p)
    X = [rng.standard_normal((d2, d1)) for _ in range(p)]
    for Xc in X:
        for i in range(a, s3):
            for j in range(b, s1):
                Xc[i, j] = 0.0
    A2 = poly_mul(poly_mul(H3i, X, p), G1i, p)
    return A3, A2, A1


def jac_rank_L3(A3, A2, A1, p, tol=1e-8):
    """Rank of the Jacobian of jets (A3, A2, A1) -> coeffs 0..p-1 of A3 A2 A1."""
    mats = [A3, A2, A1]
    cols = []
    for which in range(3):
        shape = mats[which][0].shape
        for a in range(p):
            for idx in range(shape[0] * shape[1]):
                E = np.zeros(shape[0] * shape[1]); E[idx] = 1.0
                Ej = [np.zeros(shape) for _ in range(p)]
                Ej[a] = E.reshape(shape)
                parts = [Ej if k == which else mats[k] for k in range(3)]
                prod = poly_mul(poly_mul(parts[0], parts[1], p), parts[2], p)
                cols.append(np.concatenate([c.ravel() for c in prod]))
    J = np.stack(cols, axis=1)
    sv = np.linalg.svd(J, compute_uv=False)
    return int((sv > tol * sv[0]).sum())


def main():
    print("1. L = 2: PLR QIP vs PLR series vs min_k g(k), all d in [1,7]^3")
    bad = n = 0
    for d in itertools.product(range(1, 8), repeat=3):
        q, s, o = plr_qip(d), plr_series(d), ours_L2(d)
        n += 1
        if not (q == s == o):
            bad += 1
            if bad <= 5:
                print("   mismatch", d, "QIP", q, "series", s, "ours", o)
    print(f"   {n} dimension vectors, {bad} mismatches in (C, theta)")

    print("\n2. L = 3: PLR QIP vs PLR series vs min F (integer) vs LP relaxation, d in [1,5]^4")
    bad = lp_gap = n = 0
    for d in itertools.product(range(1, 6), repeat=4):
        q, s, o = plr_qip(d), plr_series(d), ours_L3(d)
        lp = lp_L3(d)
        n += 1
        if not (q == s == o):
            bad += 1
            if bad <= 5:
                print("   mismatch", d, "QIP", q, "series", s, "ours", o)
        if abs(lp - o[0]) > 1e-9:
            lp_gap += 1
            if lp_gap <= 5:
                print("   LP below integer min", d, lp, o[0])
    print(f"   {n} dimension vectors, {bad} (C, theta) mismatches, "
          f"{lp_gap} with LP relaxation below the integer min")

    print("\n3. L = 3 strata: Jacobian rank of the jet scheme vs p * F(a, b)")
    print("   Rank <= local codim always; equality is expected only where the scheme is")
    print("   smooth, i.e. on strata open in a component. Over-killed strata (e.g. A3 =")
    print("   A1 = 0) sit inside bigger components and show lower rank, as they should.")
    rng = np.random.default_rng(0)
    ok_ineq, ok_top, n_eq, n = True, True, 0, 0
    for d in [(2, 2, 2, 2), (3, 2, 3, 2), (2, 3, 3, 4), (4, 2, 3, 3), (3, 3, 2, 4)]:
        C = ours_L3(d)[0]
        for p in [1, 2]:
            for a in range(min(d[3], d[2]) + 1):
                for b in range(min(d[1], d[0]) + 1):
                    rk = jac_rank_L3(*L3_stratum_point(d, a, b, p, rng), p)
                    pred = p * F_int(a, b, d)
                    n += 1; n_eq += rk == pred
                    ok_ineq &= rk <= pred
                    if F_int(a, b, d) == C:
                        ok_top &= rk == pred
                        if rk != pred:
                            print(f"   top stratum {d} a={a} b={b} p={p}: rank {rk} vs {pred}")
    print(f"   rank <= p*F on all {n} strata: {ok_ineq}; equality on {n_eq}/{n}; "
          f"equality on every minimizing (top) stratum: {ok_top}")


if __name__ == "__main__":
    main()
