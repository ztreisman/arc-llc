"""Depth >= 4 exploration: monomial strata of the contact loci of the multiplication map.

A monomial configuration of matrix arcs (A_1, ..., A_L) over R_p = C[t]/t^p is a
partial-permutation chain with a t-exponent on each arrow. Up to vertex permutations it
is a multiset of LACES (a, b, (mu_{a+1}, ..., mu_b)): a strand from vertex a to vertex b
with exponents mu in {0, ..., p-1} on its arrows. At p = 1 these are exactly PLR's lace
diagrams. The product A_L ... A_1 vanishes mod t^p iff every full lace [0, L] has
exponent sum >= p.

For each multiset we compute the codimension of its G = prod GL_{d_v}(R_p) orbit in
J_{p-1} from the rank of the infinitesimal action X -> (X_k A_k - A_k X_{k-1})_k. Exact
for the orbit. These orbits lie in Cont^{>=p}, but need not exhaust it: at depth >= 4
the classification contains submodule-position problems, so non-monomial families can
exist. min codim / p over monomial strata is therefore an UPPER bound on the true value
at each p.

Outputs per dimension vector:
  * C_mono(p) = min codim/p over monomial strata, p = 1..P (compare PLR C);
  * N_p = number of monomial strata attaining codim = p*C. The growth of N_p in p
    probes the pole order: the motivic zeta function has a pole of order theta when the
    minimizing strata at level p grow like p^(theta - 1).
    Compare: Aoyagi theta (pole order) and PLR theta (number of top components).
Sanity: depth 2 and 3, where the arc argument is complete.
"""
import itertools
import sys
import numpy as np

from deep_linear_orbit import plr_qip
from aoyagi_compare import aoyagi


def lace_types(L, p):
    types = []
    for a in range(L + 1):
        for b in range(a, L + 1):
            for mus in itertools.product(range(p), repeat=b - a):
                types.append((a, b, mus))
    return types


def multisets(types, d):
    """All multisets of lace types covering vertex v exactly d[v] times."""
    out = []
    cap = list(d)

    def rec(start, chosen):
        if all(c == 0 for c in cap):
            out.append(tuple(chosen))
            return
        # first vertex with remaining capacity must be covered by a lace starting
        # there or earlier; enforce canonical order by type index
        v0 = next(v for v, c in enumerate(cap) if c > 0)
        for ti in range(start, len(types)):
            a, b, _ = types[ti]
            if a > v0:
                break
            if a < v0:
                continue  # would need capacity at a, which is 0
            if all(cap[v] > 0 for v in range(a, b + 1)):
                for v in range(a, b + 1):
                    cap[v] -= 1
                chosen.append(ti)
                rec(ti, chosen)
                chosen.pop()
                for v in range(a, b + 1):
                    cap[v] += 1

    rec(0, [])
    return out


def config_from(ms, types, d, p):
    """Build monomial matrices A_k (d_k x d_{k-1}) as {(row, col): exponent}."""
    L = len(d) - 1
    nxt = [0] * (L + 1)
    A = [dict() for _ in range(L + 1)]  # A[k] for k = 1..L
    for ti in ms:
        a, b, mus = types[ti]
        idx = {}
        for v in range(a, b + 1):
            idx[v] = nxt[v]; nxt[v] += 1
        for s, v in enumerate(range(a + 1, b + 1)):
            A[v][(idx[v], idx[v - 1])] = mus[s]
    return A


def in_contact(ms, types, L, p):
    for ti in ms:
        a, b, mus = types[ti]
        if a == 0 and b == L and sum(mus) < p:
            return False
    return True


def orbit_codim(A, d, p):
    """codim in J_{p-1} of the G-orbit of the monomial configuration A."""
    L = len(d) - 1
    row_off, r = {}, 0
    for k in range(1, L + 1):
        for n in range(p):
            row_off[(k, n)] = r; r += d[k] * d[k - 1]
    nrows = r
    col_off, c = {}, 0
    for v in range(L + 1):
        for e in range(p):
            col_off[(v, e)] = c; c += d[v] * d[v]
    ncols = c
    T = np.zeros((nrows, ncols))
    for k in range(1, L + 1):
        for (rr, cc), mu in A[k].items():
            for e in range(p - mu):
                n = e + mu
                # X_k A_k with X_k = E_{a,rr} t^e: contributes E_{a,cc} t^n
                for a_ in range(d[k]):
                    T[row_off[(k, n)] + a_ * d[k - 1] + cc,
                      col_off[(k, e)] + a_ * d[k] + rr] += 1
                # -A_k X_{k-1} with X_{k-1} = E_{cc,b} t^e: contributes -E_{rr,b} t^n
                for b_ in range(d[k - 1]):
                    T[row_off[(k, n)] + rr * d[k - 1] + b_,
                      col_off[(k - 1, e)] + cc * d[k - 1] + b_] -= 1
    rank = np.linalg.matrix_rank(T) if T.any() else 0
    return nrows - rank


def analyze(d, P):
    L = len(d) - 1
    C, th_plr = plr_qip(d)
    _, th_aoy = aoyagi(tuple(reversed(d)), 0)
    res = []
    for p in range(1, P + 1):
        types = lace_types(L, p)
        best, count, n = None, 0, 0
        for ms in multisets(types, d):
            if not in_contact(ms, types, L, p):
                continue
            n += 1
            cd = orbit_codim(config_from(ms, types, d, p), d, p)
            if best is None or cd < best:
                best, count = cd, 1
            elif cd == best:
                count += 1
        res.append((p, best / p, count, n))
    return C, th_plr, th_aoy, res


def main():
    P = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cases = [(3, 2, 4), (2, 2, 2), (3, 3, 3), (2, 2, 2, 2), (2, 3, 2, 3),
             (2, 2, 2, 2, 2), (1, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2)]
    for d in cases:
        C, th_plr, th_aoy, res = analyze(d, P if len(d) <= 5 else min(P, 2))
        row = " ".join(f"p={p}: {r:.2f} N={c} (of {n})" for p, r, c, n in res)
        print(f"d={d}  C={C}  theta_PLR={th_plr}  theta_Aoyagi={th_aoy} | {row}", flush=True)


if __name__ == "__main__":
    main()


def complete_L3_count(d, p):
    """Depth 3, COMPLETE stratification (Smith A3, Smith A1, entrywise A2):
    number of strata (sorted lambda, nu in [0,p]) with codim exactly p*C."""
    d0, d1, d2, d3 = d
    s3, s1 = min(d3, d2), min(d1, d0)
    C = plr_qip(d)[0]
    dec = lambda s: [v for v in itertools.product(range(p + 1), repeat=s)
                     if all(v[i] >= v[i + 1] for i in range(s - 1))]
    n = 0
    for lam in dec(s3):
        for nu in dec(s1):
            cd = sum(l * (abs(d3 - d2) + 2 * i + 1) for i, l in enumerate(lam)) \
                + sum(v * (abs(d1 - d0) + 2 * j + 1) for j, v in enumerate(nu)) \
                + sum(max(0, p - l - v) for l in lam for v in nu)
            n += cd == p * C
    return n


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "L3":
    for d in [(2, 2, 2, 2), (2, 3, 2, 3), (3, 3, 3, 3), (1, 2, 2, 2), (2, 2, 3, 3)]:
        _, th_plr = plr_qip(d)
        _, th_aoy = aoyagi(tuple(reversed(d)), 0)
        counts = [complete_L3_count(d, p) for p in range(1, 7)]
        print(f"d={d} theta_PLR={th_plr} theta_Aoyagi={th_aoy}  complete N_p, p=1..6: {counts}")
