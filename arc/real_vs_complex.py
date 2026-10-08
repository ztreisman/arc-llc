"""Step 3: where the complex fiber-ideal formula rlct(K) = lct_C(a)/2 fails.

Lin (arXiv:1003.5338, Thm 1.2) proves rlct(K) = RLCT_R(a)/2 for any a = (f_i) with
K ~ sum f_i^2. RLCT_R(a) is a real invariant: equivalent sums of squares give the same
value (his Cor 3.5). lct_C(a) is not: it depends on which ideal you pick, and
RLCT_R(a) >= lct_C(a), with strict inequality when every divisor computing lct_C(a) has
too few real points.

1. Smallest case. K = (x^2+y^2+z^2)^2 = sum of squares of the generators of either
   a1 = (x^2+y^2+z^2)  -> lct_C = 1 (generic point of the cone; real points: only 0)
   a2 = m^2            -> lct_C = 3/2
   True rlct = 3/4 (K ~ r^4 in R^3). a1 predicts 1/2, wrong; a2 is right. Real arcs
   count correctly: ord(x^2+y^2+z^2)(gamma) >= 2q iff every coordinate has ord >= q,
   codim 3q in J, ratio 3q/2q = 3/2 = 2*rlct. Complex arcs on the cone give ratio 1.

2. Symmetric factorization W W^T = M* (W n x k, rank M* = r0). Same reduction as RRR:
   rank H = r0 n - r0(r0-1)/2, plus V in R^{m x h}, m = n-r0, h = k-r0, with fiber ideal
   entries(V V^T). Over R, ||V V^T||^2 ~ ||V||^4 and the zero set is {V = 0}, so
       lambda_sym = (r0 n - r0(r0-1)/2)/2 + m h / 4.
   Unlike RRR, the real true set is smooth here; the gap 2*lambda - rank H = mh/2 is
   genuinely higher-order (quartic) contact, the DDS additive picture with k = 2.
   Over C, V V^T = 0 has totally isotropic components with no real points except 0,
   and lct_C(a') <= their codim, which can drop below mh/2: the complex formula
   undercounts.

Checks: Monte Carlo volume scaling for lambda_sym; isotropic codim by formula and by
Jacobian rank at random complex points.
"""
import numpy as np


# ---- Monte Carlo volume scaling ---------------------------------------------

def mc_lambda(K_fn, w0, radius, n_total, eps_grid, rng, chunk=1_000_000):
    """Fit log P(K < eps) ~ lambda log eps (+ const) over eps_grid.
    Assumes multiplicity 1; reports the slope and counts."""
    d = w0.size
    counts = np.zeros(len(eps_grid), dtype=np.int64)
    done = 0
    while done < n_total:
        n = min(chunk, n_total - done)
        w = w0 + radius * rng.uniform(-1, 1, size=(n, d))
        K = K_fn(w)
        counts += (K[:, None] < eps_grid[None, :]).sum(axis=0)
        done += n
    ok = counts >= 30
    slope = np.polyfit(np.log(eps_grid[ok]), np.log(counts[ok] / n_total), 1)[0]
    return slope, counts


def K_quartic_norm(w):
    return (w ** 2).sum(axis=1) ** 2


def make_K_sym(n, k, r0):
    Mstar = np.zeros((n, n)); Mstar[np.arange(r0), np.arange(r0)] = 1.0

    def K(w):
        W = w.reshape(-1, n, k)
        R = W @ np.transpose(W, (0, 2, 1)) - Mstar
        return (R ** 2).sum(axis=(1, 2))
    return K


def sym_w0(n, k, r0):
    W = np.zeros((n, k)); W[np.arange(r0), np.arange(r0)] = 1.0
    return W.ravel()


def lambda_sym(n, k, r0):
    return (r0 * n - r0 * (r0 - 1) / 2) / 2 + (n - r0) * (k - r0) / 4


# ---- complex isotropic components -------------------------------------------

def isotropic_codim_formula(m, h):
    """min over q of codim in C^{m x h} of {V : rows span a totally isotropic
    q-space}, q <= min(m, h/2). dim OG(q,h) = q(h-q) - q(q+1)/2."""
    best = (m * h, 0)
    for q in range(1, min(m, h // 2) + 1):
        dim = q * (h - q) - q * (q + 1) // 2 + m * q
        best = min(best, (m * h - dim, q))
    return best


def isotropic_point(m, h, q, rng):
    """Random complex V (m x h) whose rows span a random totally isotropic q-space."""
    # isotropic basis e_{2j} + i e_{2j+1}, then a random real orthogonal change
    U = np.zeros((q, h), dtype=complex)
    for j in range(q):
        U[j, 2 * j], U[j, 2 * j + 1] = 1.0, 1j
    Q, _ = np.linalg.qr(rng.standard_normal((h, h)))
    U = U @ Q
    V = (rng.standard_normal((m, q)) + 1j * rng.standard_normal((m, q))) @ U
    assert np.allclose(V @ V.T, 0)
    return V


def jac_VVt(V):
    """Complex Jacobian of V -> upper-triangular entries of V V^T."""
    m, h = V.shape
    rows = []
    for i in range(m):
        for j in range(i, m):
            g = np.zeros((m, h), dtype=complex)
            g[i] += V[j]; g[j] += V[i]
            rows.append(g.ravel())
    return np.array(rows)


def main():
    rng = np.random.default_rng(0)
    eps = np.logspace(-6, -1, 11)

    print("1. K = (x^2+y^2+z^2)^2: rlct by volume scaling (expect 0.75)")
    s, _ = mc_lambda(K_quartic_norm, np.zeros(3), 1.0, 4_000_000, eps, rng)
    print(f"   slope {s:.3f}   lct_C(x^2+y^2+z^2)/2 = 0.5,  lct_C(m^2)/2 = 0.75")

    print("\n2. Symmetric factorization W W^T = M*: lambda_sym vs Monte Carlo")
    print("   (n,k,r0) d | lambda_sym | MC slope | rankH/2 | complex bound on lambda")
    for (n, k, r0) in [(1, 3, 0), (1, 4, 0), (2, 2, 1), (2, 3, 1), (2, 4, 1), (3, 2, 1)]:
        d = n * k
        m, h = n - r0, k - r0
        rank_half = (r0 * n - r0 * (r0 - 1) / 2) / 2
        cod, q = isotropic_codim_formula(m, h)
        cbound = rank_half + min(cod, m * h / 2) / 2
        eps_c = np.logspace(-5, -1, 9)
        s, cnt = mc_lambda(make_K_sym(n, k, r0), sym_w0(n, k, r0), 0.5, 20_000_000,
                           eps_c, rng)
        print(f"   ({n},{k},{r0}) {d:2d} | {lambda_sym(n, k, r0):10.3f} | {s:8.3f} | "
              f"{rank_half:7.2f} | {cbound:.3f}" + ("  < lambda" if cbound < lambda_sym(n, k, r0) else ""))

    print("\n3. Complex isotropic components of V V^T = 0: formula codim vs Jacobian rank")
    print("   (m,h) | q | codim (formula) | Jacobian rank | mh/2 | complex < real?")
    for (m, h) in [(1, 2), (1, 3), (1, 5), (2, 3), (2, 4), (2, 6), (3, 4), (3, 6), (4, 8)]:
        cod, q = isotropic_codim_formula(m, h)
        if q == 0:
            print(f"   ({m},{h}) | - | no isotropic component | | {m*h/2} |")
            continue
        V = isotropic_point(m, h, q, rng)
        sv = np.linalg.svd(jac_VVt(V), compute_uv=False)
        rk = int((sv > 1e-9 * sv[0]).sum())
        print(f"   ({m},{h}) | {q} | {cod:15d} | {rk:13d} | {m*h/2:4} | "
              f"{'yes' if cod < m*h/2 else 'no'}")


if __name__ == "__main__":
    main()
