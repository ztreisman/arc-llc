"""
Contact-hierarchy experiment.

Design A: vanishing-order estimation along sampled linear arcs.
Design B: tube-volume exponent of the coefficient map C_p on jet space,
          testing whether  min_p  chat_p / p  recovers the rlct.

All test cases are polynomial, so jets are computed exactly by truncated
polynomial arithmetic (no autodiff, no truncation error).
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- truncated polynomial arithmetic, arrays of shape (N, p) ----------

def tmul(A, B):
    """Multiply two truncated power series mod t^p, batched over axis 0."""
    N, p = A.shape
    C = np.zeros_like(A)
    for k in range(p):
        acc = np.zeros(N)
        for i in range(k + 1):
            acc += A[:, i] * B[:, k - i]
        C[:, k] = acc
    return C

def tpow(A, e):
    R = np.zeros_like(A)
    R[:, 0] = 1.0
    base = A
    while e:
        if e & 1:
            R = tmul(R, base)
        e >>= 1
        if e:
            base = tmul(base, base)
    return R

# ---------- test cases ----------
# each: (name, n_vars, lambda_true, K(list of (N,p) jets) -> (N,p))

CASES = [
    ("x^4",            1, 0.25,      lambda v: tpow(v[0], 4)),
    ("x^2 + y^2",      2, 1.0,       lambda v: tpow(v[0], 2) + tpow(v[1], 2)),
    ("a^2 b^2",        2, 0.5,       lambda v: tmul(tpow(v[0], 2), tpow(v[1], 2))),
    ("a^2 + b^6",      2, 2.0 / 3,   lambda v: tpow(v[0], 2) + tpow(v[1], 6)),
    ("(a^3 - b^2)^2",  2, 5.0 / 12,  lambda v: tpow(tpow(v[0], 3) - tpow(v[1], 2), 2)),
    ("||a||^2||b||^2", 4, 1.0,       lambda v: tmul(tpow(v[0],2)+tpow(v[1],2),
                                                    tpow(v[2],2)+tpow(v[3],2))),
]

# ---------- sampling ----------

def ball(N, n):
    """Uniform in the unit ball of R^n."""
    g = rng.standard_normal((N, n))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = rng.random((N, 1)) ** (1.0 / n)
    return g * r

def jets(N, n, p):
    """Sample gamma in J_{p-1}: base point c_0 in unit ball, higher coeffs too.
    Returns list of n arrays of shape (N, p)."""
    coeffs = [ball(N, n) for _ in range(p)]      # c_0 ... c_{p-1}, each (N, n)
    return [np.stack([c[:, i] for c in coeffs], axis=1) for i in range(n)]

# ---------- Design A: vanishing order along linear arcs ----------

def design_A(case, N=20000, n_t=40):
    name, n, lam, K = case
    u = rng.standard_normal((N, n))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    ts = np.logspace(-6, -1, n_t)
    # K(t u) with jets of length 1 is just K evaluated pointwise; build (N,1) jets
    orders = []
    vals = np.empty((N, n_t))
    for j, t in enumerate(ts):
        v = [np.ascontiguousarray((t * u[:, i]).reshape(N, 1)) for i in range(n)]
        vals[:, j] = K(v)[:, 0]
    good = vals[:, 0] > 1e-300
    lv = np.log(np.maximum(vals, 1e-300))
    lt = np.log(ts)
    slope = np.polyfit(lt, lv[good].T, 1)[0]
    med = np.median(slope)
    frac = np.mean(np.abs(slope - med) < 0.05)
    return med, frac, 1.0 - good.mean()

# ---------- Design B: tube-volume exponent of C_p ----------

def tube_exponent(case, p, N=400_000, chunk=100_000):
    name, n, lam, K = case
    eps_grid = np.logspace(-9, -1, 33)
    counts = np.zeros_like(eps_grid)
    done = 0
    while done < N:
        m = min(chunk, N - done)
        v = jets(m, n, p)
        A = K(v)                       # (m, p) coefficients a_0..a_{p-1}
        worst = np.max(np.abs(A[:, :p]), axis=1)
        counts += (worst[:, None] <= eps_grid[None, :]).sum(axis=0)
        done += m
    V = counts / N
    # fit log V = c * log eps + const, on a window with decent statistics
    mask = (V > 5e-5) & (V < 0.2)
    if mask.sum() < 5:
        return np.nan, V, eps_grid
    c = np.polyfit(np.log(eps_grid[mask]), np.log(V[mask]), 1)[0]
    return c, V, eps_grid

# ---------- run ----------

if __name__ == "__main__":
    print("=== Design A: vanishing order 2k along generic linear arcs ===")
    for case in CASES:
        med, frac, deg = design_A(case)
        print(f"  {case[0]:>16s}   generic order = {med:5.2f}"
              f"   (frac at generic = {frac:.3f}, frac identically zero = {deg:.3f})")

    print()
    print("=== Design B: tube-volume exponent chat_p and chat_p / p ===")
    header = f"{'case':>16s} {'lambda':>8s} " + " ".join(
        f"{'c'+str(p)+'/'+str(p):>9s}" for p in (1, 2, 3))
    print(header)
    for case in CASES:
        row = f"{case[0]:>16s} {case[2]:8.4f} "
        for p in (1, 2, 3):
            c, V, e = tube_exponent(case, p)
            row += f" {c/p:9.4f}"
        print(row)
