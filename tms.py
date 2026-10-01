"""Toy Model of Superposition (TMS) in the high-sparsity limit, following
Chen, Lau, Mendel, Wei & Murfet, "Dynamical versus Bayesian Phase Transitions
in a Toy Model of Superposition" (arXiv:2310.06301).

Model: f(x; W, b) = ReLU(W^T W x + b), W in R^{r x c}, b in R^c, x = mu * e_i
with i uniform on {1..c}, mu ~ U[0,1]. Population loss
    L(W, b) = E_x ||x - f(x; W, b)||^2
(unhalved, the convention under which the paper's tabulated losses and
critical sample sizes are stated: n_cr = 601 for the 5->6 transition is
reproduced with exactly this L).

L is computed exactly (no quadrature): for x = mu e_i the j-th output is
ReLU(mu*g_ij + b_j) with g = W^T W, which is active on an interval of mu in
[0,1], so each component's integral is a closed-form polynomial in the
interval endpoints. Everything is batched over a leading dimension so many
chains / trajectories run in parallel.
"""
import numpy as np
import torch

DT = torch.float64


def population_loss(W, b):
    """W: (B, r, c), b: (B, c) -> (B,) exact population loss."""
    c = W.shape[-1]
    g = W.transpose(-1, -2) @ W                      # (B, c, c), g[i, j] = W_i . W_j
    # component j of the output for input mu*e_i: ReLU(mu*g[i,j] + b[j]).
    G = g                                             # index [B, i, j]
    bj = b[:, None, :].expand_as(G)
    a = torch.eye(c, dtype=W.dtype, device=W.device)[None].expand_as(G)

    eps = 1e-12
    mu0 = -bj / torch.where(G.abs() < eps, torch.full_like(G, eps), G)
    pos = G > eps
    neg = G < -eps
    flat = ~(pos | neg)
    lo = torch.where(pos, mu0, torch.zeros_like(G))
    hi = torch.where(neg, mu0, torch.ones_like(G))
    lo = lo.clamp(0, 1)
    hi = hi.clamp(0, 1)
    # flat (g ~ 0): active on all of [0,1] iff b > 0
    lo = torch.where(flat, torch.zeros_like(G), lo)
    hi = torch.where(flat, (bj > 0).to(G.dtype), hi)
    hi = torch.maximum(hi, lo)
    length = hi - lo
    p = a - G
    active = p ** 2 * (hi ** 3 - lo ** 3) / 3 - p * bj * (hi ** 2 - lo ** 2) + bj ** 2 * length
    inactive = a ** 2 * (1.0 / 3 - (hi ** 3 - lo ** 3) / 3)
    return (active + inactive).sum((-1, -2)) / c


def population_loss_np(W, b):
    return float(population_loss(torch.tensor(W[None], dtype=DT), torch.tensor(b[None], dtype=DT))[0])


def quadrature_loss(W, b, n_mu=200_001):
    """Independent brute-force check of population_loss (single point)."""
    r, c = W.shape
    mu = np.linspace(0, 1, n_mu)
    g = W.T @ W
    tot = 0.0
    for i in range(c):
        out = np.maximum(mu[:, None] * g[i][None, :] + b[None, :], 0.0)
        tgt = np.zeros_like(out)
        tgt[:, i] = mu
        tot += np.trapezoid(((tgt - out) ** 2).sum(1), mu)
    return tot / c


# l*, b* from Table A.1 of the paper (c = 6): optimal column length / bias
# of the standard k-gon. The 5-gon's sixth column is zero with a negative
# bias (any bias < 0 works; -1 is a representative vestigial value).
KGON_PARAMS = {4: (1.0, 0.0), 5: (1.17046, -0.28230), 6: (1.32053, -0.61814)}
KGON_LOSS_REF = {5: 0.06874, 6: 0.04819}
KGON_LLC_REF = {5: 7.0, "5+": 8.5, 6: 8.5}


def kgon(k, c=6, n_pos=0, vestigial_bias=-1.0, rotation=0.0):
    """Standard k-gon (r=2): k columns of length l* at angles 2*pi*j/k, the
    remaining c-k columns zero with negative bias, optionally `n_pos` of those
    biases set to the positive critical value 1/(2c) (the k^{sigma+}-gons)."""
    l, bstar = KGON_PARAMS[k]
    W = np.zeros((2, c))
    b = np.full(c, vestigial_bias)
    for j in range(k):
        ang = rotation + 2 * np.pi * j / k
        W[:, j] = l * np.array([np.cos(ang), np.sin(ang)])
        b[j] = bstar
    for j in range(k, k + n_pos):
        b[j] = 1.0 / (2 * c)
    return W, b


def pack(W, b):
    return torch.cat([W.reshape(W.shape[0], -1), b], dim=1)


def unpack(w, r=2, c=6):
    return w[:, : r * c].reshape(-1, r, c), w[:, r * c:]


def loss_flat(w, r=2, c=6):
    W, b = unpack(w, r, c)
    return population_loss(W, b)


def count_vertices(W, tol=0.1):
    """Number of vertices of the convex hull of the columns of W (r=2), counting
    only columns of non-negligible length -- the paper's k classifier (it also
    counts positive biases for sigma)."""
    from scipy.spatial import ConvexHull
    pts = W.T
    keep = np.linalg.norm(pts, axis=1) > tol
    pts = pts[keep]
    if len(pts) < 3:
        return len(pts)
    try:
        return len(ConvexHull(pts).vertices)
    except Exception:
        return len(pts)


def sgld_llc(w_star, n=5000, gamma=0.1, eps=5e-5, steps=10_000, chains=10, burn=0,
             seed=0, r=2, c=6, escape_frac=0.05, noise_scale=1.0):
    """Localized SGLD estimate of the local learning coefficient at w_star
    (devinterp-style, on the exact population loss):
        lambda_hat = n*beta*(E[L(w)] - L(w*)),  beta = 1/log n,
    posterior  exp(-n*beta*L(w) - gamma/2 * ||w - w*||^2).

    Following the paper's Appendix K protocol, a chain is discarded if more
    than `escape_frac` of its samples have loss below L(w*) (it has fallen
    into a lower-loss phase and is no longer measuring this critical point).
    Returns (lambda_hat, std over kept chains, n_kept, per-chain estimates).
    """
    g = torch.Generator().manual_seed(seed)
    w0 = w_star.clone().detach().reshape(1, -1).to(DT)
    L0 = loss_flat(w0, r, c).item()
    nb = n / np.log(n)
    w = w0.repeat(chains, 1).clone()
    acc = torch.zeros(chains, dtype=DT)
    below = torch.zeros(chains, dtype=DT)
    cnt = 0
    for t in range(steps):
        w.requires_grad_(True)
        L = loss_flat(w, r, c)
        grad, = torch.autograd.grad(L.sum(), w)
        with torch.no_grad():
            drift = nb * grad + gamma * (w - w0)
            w = w - 0.5 * eps * drift + np.sqrt(eps) * noise_scale * torch.randn(w.shape, generator=g, dtype=DT)
            if t >= burn:
                acc += L.detach()
                below += (L.detach() < L0 - 1e-9).to(DT)
                cnt += 1
    per_chain = nb * (acc / cnt - L0)
    kept = (below / cnt) <= escape_frac
    est = per_chain[kept]
    if kept.sum() == 0:
        return float("nan"), float("nan"), 0, per_chain.numpy()
    return float(est.mean()), float(est.std()), int(kept.sum()), per_chain.numpy()


def make_dataset(S, n, c, gen):
    """S independent datasets of n samples x = mu * e_j (high-sparsity TMS)."""
    j = torch.randint(0, c, (S, n), generator=gen)
    mu = torch.rand(S, n, generator=gen, dtype=DT)
    X = torch.zeros(S, n, c, dtype=DT)
    X.scatter_(2, j[..., None], mu[..., None])
    return X


def batch_loss(w, X, r=2, c=6):
    """Mean empirical loss of S parameter vectors w (S,18) on S batches X (S,B,c)."""
    W, b = unpack(w, r, c)
    g = W.transpose(-1, -2) @ W                              # (S, c, c)
    out = torch.relu(X @ g + b[:, None, :])                  # g symmetric
    return ((X - out) ** 2).sum(-1).mean(-1)                 # (S,)


def sgd_trajectories(w_init, n=1000, batch=20, lr=0.005, epochs=4500, record_every=30,
                     seed=0, r=2, c=6):
    """Minibatch SGD (the paper's Sec. 5 protocol: n=1000 training samples,
    batch 20, lr 0.005, 4500 epochs) for S trajectories in parallel, each with
    its own dataset. Records the exact population loss every `record_every`
    epochs. Returns (recorded params (T,S,d), epochs (T,), pop. loss (T,S))."""
    gen = torch.Generator().manual_seed(seed)
    S = w_init.shape[0]
    X = make_dataset(S, n, c, gen)
    w = w_init.clone().to(DT)
    steps_per_epoch = n // batch
    rec_w, rec_L, rec_ep = [], [], []
    ar = torch.arange(S)[:, None]
    for ep in range(epochs + 1):
        if ep % record_every == 0:
            rec_w.append(w.clone())
            rec_L.append(loss_flat(w, r, c).detach())
            rec_ep.append(ep)
        if ep == epochs:
            break
        perm = torch.argsort(torch.rand(S, n, generator=gen), dim=1)
        for s in range(steps_per_epoch):
            idx = perm[:, s * batch:(s + 1) * batch]
            Xb = X[ar, idx]
            w = w.detach().requires_grad_(True)
            L = batch_loss(w, Xb, r, c)
            grad, = torch.autograd.grad(L.sum(), w)
            w = w.detach() - lr * grad
    return torch.stack(rec_w), np.array(rec_ep), torch.stack(rec_L)


def sgld_llc_points(w_points, n=1000, gamma=1.0, eps=1e-3, steps=500, chains=2, seed=0, r=2, c=6):
    """One short localized-SGLD LLC estimate per row of w_points (M, d), all in
    parallel -- the paper's trajectory protocol (full-batch, eps=1e-3, gamma=1,
    500 steps), here on the exact population loss. Returns (M,) estimates
    (can be negative for high-loss points where the chain escapes to a lower
    phase, as the paper also notes)."""
    gen = torch.Generator().manual_seed(seed)
    M, d = w_points.shape
    w0 = w_points.detach().to(DT).repeat_interleave(chains, 0)
    L0 = loss_flat(w0, r, c).detach()
    nb = n / np.log(n)
    w = w0.clone()
    acc = torch.zeros(M * chains, dtype=DT)
    for _ in range(steps):
        w.requires_grad_(True)
        L = loss_flat(w, r, c)
        grad, = torch.autograd.grad(L.sum(), w)
        with torch.no_grad():
            w = w - 0.5 * eps * (nb * grad + gamma * (w - w0)) \
                + np.sqrt(eps) * torch.randn(w.shape, generator=gen, dtype=DT)
            acc += L.detach()
    est = nb * (acc / steps - L0)
    return est.reshape(M, chains).mean(1).numpy()


def critical_sample_size(dL, dlam, dc, n_lo=3.0, n_hi=1e6):
    """Solve n*dL + dlam*log(n) + dc = 0 (free-energy crossing, dL < 0) for n."""
    from scipy.optimize import brentq
    f = lambda n: n * dL + dlam * np.log(n) + dc
    grid = np.logspace(np.log10(n_lo), np.log10(n_hi), 4000)
    vals = np.array([f(x) for x in grid])
    sign = np.where(np.diff(np.sign(vals)) != 0)[0]
    return float(brentq(f, grid[sign[-1]], grid[sign[-1] + 1])) if len(sign) else float("nan")
