"""Exploratory (not pre-registered): per-chain diagnosis of the P1 estimate at n=5e4."""
import sys
import numpy as np, torch
sys.path.insert(0, ".")
import tms
torch.set_num_threads(2)

d = np.load("param_decomp_compare/target_5-2_polished.npz")
W, b, ev = d["W"], d["b"], d["hess_eig"]
c = 5
w0 = tms.pack(torch.tensor(W[None]), torch.tensor(b[None])).to(tms.DT)
loss_fn = lambda w: tms.population_loss_bernoulli(*tms.unpack(w, 2, c))
n = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
nb = n / np.log(n); h_max, h_min = ev.max(), ev[ev > 1e-6 * ev.max()].min()
eps = 0.02 / (nb * h_max); steps = int(25 * 2 / (eps * nb * h_min))
g = torch.Generator().manual_seed(0)
L0 = loss_fn(w0).item()
w = w0.repeat(10, 1).clone(); trace = []; dist = []
for t in range(steps):
    w.requires_grad_(True)
    L = loss_fn(w); gr, = torch.autograd.grad(L.sum(), w)
    with torch.no_grad():
        w = w - 0.5 * eps * (nb * gr + 0.1 * (w - w0)) + np.sqrt(eps) * torch.randn(w.shape, generator=g, dtype=tms.DT)
    if t % 100 == 0:
        trace.append(L.detach().numpy().copy()); dist.append((w - w0).norm(dim=1).numpy().copy())
trace = np.array(trace); dist = np.array(dist); burn = len(trace) // 5
lam = nb * (trace[burn:].mean(0) - L0)
half = len(trace[burn:]) // 2
lam1 = nb * (trace[burn:burn + half].mean(0) - L0); lam2 = nb * (trace[burn + half:].mean(0) - L0)
print(f"n={n} per-chain lambda_hat:", lam.round(2))
print("  first/second half:", lam1.round(2), lam2.round(2))
print("  max |w - w*| per chain:", dist.max(0).round(3), " (pentagon column length 1.13)")
np.savez(f"param_decomp_compare/p1_diag_{n}.npz", trace=trace, dist=dist, L0=L0, nb=nb)
