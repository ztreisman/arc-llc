"""Matching check: is param-decomp's trained 5->2 TMS target a critical point of
the population loss for its own data distribution, and what is the Hessian
spectrum there?"""
import numpy as np, torch
import sys; sys.path.insert(0, ".")
import tms

d = np.load("param_decomp_compare/target_5-2.npz")
W0, b0 = d["W"].astype(np.float64), d["b"].astype(np.float64)
c = W0.shape[1]

def L(w):
    return tms.population_loss_bernoulli(w[:10].reshape(1, 2, c), w[10:].reshape(1, c))[0]

def grad(w):
    w = torch.tensor(w, dtype=tms.DT, requires_grad=True)
    g, = torch.autograd.grad(L(w), w)
    return g.numpy()

w0 = np.r_[W0.ravel(), b0]
print(f"target: L = {L(torch.tensor(w0)).item():.6f}, |grad| = {np.linalg.norm(grad(w0)):.2e}")

# nearest critical point (polish with BFGS from the target)
from scipy.optimize import minimize
r = minimize(lambda w: L(torch.tensor(w)).item(), w0, jac=grad, method="BFGS",
             options=dict(gtol=1e-11, maxiter=5000))
ws = r.x
Ws, bs = ws[:10].reshape(2, c), ws[10:]
print(f"polished: L = {r.fun:.6f}, |grad| = {np.linalg.norm(grad(ws)):.2e}, "
      f"moved |dw| = {np.linalg.norm(ws - w0):.3f}")
norms = np.linalg.norm(Ws, axis=0)
ang = np.sort(np.degrees(np.arctan2(Ws[1], Ws[0])))
print("  norms", norms.round(4), " biases", bs.round(4))
print("  angle gaps", np.diff(np.r_[ang, ang[0] + 360]).round(3))

H = torch.autograd.functional.hessian(L, torch.tensor(ws, dtype=tms.DT)).numpy()
ev = np.sort(np.linalg.eigvalsh(H))
print("Hessian eigenvalues:", np.array2string(ev, precision=4))
rank = int((ev > 1e-6 * ev.max()).sum())
print(f"Hessian rank {rank} of {len(ev)} -> Morse-Bott count lambda = rank/2 = {rank/2}")
np.savez("param_decomp_compare/target_5-2_polished.npz", W=Ws, b=bs, hess_eig=ev)
