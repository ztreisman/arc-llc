"""P1 of PREREGISTRATION.md: localized-SGLD LLC at the polished param-decomp 5->2
target, under the population loss of its own data distribution."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
import tms
torch.set_num_threads(2)

d = np.load("param_decomp_compare/target_5-2_polished.npz")
W, b, ev = d["W"], d["b"], d["hess_eig"]
c = W.shape[1]
w_star = tms.pack(torch.tensor(W[None]), torch.tensor(b[None]))
loss_fn = lambda w: tms.population_loss_bernoulli(*tms.unpack(w, 2, c))
h_max, h_min = ev.max(), ev[ev > 1e-6 * ev.max()].min()

out = {}
for n in (5_000, 50_000, 500_000):
    nb = n / np.log(n)
    eps = 0.02 / (nb * h_max)                    # eps * nb * h_max / 2 = 0.01 (stiff mode)
    relax = 2 / (eps * nb * h_min)               # slowest non-zero mode, in steps
    steps = int(25 * relax)
    t = time.time()
    m, sd, kept, per = tms.sgld_llc(w_star, n=n, gamma=0.1, eps=eps, steps=steps, chains=10,
                                    burn=steps // 5, seed=0, c=c, loss_fn=loss_fn)
    out[n] = {"mean": m, "std": sd, "kept": kept, "eps": eps, "steps": steps}
    print(f"n={n:>7}: lambda_hat = {m:.3f} +/- {sd:.3f} (kept {kept}/10, eps={eps:.2e}, "
          f"{steps} steps, {time.time()-t:.0f}s)", flush=True)
json.dump({str(k): v for k, v in out.items()}, open("param_decomp_compare/p1_llc.json", "w"), indent=2)
