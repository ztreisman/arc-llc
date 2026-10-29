"""How well does the min-over-restarts Hessian estimator recover lambda on RRR in practice?

Background (arc/README.md, section 2): on reduced-rank regression lambda is half the
codimension of the largest component of the true set Z. At a generic point of that
component rank(Hess K) = 2*lambda. At the minimum-norm point, where the components meet,
rank = r0(N+M-r0) < 2*lambda. step0_tangent_excess.py polished every run exactly onto Z.
Here we do what a practitioner would do:

  * full-batch GD on K + ridge*||w||^2 for a fixed budget (no polishing onto Z);
  * the exact Hessian of K (not of the regularized objective) at the final point,
    H = 2 J^T J + 2 S(R), with S the residual term <R, dW2 dW1>;
  * rank by three rules: a fixed relative tolerance (1e-6, the repo default); the
    largest gap in the log spectrum; and a residual-aware tolerance that ignores
    eigenvalues below 4*||R||_F, the size of the residual-curvature term (which is
    nonzero whenever the run has not reached the true set, e.g. under ridge);
  * lambda_hat(R) = min rank / 2 over R restarts, bootstrapped from 50 runs.

Grid: the 14 anchor cells (M=10, N=5), ridge in {0, 1e-4, 1e-3, 1e-2}, init scale in
{0.05, 0.5, 1.5}, R in {1, 2, 5, 10, 20, 50}.
Outputs: arc/results/hessian_restarts.json and a printed summary.
"""
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rrr_model import aoyagi_2005_anchor_cells, make_teacher  # noqa: E402

M_IN, N_OUT = 10, 5
RIDGES = [0.0, 1e-4, 1e-3, 1e-2]
SCALES = [0.05, 0.5, 1.5]
RS = [1, 2, 5, 10, 20, 50]
N_RUNS = 50
STEPS = 20_000
TOL = 1e-6
N_BOOT = 2000


def train(H, r0, ridge, scale, rng):
    Mstar = make_teacher(N_OUT, M_IN, r0)
    W1 = rng.standard_normal((N_RUNS, H, M_IN)) * scale
    W2 = rng.standard_normal((N_RUNS, N_OUT, H)) * scale
    for _ in range(STEPS):
        R = W2 @ W1 - Mstar
        L = np.linalg.norm(W1, 2, axis=(1, 2)) ** 2 + np.linalg.norm(W2, 2, axis=(1, 2)) ** 2
        lr = (0.1 / (np.maximum(L, 1e-3) + ridge))[:, None, None]
        g1 = 2 * np.transpose(W2, (0, 2, 1)) @ R + 2 * ridge * W1
        g2 = 2 * R @ np.transpose(W1, (0, 2, 1)) + 2 * ridge * W2
        W1, W2 = W1 - lr * g1, W2 - lr * g2
    K = ((W2 @ W1 - Mstar) ** 2).sum(axis=(1, 2))
    return W1, W2, K, Mstar


def hessian_K(W1, W2, Mstar):
    """Exact Hessian of K = ||W2 W1 - M*||^2 in w = [vec_r(W1), vec_r(W2)]."""
    H, M = W1.shape
    N = W2.shape[0]
    J = np.hstack([np.kron(W2, np.eye(M)), np.kron(np.eye(N), W1.T)])
    Hs = 2 * J.T @ J
    R = W2 @ W1 - Mstar
    n1 = H * M
    # 2 <R, dW2 dW1>: d/dW2[a,h] d/dW1[h,b] -> 2 R[a,b]
    for a in range(N):
        for h in range(H):
            i2 = n1 + a * H + h
            for b in range(M):
                i1 = h * M + b
                Hs[i1, i2] += 2 * R[a, b]
                Hs[i2, i1] += 2 * R[a, b]
    return Hs


def ranks(Hs, res_norm=0.0):
    ev = np.sort(np.abs(np.linalg.eigvalsh(Hs)))[::-1]
    top = ev[0]
    r_tol = int((ev > TOL * top).sum())
    r_res = int((ev > max(TOL * top, 4 * res_norm)).sum())
    pos = ev[ev > 1e-14 * top]
    if len(pos) < len(ev):
        logs = np.log(np.concatenate([pos, [1e-14 * top]]))
    else:
        logs = np.log(pos)
    gaps = logs[:-1] - logs[1:]
    r_gap = int(np.argmax(gaps) + 1) if len(gaps) else len(pos)
    return r_tol, r_gap, r_res, ev


def boot_min(ranks_arr, R, rng):
    idx = rng.integers(0, len(ranks_arr), size=(N_BOOT, R))
    return ranks_arr[idx].min(axis=1)


def main():
    rng = np.random.default_rng(0)
    cells = aoyagi_2005_anchor_cells(M=M_IN, N=N_OUT)
    out = []
    t0 = time.time()
    for c in cells:
        H, r0 = c["H"], c["r0"]
        two_lam = int(round(2 * c["lambda_true"]))
        rank_minnorm = r0 * (N_OUT + M_IN - r0)
        for ridge in RIDGES:
            for scale in SCALES:
                W1s, W2s, Ks, Mstar = train(H, r0, ridge, scale, rng)
                rt, rg, rr, small = [], [], [], []
                for W1, W2 in zip(W1s, W2s):
                    res = np.linalg.norm(W2 @ W1 - Mstar)
                    a, b, cc, ev = ranks(hessian_K(W1, W2, Mstar), res)
                    rt.append(a); rg.append(b); rr.append(cc)
                    small.append(float(ev[min(two_lam, len(ev)) - 1] / ev[0]))
                rt, rg, rr = np.array(rt), np.array(rg), np.array(rr)
                rec = {"H": H, "r0": r0, "two_lambda": two_lam, "rank_minnorm": rank_minnorm,
                       "ridge": ridge, "scale": scale,
                       "K_median": float(np.median(Ks)),
                       "rank_tol": rt.tolist(), "rank_gap": rg.tolist(), "rank_res": rr.tolist(),
                       "ev_at_2lambda_rel_median": float(np.median(small))}
                for rule, arr in [("tol", rt), ("gap", rg), ("res", rr)]:
                    for R in RS:
                        mins = boot_min(arr, R, rng)
                        rec[f"{rule}_exact_R{R}"] = float(np.mean(mins == two_lam))
                        rec[f"{rule}_relerr_R{R}"] = float(np.mean(np.abs(mins - two_lam)) / two_lam)
                        rec[f"{rule}_mean_R{R}"] = float(np.mean(mins))
                out.append(rec)
        print(f"cell H={H} r0={r0} done ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", "hessian_restarts.json")
    with open(path, "w") as f:
        json.dump(out, f)

    print("\nFraction of the 14 cells (averaged over bootstrap) where min rank over R restarts")
    print("equals 2*lambda exactly. Rule: tol / gap / res.")
    for ridge in RIDGES:
        for scale in SCALES:
            recs = [r for r in out if r["ridge"] == ridge and r["scale"] == scale]
            s = " ".join(f"R={R}: " + "/".join(f"{np.mean([r[f'{u}_exact_R{R}'] for r in recs]):.2f}"
                                                for u in ("tol", "gap", "res")) for R in RS)
            kmed = np.median([r["K_median"] for r in recs])
            print(f"  ridge={ridge:<6g} init={scale:<4g} K~{kmed:.0e} | {s}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
