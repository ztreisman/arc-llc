"""Rule A / DL_raw analysis of one extracted run (PREREGISTRATION.md)."""
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

def identity_ci_error(ci, tol):            # re-implementation of param-decomp's metric
    n, C = ci.shape; size = min(n, C)
    _, cols = linear_sum_assignment(-ci[:size])
    perm = list(cols) + [c for c in range(C) if c not in set(cols)]
    ci = ci[:, perm]; off = np.ones(ci.shape, bool); off[np.arange(size), np.arange(size)] = False
    return int((ci[off] > tol).sum() + (np.diagonal(ci[:size, :size]) < 1 - tol).sum()), ci

def analyze(path, alive_thr=0.5, verbose=True):
    d = np.load(path)
    out = {}
    alive_feat = {}
    for site in ("linear1", "linear2"):
        ci = d[f"ci_{site}"]
        err, ci_p = identity_ci_error(ci, 0.1)
        alive_cols = np.where((ci > alive_thr).any(0))[0]
        alive_feat[site] = (ci > alive_thr).any(1)          # feature has >=1 alive component
        out[site] = {"identity_ci_error_tol0.1": err, "n_alive": len(alive_cols),
                     "alive_per_feature": (ci > alive_thr).sum(1).tolist()}
        if verbose:
            keep = np.where((ci_p > 0.05).any(0))[0]
            print(f"--- {site}: identity_ci_error(tol 0.1) = {err}; alive (CI>{alive_thr}) "
                  f"components = {len(alive_cols)}; alive per feature = "
                  f"{out[site]['alive_per_feature']}")
            print("    permuted CI, columns with any entry > 0.05:")
            print(np.array2string(ci_p[:, keep], precision=3, suppress_small=True,
                                  max_line_width=150))
    k = int((alive_feat["linear1"] & alive_feat["linear2"]).sum())
    out["k"] = k; out["DL_A"] = 3 * k - 1
    out["DL_raw"] = 6 * (out["linear1"]["n_alive"] + out["linear2"]["n_alive"]) + 5
    return out

if __name__ == "__main__":
    r = analyze(sys.argv[1])
    print(f"k = {r['k']}  DL_A = 3k-1 = {r['DL_A']}  (2*lambda = 14)   DL_raw = {r['DL_raw']}")
