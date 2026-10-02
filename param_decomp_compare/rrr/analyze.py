"""Score the RRR SPD runs against PREREGISTRATION.md (P1-P3)."""
import glob
import os

import numpy as np

M, N = 10, 5
RES = os.path.expanduser("~/rrr-spd/results")


def two_lam(H, r0):
    return N * H + (M - H) * r0


def hess_rank(r0):
    return r0 * (M + N - r0)


rows = []
for f in sorted(glob.glob(f"{RES}/*.npz")):
    d = np.load(f)
    H, r0, cf = int(d["H"]), int(d["r0"]), float(d["coeff_factor"])
    a1, a2 = len(d["alive1"]), len(d["alive2"])
    conv = max(float(d["rel_err1"]), float(d["rel_err2"])) < 0.05
    k = a2
    rows.append(dict(H=H, r0=r0, cf=cf, a1=a1, a2=a2, conv=conv,
                     rel=max(float(d["rel_err1"]), float(d["rel_err2"])),
                     recon=float(d["recon_rel"]), DL_fn=k * (M + N - k),
                     DL_raw=a1 * (M + H - 1) + a2 * (H + N - 1),
                     two_lam=two_lam(H, r0), hrank=hess_rank(r0),
                     ci1=np.sort(d["ci_frac1"])[::-1][:6], ci2=np.sort(d["ci_frac2"])[::-1][:6]))

for cf in sorted({r["cf"] for r in rows}, reverse=True):
    sub = sorted([r for r in rows if r["cf"] == cf], key=lambda r: (r["r0"], r["H"]))
    print(f"\n=== importance-minimality x{cf:.4g}  ({len(sub)} runs)")
    print(" H r0 | alive L1/L2 | conv  max rel err | recon rel | DL_fn  Hess rank  2*lam | DL_raw")
    for r in sub:
        print(f" {r['H']}  {r['r0']}  |    {r['a1']} / {r['a2']}    | {'yes' if r['conv'] else 'NO '}"
              f"  {r['rel']:.1e}   | {r['recon']:.1e}  | {r['DL_fn']:4d}  {r['hrank']:6d}  "
              f"{r['two_lam']:7d} | {r['DL_raw']:4d}")
    c = [r for r in sub if r["conv"]]
    p1 = all(r["a1"] == r["r0"] and r["a2"] == r["r0"] for r in c)
    p3a = all(r["DL_fn"] == r["hrank"] for r in c)
    p3b = all(r["DL_fn"] != r["two_lam"] for r in c if r["H"] > r["r0"])
    # within-r0: does the alive count move with H?
    moves = [r0 for r0 in {r["r0"] for r in c}
             if len({(r["a1"], r["a2"]) for r in c if r["r0"] == r0}) > 1]
    print(f" converged {len(c)}/{len(sub)}.  P1 (alive = r0 at both sites): {p1}.  "
          f"P2 (count independent of H at fixed r0): {not moves}"
          f"{'' if not moves else f' -- moves at r0 in {sorted(moves)}'}.")
    print(f" P3: DL_fn == Hessian rank in all converged: {p3a};  DL_fn != 2*lambda wherever "
          f"H > r0: {p3b}")
    print(" largest fraction-alive among the top-6 components (L1 | L2), to show the margin:")
    for r in sub:
        print(f"   H{r['H']} r{r['r0']}: {np.round(r['ci1'], 2)} | {np.round(r['ci2'], 2)}")
