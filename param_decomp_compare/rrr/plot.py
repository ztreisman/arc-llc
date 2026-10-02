"""Figure: SPD's function-level description length vs 2*lambda and Hessian rank, by H."""
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M, N = 10, 5
rows = [np.load(f) for f in glob.glob(os.path.expanduser("~/rrr-spd/results/*.npz"))]
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for r0 in range(1, 6):
    Hs = np.arange(max(r0, 2), 6)
    c = f"C{r0 - 1}"
    ax.plot(Hs, N * Hs + (M - Hs) * r0, "-", color=c, lw=2, label=f"r0={r0}: 2λ" if r0 == 1 else f"r0={r0}")
    ax.plot(Hs, np.full(len(Hs), r0 * (M + N - r0)), ":", color=c, lw=1.5)
    for cf, mk in ((1.0, "o"), (1 / 3, "x")):
        pts = [(int(d["H"]), len(d["alive2"]) * (M + N - len(d["alive2"]))) for d in rows
               if int(d["r0"]) == r0 and abs(float(d["coeff_factor"]) - cf) < 1e-6]
        if pts:
            h, v = zip(*sorted(pts))
            ax.scatter(np.array(h) + (0.06 if mk == "x" else -0.06), v, marker=mk, color=c, s=40, zorder=5)
ax.plot([], [], "k-", lw=2, label="2λ (Aoyagi-Watanabe)")
ax.plot([], [], "k:", lw=1.5, label="Hessian rank r0(M+N−r0)")
ax.scatter([], [], marker="o", color="k", label="SPD, default minimality")
ax.scatter([], [], marker="x", color="k", label="SPD, ×1/3 minimality")
ax.set_xlabel("hidden width H (excess capacity H − r0)")
ax.set_ylabel("parameter count")
ax.set_xticks(range(2, 6))
ax.set_title("Reduced-rank regression (M=10, N=5): SPD's description length\n"
             "follows the Hessian rank, not 2λ, as capacity grows", fontsize=10)
handles, labels = ax.get_legend_handles_labels()
keep = [i for i, l in enumerate(labels) if not l.startswith("r0=") or l.endswith("2λ")]
ax.legend([handles[i] for i in keep[1:]], [labels[i] for i in keep[1:]], fontsize=8, loc="upper left")
for r0 in range(1, 6):
    ax.text(5.2, r0 * (M + N - r0), f"r0={r0}", color=f"C{r0-1}", va="center", fontsize=8)
ax.set_xlim(1.8, 5.6)
fig.tight_layout()
fig.savefig("spd_vs_llc.png", dpi=140)
