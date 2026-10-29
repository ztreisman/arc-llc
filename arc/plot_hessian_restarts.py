"""Figure for hessian_restarts.py: mean lambda_hat / lambda of the min-over-restarts
Hessian estimator vs number of restarts R. Columns: ridge. Rows: rank rule (the repo's
fixed relative tolerance; the residual-aware tolerance). Lines: init scale."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SERIES = {0.05: ("#2a78d6", "o"), 0.5: ("#eb6834", "s"), 1.5: ("#1baf7a", "^")}
INK, MUTED, GRID, SURFACE = "#2b2b29", "#6b6b66", "#e4e3de", "#fcfcfb"
RULES = [("tol", "fixed tolerance (1e-6)"), ("res", "residual-aware tolerance")]


def main():
    with open(os.path.join(HERE, "results", "hessian_restarts.json")) as f:
        recs = json.load(f)
    ridges = sorted({r["ridge"] for r in recs})
    Rs = sorted(int(k.split("_R")[1]) for k in recs[0] if k.startswith("tol_exact_R"))

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(len(RULES), len(ridges), figsize=(3.1 * len(ridges), 5.8),
                             sharex=True, sharey=True, facecolor=SURFACE)
    for i, (rule, rule_name) in enumerate(RULES):
        for j, ridge in enumerate(ridges):
            ax = axes[i, j]
            for scale, (col, mk) in SERIES.items():
                sub = [r for r in recs if r["ridge"] == ridge and r["scale"] == scale]
                ys = [np.mean([r[f"{rule}_mean_R{R}"] / r["two_lambda"] for r in sub]) for R in Rs]
                ax.plot(Rs, ys, color=col, lw=2, marker=mk, ms=6, markeredgecolor=SURFACE,
                        markeredgewidth=1.5, label=f"init scale {scale:g}")
            ax.axhline(1.0, color=MUTED, lw=1, ls="--")
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            ax.set_xticks(Rs)
            ax.set_xticklabels([str(R) for R in Rs])
            ax.grid(True, color=GRID, lw=0.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if i == 0:
                ax.set_title(f"ridge = {ridge:g}", color=INK)
            if i == len(RULES) - 1:
                ax.set_xlabel("restarts R")
            if j == 0:
                ax.set_ylabel(f"{rule_name}\nmean λ̂ / λ")
    axes[0, 0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle("Min-over-restarts Hessian estimate of λ on the 14 RRR anchor cells "
                 "(1 = exact; dashed)", color=INK)
    fig.tight_layout()
    out = os.path.join(HERE, "results", "hessian_restarts.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
