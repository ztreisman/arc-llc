"""Plotting utilities: log-log volume plots, SGLD free-energy plots,
arc-direction histograms, and the training-trajectory ratio plot.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_volume_scaling(exp, out_path):
    vol = exp["vol"]
    log_eps = vol["log_eps"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(log_eps, vol["log_vol"], "o", ms=4, label="empirical")

    xs = np.linspace(log_eps.min(), log_eps.max(), 200)
    # lambda*log(eps) + k*log(-log(eps)) + const, matching the fit used to
    # produce lambda_estimate (accounts for the log-multiplicity correction).
    const = vol["log_vol"].mean() - vol["lambda_estimate"] * log_eps.mean() \
        - vol["log_mult_k"] * np.log(-log_eps).mean()
    fit_curve = vol["lambda_estimate"] * xs + vol["log_mult_k"] * np.log(-xs) + const
    ax.plot(xs, fit_curve, "-",
            label=f"fit: lambda={vol['lambda_estimate']:.3f}, k={vol['log_mult_k']:.2f}")

    naive_coeffs = np.polyfit(log_eps, vol["log_vol"], 1)
    ax.plot(xs, np.polyval(naive_coeffs, xs), ":", color="C2",
            label=f"naive power-law fit={naive_coeffs[0]:.3f}")

    true_line_b = np.polyval(naive_coeffs, xs)[0] - exp["lambda_true"] * xs[0]
    ax.plot(xs, exp["lambda_true"] * xs + true_line_b,
            "--", color="gray", label=f"true slope={exp['lambda_true']:.3f}")
    ax.set_xlabel("log eps")
    ax.set_ylabel("log Vol{K <= eps}")
    ax.set_title(f"Volume scaling: {exp['tag']}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sgld_free_energy(exp, out_path):
    sgld = exp["sgld"]
    fig, ax = plt.subplots(figsize=(5, 4))
    log_n = sgld["log_n"]
    F = np.array(sgld["free_energies"])
    ax.plot(log_n, F, "o", ms=5, label="empirical F(n)")
    coeffs = np.polyfit(log_n, F, 1)
    xs = np.array([log_n.min(), log_n.max()])
    ax.plot(xs, np.polyval(coeffs, xs), "-", label=f"fit slope={coeffs[0]:.3f}")
    intercept_true = np.polyval(coeffs, xs)[0] - exp["lambda_true"] * xs[0]
    ax.plot(xs, exp["lambda_true"] * xs + intercept_true, "--", color="gray",
            label=f"true slope={exp['lambda_true']:.3f}")
    ax.set_xlabel("log n")
    ax.set_ylabel("F(n) = n * E[K(w)]")
    ax.set_title(f"SGLD free energy: {exp['tag']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_arc_direction(exp5, out_path):
    arc = exp5["arc"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    axes[0].hist(np.log10(arc["K_on_sphere"] + 1e-300), bins=60, color="C0")
    axes[0].set_xlabel("log10 K(delta), delta on unit sphere")
    axes[0].set_ylabel("count")
    axes[0].set_title("Distribution of K on unit sphere")

    axes[1].loglog(arc["thresholds"], arc["fractions"], "o-", ms=3)
    axes[1].set_xlabel("threshold eps")
    axes[1].set_ylabel("fraction K(delta) < eps")
    axes[1].set_title(f"Arc-direction scaling (fit exponent={exp5['exponent']:.3f})")

    fig.suptitle(exp5["tag"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_trajectory(exp4, out_path):
    traj = exp4["trajectory"]
    steps = [t["step"] for t in traj]
    ratios = [t["ratio"] for t in traj]
    dists = [t["dist_to_origin"] for t in traj]
    ks = [t.get("log_mult_k", 0.0) for t in traj]

    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    ax1 = axes[0]
    ax1.loglog(steps, ratios, "o-", color="C0", label="lambda/(d/2) (local volume scaling)")
    ax1.axhline(exp4["ratio_true"], color="gray", ls="--", label=f"branch/true ratio={exp4['ratio_true']:.3f}")
    ax1.axhline(1.0, color="C3", ls=":", label="naive 'regular' ratio=1")
    ax1.set_ylabel("lambda / (d/2)  [log scale]")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_title(exp4["tag"] +
                   "\n(large early ratios: local ball not yet near a zero of K)", fontsize=9)

    ax2 = axes[1]
    ax2.semilogx(steps, dists, "-", color="C2", label="||w_t|| (dist to origin)")
    ax2.set_ylabel("||w_t||", color="C2")
    ax2.set_xlabel("training step")
    ax2b = ax2.twinx()
    ax2b.semilogx(steps, ks, "-", color="C4", label="fitted log-multiplicity k")
    ax2b.set_ylabel("log-multiplicity k", color="C4")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dds_validation(exp7, out_path):
    a = exp7["analytic"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(a["log_t"], a["log_lam_h1"], "o-", ms=3, label=f"log lambda+_min(G_h1), slope={a['slope_lam_h1']:.2f}")
    ax.plot(a["log_t"], a["log_sigma_h1_sq"], "s-", ms=3, label=f"log sigma_min(X_h1)^2, slope={a['slope_sigma_h1']:.2f}")
    ax.plot(a["log_t"], a["log_lam_h2"], "^-", ms=3, label=f"log lambda+_min(G_h2), slope={a['slope_lam_h2']:.2f}")
    ax.set_xlabel("log t (distance to {B=0} branch)")
    ax.set_ylabel("log observable")
    ax.set_title(f"Analytic-limit rate check\nstructural correlation rho={a['rho_structural']:.4f}")
    ax.legend(fontsize=8)

    traj = exp7["trajectory_dds"]
    steps = [c["step"] for c in traj]
    lam_h1 = [c["lambda_plus_min_h1"] for c in traj]
    lam_h2 = [c["lambda_plus_min_h2"] for c in traj]
    sigma_h1 = [c["sigma_min_h1"] for c in traj]

    ax2 = axes[1]
    ax2.loglog(steps, lam_h1, "o-", ms=3, label="lambda+_min(G_h1)")
    ax2.loglog(steps, lam_h2, "^-", ms=3, label="lambda+_min(G_h2)")
    ax2.loglog(steps, np.array(sigma_h1) ** 2, "s-", ms=3, label="sigma_min(X_h1)^2")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("observable value (log scale)")
    ax2.set_title(f"Real GD trajectory (both layers collapse together)\n"
                   f"rho(h1,sigma_h1)={exp7['rho_structural_trajectory']:.3f}, "
                   f"rho(h1,h2)={exp7['rho_h1_h2_trajectory']:.3f}")
    ax2.legend(fontsize=8)

    fig.suptitle(exp7["tag"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_dds_cross_cell(exp8, out_path):
    cells = exp8["cells"]
    lam_true = exp8["lambda_true"]
    Hs = np.array([c["H"] for c in cells])
    within = exp8["within_H_concordance"]
    panels = [("llc", "SGLD LLC  n*beta*E[K]", exp8["llc_hat"], exp8["rho_llc"], None)]
    for name in exp8["observables"]:
        panels.append((name, "log " + name if not name.startswith("log") else name,
                       exp8["per_cell_mean"][name], exp8["cross_cell_rho"][name],
                       exp8["cross_cell_rho_ci"][name]))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (key, label, vals, rho, ci) in zip(axes, panels):
        for h in [2, 3, 4, 5]:
            sel = Hs == h
            order = np.argsort(lam_true[sel])
            ax.plot(lam_true[sel][order], np.asarray(vals)[sel][order], "o-", color=f"C{h-2}",
                    ms=5, lw=1)
        c, t = within[key]
        if key == "llc":
            lim = [lam_true.min() - 1, lam_true.max() + 3]
            ax.plot(lim, lim, "k--", lw=0.8)
            title = f"{label}\nrho={rho:.3f}, within-H {c}/{t}"
        else:
            title = f"{label}\nrho={rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}], within-H {c}/{t}"
        ax.set_xlabel("closed-form lambda")
        ax.set_title(title, fontsize=9)

    handles = [plt.Line2D([0], [0], marker="o", color=f"C{h-2}", label=f"H={h}", markersize=6)
               for h in [2, 3, 4, 5]]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle(exp8["tag"] + "  (lines join cells of equal width H, r0 increasing)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_deep_linear_counting(exp9, out_path):
    rs = exp9["rs"]
    ratio_summary = exp9["ratio_summary"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xs = [1] + [r for r in rs if r != 1]
    ys_logdet = [1.0] + [ratio_summary[r]["log_det_plus_ratio_mean"] for r in rs if r != 1]
    ys_lam = [1.0] + [ratio_summary[r]["lambda_plus_min_ratio_mean"] for r in rs if r != 1]

    ax.plot(rs, rs, "--", color="gray", label="predicted y=r (log_det_plus)")
    ax.plot(xs, ys_logdet, "o-", ms=8, color="C0", label="log_det_plus(G) slope ratio (counts r)")
    ax.plot(xs, ys_lam, "s-", ms=8, color="C3", label="lambda_plus_min(G) slope ratio (rank-blind)")
    ax.axhline(1.0, color="C3", ls=":", alpha=0.5)

    ax.set_xlabel("rank-deficit r (number of simultaneously-dead directions)")
    ax.set_ylabel("slope ratio (vs r=1)")
    ax.set_title(exp9["tag"] +
                  "\n(closed-form, symmetric construction: log_det_plus=r is a\n"
                  "mathematical necessity here, not an independent empirical test --\n"
                  "see RESULTS.md)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tms(exp10, out_path):
    sgd = exp10["sgd"]
    L, lam, ep = sgd["loss"], sgd["lam_hat"], sgd["epochs"]
    levels = sgd["ref_levels"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    # (1) LLC estimate vs n at the 5, 5+, 6-gons
    ax = axes[0]
    for i, (name, row) in enumerate(exp10["llc"].items()):
        ns = sorted(row["est"])
        m = [row["est"][n]["mean"] for n in ns]
        sd = [row["est"][n]["std"] for n in ns]
        ax.errorbar(np.array(ns) * (1 + 0.06 * (i - 1)), m, yerr=sd, fmt="o-", color=f"C{i}",
                    capsize=3, label=f"{name}-gon (theory {row['theory']})")
        ax.axhline(row["theory"], color=f"C{i}", ls="--", lw=0.8)
        ax.scatter([5000 * (1 + 0.06 * (i - 1))], [row["paper_hat_n5000"]], marker="x",
                   color=f"C{i}", s=60, zorder=5)
    ax.set_xscale("log")
    ax.set_xlabel("n (sets SGLD temperature n*beta, beta=1/log n)")
    ax.set_ylabel("LLC estimate")
    ax.set_title("SGLD LLC at k-gon critical points\n(dashed: theory, x: Chen et al. Table K.1)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # (2) one trajectory: opposing staircases
    ax = axes[1]
    n_plateaus = [len({round(v, 3) for v in L[:, s] if min(abs(v - lv) for lv in levels.values()) < 3e-4})
                  for s in range(L.shape[1])]
    s = int(np.lexsort((L[-1], -np.array(n_plateaus)))[0])
    smooth = np.full(len(ep), np.nan)
    smooth[5:] = np.convolve(lam[:, s], np.ones(6) / 6, mode="valid")
    ax.plot(ep, L[:, s], color="C0")
    for nm, lv in levels.items():
        ax.axhline(lv, color="gray", lw=0.5, ls=":")
        ax.text(ep[-1], lv, f" {nm}", fontsize=7, va="center", color="gray")
    ax.set_xlabel("epoch")
    ax.set_ylabel("population loss", color="C0")
    ax2 = ax.twinx()
    ax2.plot(ep, smooth, color="C3")
    ax2.set_ylabel("LLC estimate (smoothed, window 6)", color="C3")
    ax.set_title(f"SGD from a 4-gon (trajectory {s}):\nloss falls, LLC rises", fontsize=10)

    # (3) all checkpoints: loss vs LLC
    ax = axes[2]
    Lf, lf = L.ravel(), lam.ravel()
    other = np.ones_like(Lf, dtype=bool)
    for i, (nm, lv) in enumerate(levels.items()):
        sel = np.abs(Lf - lv) < 3e-4
        other &= ~sel
        if sel.any():
            ax.scatter(lf[sel], Lf[sel], s=6, color=f"C{i}", label=f"{nm}  (n={sel.sum()})")
    ax.scatter(lf[other], Lf[other], s=4, color="lightgray", label="in transit")
    ax.set_xlabel("LLC estimate (eps=1e-3, gamma=1, 500 steps)")
    ax.set_ylabel("population loss")
    ax.set_title(f"All {L.shape[1]} trajectories x {L.shape[0]} checkpoints", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)

    fig.suptitle(exp10["tag"], fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
