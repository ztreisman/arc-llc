"""Experiments 1-6 from arc_llc_context.md (experiment 6 added: asymmetric
n != m, testing the multi-restart min-codim fix to the Hessian estimator).
Experiment 7 adds Dead-Direction Signatures (Shirodkar & Narayanan 2606.21158)
as a further estimator, validated on the same r=1 toy models."""
import numpy as np
import torch

from model import (
    true_lambda, dim_w, make_K, make_K_torch, make_K_regular,
)
from estimators import (
    volume_scaling_estimator, hessian_branch_estimator,
    hessian_multi_restart_estimator,
    sgld_llc_estimator, arc_direction_estimator,
)
from dds import dds_observables
from rrr_model import aoyagi_2005_anchor_cells, exact_branch_point, make_teacher
import tms
from deep_linear import canonical_layers, make_teacher_diag, dds_observables_deep


def _ground_truth_experiment(r, n, m, seed, sgld_n_steps=15_000, sgld_n_values=None,
                              vol_n_samples=5_000_000, tag=""):
    rng = np.random.default_rng(seed)
    d = dim_w(r, n, m)
    lam_true = true_lambda(r, n, m)
    K = make_K(r, n, m)
    K_t = make_K_torch(r, n, m)

    vol = volume_scaling_estimator(K, d, n_samples=vol_n_samples, rng=rng)
    hess_A = hessian_branch_estimator(K_t, r, n, m, d, branch="A", rng=rng)
    hess_B = hessian_branch_estimator(K_t, r, n, m, d, branch="B", rng=rng)
    sgld = sgld_llc_estimator(K_t, d, n_steps=sgld_n_steps, n_values=sgld_n_values,
                               seed=seed)

    rows = [
        ("Volume scaling", vol["lambda_estimate"], vol["ratio"]),
        (f"Hessian null space (branch='A': A free, on the {{B=0}} plane)", hess_A["lambda_estimate"], hess_A["ratio"]),
        (f"Hessian null space (branch='B': B free, on the {{A=0}} plane)", hess_B["lambda_estimate"], hess_B["ratio"]),
        ("SGLD (devinterp-style)", sgld["lambda_estimate"], sgld["ratio"]),
    ]

    return {
        "tag": tag, "r": r, "n": n, "m": m, "d": d,
        "lambda_true": lam_true, "ratio_true": lam_true / (d / 2),
        "rows": rows,
        "vol": vol, "hess_A": hess_A, "hess_B": hess_B, "sgld": sgld,
    }


def experiment_1(seed=0):
    """r=1, n=m=1, d=2. True lambda=1/2, ratio=1/2."""
    return _ground_truth_experiment(1, 1, 1, seed, tag="Experiment 1 (r=1,n=1,m=1,d=2)",
                                     sgld_n_values=[50, 100, 200, 500, 1000, 2000, 5000])


def experiment_2(seed=1):
    """r=1, n=m=2, d=4. True lambda=1, ratio=1/2."""
    return _ground_truth_experiment(1, 2, 2, seed, tag="Experiment 2 (r=1,n=2,m=2,d=4)",
                                     sgld_n_values=[50, 100, 200, 500, 1000, 2000, 5000])


def experiment_3(seed=2, d=4):
    """Regular model K(w) = ||w||^2. True lambda = d/2, ratio = 1."""
    rng = np.random.default_rng(seed)
    K, K_t = make_K_regular(d)
    lam_true = d / 2.0

    vol = volume_scaling_estimator(K, d, n_samples=5_000_000, rng=rng)
    # Hessian is 2*Identity everywhere -> no null space anywhere; evaluate at origin.
    from model import K_grad_hess
    _, _, H = K_grad_hess(np.zeros(d), K_t)
    eigvals = np.linalg.eigvalsh(H)
    null_dim = int(np.sum(np.abs(eigvals) < 1e-8))
    lambda_hess = (d - null_dim) / 2.0
    sgld = sgld_llc_estimator(K_t, d, n_steps=15_000,
                               n_values=[50, 100, 200, 500, 1000, 2000, 5000], seed=seed)

    rows = [
        ("Volume scaling", vol["lambda_estimate"], vol["ratio"]),
        ("Hessian null space (at origin)", lambda_hess, lambda_hess / (d / 2)),
        ("SGLD (devinterp-style)", sgld["lambda_estimate"], sgld["ratio"]),
    ]

    return {
        "tag": f"Experiment 3 (regular model, d={d})", "r": None, "n": None, "m": None, "d": d,
        "lambda_true": lam_true, "ratio_true": 1.0,
        "rows": rows, "vol": vol, "sgld": sgld,
        "eigenvalues": eigvals, "null_dim": null_dim,
    }


def experiment_4(seed=3, r=1, n=2, m=2, n_steps=15_000, ridge=0.02, adam_lr=0.02,
                  init_scale=3.0, local_radius=0.15,
                  n_checkpoints=20, vol_n_samples=500_000):
    """Track lambda/(d/2) via LOCAL volume scaling (a small ball of
    `local_radius` around the current iterate w_t) across an Adam
    trajectory minimizing K(w) + ridge*||w||^2 from a deliberately
    "far from the singular locus" random initialization.

    K has no positive-K critical points for this model (its only critical
    points are on W0 = {AB=0} itself), so any minimizer eventually reaches
    W0, and the ridge term additionally breaks the classic deep-linear-net
    conservation law (||A||^2-||B||^2 = const) so it decays to 0 and the
    trajectory converges specifically to the singular origin rather than
    an arbitrary smooth point of a branch.

    Important, and somewhat surprising, finding from developing this
    experiment (see RESULTS.md): the naive expectation "ratio starts near
    1 (regular) and decays to 1/2 (singular)" is not quite what happens.
    Early on, w_t is far from ANY zero of K, so the local ball at radius
    `local_radius` contains no near-zero of K at all -- the log-log
    volume-vs-eps fit is then measuring the shape of a locally linear (not
    quadratic) function and returns large, not-really-interpretable ratio
    values, not the "regular ratio ~= 1" of experiment 3. Only once the
    trajectory gets close enough to W0 for the ball to contain a genuine
    near-zero does the ratio snap down to ~0.5 (matching the branch-local
    RLCT from hessian_branch_estimator) and stay there; the fitted
    log-multiplicity constant k then further jumps up right as the
    trajectory reaches the true origin (where both branches of W0 cross),
    distinguishing "near a generic point of W0" from "at the singular
    point itself" even though lambda itself reads the same in both cases
    for this symmetric (n==m) model.
    """
    import torch
    rng = np.random.default_rng(seed)
    d = dim_w(r, n, m)
    lam_true = true_lambda(r, n, m)
    K = make_K(r, n, m)
    K_t = make_K_torch(r, n, m)

    w = torch.tensor(rng.standard_normal(d) * init_scale, dtype=torch.float64,
                      requires_grad=True)
    opt = torch.optim.Adam([w], lr=adam_lr)
    checkpoint_steps = np.unique(np.geomspace(1, n_steps, n_checkpoints).astype(int))

    trajectory = []
    step = 0
    for target in checkpoint_steps:
        while step < target:
            step += 1
            opt.zero_grad()
            loss = K_t(w) + ridge * (w ** 2).sum()
            loss.backward()
            opt.step()

        w_t_np = w.detach().numpy().copy()
        dist_to_origin = float(np.linalg.norm(w_t_np))

        def K_shifted(delta, w0=w_t_np):
            return K(w0 + delta)

        vol = volume_scaling_estimator(K_shifted, d, n_samples=vol_n_samples,
                                        radius=local_radius, rng=rng)
        trajectory.append({
            "step": step,
            "w": w_t_np,
            "dist_to_origin": dist_to_origin,
            "K_val": K(w_t_np),
            "ratio": vol["ratio"],
            "lambda_estimate": vol["lambda_estimate"],
            "log_mult_k": vol["log_mult_k"],
        })

    return {
        "tag": f"Experiment 4 (training trajectory, r={r},n={n},m={m},d={d})",
        "d": d, "lambda_true": lam_true, "ratio_true": lam_true / (d / 2),
        "trajectory": trajectory,
    }


def experiment_5(seed=4, r=1, n=2, m=2, n_directions=2_000_000):
    """Distribution of K over the unit sphere at the singular origin."""
    rng = np.random.default_rng(seed)
    d = dim_w(r, n, m)
    K = make_K(r, n, m)
    arc = arc_direction_estimator(K, d, n_directions=n_directions, rng=rng)

    # Fit power-law exponent of fraction(K < t) vs t in the small-t regime.
    thresholds, fractions = arc["thresholds"], arc["fractions"]
    mask = (fractions > 0) & (fractions <= 0.2)
    if mask.sum() >= 2:
        coeffs = np.polyfit(np.log(thresholds[mask]), np.log(fractions[mask]), 1)
        exponent = coeffs[0]
    else:
        exponent = float("nan")

    return {
        "tag": f"Experiment 5 (arc direction distribution, r={r},n={n},m={m},d={d})",
        "d": d, "arc": arc, "exponent": exponent,
    }


def experiment_6(seed=5, r=1, n=1, m=4, n_init=20):
    """Asymmetric n != m: stress-tests the Hessian estimator's branch
    dependence (flagged as a caveat in experiments 1/2, where n==m hid it)
    and validates the multi-restart min-codim fix.

    True lambda = min(n,m)/2. The {A=0} branch (B free, dimension m) has
    codim n; the {B=0} branch (A free, dimension n) has codim m --
    landing on the wrong (larger-codim) branch from a single gradient
    descent run overstates lambda by a factor of m/n.
    """
    rng = np.random.default_rng(seed)
    d = dim_w(r, n, m)
    lam_true = true_lambda(r, n, m)
    K_t = make_K_torch(r, n, m)

    hess_A = hessian_branch_estimator(K_t, r, n, m, d, branch="A", rng=rng)
    hess_B = hessian_branch_estimator(K_t, r, n, m, d, branch="B", rng=rng)
    multi = hessian_multi_restart_estimator(K_t, d, n_init=n_init, seed=seed)

    rows = [
        (f"Hessian, branch='A' (A free, on {{B=0}} plane, codim m={m})", hess_A["lambda_estimate"], hess_A["ratio"]),
        (f"Hessian, branch='B' (B free, on {{A=0}} plane, codim n={n})", hess_B["lambda_estimate"], hess_B["ratio"]),
        (f"Hessian, multi-restart min-codim ({n_init} restarts)", multi["lambda_estimate"], multi["ratio"]),
    ]

    return {
        "tag": f"Experiment 6 (asymmetric n!=m, r={r},n={n},m={m},d={d})",
        "r": r, "n": n, "m": m, "d": d,
        "lambda_true": lam_true, "ratio_true": lam_true / (d / 2),
        "rows": rows, "hess_A": hess_A, "hess_B": hess_B, "multi": multi,
    }


def experiment_7(seed=6, r=1, n=2, m=2, n_dds_samples=20_000):
    """Dead-Direction Signatures (DDS), validated on the same r=1 toy model.

    Our K(w) = ||AB||_F^2 model IS the two-layer linear network DDS reads
    (x -> hidden=Bx [layer "h1", the bottleneck] -> output=A(Bx) [layer
    "h2"]) with a zero teacher, so no new model is needed for this part.

    Two checks, mirroring the paper's own validation structure:
    (a) Analytic-limit rate check: approach the {B=0} branch along a fixed
        transverse direction (A fixed, B=t*B_dir, t -> 0) and confirm the
        predicted Theorem-2 structural correlation
        rho(lambda_plus_min(G_h1), sigma_min(X_h1)^2) -> +1, plus the
        predicted rate exponents.
    (b) The same readout along the real (noisy, ridge-regularized) GD
        trajectory from experiment_4, checking whether h1 and h2 collapse
        together or separately.

    Important finding surfaced here, not hidden: h1 and h2 collapse at THE
    SAME rate in our r0=0 (true rank zero) toy models, unlike the DDS
    paper's own r0>0 anchor where the boundary layer h2 stays flat while
    only the bottleneck h1 collapses. This is a real structural difference,
    not a bug: with truth rank 0, the entire map (both layers) must vanish
    at the true optimum, so there is no "surviving signal" for h2 to carry.
    See RESULTS.md for discussion.
    """
    rng = np.random.default_rng(seed)
    d = dim_w(r, n, m)
    lam_true = true_lambda(r, n, m)

    # --- (a) analytic-limit rate check, approaching the {B=0} branch ---
    A_dir = rng.standard_normal((n, r))
    B_dir = rng.standard_normal((r, m))
    ts = np.logspace(0, -4, 25)
    log_lam_h1, log_sigma_h1_sq, log_lam_h2, log_sigma_h2_sq = [], [], [], []
    for t in ts:
        B = t * B_dir
        obs = dds_observables(A_dir, B, N=n_dds_samples, rng=np.random.default_rng(seed))
        log_lam_h1.append(np.log(obs["lambda_plus_min_h1"]))
        log_sigma_h1_sq.append(np.log(obs["sigma_min_h1"] ** 2))
        log_lam_h2.append(np.log(obs["lambda_plus_min_h2"]))
        log_sigma_h2_sq.append(np.log(max(obs["sigma_min_h2"], 1e-300) ** 2))

    from scipy.stats import spearmanr
    log_t = np.log(ts)
    rho_structural, _ = spearmanr(log_lam_h1, log_sigma_h1_sq)
    slope_lam_h1 = np.polyfit(log_t, log_lam_h1, 1)[0]
    slope_sigma_h1 = np.polyfit(log_t, log_sigma_h1_sq, 1)[0]
    slope_lam_h2 = np.polyfit(log_t, log_lam_h2, 1)[0]

    analytic = {
        "ts": ts, "log_t": log_t,
        "log_lam_h1": np.array(log_lam_h1), "log_sigma_h1_sq": np.array(log_sigma_h1_sq),
        "log_lam_h2": np.array(log_lam_h2),
        "rho_structural": rho_structural,
        "slope_lam_h1": slope_lam_h1, "slope_sigma_h1": slope_sigma_h1,
        "slope_lam_h2": slope_lam_h2,
    }

    # --- (b) along the real GD+ridge trajectory (reuse experiment_4's setup) ---
    traj4 = experiment_4(seed=seed, r=r, n=n, m=m)
    traj_dds = []
    for ckpt in traj4["trajectory"]:
        w = ckpt["w"]
        A = w[: r * n].reshape(n, r)
        B = w[r * n :].reshape(r, m)
        obs = dds_observables(A, B, N=n_dds_samples, rng=np.random.default_rng(seed))
        traj_dds.append({
            "step": ckpt["step"], "dist_to_origin": ckpt["dist_to_origin"],
            "lambda_plus_min_h1": obs["lambda_plus_min_h1"],
            "lambda_plus_min_h2": obs["lambda_plus_min_h2"],
            "sigma_min_h1": obs["sigma_min_h1"],
        })

    # restrict to points where the bottleneck hasn't fully numerically
    # vanished (log(0) is uninformative), matching the paper's own
    # "measurable Phase A" precondition
    valid = [c for c in traj_dds if c["lambda_plus_min_h1"] > 0 and c["sigma_min_h1"] > 0]
    log_lam_h1_traj = np.log([c["lambda_plus_min_h1"] for c in valid])
    log_sigma_h1_traj = np.log([c["sigma_min_h1"] ** 2 for c in valid])
    log_lam_h2_traj = np.log([c["lambda_plus_min_h2"] for c in valid if c["lambda_plus_min_h2"] > 0])
    rho_traj, _ = spearmanr(log_lam_h1_traj, log_sigma_h1_traj) if len(valid) >= 3 else (float("nan"), None)
    rho_h1_h2_traj, _ = spearmanr(log_lam_h1_traj[:len(log_lam_h2_traj)], log_lam_h2_traj) \
        if len(log_lam_h2_traj) >= 3 else (float("nan"), None)

    return {
        "tag": f"Experiment 7 (DDS validation, r={r},n={n},m={m},d={d})",
        "d": d, "lambda_true": lam_true,
        "analytic": analytic,
        "trajectory_dds": traj_dds,
        "rho_structural_trajectory": rho_traj,
        "rho_h1_h2_trajectory": rho_h1_h2_traj,
    }


def _cross_cell_rho(per_cell, lam_true, names):
    from scipy.stats import spearmanr
    return {n: float(spearmanr(per_cell[n], lam_true)[0]) for n in names}


def experiment_8(M=10, N=5, n_chains=8, n_boot=500, seed=7, scales=(0.01, 0.1, 0.3),
                 n_ablation_seeds=10, n_dds_samples=5_000):
    """Cross-cell DDS rank-tracking against closed-form RLCT on the 14-cell
    Aoyagi-Watanabe grid (M=10, N=5, H in {2,...,5}, truth rank r0 in
    {1,...,min(N,H)}; lambda = (NH + (M-H) r0)/2).

    Main protocol: read the DDS observables off a localized SGLD *posterior
    ensemble* around the exact minimum-norm solution (rrr_model.sgld_rrr_chains),
    averaging log-observables over snapshots and chains. Two checks come with it:
      - the sampler's own LLC estimate n*beta*E[K] must recover the closed-form
        lambda (it does: rho ~ 0.999, 5-9% high), so the ensemble is
        equilibrated and its fluctuations carry the local geometry;
      - bootstrap CIs over chains for every cross-cell rho.
    Only the output layer h2 (width N, fixed across cells) is compared across
    cells; h1 has width H, so its spectrum is not comparable between cells.

    Ablation: the same readings at a single hand-constructed point (exact
    solution + a fixed-Frobenius-norm transverse perturbation). Its cross-cell
    rho is the same at every perturbation scale, i.e. it reflects the geometry
    of the perturbation, not lambda.
    """
    from rrr_model import sgld_rrr_chains, exact_branch_point
    from scipy.stats import spearmanr
    cells = aoyagi_2005_anchor_cells(M=M, N=N)
    lam_true = np.array([c["lambda_true"] for c in cells])
    names = ["lambda_plus_min_h2", "log_det_plus_h2", "sigma_min_h2", "sigma_plus_min_h2"]

    # --- main: posterior ensemble ---
    per_chain = {n: np.zeros((len(cells), n_chains)) for n in names}
    llc_hat = np.zeros((len(cells), n_chains))
    for i, cell in enumerate(cells):
        lam_hat, snaps, T = sgld_rrr_chains(M, N, cell["H"], cell["r0"], chains=n_chains,
                                            seed=seed + i)
        llc_hat[i] = lam_hat
        vals = {n: [[] for _ in range(n_chains)] for n in names}
        for W1, W2 in snaps:
            for k in range(n_chains):
                o = dds_observables(W2[k], W1[k], N=n_dds_samples, target=T,
                                    rng=np.random.default_rng(seed))
                vals["lambda_plus_min_h2"][k].append(np.log(o["lambda_plus_min_h2"]))
                vals["log_det_plus_h2"][k].append(o["log_det_plus_h2"])
                vals["sigma_min_h2"][k].append(np.log(o["sigma_min_h2"]))
                vals["sigma_plus_min_h2"][k].append(np.log(o["sigma_plus_min_h2"]))
        for n in names:
            per_chain[n][i] = [np.mean(v) for v in vals[n]]

    point = {n: per_chain[n].mean(1) for n in names}
    llc_mean = llc_hat.mean(1)
    rho = _cross_cell_rho(point, lam_true, names)
    rho_llc = float(spearmanr(llc_mean, lam_true)[0])

    # within-H pairwise concordance: cells with the same width H share every
    # layer dimension, so this comparison is free of dimension artifacts
    H_arr = np.array([c["H"] for c in cells])

    def within_H(v):
        conc = tot = 0
        for h in np.unique(H_arr):
            idx = np.where(H_arr == h)[0]
            for i in idx:
                for j in idx:
                    if lam_true[i] < lam_true[j]:
                        tot += 1
                        conc += v[i] < v[j]
        return int(conc), int(tot)

    within = {n: within_H(point[n]) for n in names}
    within["llc"] = within_H(llc_mean)

    rng = np.random.default_rng(seed)
    boot = {n: [] for n in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n_chains, size=(len(cells), n_chains))
        for n in names:
            v = np.take_along_axis(per_chain[n], idx, axis=1).mean(1)
            boot[n].append(spearmanr(v, lam_true)[0])
    rho_ci = {n: (float(np.nanpercentile(boot[n], 2.5)), float(np.nanpercentile(boot[n], 97.5)))
              for n in names}

    # --- ablation: single constructed point, several perturbation scales ---
    ablation = {}
    for scale in scales:
        rhos = {n: [] for n in names}
        for s in range(n_ablation_seeds):
            obs_rows = []
            for i, cell in enumerate(cells):
                rg = np.random.default_rng(seed + 1000 * s + i)
                W1, W2 = exact_branch_point(M, N, cell["H"], cell["r0"], scale=scale, rng=rg)
                obs_rows.append(dds_observables(W2, W1, N=n_dds_samples, rng=rg,
                                                target=make_teacher(N, M, cell["r0"])))
            for n in names:
                rhos[n].append(spearmanr([r[n] for r in obs_rows], lam_true)[0])
        ablation[scale] = {n: float(np.mean(v)) for n, v in rhos.items()}

    return {
        "tag": f"Experiment 8 (DDS cross-cell rank-tracking, Aoyagi-Watanabe grid, "
               f"M={M}, N={N}, {len(cells)} cells)",
        "cells": cells, "lambda_true": lam_true, "observables": names,
        "per_cell_mean": point, "cross_cell_rho": rho, "cross_cell_rho_ci": rho_ci,
        "llc_hat": llc_mean, "llc_rel_err": (llc_mean - lam_true) / lam_true,
        "rho_llc": rho_llc, "ablation_rho": ablation, "within_H_concordance": within,
    }


def experiment_9(D=20, Ls=(4, 6, 8), rs=(1, 2, 3, 4), n_taus=15, n_dds_samples=20_000, seed=8):
    """The DDS paper's own most discriminating claim: the rank-multiplicative
    volume identity. On an L-layer deep-linear "noisy bridge" (their Sec.
    4.2 / App. B.8.2) with r simultaneously-dead directions, log_det_plus(G)
    slope (vs log distance-to-singularity) should scale as r times the
    rank-1 slope, while lambda_plus_min(G) slope should be r-invariant.
    This needs layer width >= 2 dead directions at once to test at all --
    our r=1 bottleneck models (experiments 1-8) cannot touch it.

    Construction: every layer is diagonal, diag(1,...,1,tau,...,tau) with r
    copies of a shared tau on the dead coordinates (generalizing the L=2
    approach validated in experiment_7 to L layers and r simultaneous dead
    directions -- see deep_linear.py's module docstring). This is fully
    closed-form (no training, no convergence-criterion tuning), sweeping
    tau directly rather than gradient-descending to it.

    Caveat, stated plainly: our per-layer rate EXPONENTS do not match the
    paper's reported 2(L-ell) formula (we get 4L-2*ell here; both decrease
    by exactly 2 per layer, but ours is offset by 2L, evidently because our
    "every layer shares the same tau" construction differs from their
    unspecified exact Wl(t)=W*l+t*deltal recipe -- we don't have enough
    detail to reproduce that construction exactly). What we DO reproduce,
    exactly and robustly regardless of that offset, is the discriminating
    claim itself: because log_det_plus sums log-eigenvalues over all r
    (symmetric, identically-scaling) dead directions, its slope is
    necessarily r times the single-direction slope; because
    lambda_plus_min reads whichever eigenvalue is smallest and all r dead
    eigenvalues are identical by construction, it's necessarily
    r-invariant. This holds for any L, D, or exponent convention.
    """
    rng_master = np.random.default_rng(seed)
    taus = np.logspace(0, -3, n_taus)
    log_taus = np.log(taus)

    per_L = {}
    for L in Ls:
        per_r_layer_slopes = {r: {} for r in rs}
        for r in rs:
            teacher = make_teacher_diag(D, r)
            log_logdet_by_layer = {ell: [] for ell in range(1, L)}
            log_lam_by_layer = {ell: [] for ell in range(1, L)}
            for tau in taus:
                Ws = canonical_layers(D, L, r, tau)
                obs = dds_observables_deep(Ws, N=n_dds_samples, target=teacher,
                                            rng=np.random.default_rng(seed))
                for ell in range(1, L):
                    log_logdet_by_layer[ell].append(obs[ell]["log_det_plus"])
                    log_lam_by_layer[ell].append(np.log(obs[ell]["lambda_plus_min"]))
            for ell in range(1, L):
                s_logdet = np.polyfit(log_taus, log_logdet_by_layer[ell], 1)[0]
                s_lam = np.polyfit(log_taus, log_lam_by_layer[ell], 1)[0]
                per_r_layer_slopes[r][ell] = {"slope_log_det_plus": s_logdet, "slope_lambda_plus_min": s_lam}
        per_L[L] = per_r_layer_slopes

    # aggregate slope RATIOS (vs r=1) across all (L, layer) cells, mirroring
    # the paper's own pooling across their 7 noisy-bridge configurations
    ratios_logdet = {r: [] for r in rs if r != 1}
    ratios_lam = {r: [] for r in rs if r != 1}
    for L in Ls:
        for ell in range(1, L):
            base_logdet = per_L[L][1][ell]["slope_log_det_plus"]
            base_lam = per_L[L][1][ell]["slope_lambda_plus_min"]
            for r in rs:
                if r == 1:
                    continue
                ratios_logdet[r].append(per_L[L][r][ell]["slope_log_det_plus"] / base_logdet)
                ratios_lam[r].append(per_L[L][r][ell]["slope_lambda_plus_min"] / base_lam)

    ratio_summary = {
        r: {"log_det_plus_ratio_mean": float(np.mean(ratios_logdet[r])),
            "log_det_plus_ratio_std": float(np.std(ratios_logdet[r])),
            "lambda_plus_min_ratio_mean": float(np.mean(ratios_lam[r])),
            "lambda_plus_min_ratio_std": float(np.std(ratios_lam[r]))}
        for r in rs if r != 1
    }

    return {
        "tag": f"Experiment 9 (deep-linear noisy bridge, D={D}, L in {list(Ls)}, r in {list(rs)})",
        "D": D, "Ls": list(Ls), "rs": list(rs), "taus": taus,
        "per_L": per_L, "ratio_summary": ratio_summary,
    }


def experiment_10(llc_ns=(5_000, 50_000, 500_000), n_traj=30, epochs=4500, seed=10,
                  record_every=30):
    """Reproduce the core findings of Chen et al. (arXiv:2310.06301) on the
    Toy Model of Superposition, r=2 hidden dims, c=6 inputs.

    (a) The k-gon critical points: exact population loss, gradient norm, and
        prior factor 1/2*||w*||^2 against the paper's tabulated values.
    (b) Theoretical LLCs (7, 8.5, 8.5 for 5, 5+, 6) vs localized-SGLD
        estimates at several n -- the n=5000 row is the paper's Table K.1
        protocol; larger n shows the residual bias is a finite-n effect.
    (c) Critical sample sizes of the Bayesian phase transitions from the
        free-energy crossing n*dL + dlam*log n + dc = 0.
    (d) SGD trajectories (paper Sec. 5 protocol) initialized at a 4-gon plus
        N(0, 0.01^2) noise: loss plateaus at k-gon levels, with the SGLD LLC
        estimate rising as loss falls ("opposing staircases").
    """
    out = {"tag": "Experiment 10 (Toy Model of Superposition, r=2, c=6; Chen et al. 2023)"}

    # (a) critical points
    cases = [("4", 4, 0), ("4+", 4, 1), ("5", 5, 0), ("5+", 5, 1), ("6", 6, 0)]
    paper_L = {"4": 1 / 9, "4+": 0.10417, "5": 0.06874, "5+": 0.06180, "6": 0.04819}
    paper_prior = {"4": 2.0, "4+": 2.00347, "5": 3.62417, "5+": 3.62764, "6": 6.37767}
    crit = {}
    for name, k, npos in cases:
        W, b = tms.kgon(k, n_pos=npos, vestigial_bias=0.0)
        w = tms.pack(torch.tensor(W[None]), torch.tensor(b[None])).requires_grad_(True)
        L = tms.loss_flat(w)
        g, = torch.autograd.grad(L.sum(), w)
        # prior factor counts only the k-gon block and positive biases (vestigial
        # negative biases are free and contribute >= 0, so this is the paper's
        # lower bound for those)
        prior = 0.5 * ((W ** 2).sum() + (b[:k] ** 2).sum() + (b[k:][b[k:] > 0] ** 2).sum())
        crit[name] = {"loss": L.item(), "paper_loss": paper_L[name],
                      "grad_norm": g.norm().item(), "prior": float(prior),
                      "paper_prior": paper_prior[name]}
    out["critical_points"] = crit

    # (b) LLC estimates vs theory
    theory = {"5": 7.0, "5+": 8.5, "6": 8.5}
    paper_hat = {"5": 7.705, "5+": 9.906, "6": 9.027}
    steps_for = {5_000: (1e-5, 50_000), 50_000: (1e-6, 100_000), 500_000: (1e-7, 200_000)}
    llc = {}
    for name, k, npos in [c for c in cases if c[0] in theory]:
        # vestigial bias strictly negative (-1): at exactly 0 the dead column sits
        # on the boundary of a lower-loss chamber and SGLD chains escape into it
        W, b = tms.kgon(k, n_pos=npos, vestigial_bias=-1.0)
        w = tms.pack(torch.tensor(W[None]), torch.tensor(b[None]))
        row = {"theory": theory[name], "paper_hat_n5000": paper_hat[name], "est": {}}
        for n in llc_ns:
            eps, steps = steps_for[n]
            m, sd, kept, _ = tms.sgld_llc(w, n=n, gamma=0.1, eps=eps, steps=steps,
                                          burn=steps // 5, seed=seed)
            row["est"][n] = {"mean": m, "std": sd, "kept": kept}
        llc[name] = row
    out["llc"] = llc

    # (c) critical sample sizes
    def dd(a, bb):
        return (crit[bb]["loss"] - crit[a]["loss"], theory_all[bb] - theory_all[a],
                crit[bb]["prior"] - crit[a]["prior"])
    theory_all = {"5": 7.0, "6": 8.5}
    out["n_crit"] = {
        "5->6": {"ours": tms.critical_sample_size(*dd("5", "6")), "paper": 601},
    }
    dL, dlam, dc = dd("5", "6")
    out["n_crit"]["5->6"]["terms"] = {"dL": dL, "dlam": dlam, "dc": dc}
    out["n_crit"]["5->6"]["no_constant"] = tms.critical_sample_size(dL, dlam, 0.0)

    # (d) SGD trajectories
    torch.manual_seed(seed)
    W, b = tms.kgon(4, vestigial_bias=0.0)
    w0 = tms.pack(torch.tensor(W[None]), torch.tensor(b[None])).repeat(n_traj, 1)
    w0 = w0 + 0.01 * torch.randn_like(w0)
    rw, eps_rec, Ltraj = tms.sgd_trajectories(w0, epochs=epochs, record_every=record_every, seed=seed)
    T, S, d = rw.shape
    lam_hat = tms.sgld_llc_points(rw.reshape(T * S, d), seed=seed).reshape(T, S)
    ref_levels = {"4": 1 / 9, "4+": 0.10417, "4++": 0.09722, "5": 0.06874,
                  "5+": 0.06180, "6": 0.04819}
    near = np.full((T, S), "", dtype=object)
    for nm, lv in ref_levels.items():
        near[np.abs(Ltraj.numpy() - lv) < 3e-4] = nm
    out["sgd"] = {"epochs": eps_rec, "loss": Ltraj.numpy(), "lam_hat": lam_hat,
                  "near": near, "ref_levels": ref_levels}
    return out


def format_table(exp):
    lines = [f"### {exp['tag']}", ""]
    lines.append(f"d = {exp['d']}, true lambda = {exp['lambda_true']:.4f}, "
                 f"true ratio lambda/(d/2) = {exp['ratio_true']:.4f}")
    lines.append("")
    lines.append("| Method | lambda estimate | d/2 | lambda/(d/2) | True lambda/(d/2) |")
    lines.append("|---|---|---|---|---|")
    for name, lam, ratio in exp["rows"]:
        lines.append(f"| {name} | {lam:.4f} | {exp['d']/2:.2f} | {ratio:.4f} | {exp['ratio_true']:.4f} |")
    if "vol" in exp:
        vol = exp["vol"]
        lines.append("")
        lines.append(f"(Volume scaling diagnostics: naive power-law fit (no log-multiplicity "
                      f"correction) = {vol['lambda_estimate_naive']:.4f}, fitted log-multiplicity "
                      f"exponent k = {vol['log_mult_k']:.3f}, fit R^2 = {vol['r2']:.5f})")
    return "\n".join(lines)
