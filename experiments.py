"""Experiments 1-6 from arc_llc_context.md (experiment 6 added: asymmetric
n != m, testing the multi-restart min-codim fix to the Hessian estimator)."""
import numpy as np

from model import (
    true_lambda, dim_w, make_K, make_K_torch, make_K_regular,
)
from estimators import (
    volume_scaling_estimator, hessian_branch_estimator,
    hessian_multi_restart_estimator,
    sgld_llc_estimator, arc_direction_estimator,
)


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
