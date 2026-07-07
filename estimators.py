"""Estimators for the RLCT / local learning coefficient lambda.

Two estimators are the main comparison:
  - hessian_branch_estimator: geometric, from the null space of the
    Hessian of K evaluated at a point on one branch of W0 (away from
    the singular origin, since the Hessian vanishes identically at 0
    for this quartic model).
  - sgld_llc_estimator: the devinterp-style estimator, sampling from a
    tempered, localized posterior with SGLD and reading off lambda from
    how the free energy (WBIC) scales with log(n).

volume_scaling_estimator is a third, more robust estimator used mainly
to validate ground truth and sanity-check the other two.
arc_direction_estimator is a diagnostic on the distribution of K over
the unit sphere.
"""
import numpy as np
import torch

from model import K_grad_hess, sample_on_branch


def volume_scaling_estimator(K_fn, d, n_samples=200_000, eps_values=None,
                              fit_vol_max=0.3, min_count=20, radius=1.0, rng=None):
    """Estimate lambda from Vol{w : K(w) <= eps} ~ C * eps^lambda * |log eps|^k
    as eps -> 0.

    The |log eps|^k factor is not a nicety: Watanabe's theory says the RLCT
    zeta function ζ(z) = ∫ K(w)^z dw has poles of some multiplicity m, and
    multiplicity m > 1 (as happens even in the simplest r=n=m=1 case here,
    where the pole at z=-1/2 has multiplicity 2) produces exactly this log
    correction to the volume asymptotics. A plain power-law fit (k forced to
    0) is measurably biased low for this model (empirically ~0.42 instead of
    the true 0.5) -- fitting k alongside lambda removes most of that bias.

    Samples w uniformly from the unit ball in R^d, evaluates K, and does a
    linear regression of log(Vol) on [1, log(eps), log(-log(eps))], restricted
    to the small-eps regime (Vol <= fit_vol_max, eps < 1) to stay clear of
    saturation as eps -> O(1), and to eps large enough that at least
    `min_count` of the n_samples points fall below it (so the empirical
    volume isn't dominated by sampling noise).
    """
    if rng is None:
        rng = np.random.default_rng()

    raw = rng.standard_normal((n_samples, d))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    radii = radius * rng.uniform(0, 1, (n_samples, 1)) ** (1.0 / d)
    w_samples = raw / norms * radii

    K_vals = np.asarray(K_fn(w_samples))

    if eps_values is None:
        # Scale-adaptive: build the eps grid from the empirical range of
        # K_vals itself rather than an absolute range, so this works
        # whether K_vals ~ 1e-30 (near a true zero of K) or ~1e3 (K_fn
        # evaluated in a region far from any zero of K, e.g. experiment 4
        # early in training).
        positive = K_vals[K_vals > 0]
        if positive.size < min_count:
            raise RuntimeError("volume_scaling_estimator: fewer than min_count "
                                "samples have K>0; increase n_samples")
        lo = np.quantile(positive, min_count / len(K_vals))
        hi = np.quantile(positive, 0.9)
        eps_values = np.logspace(np.log10(lo), np.log10(hi), 50)

    volumes = np.array([np.mean(K_vals <= eps) for eps in eps_values])
    counts = np.array([np.sum(K_vals <= eps) for eps in eps_values])

    mask = (volumes > 0) & (volumes <= fit_vol_max) & (counts >= min_count)
    log_eps = np.log(eps_values[mask])
    log_vol = np.log(volumes[mask])

    if mask.sum() < 3:
        raise RuntimeError("volume_scaling_estimator: not enough points in fit "
                            "region; increase n_samples or widen eps_values")

    # The log(-log(eps)) log-multiplicity basis function is only meaningful
    # in the eps -> 0 (eps < 1) asymptotic regime the RLCT theory describes.
    # When K_fn's natural scale puts the fit region at eps >= 1 (e.g. K
    # evaluated far from any zero of K, as happens transiently in
    # experiment 4), fall back to a plain power-law fit instead.
    if np.all(eps_values[mask] < 1):
        log_log_eps = np.log(-log_eps)
        X = np.column_stack([np.ones_like(log_eps), log_eps, log_log_eps])
        coeffs, *_ = np.linalg.lstsq(X, log_vol, rcond=None)
        const, lambda_estimate, k_log = coeffs
        pred = X @ coeffs
    else:
        naive = np.polyfit(log_eps, log_vol, 1)
        lambda_estimate, const = naive
        k_log = 0.0
        pred = np.polyval(naive, log_eps)
    ss_res = np.sum((log_vol - pred) ** 2)
    ss_tot = np.sum((log_vol - log_vol.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    naive_coeffs = np.polyfit(log_eps, log_vol, 1)

    return {
        "eps_values": eps_values,
        "volumes": volumes,
        "lambda_estimate": lambda_estimate,
        "lambda_estimate_naive": naive_coeffs[0],
        "log_mult_k": k_log,
        "ratio": lambda_estimate / (d / 2),
        "log_eps": log_eps,
        "log_vol": log_vol,
        "r2": r2,
        "n_fit_points": mask.sum(),
    }


def hessian_branch_estimator(K_t_fn, r, n, m, d, branch="A", scale=1.0,
                              null_rel_tol=1e-8, rng=None):
    """Estimate lambda from the Hessian null space at a point on one
    branch of W0, away from the singular origin.

    Important caveat (surfaced deliberately, not hidden): this measures
    the tangent dimension of a single smooth branch of W0, which equals
    the true RLCT only in special cases (e.g. n == m). See RESULTS.md.
    """
    if rng is None:
        rng = np.random.default_rng()
    w_star = sample_on_branch(r, n, m, branch=branch, scale=scale, rng=rng)
    K_val, grad, H = K_grad_hess(w_star, K_t_fn)

    eigvals = np.linalg.eigvalsh(H)
    tol = null_rel_tol * np.abs(eigvals).max() if np.abs(eigvals).max() > 0 else null_rel_tol
    null_dim = int(np.sum(np.abs(eigvals) < tol))
    lambda_estimate = (d - null_dim) / 2.0

    return {
        "branch": branch,
        "w_star": w_star,
        "K_at_w_star": K_val,
        "grad_norm": float(np.linalg.norm(grad)),
        "eigenvalues": eigvals,
        "null_dim": null_dim,
        "d": d,
        "lambda_estimate": lambda_estimate,
        "ratio": lambda_estimate / (d / 2),
    }


def hessian_multi_restart_estimator(K_t_fn, d, n_init=20, init_scale=0.5,
                                     lr=0.05, n_steps=1000,
                                     null_rel_tol=1e-6, seed=None):
    """Run plain gradient descent on K from n_init random starts, evaluate
    the Hessian null space at whatever branch point each run converges to,
    and return min(codim over runs) / 2.

    Why the minimum (not an arbitrary single run) is the right thing to
    take: for this r=1 model, each smooth branch of W0 contributes its own
    codimension as a candidate "local RLCT", and W0's true RLCT is the
    *smallest* of these (the zeta function's rightmost pole is set by
    whichever stratum has the smallest codimension -- see RESULTS.md). A
    single gradient descent run converges to whichever branch's basin
    contains the initialization, which is a matter of chance, so one run
    can land on a branch with too-large codimension. Multiple restarts
    make it overwhelmingly likely that at least one run lands in the
    smallest-codimension branch's basin, and the minimum over runs then
    recovers it exactly.

    This is a fact about the specific union-of-linear-subspaces geometry
    of the r=1 model (a normal-crossing arrangement), not a generic
    property of RLCT estimation -- see the caveat in RESULTS.md about why
    this would not obviously generalize past r=1 or past normal-crossing
    singularities.
    """
    if seed is not None:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    codims = []
    w_finals = []
    for _ in range(n_init):
        w = torch.tensor(rng.standard_normal(d) * init_scale, dtype=torch.float64,
                          requires_grad=True)
        for _ in range(n_steps):
            if w.grad is not None:
                w.grad = None
            K_t_fn(w).backward()
            with torch.no_grad():
                w -= lr * w.grad

        w_np = w.detach().numpy().copy()
        _, _, H = K_grad_hess(w_np, K_t_fn)
        eigvals = np.linalg.eigvalsh(H)
        tol = null_rel_tol * np.abs(eigvals).max() if np.abs(eigvals).max() > 0 else null_rel_tol
        null_dim = int(np.sum(np.abs(eigvals) < tol))
        codims.append(d - null_dim)
        w_finals.append(w_np)

    codims = np.array(codims)
    lambda_estimate = codims.min() / 2.0

    return {
        "codims": codims,
        "distinct_codims": sorted(set(codims.tolist())),
        "lambda_estimate": lambda_estimate,
        "ratio": lambda_estimate / (d / 2),
        "w_finals": w_finals,
    }


def sgld_llc_estimator(K_t_fn, d, w_star=None, n_values=None, n_steps=15_000,
                        burn_in_frac=0.5, lr=2e-4, num_chains=5,
                        localization_strength=1.0, seed=None):
    """Devinterp-style LLC estimator via SGLD.

    For each n in n_values, sample from the tempered, localized measure
        pi_n(w) ~ exp(-n*beta_n*K(w) - (gamma/2)||w - w*||^2),  beta_n = 1/log(n)
    via Langevin dynamics, and form the WBIC-like free energy proxy
        F(n) = n * E_{pi_n}[K(w)].
    Watanabe's asymptotic theory gives F(n) ~ lambda*log(n) + const (since
    K(w*) = 0 here), so lambda is read off as the slope of F(n) vs log(n).

    Note on step size: an earlier version scaled lr down by 1/(n*beta_n) to
    keep the *drift* step bounded, but that also shrinks the OU relaxation
    rate near w* (which goes like localization_strength * lr) by the same
    factor, so the chain no longer equilibrates within a fixed step budget
    -- larger n silently gave garbage (non-stationary) traces. Near w* the
    quartic K has a tiny gradient, so a single fixed lr (tuned for stability
    at the largest n*beta_n used) is both stable and lets the chain actually
    mix for every n.

    Following devinterp practice, `num_chains` independent chains are run
    per n and averaged, since a single SGLD chain's K_trace mean is noisy
    enough to visibly bias the fitted slope from run to run.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if w_star is None:
        w_star = np.zeros(d)
    if n_values is None:
        n_values = [50, 100, 200, 500, 1000, 2000, 5000]

    w_star_t = torch.tensor(w_star, dtype=torch.float64)
    free_energies = []
    traces = {}

    for n in n_values:
        beta_n = 1.0 / np.log(n)
        n_beta = n * beta_n  # = n / log(n)

        burn_in = int(n_steps * burn_in_frac)
        chain_means = []
        all_traces = []
        for _chain in range(num_chains):
            w = w_star_t.clone() + 0.01 * torch.randn(d, dtype=torch.float64)
            w.requires_grad_(True)

            K_trace = []
            for step in range(n_steps):
                if w.grad is not None:
                    w.grad = None
                K_val = K_t_fn(w)
                loc_prior = 0.5 * localization_strength * ((w - w_star_t) ** 2).sum()
                loss = n_beta * K_val + loc_prior
                loss.backward()

                with torch.no_grad():
                    noise = np.sqrt(2 * lr) * torch.randn(d, dtype=torch.float64)
                    w -= lr * w.grad
                    w += noise
                w.requires_grad_(True)

                if step >= burn_in:
                    K_trace.append(K_val.item())

            chain_means.append(np.mean(K_trace))
            all_traces.append(K_trace)

        F_n = n * float(np.mean(chain_means))
        free_energies.append(F_n)
        traces[n] = all_traces

    log_n = np.log(n_values)
    coeffs = np.polyfit(log_n, free_energies, 1)
    lambda_estimate = coeffs[0]

    return {
        "n_values": n_values,
        "free_energies": free_energies,
        "log_n": log_n,
        "lambda_estimate": lambda_estimate,
        "ratio": lambda_estimate / (d / 2),
        "traces": traces,
    }


def arc_direction_estimator(K_fn, d, n_directions=20_000, thresholds=None, rng=None):
    """Distribution of K over the unit sphere: diagnostic for how much of
    parameter space is 'flat' (close to W0) in direction only.
    """
    if rng is None:
        rng = np.random.default_rng()
    if thresholds is None:
        thresholds = np.logspace(-8, 0, 30)

    directions = rng.standard_normal((n_directions, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    K_on_sphere = np.asarray(K_fn(directions))
    fractions = np.array([np.mean(K_on_sphere < t) for t in thresholds])

    return {
        "K_on_sphere": K_on_sphere,
        "thresholds": thresholds,
        "fractions": fractions,
        "directions": directions,
    }
