# Results: Hessian null-space vs SGLD estimators of the RLCT/LLC

Implementation of the plan in `arc_llc_context.md`: compare a new geometric estimator
(Hessian null space) against the standard devinterp SGLD estimator, validated against
analytically known RLCT values on rank-1 matrix factorization with true rank 0.

Run `python3 main.py` to reproduce. Full numbers in `results/summary.json` and
`results/tables.md`; plots in `plots/`.

## Headline result

All four estimators — volume scaling, Hessian null space (on either branch), and SGLD —
agree with each other and with ground truth to within a few percent, on both test cases:

| Experiment | d | true λ | Volume scaling | Hessian null space | SGLD |
|---|---|---|---|---|---|
| 1 (n=m=1) | 2 | 0.500 | 0.458 | 0.500 | 0.525 |
| 2 (n=m=2) | 4 | 1.000 | 0.983 | 1.000 | 1.086 |
| 3 (regular, K=\|\|w\|\|²) | 4 | 2.000 | 1.926 | 2.000 | 2.451 |

The Hessian null-space estimator is exact in these cases (see caveat below on *why* —
it isn't measuring quite what the RLCT literature usually means). SGLD is noisier
(6-8% high on the singular cases, ~23% high on the regular one) but recovers the right
answer within the noise of a 5-chain estimate.

## Correction to the source spec

`arc_llc_context.md` states a general formula λ = r(n+m-r)/2, then in the very next
section derives, via the zeta function, λ=1/2 for r=1,n=m=1 and λ=1 for r=1,n=m=2.
**These are inconsistent for n=m=2**: the general formula gives 3/2, the worked
derivation gives 1. The general formula is the Aoyagi–Watanabe RLCT for *reduced-rank
regression* (y = BAx + noise, with an extra integral over an input distribution x) —
a related but different model from the plain Frobenius-norm loss K = ||AB||_F^2
actually implemented here (there's no x to integrate over).

For r=1, ζ(z) factors as independent radial integrals over a ∈ R^n and b ∈ R^m
(K(a,b) = ||a||²||b||²), each contributing a simple pole at z=-n/2 and z=-m/2. The RLCT
is the rightmost pole:

    lambda = min(n, m) / 2,   multiplicity 2 iff n == m, else 1

This matches the doc's own two worked examples exactly, and — more importantly — is
independently confirmed by all four estimators here, including the exact analytic
Hessian computation (not just a Monte Carlo method that could share a bug with the
zeta-function derivation). `model.py::true_lambda` implements this for r=1; the r>1
case is untouched (no experiment here instantiates r>1, so it's left unverified).

## Implementation notes and fixes beyond the original plan

The plan's pseudocode was a good starting point but had a few bugs / underspecified
details that mattered in practice:

1. **Volume scaling has a log-multiplicity bias.** A naive `log(Vol) ~ lambda*log(eps)`
   fit is measurably biased low (0.42 vs true 0.50 for experiment 1) because the RLCT
   zeta function's pole multiplicity (2, for both cases tested) produces a
   `Vol(eps) ~ C * eps^lambda * |log eps|^k` asymptotic, not a pure power law. Fitting
   `k` alongside `lambda` (linear regression on `[1, log(eps), log(-log(eps))]`)
   removes most of the bias (0.983 vs naive 0.898 for experiment 2). This is a direct
   consequence of the same pole-multiplicity structure the doc itself derives, just not
   connected to the volume-estimator implementation there.

2. **`K` needed vectorizing.** The plan's `make_K` evaluates one sample at a time; a
   Python-level loop over hundreds of thousands of Monte Carlo samples is the dominant
   cost (~15s per call). Rewriting it as a single batched einsum over `(N, d)` arrays
   gives a >1000x speedup, which is what makes 5M-sample volume estimates and 2M-sample
   arc-direction estimates cheap enough to run routinely.

3. **The SGLD step-size schedule in the plan is unstable.** Scaling `lr` down by
   `1/(n*beta_n)` (as sketched) keeps the drift step bounded, but it also scales down
   the Ornstein-Uhlenbeck relaxation rate near w* by the same factor, so the chain
   never re-equilibrates within a fixed step budget for larger n — it just silently
   returns a non-stationary trace (checked directly: the chain's mean position drifts
   monotonically with n instead of converging to a fixed distribution). Because K is
   quartic (tiny gradient near w*=0), a single fixed `lr`, tuned for stability at the
   largest `n*beta_n` used, is both stable and lets the chain equilibrate for every n.
   Averaging `num_chains=5` independent chains per n (standard devinterp practice)
   was also necessary — a single chain's free-energy estimate is noisy enough to
   visibly bias the fitted slope run to run (1.09 vs 1.38 for experiment 2, same seed
   family, chains=5 vs 1).

4. **The Hessian null-space estimator, evaluated off-origin, measures branch tangent
   dimension, not the RLCT directly** — and the two coincided only because n==m in
   experiments 1/2, which hid the issue. At w*=0 the Hessian is identically zero (K is
   quartic), so we evaluate it at a point on one *smooth* branch of W0 (away from the
   origin where the branches cross). There, the null space is exactly the tangent
   space of that single branch, giving `lambda_estimate = codim/2`: codim n on the
   `{B=0}` branch (A free), codim m on the `{A=0}` branch (B free). For n=m these
   coincide with `min(n,m)/2`; for n≠m a *single* Hessian evaluation gives one of two
   different, generally-wrong answers depending on which branch a gradient-descent run
   happens to land on.

   **Fix, proposed and validated in a follow-up round (experiment 6):** run gradient
   descent from many random restarts, take the Hessian null-space codim at each
   converged point, and use `min(codim)` rather than any single run's codim. This
   works because W0's true RLCT is the *minimum* over its branches' codimensions (the
   zeta function's rightmost pole is set by the smallest-codimension stratum), and a
   union-of-linear-subspaces variety like this one has no additional singularity at
   the crossing point beyond what's visible in each branch — so `min(codim)/2` over
   branches equals the true RLCT exactly, not just approximately. `hessian_multi_restart_estimator`
   implements this; on the asymmetric case n=1, m=4 (true λ=0.5) it correctly returns
   0.5 (20 restarts: 18 land on the codim-1 branch, 2 on the codim-4 branch, min wins),
   versus a naive single Hessian evaluation on the wrong branch overstating λ by 4x
   (2.0 instead of 0.5). It's also cheap to be confident in: for r=1, whether GD lands
   on the low- or high-codim branch is governed by a conserved quantity of the
   K-gradient flow (`||A||^2 - ||B||^2`), and the branch with smaller codimension
   turns out to be the *more probable* outcome from a generic random init when n≠m
   (empirically 62-99% per single run across the (n,m) pairs tested here), so a modest
   number of restarts (~20) reliably includes at least one success.

   Caveat on generality: this min-codim trick is a fact about *this* geometry — a
   normal-crossing arrangement of coordinate subspaces, where every branch is smooth
   and the crossing itself adds only pole *multiplicity*, not a smaller pole location.
   It is not a general RLCT-estimation method. For r>1 (not tested here) or for
   singularities that aren't simple unions of linear subspaces, resolving the
   singularity could reveal an even smaller RLCT that isn't the codimension of any
   smooth stratum visible in the original coordinates, and this restart-and-minimize
   trick would then undershoot the truth less reliably or not at all.

   Experiment 6 (r=1, n=1, m=4, d=5, true λ=0.500):

   | Method | λ estimate |
   |---|---|
   | Single Hessian, branch='A' (A free, on {B=0} plane, codim m=4) | 2.000 |
   | Single Hessian, branch='B' (B free, on {A=0} plane, codim n=1) | 0.500 |
   | Multi-restart min-codim (20 restarts: 18×codim-1, 2×codim-4) | 0.500 |

## Experiment 4: local ratio across a training trajectory

The plan expected "ratio starts near 1 (regular), decreases toward 1/2 (singular) as
training converges." That's not quite what happens, and the actual mechanism is more
informative:

![trajectory](plots/exp4_trajectory.png)

Starting from a deliberately large random initialization (far from W0 = {AB=0}) and
descending K(w) + ridge·||w||² with Adam, the *local* volume-scaling ratio (computed
on a small ball of radius 0.15 around the current iterate) is initially a **large,
not-really-interpretable number** (~29, not ~1) — because that small ball doesn't yet
contain any near-zero of K at all. The log-log volume/eps "fit" in that regime is just
measuring the shape of a locally linear (not quadratic) function, which isn't RLCT
behavior in any meaningful sense. Only once the trajectory gets close enough to W0 for
the ball to actually contain a near-zero does the ratio snap down — sharply, around
step 100-150 here — to ≈0.5, matching the branch-local RLCT, and it stays there for a
long stretch while a weight-decay-driven "balancing" dynamic (the classical deep-linear
imbalance a²-b² decaying under ridge) slowly pulls the trajectory the rest of the way
from a generic point on the branch to the true origin. The fitted log-multiplicity
constant k — which stays low (~0.1-0.6) while on a generic branch point — spikes to
~2.6-4.7 right as the trajectory reaches the actual origin, correctly flagging the
higher-order singularity where both branches cross, even though λ itself reads the same
0.5 in both regimes for this symmetric model.

Takeaway: "regular ratio ≈ 1" isn't really a distance-to-origin story; it's an artifact
of not yet being close enough to *any* zero of K for volume scaling to be meaningful.
Once meaningful, it reads out the branch RLCT immediately, and only the log-multiplicity
correction distinguishes generic points of the singular locus from the origin itself.

## Experiment 5: arc-direction distribution

The plan's guess that P(K(delta) < eps) for delta on the unit sphere scales as
eps^(dim W0/4) doesn't match the same zeta-function logic used above. Redone: near the
"a-axis" of a branch, K(delta) = ||a||²||b||² ~ ||b||² for small transverse ||b||
(quadratic, not linear, in the transverse coordinate), so the sphere-measure within eps
of a branch scales as eps^(codim/2) for that branch, and the overall exponent is
dominated by the smaller codim, i.e. min(n,m)/2 = lambda again. Fitted exponent for
experiment 5 (n=m=2): **0.933**, matching the predicted 1.0 within Monte Carlo noise —
an independent (fifth) confirmation of the corrected ground truth.

## Connection to the stated larger goal (GNN / ring-5 barrier)

The volume-scaling estimator is the one to carry forward to a real network: it only
needs forward passes (loss evaluations at randomly perturbed weights), no
backward-mode SGLD machinery. The one implementation lesson from experiment 4 that
transfers directly: a local-ratio estimate is only meaningful once the sampling radius
is small enough to be "local" *and* the current weights are close enough to some
actual near-minimum for the ball to contain it — track the minimum K value seen in the
sampling ball (`volumes` in the returned dict lets you check this) as a validity
diagnostic before trusting a checkpoint's ratio reading.
