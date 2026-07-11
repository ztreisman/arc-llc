# arc-llc

Comparing two estimators of the RLCT (real log canonical threshold) / local learning
coefficient (LLC) on a small, analytically tractable singular statistical model:

1. **Hessian null-space estimator** (geometric): find the zero-curvature directions of
   the loss K at a point w*, measure their dimension, infer λ from the codimension.
2. **SGLD estimator** (devinterp-style): sample from a tempered, localized posterior at
   inverse temperature β = 1/log(n) and read λ off the slope of the free energy vs
   log(n).

Both are validated against a **volume-scaling estimator** and, more recently, against
**Dead-Direction Signatures** (Shirodkar & Narayanan, arXiv:2606.21158) — a family of
cheap closed-form spectral reads of a network's activations and per-sample-gradient
Fisher-Gram — plus ground truth on rank-1 matrix factorization with true rank 0
(K(A,B) = ||AB||_F², true distribution = zero matrix), where λ is known analytically:
λ = min(n,m)/2 for r=1.

The geometric estimators are motivated by Mustață's jet-scheme characterization of log canonical thresholds (arXiv:math/0102201): the RLCT is determined by dimensions of contact loci in arc space, and the Hessian null space computes the first-order contact data. The longer-term question is whether higher-order jet/contact statistics yield cheaper or more informative LLC estimators than sampling-based methods.

See [`RESULTS.md`](RESULTS.md) for the full write-up — headline numbers, a correction to
an internal inconsistency in the original spec's ground-truth formula, several
implementation bugs found and fixed along the way (a log-multiplicity bias in the
volume estimator, an unstable SGLD step schedule, branch-dependence in the Hessian
estimator and a multi-restart fix for it), discussion of experiments 4-6, and the DDS
validation (experiments 7-8: the core rate/structural-correlation claim holds exactly;
cross-cell magnitude-tracking is a harder, more protocol-sensitive story, reported
honestly rather than smoothed over).

## Quickstart

```bash
pip install torch numpy scipy matplotlib
python3 main.py
```

Runs all 8 experiments (~7-9 minutes), prints result tables, and writes:
- `plots/*.png` — volume-scaling and SGLD free-energy fits, the training-trajectory
  ratio plot, and the arc-direction distribution
- `results/summary.json`, `results/tables.md` — machine-readable and Markdown results

## Files

| File | Contents |
|---|---|
| `model.py` | The K(w) = \|\|AB\|\|² loss (numpy + torch), gradient/Hessian utilities, analytic ground truth |
| `estimators.py` | `volume_scaling_estimator`, `hessian_branch_estimator`, `hessian_multi_restart_estimator`, `sgld_llc_estimator`, `arc_direction_estimator` |
| `dds.py` | Dead-Direction Signatures: activation/Fisher-Gram spectral observables |
| `rrr_model.py` | Aoyagi-Watanabe reduced-rank-regression model (general truth rank r0), external closed-form RLCT |
| `experiments.py` | Experiments 1-8 and table formatting |
| `plots.py` | Plotting utilities |
| `main.py` | Runs everything, saves plots + results |
| `RESULTS.md` | Full write-up of findings, corrections, and caveats |
| `arc_llc_context.md` | Original project spec this implements |

## Headline result

Volume-scaling and Hessian estimators agree with corrected ground truth to within a few percent across all cases. SGLD is accurate on the singular models but overshoots the regular case by ~20% (likely under-equilibration at higher λ; see RESULTS.md).

| Experiment | d | true λ | Volume scaling | Hessian null space | SGLD |
|---|---|---|---|---|---|
| 1 (n=m=1) | 2 | 0.500 | 0.458 | 0.500 | 0.525 |
| 2 (n=m=2) | 4 | 1.000 | 0.983 | 1.000 | 1.086 |
| 3 (regular, K=\|\|w\|\|²) | 4 | 2.000 | 1.926 | 2.000 | 2.451 |

Experiment 6 (n=1, m=4, asymmetric) shows the naive Hessian estimator is
branch-dependent — a single run can overstate λ by up to 4x — and validates a
multi-restart min-codim fix that recovers the exact value. Details in
[`RESULTS.md`](RESULTS.md).
