# Pre-registration: does SPD's description length track λ or Hessian rank?

**Drafted 2026-10-01, before any SPD run on these targets (including smoke tests).**
Amendments go in dated sections at the end. Hashes are in `PREREGISTRATION.sha256`.

## Why this testbed

On TMS 5→2 (`../RESULTS.md`) the critical points are Morse–Bott, so λ = (Hessian rank)/2
and no comparison can separate the two. In reduced-rank regression (RRR) with excess
capacity (H > r0), 2λ exceeds the Hessian rank. The quartic excess directions contribute
to λ but not to the Hessian. The question is which of the two SPD's decomposition follows.

## Targets

Two-layer linear network y = W₂W₁x with W₁ ∈ ℝ^{H×M}, W₂ ∈ ℝ^{N×H}, M = 10, N = 5, no bias,
no nonlinearity, x ~ N(0, I_M). Teacher M* = diag(1,…,1,0,…) of rank r0.

The target weights are the exact minimum-norm solution: identity blocks on the first r0
hidden units, exact zeros elsewhere. This is the point at which experiment 8's SGLD
recovered the Aoyagi-Watanabe λ (ρ = 0.999).

**Cells:** all (H, r0) with H ∈ {1,…,5}, r0 ∈ {0,…,min(N, H)}, satisfying Case-3 validity
N + H < M + r0. That gives **19 cells**.
- The 4 cells with r0 = 0 are matrix factorization in network form: E‖W₂W₁x‖² = ‖W₂W₁‖²_F
  for isotropic x, so they are the K = ‖AB‖² model of experiments 1-6, with A = W₂, B = W₁.
  At H = 1 their λ agrees with min(n, m)/2.

**Known quantities, already computed:**
- 2λ = NH + (M − H)r0, which increases with H at fixed r0 (slope N − r0 > 0).
- Hessian rank at the target = r0(M + N − r0), independent of H. This is verified
  numerically for the 14 r0 ≥ 1 cells with H ≥ 2; it is 0 for r0 = 0.

## SPD protocol (fixed now)

- param-decomp's JAX trainer, run on this target through a driver
  (`run_rrr_spd.py`). The driver swaps the TMS forward for the linear, bias-free
  M → H → N forward and samples Gaussian inputs.
- Everything else is taken from `tms_5-2.yaml`: loss list, CI function (layerwise MLP,
  hidden 50), optimizers, schedules, 20k steps, batch 4096, seed 0. C = 20 per site
  (≥ H at every cell). The decomposition is untied: `linear1` = W₁, `linear2` = W₂.
- Two importance-minimality settings: the config default (8e-4, freq 4e-4) and ×1/3
  (the setting that recovered TMS ground truth). That gives **38 runs**.
- **Alive component:** lower-leaky CI > 0.5 on at least 50% of 4096 fresh Gaussian probe
  inputs.
- **Faithfulness check:** relative error of the summed components,
  ‖Σ_c V_cU_c − W‖_F / ‖W‖_F per site, and the masked-output recon MSE relative to output
  variance. A run counts as **converged** if every site with ‖W‖ > 0 has relative error
  < 5%. Runs that fail are reported and excluded from P1-P3.

## Predictions

- **P1 (count).** In every converged run, both sites have exactly r0 alive components
  (0 for r0 = 0).
- **P2 (H-independence).** At fixed r0 the alive counts do not change with H. Across all
  H > r0 cells, the excess capacity H − r0 produces **no** alive components.
- **P3 (which quantity SPD tracks).** Define the function-level description length
  DL_fn = k(M + N − k), with k = alive count at `linear2` (the parameter count of a
  rank-k N × M map). Then DL_fn = Hessian rank exactly in every converged run, and
  DL_fn ≠ 2λ in every cell with H > r0. Within each r0 ≥ 1, the H-ordered pairs (where
  2λ strictly increases) give DL_fn ties in all of them. For r0 = 0, DL_fn = 0 while
  2λ = 5, 10, 15, 20.
- **Descriptive only:** DL_raw = Σ_sites (alive)·(d_in + d_out − 1). It depends on H
  through the site dimensions, so it is not used as a test.

## What each outcome means

- **P1-P3 hold:** SPD's description length is a function-level quantity. It counts the
  rank of the computed map, which the Hessian also sees, and is blind to the singular
  geometry of unused capacity that λ measures. λ and SPD minimality are then different
  quantities, and λ is not a candidate MDL target for SPD in this sense. The prediction
  comes from theory (dense inputs give minimality nothing to separate), and the run
  tests whether SPD's actual optimizer realizes it.
- **Alive count grows with H at fixed r0** (excess directions kept alive): SPD *does*
  respond to excess capacity. Compare the H-slope of the counts with the λ slope
  (N − r0)/2.
- **Count ≠ r0 but H-independent** (e.g. SPD splits the rank-r0 map into more than r0
  components): SPD tracks neither λ nor Hessian rank. Report what it tracks.

---

## Amendment 1 (2026-10-01, after the freeze; before any production run finished)

- **Smoke tests.** Run after the freeze, for engineering only: one 400-step run of
  H = 2, r0 = 1, ×1, deleted afterwards, plus a check that the patched forward equals
  x W₁ᵀ W₂ᵀ exactly. Their outputs are not used.
- **Recon is defined concretely.** "Masked-output recon MSE relative to output variance"
  is implemented as hard masking: y_hat = x (V₁U₁)[alive] (V₂U₂)[alive], i.e. only the
  alive components at each site, compared with the target output on the 4096 probe
  inputs. For r0 = 0 the target output is 0, so the raw mean of y_hat² is reported.
- **Production runs:** `~/rrr-spd/run_all.sh` (38 jobs listed in `~/rrr-spd/jobs.txt`).

## Amendment 2 (2026-10-01; after the first 9 jobs finished, 5 of which succeeded)

- **r0 = 0 cells (8 runs) cannot run.** param-decomp asserts ‖W‖² > 0 per site
  ("faithfulness needs finite positive ‖W_s‖²"), and the exact min-norm target for a zero
  teacher is W₁ = W₂ = 0. The target is not perturbed to get around this, because that
  would be a different experiment. These cells are reported as **"SPD undefined: there is
  nothing to decompose"**, with an implied description length of 0 against
  2λ = 5, 10, 15, 20. P1-P3 are evaluated on the r0 ≥ 1 cells only.
- **H = 1, r0 = 1 (2 runs) cannot run.** The trainer raises a sharding error at hidden
  width 1. This cell has H = r0, so it has no excess capacity and is not a discriminating
  cell. It is excluded rather than patched.
- **Remaining:** 14 cells × 2 coefficients = 28 runs, all H ≥ 2 and r0 ≥ 1.
