# Pre-registration: LLC vs parameter-decomposition description length on TMS 5→2

**Drafted 2026-09-30, before** (i) any SGLD estimate at c = 5 and (ii) opening any output
of the param-decomp run `~/tms-5-2-run/data/runs/p-e78416be` (metrics, logs, checkpoints).
Nothing below may be edited after either happens; amendments go in a dated section at
the end.

## Object

- Target: param-decomp `tms_5-2.yaml`, regenerated deterministically with
  `pretrain_tms_target` (seed 0, 5000 steps) → `target_5-2.npz`.
  - f(x) = ReLU(WᵀWx + b), W ∈ ℝ^{2×5}, tied weights, d = 15.
  - Data: each feature independently active w.p. 0.05, value U[0,1].
- Loss: `tms.population_loss_bernoulli`. Exact for ≤ 1 active feature, quasi-MC
  otherwise; it agrees with brute-force MC of their generator to 0.14σ.

## Already observed (not predictions)

- At the target under this loss, ‖∇L‖ = 5.1e-4. Polishing moves w by 0.018 and reaches an
  exact critical point (‖∇L‖ = 1.6e-10): a regular pentagon with l = 1.1346,
  b = −0.2463, and angle gaps of 72.000°.
- The Hessian there has rank 14 of 15: one zero mode (the SO(2) rotation) plus 14 positive
  eigenvalues in [3.6e-3, 6.2e-2]. The Morse–Bott count gives λ = 7. This matches
  Chen et al.'s Appendix H theorem for the c = 5 pentagon under the one-hot loss,
  λ = (3c − 1)/2 = 7.

## Counting rule (confirmed by Z.T., 2026-09-30: Rule A is the tested rule; DL_raw reported descriptively)

**Rule A (per-feature count).** A feature is *represented* if it has an alive component
at both `linear1` and `linear2`. "Alive" means lower-leaky CI on the single-feature
probe > 0.5, the threshold `identity_ci_error` uses.

DL_A = 3·k − 1, where k is the number of represented features. The per-feature 3 is the
2 hidden coordinates of the feature direction plus 1 bias entry. The −1 is the global
rotation gauge.

For comparison only (not the tested rule), the raw count of SPD parameters in use:
- each alive rank-1 component in a 2×5 or 5×2 matrix has 2 + 5 − 1 = 6 parameters
  (after the V/U scale gauge);
- plus the 5 undecomposed bias entries.

DL_raw = 6·(alive₁ + alive₂) + 5. This is 65 at the identity pattern. It is expected to
disagree with 2λ, and recording it shows the rule matters.

## Predictions

- **P1 (LLC).** Localized SGLD at the polished target under the Bernoulli loss, using the
  exp-10 protocol (γ = 0.1, 10 chains, discard escaping chains): λ̂ decreases toward 7 as
  n goes 5·10³ → 5·10⁴ → 5·10⁵, and is within one chain-std of 7 at n = 5·10⁵.
- **P2 (decomposition).** At the final checkpoint the decomposition recovers the identity
  pattern: 5 alive components per site, `identity_ci_error` = 0 at both sites. Hence
  k = 5 and DL_A = 14 = 2λ.
- **P3 (sweep; the existing run alone can't test it).**
  - **Sweep design, fixed now.** Scale the ImportanceMinimalityLoss `coeff` and its
    `frequency.coeff` together, keeping their 2:1 ratio, by factors
    {1/27, 1/9, 1/3, 1, 3, 9, 27} around the config's (8e-4, 4e-4). That's 7 runs,
    seed 0, everything else exactly as in `tms_5-2.yaml`. The ×1 run is a fresh rerun,
    not the existing `p-e78416be`.
  - **Recovered** means `identity_ci_error` = 0 at both sites at the final step.
  - **Prediction.**
    - Every recovered run has k = 5, so DL_A = 14 = 2λ (this follows from the definitions).
    - The substantive part is the shape: recovered runs form one contiguous range of
      coefficients that includes ×1.
    - Below that range (weak minimality): k stays 5, so DL_A = 14, but recovery fails
      through *extra* alive components (more than 5 alive at a site). DL_A is blind to
      this by construction (k ≤ 5 caps DL_A at 14); DL_raw and the per-site alive counts
      record it.
    - Above it (strong minimality): features are dropped, k < 5, so DL_A < 14.
  - **The non-trivial question P3 asks.** Does the edge of the recovered range sit where
    the decomposition's minimality pressure stops paying for a 15th parameter? This is
    read off the alive-count curve and is reported descriptively, with no threshold set in
    advance.

## What each outcome means

- P1 fails: the SGLD machinery does not transfer to this distribution, or the critical
  point is degenerate beyond second order. Investigate before any comparison.
- P2 fails: report alive counts and DL_A as observed. A mismatch with 2λ = 14 at a run
  that the paper's own metric scores as successful would itself be a result.
- P1 and P2 both hold: TMS is a **calibration** case. At a Morse–Bott point any
  description length that counts non-degenerate parameters agrees with λ and with
  Hessian rank. The discriminating test is RRR with H > r0, where 2λ exceeds Hessian
  rank (e.g. H = 2, r0 = 1: rank 14 vs 2λ = 18).

---

## Amendment 1 (2026-09-30, after the freeze; recorded before any SGLD result or sweep output was read)

- **Threshold error in the frozen text.** Rule A says "alive" means CI > 0.5, "the
  threshold `identity_ci_error` uses". The parenthetical is wrong: param-decomp calls
  `identity_ci_error(ci, tolerance=0.1)`, which counts off-diagonal CI > 0.1 and
  on-diagonal CI < 0.9 as errors. The tested rule is unchanged: alive = CI > 0.5, as
  written. "Recovered" in P2/P3 is the logged metric as written (tol 0.1). Both are
  reported.
- **P1 restarted.** The first P1 process was killed after 24 min with no output, because
  torch thread oversubscription starved the machine. It was restarted unchanged except for
  `torch.set_num_threads(2)`. No estimates had been produced.
- **P2 already read.** The existing run `p-e78416be` was opened after the freeze and
  before this amendment. P2 is evaluated on it as pre-registered, below.

## Terminology amendment (2026-10-02; no protocol change)

The method was called "SPD" above. The param-decomp trainer and configs used here
implement **VPD** (adVersarial Parameter Decomposition, Bushnaq et al. 2026). VPD is SPD
plus an adversarial reconstruction loss (`MergedStochasticSubsetPPGDReconLoss`) and
frequency minimality, both of which are in the configs run. Read "SPD" as "VPD"
throughout. Predictions, definitions and runs are unchanged.
