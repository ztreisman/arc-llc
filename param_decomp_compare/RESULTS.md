# LLC vs parameter-decomposition description length: TMS 5→2

Predictions were registered in [`PREREGISTRATION.md`](PREREGISTRATION.md), frozen
2026-09-30 21:37, with one dated amendment; hashes are in `PREREGISTRATION.sha256`.
Sections marked *exploratory* were not pre-registered.

## Matching check: both codebases describe the same object

The param-decomp target (`tms_5-2.yaml`) is f(x) = ReLU(WᵀWx + b) with W ∈ ℝ^{2×5} and
tied weights, the same functional form as `tms.py`. The data differ. In param-decomp each
feature is independently active with p = 0.05 and value U[0,1]. In Chen et al. exactly
one feature is active.

- **Under the one-hot loss, the target is not critical** (‖∇L‖ = 1.9e-2). It is a regular
  pentagon at l = 1.137 instead of the one-hot optimum l* = 1.17046, which is Chen et al.'s
  5-gon value.
- **Under its own distribution, it is critical.** `tms.population_loss_bernoulli` is exact
  for ≤ 1 active feature and quasi-MC for ≥ 2, and matches brute-force MC of their sampler
  to 0.14σ. Under it, ‖∇L‖ = 5e-4. Polishing moves w by 0.018 and reaches an exact
  regular pentagon: angle gaps 72.000°, l = 1.1346, b = −0.2463.
- **Tied Hessian:** rank 14 of 15. The one zero mode is SO(2), so the critical point is
  Morse–Bott and λ = 7. This matches Chen et al.'s Appendix H theorem for the c = 5
  pentagon (λ = (3c − 1)/2) and the hypothesis 2λ = 3k − 1.
- **The regenerated target equals the run's restored target** to 5e-7 (float32
  non-determinism).

**P1** (SGLD λ̂ at the polished target under the Bernoulli loss; 10 chains, all kept):

| n | 5·10³ | 5·10⁴ | 5·10⁵ |
|---|---|---|---|
| λ̂ ± chain-std | 6.62 ± 0.37 | 8.75 ± 3.01 | **7.38 ± 0.69** |

- **Passes the pre-registered criterion:** within one chain-std of 7 at n = 5·10⁵.
- **Fails the predicted monotone decrease toward 7:** the estimate starts *below* 7, unlike
  the one-hot case.
- The n = 5·10⁴ spread is ~8× larger than at n = 5·10³. *Exploratory diagnosis*
  (`p1_diag.py`, same settings and seed, per-chain output):
  - **One outlier chain.** Chain 7 reads λ̂ = 17.1 and has left the pentagon basin
    *upward*: nβ(L − L₀) sits at ~16 from the end of burn-in and climbs to ~23. The
    paper's chain filter, reused here, only discards chains that fall below L₀, so it
    cannot catch an upward escape.
  - **The other nine chains** give 6.7-9.1, mean 7.79 ± 0.75.
  - **The large distances |w − w*| (1.3-3.1) are mostly expected motion along the
    rotation orbit**, which γ = 0.1 barely confines (displacement ~√(1/γ) ≈ 3). A
    rotation-invariant distance is needed to tell an escape from orbit motion.
  - **Net:** λ̂ is consistent with 7 at large n, but noisier than the one-hot c = 6 case
    (8.46 ± 0.58 vs theory 8.5). A filter for upward escapes, e.g. a stationarity test on
    the loss trace, is the obvious protocol fix for any future run.

## P2: the existing run (importance-minimality coeff 8e-4): **fails as stated**

| Site | Alive (CI > 0.5) | `identity_ci_error` (tol 0.1) |
|---|---|---|
| `linear1` | 5, one per feature | 0 |
| `linear2` | **4** | 2 |

At `linear2`, features 1 and 4 share a single rank-1 component. They sit at −42° and
173°, 145° apart: nearly antipodal and not adjacent on the pentagon. The shared
component's hidden direction reads −1.93 on feature 1 and +1.91 on feature 4. Its output
vector is −0.67 on output 1 and +0.68 on output 4. Each feature therefore drives its own
output positive, and the ReLU (bias −0.25) removes the wrong-sign half. Masked to the alive
components, single-feature reconstruction error is the same for the shared pair
(0.065, 0.052) as for the other features (0.049-0.058). The ~0.05 floor is the same for
every feature and comes from hard-thresholding the CI.

Rule A still gives k = 5 and DL_A = 14 = 2λ, because every feature has an alive
component at both sites. **The pre-registered rule cannot see component sharing.**

## P3: importance-minimality sweep

Seven runs, coefficients ×{1/27, …, 27} around (8e-4, 4e-4), seed 0, everything else as
in the config.

| Coefficient | Alive L1 / L2 | Recovered | k | DL_A | DL_raw |
|---|---|---|---|---|---|
| ×1/27 | 8 / 8 | no (7 / 15 errors) | 5 | 14 | 101 |
| ×1/9 | 5 / 7 | no (1 / 11) | 5 | 14 | 77 |
| **×1/3** | **5 / 5** | **yes** | 5 | 14 | **65** |
| ×1 (rerun) | 5 / 4 | no (0 / 2) | 5 | 14 | 59 |
| ×3 | 5 / 4 | no (0 / 2) | 5 | 14 | 59 |
| ×9 | 2 / 4 | no (5 / 6) | 2 | 5 | 41 |
| ×27 | 0 / 0 | no (5 / 5) | 0 | −1 | 5 |

Against the pre-registered prediction:

- **"Recovered runs form one contiguous range that includes ×1": fails.** Only ×1/3
  recovers the identity. The config's default (×1) does not, and neither does a fresh
  rerun of it.
- **"Below the range, k = 5 and recovery fails through extra components": holds**
  (×1/9, ×1/27).
- **"Above it, features are dropped (k < 5)": holds only from ×9 up.** Between ×1/3 and
  ×9 there is a regime the prediction did not anticipate. At ×1 and ×3 the decomposition
  merges features 1 and 4 into one `linear2` component: the same pair in the original run,
  the rerun, and ×3. That gives a description *shorter* than the ground truth
  (DL_raw 59 < 65) with k still 5.

DL_A = 2λ = 14 holds at every run that keeps all five features (×1/27 through ×3). That
follows from the definition of Rule A, so it carries no evidence. The rule is blind to
splitting (below ×1/3) and to merging (×1, ×3). DL_raw follows both, and has a plateau at
59 across ×1-×3, the setting VPD's minimality actually prefers.

## Exploratory: VPD decomposes the untied model, whose λ is 10.5

VPD decomposes `linear1` and `linear2` as independent matrices. The model it describes is
therefore the *untied* f = ReLU(W₂W₁x + b), with 25 parameters and a GL(2) gauge. At the
target (W₂ = W₁ᵀ):

- the untied loss is also critical (‖∇L‖ = 1.5e-6);
- the Hessian has rank 21 of 25: four zero modes (exactly GL(2)), no negative directions.

So this is again Morse–Bott, with **λ_untied = 10.5**. The tied rule 3k − 1 counts the
wrong model for VPD. The untied analogue is 5k − 4 = 21, from 2 + 2 hidden coordinates
plus a bias per feature, minus the 4-dimensional gauge. That rule was formed after seeing
the data and is not tested here.

## What this says about LLC as a description length for parameter decomposition

1. **TMS 5→2 is a calibration case, as expected.** The relevant critical points (tied and
   untied) are Morse–Bott, so λ is half the Hessian rank. Any rule that counts
   non-degenerate parameters agrees with it by construction, and agreement here is
   evidence of nothing beyond correct counting.
2. **The useful finding is a mismatch in kind, not in number.** At its default
   coefficient VPD prefers a description with fewer components than the ground truth.
   It merges two near-antipodal features through the ReLU, which is valid on the
   single-feature inputs VPD's minimality is effectively measured on. λ is a property of
   the target and does not change across the sweep. So λ cannot be the quantity VPD's
   minimality is tracking; at most it is a fixed reference that one regime of the sweep
   (×1/3) happens to sit at. Whether the merged description is "wrong" depends on
   whether rare multi-feature inputs (~2% of data at p = 0.05) are reconstructed. This
   is the next check.
3. **The discriminating testbed is reduced-rank regression with H > r0.** There 2λ
   exceeds the Hessian rank: at the minimum-norm solution, H = 2, r0 = 1 gives rank 14
   vs 2λ = 18, and the gap grows to 16 at H = 5, r0 = 1. A decomposition's description
   length can then track λ, track the Hessian rank, or track neither. Note that the
   Hessian rank is a **lower** bound on 2λ (degenerate directions contribute fractional
   amounts), so it undercounts there rather than overcounts.
   **Done, in [`rrr/RESULTS.md`](rrr/RESULTS.md) (pre-registered, 28 runs).** VPD tracks
   the Hessian rank exactly in every run. It finds r0 components whatever the excess
   capacity, while 2λ grows by N − r0 per extra hidden unit.

## Files

- `export_target.py`: regenerate the target weights (param-decomp venv).
- `check_target.py`: matching check and tied Hessian.
- `extract_ci.py`: restore a run and read the single-feature CI (param-decomp venv).
- `analyze_run.py`: Rule A / DL_raw.
- `p1_llc.py`: P1.
- Extracted CI matrices and numeric outputs (`*.npz`, `*.json`) are not committed. Regenerate them with `extract_ci.py` from the run directories.
- Sweep configs and launcher: `~/tms-5-2-sweep/`.
