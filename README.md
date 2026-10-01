# arc-llc

Estimators of the RLCT / local learning coefficient (LLC) λ, tested on singular models
where λ is known in closed form:

1. **Hessian null space** (geometric): codimension of the zero-curvature directions of the
   loss at a point on the singular set.
2. **SGLD** (devinterp-style): localized tempered-posterior sampling at β = 1/log n.
3. **Volume scaling**: Vol{K ≤ ε} ~ ε^λ |log ε|^{m−1}, fit with the log-multiplicity term.
4. **Dead-Direction Signatures** (Shirodkar & Narayanan, arXiv:2606.21158): closed-form
   spectral reads of activations and the per-sample Fisher-Gram.

Testbeds: rank-1 matrix factorization (λ = min(n,m)/2), the Aoyagi-Watanabe
reduced-rank-regression grid, an L-layer deep-linear bridge, and the **Toy Model of
Superposition**, where we reproduce the k-gon critical points, LLCs, and phase
transitions of Chen et al. (arXiv:2310.06301).

The geometric estimators are motivated by Mustață's jet-scheme characterization of log
canonical thresholds (arXiv:math/0102201): the RLCT is determined by dimensions of contact
loci in arc space, and the Hessian null space is the first-order contact data. The
longer-term question is whether higher-order jet/contact statistics give cheaper or more
informative LLC estimators than sampling.

See [`RESULTS.md`](RESULTS.md) for the full write-up.

## Highlights

- **Ground truth recovered** on matrix factorization by volume scaling and the Hessian
  estimator (within 2-8%; the Hessian is exact) and by SGLD (5-9% on the singular cases).
  A multi-restart min-codim fix makes the Hessian estimator exact on asymmetric branches,
  where a single read is up to 4× off (experiments 1-6).
- **Toy Model of Superposition** (experiment 10). Exact closed-form population loss.
  - k-gon losses and prior terms match the paper to 5 digits.
  - SGLD LLC estimates converge to the theoretical 7 / 8.5 / 8.5 as n grows (6.90, 8.41,
    8.46 at n = 5·10⁵), showing the overshoot in the paper's own Table K.1 is a finite-n
    effect.
  - The 5→6 Bayesian transition comes out at n_cr = 601 (paper: 601).
  - SGD runs from a 4-gon spend 95% of checkpoints on k-gon plateaus, with λ̂ rising as
    loss falls.
- **DDS** (experiments 7-9). The rate claim holds exactly (ρ = 1.000). Across the 14-cell
  RRR grid, read from a calibrated SGLD posterior ensemble whose own LLC tracks the closed
  form at ρ = 0.999, DDS orders cells by truth rank within a fixed width (17-20/20 pairs)
  but does not read λ's magnitude across widths. The apparent σ_min success (ρ = 0.86) is
  a layer-rank artifact.

![tms](plots/exp10_tms.png)

## Quickstart

```bash
pip install torch numpy scipy matplotlib
python3 main.py
```

Runs all 10 experiments (~10 min on CPU), prints result tables, and writes `plots/*.png`,
`results/summary.json`, `results/tables.md`.

## Files

| File | Contents |
|---|---|
| `model.py` | K(w) = ‖AB‖² (numpy + torch), gradient/Hessian utilities, ground truth |
| `estimators.py` | Volume scaling, Hessian branch / multi-restart, SGLD, arc-direction estimators |
| `dds.py` | Dead-Direction Signatures: activation and Fisher-Gram spectral observables |
| `rrr_model.py` | Reduced-rank regression (truth rank r0), Aoyagi-Watanabe closed form, batched localized SGLD |
| `deep_linear.py` | L-layer deep-linear bridge for the DDS counting identity |
| `tms.py` | Toy Model of Superposition: exact population loss, k-gons, SGLD LLC, batched SGD |
| `experiments.py` | Experiments 1-10 |
| `plots.py`, `main.py` | Plotting; run everything |
| `RESULTS.md` | Write-up |
| `notes/dds_arc.tex` | Note relating DDS to the arc-space picture |
| `arc_llc_context.md` | Original project spec |
