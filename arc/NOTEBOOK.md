# Arc-space notebook: reading the RLCT from contact loci

**This is a working notebook, kept chronologically. For the results, start with
[`README.md`](README.md) in this folder.** Sections 1-3 record the starting hypotheses;
some were later revised, and each revision is marked where it applies. Section 4 holds
the results, 4a-4k in the order they were obtained.

The rest of the repo covers volume scaling, Hessian null space (with the multi-restart
min-codim fix), SGLD, DDS, reduced-rank regression (RRR), deep-linear, the Toy Model of
Superposition (TMS), and a pre-registered LLC vs VPD comparison. This notebook covers the
arc-space line: reading the RLCT from contact loci, aimed at a specific gap the repo
measured.

---

## 1. Motivation from existing repo results

Three independent cheap observables all track the FUNCTION and miss part of lambda:

- **Hessian rank**: first-order contact data by construction.
- **DDS** (exps 7-9): orders RRR cells by truth within a fixed width (17-20/20 pairs) but
  does not read lambda's magnitude across widths.
- **VPD** (`param_decomp_compare/rrr/`): in all 28 runs finds exactly r0 components
  regardless of hidden width, so its description length equals the Hessian rank, not
  2*lambda (off by up to 16).

The gap **2*lambda - rank(H)** is the contribution of unused capacity. The starting
hypothesis was that it is **higher-order contact data** (directions where K vanishes to
order > 2). Goal: an estimator or exact computation that reads this gap, validated on
the RRR grid where lambda is known in closed form (Aoyagi-Watanabe, in `rrr_model.py`).

**Revised (4f/4g):** for RRR the gap is not higher-order contact. It is tangent excess
at the min-norm point, where components of the true set meet.

---

## 2. Starting principle: the fiber ideal (revised in 4i)

Write K locally as a sum of squares K = sum_i f_i^2 and let a = (f_1, ..., f_r) be the
fiber ideal. Then (expected, see caveats):

    rlct(K) = lct_C(a) / 2

provided the divisors computing lct(a) have real points. **(See 4h/4i: the safe statement
is rlct = RLCT_R(a)/2, Lin Thm 1.2. The complex version fails for a = (x^2+y^2+z^2) and
for symmetric factorization.)** lct of an ideal is computed by
the complex Mustață / Ein-Lazarsfeld-Mustață contact-locus formula,

    lct(a) = min_p  codim Cont^{>=p}(a) / p ,   codim taken in J_{p-1}.

The original plan treated this as superseding the "real contact conjecture" (counting
real arcs of K), since the ideal version reduces the R vs C issue to one checkable
condition (real centers of the minimizing divisors). **Revised (4i):** that condition
genuinely fails in natural examples, and real arc counting is the robust formulation.

The fiber-ideal RLCT and its factor of 2 are Shaowei Lin's (4h).

Exact confirmations (consistent with both formulations):

| K | ideal a | lct(a) | lct(a)/2 | rlct |
|---|---|---|---|---|
| x^4 = (x^2)^2 | (x^2) | 1/2 | 1/4 | 1/4 |
| x^2+y^2 | (x, y) | 2 | 1 | 1 |
| x^2+y^2+z^2 | (x, y, z) | 3 | 3/2 | 3/2 |
| (ab)^2 | (ab) | 1 | 1/2 | 1/2 |
| a^2 + (b^3)^2 | (a, b^3) | 4/3 | 2/3 | 2/3 |

Note lct_C of the POLYNOMIAL x^2+y^2+z^2 is 1 (wrong for SLT); lct of the IDEAL (x,y,z)
is 3, and halving gives the right answer. Single-square cases like (a^3 - b^2)^2 are
trivially lct(f)/2 and no longer count as tests; genuine tests need several generators
with cross-cancellation, which is exactly RRR.

---

## 3. Conventions

- Jet of order p-1: gamma(t) = c_0 + c_1 t + ... + c_{p-1} t^{p-1}, base point c_0 free.
  J_{p-1}(R^d) has dimension d*p. Normalize codim by p, codim taken in J_{p-1}.
  Sanity check: x^k gives codim ceil(p/k), min over p is 1/k at p = k.
- No factor of 2 between lct and rlct in general. The 2 comes from writing K as squares:
  rlct(f^2) = rlct(f)/2, and rlct(sum f_i^2) = RLCT_R(a)/2 (Lin; see 4h, 4i for when
  the complex lct can replace RLCT_R).
- ||AB||^2 with inner dimension r (zero teacher) is RRR with r0 = 0 and H = r, for
  isotropic inputs. Its lambda is given by Aoyagi-Watanabe. It equals min(n,m)/2 for
  r = 1 but not in general: n = m = r = 2 gives 3/2, not 1. The formula r(n+m-r)/2 is
  RRR's lambda when the teacher has rank r and H = r (no excess capacity).
- Hessian estimator on asymmetric branches: the larger branch (smaller codim) gives the
  right answer; min codim over restarts is the fix (already in repo).

---

## 4. Results so far on the arc-space line

### 4a. Design A: vanishing order along linear arcs (WORKS)
Fit log K(w0 + t u) vs log t; recovers directional order 2k exactly on all test cases,
forward passes only. Better: read coefficients directly from `jet`.

### 4b. Design B: tube-volume hierarchy (FAILS, do not revisit)
c_p from V_p(eps) = vol{ gamma : |k_j(gamma)| <= eps, j < p }. p = 1 is the standard
volume LLC; every p > 1 undershoots monotonically. Exact demo, x^4, p = 2:
V_2(eps) ~ eps^{1/3}, so c_2/2 = 1/6 < 1/4. Tube volume is dominated by base points deep
in W_0, not by high-contact arcs; tube exponent is not codimension.

| case | rlct | c_1/1 | c_2/2 | c_3/3 |
|---|---|---|---|---|
| x^4 | 0.250 | 0.252 | 0.167 | 0.148 |
| a^2 b^2 | 0.500 | 0.437 | 0.305 | 0.272 |
| a^2+b^6 | 0.667 | 0.662 | 0.521 | 0.504 |

### 4c. jax.experimental.jet (VERIFIED as the arc primitive)
`jet(K, (c_0,), ([c_1..c_m],), factorial_scaled=False)` returns exact Taylor coefficients
of K(gamma(t)). The default `factorial_scaled=True` is wrong for nonlinear arcs
(neither Taylor coefficients nor j! times them). Use x64. `jet` is unmaintained with
incomplete primitive coverage; TaylorDiff.jl is the maintained alternative.

### 4d. Design C: jet deepening / lifting (WORKS on demo)
Optimize higher jet coefficients through `jet`. On K = (b - a^2)^2 at 0 with u_1 = (1,0):
linear arc has order 4; deepening recovers u_2 ≈ (0, 0.975) (true lift (0,1)).
This is the lifting question of Treisman, "Rationally generated arcs" (math/0610134).
A purely linear (directional) reading cannot see it. Code: `early/design_C_demo.py`.

### 4e. Design C + D numeric estimator (FAILS, needs redesign)
Witness search + projection + local PCA (`early/contact_dim.py`). Acceptance by S < 1e-14 is
not scale-aware: S vanishes to high order, so x^4 is "accepted" off the variety
(PCA overcounts) and x^2+y^2+z^2 never finds a witness. Jacobian rank is not a fix
(undercounts over R). Lower priority now that section 2 gives an exact algebraic route.

### 4f. RRR via the fiber ideal: orbit counting (WORKS, `arc/rrr_orbit.py`)
Reduce at the min-norm point: lct(a) = r0(N+M-r0) + lct(B'A'), primes = dims minus r0;
the first term is rank(H). Stratify B'(t) by Smith exponents l_1 >= ... >= l_s,
s = min(N',H'); Docampo stratum codim sum l_i(|N'-H'| + 2i - 1); A' rows then vanish
mod t^{p-l_i}. Linear in l, so l_i in {0,p}, and with k = number of killed factors

    codim/p = g(k) = k^2 - k(M' - |N'-H'|) + M' s,   lct(B'A') = min_{k in 0..s} g(k).

- Matches the full Aoyagi-Watanabe theorem (all 4 cases, parity split included) on all
  2754 cells with M,N,H <= 9. The parity split is integer minimization of a quadratic.
- Stratum codims confirmed by Jacobian rank of the jet scheme (5 shapes, p = 1,2,3).
- On the 14 anchor cells the gap 2*lambda - rank(H) is (H-r0)(N-r0), minimizer k = s.
- Minimizing arcs are real (Smith form over R[[t]]), so the real-center condition holds.

**Reframing.** g(k) does not depend on p: the minimum is already reached at p = 1. So
for RRR, lct(a) = codim of the true set Z (largest component through the point). The
gap is not higher-order contact. It is the excess of the Zariski tangent space over the
local dimension of Z at a singular point:

    2*lambda - rank H(w) = dim T_w Z - dim_w Z.

The min-norm (ridge-selected) point is where the components of the variety of complexes
meet, so every Hessian-based observable undercounts there. At a generic point of the
largest component, rank H = 2*lambda. lct = codim is special here. Why is open:
plausibly because the variety of complexes has mild (rational) singularities, but no
reference proves it. It fails in general, e.g. (a, b^3).

### 4g. Step 0: tangent excess confirmed on all 14 anchor cells (`arc/step0_tangent_excess.py`)
Components of Z are labelled by the excess ranks (alpha, beta) of (W1, W2), with
alpha + beta = H' and alpha = k. On component (alpha, beta):
rank H = r0(N+M-r0) + N'alpha + M'beta - alpha*beta. In case 3 the largest component is
(H', 0): every excess W1 row alive, every excess W2 column zero.
- A. Generic points on every component (102): rank J matches the formula, 0 mismatches.
- B. Unregularized GD (init scale 0.05, 0.5, 1.5; 20 restarts) + Gauss-Newton polish
  onto Z: min rank H over restarts = 2*lambda on 14/14 cells at every scale. Runs land
  on several components. The largest one can be a minority: (5,1) hit (4,0) in only
  1-3 of 20 runs, so the min-codim fix needs enough restarts.
- C. Perturb the min-norm point by eps (1e-3, 1e-2, 1e-1), project onto Z, take rank H at
  the landing point: min over 200 = 2*lambda on 14/14 at every eps. So the estimator is
  "Hessian rank at nearby generic true points", not the Hessian at w0. Local PCA of the
  landing cloud is unreliable: it reads the span of the union of components and
  overcounts dim Z.
- Rank calls are clean: smallest kept J singular value >= 2e-2 * sigma_1, largest
  dropped <= 1e-14, and 60/60 runs match their (alpha, beta) formula. Caveat: with a
  second seed, (5,1) at init 0.05 never reached (4,0) in 20 restarts (min 32, not 30).

### 4h. Step 2: Lin's normalization (arXiv:1003.5338, J. Alg. Stat. 8 (2017))
- Lin defines RLCT_R(I) via zeta(z) = int (sum f_i^2)^{-z/2}, and proves
  (Thm 1.2) 2*lambda = RLCT_R(I) whenever c1 sum f_i^2 <= K <= c2 sum f_i^2.
  It is independent of generators (Prop 3.6) and adds over disjoint variables (Prop 3.7).
- So "rlct = (ideal threshold)/2" is Lin's theorem, with the same factor of 2, but for
  the REAL threshold. Remark 3.1 defines the complex one via (sum |f_i|^2)^{-z} and
  notes it "is often different" (citing Saito, arXiv:0707.2308).
- Not in Lin: jets, arcs, contact loci, RRR. What is new here is the orbit / contact
  computation (4f), the tangent-excess explanation of the observable gap (4g), and the
  real-vs-complex analysis (4i).

### 4h'. Literature on the orbit derivation (searched 2026-10-10)
- Pepin Lehalleur & Rimanyi, "Geometry of fibers of the multiplication map of deep
  linear neural networks", arXiv:2411.19920 (v2 Dec 2024). Closest prior work. Quiver
  representations give codim C and the component count theta of {A_1...A_L = B} for any
  depth (Poincare series, a quadratic integer program, an explicit formula). Rank-r
  targets reduce to r = 0 (their Lemmas 4.5-4.6, same reduction as ours). Real points
  are Zariski dense (Thm 3.3). Their Thm 8.6, rlct = C/2, is NOT proved independently:
  they compare C with Aoyagi's formula (Aoyagi, Neural Networks 172 (2024) 106132,
  proved by blow-ups). The Pith review flags this dependency and a gap in Thm 6.1 (lace
  diagrams). No arcs, jets, Smith form, Hessian rank, or min-norm discussion.
- Nothing found that derives RRR's lambda from arc spaces / contact loci / GL(C[[t]])
  orbits. Docampo and Zhu (jet schemes) treat determinantal varieties and W^r_d loci,
  not the multiplication map.
- So our 4f is plausibly the first DIRECT proof of rlct = codim/2 for two layers,
  with the parity case. It is independent of Aoyagi and gives PLR's Thm 8.6 at L = 2.
  (Aoyagi's formula is checked against all of this in 4h''.)

### 4h''. PLR's QIP vs ours, and depth 3 (`arc/deep_linear_orbit.py`)
- PLR Thm 6.1 (QIP): sort d' increasing; e in N^N with sum e = d'_0; minimize
  G = sum_{1<=j<=i<=N} e_i (e_j + d'_j - d'_{j-1}). At N = 2 this is
  e^2 - e(d'_0 + d'_1 - d'_2) + d'_0 d'_1. When M' is the largest dim it equals our g
  under e = s - k. Other orderings agree via permutation invariance.
  Numerically: QIP = PLR Thm 5.5 series = min_k g, in C AND theta, on all 343 vectors
  in [1,7]^3.
- Depth 3, new arc argument. A_3 and A_1 share no vertex, so put both in Smith form
  (lambda, nu); the basis changes act on A_2 by a volume-preserving map, so A_2 stays
  generic and contact becomes entrywise: ord (A_2)_ij >= p - lambda_i - nu_j. Then
  codim/p = F(x,y) = sum x_i(|d3-d2|+2i-1) + sum y_j(|d1-d0|+2j-1)
  + sum_ij max(0, 1-x_i-y_j). As an LP with slacks the constraints are
  [bipartite incidence | I], totally unimodular, so the min is integral: p = 1,
  lct_C = C = min_{a,b} a(|d3-d2|+a) + b(|d1-d0|+b) + (s3-a)(s1-b).
  Real: RLCT_R >= lct_C (real log resolution) and RLCT_R <= codim of the real zero set
  = C (the minimizing strata are real), so rlct = C/2 at depth 3, independent of Aoyagi.
  Assumed, not proved here: Docampo's stratum codims (checked numerically), the
  r0 -> 0 reduction for lct (PLR prove it for codim), and the standard real-vs-complex
  inequality.
  Numerically: QIP = series = min F in (C, theta) on all 625 vectors in [1,5]^4; the LP
  relaxation never goes below the integer min; Jacobian rank of the jet scheme <= p*F
  on all 108 strata tested, with equality on every minimizing stratum. (Over-killed
  strata sit inside bigger components and show lower rank, as expected.)
- Aoyagi (arXiv:2501.12747, which restates the 2024 result and extends it
  to biases, ReLU, softmax). Def 4 / Thm 4 checked by `arc/aoyagi_compare.py`: lambda
  equals our smooth part + C_PLR/2 on every case through depth 5 (about 6,800 cases,
  r = 0 and r > 0).
  Her theta (pole order) = a(l-a)+1. PLR's theta (component count) = C(l,a) in every
  case spot-checked. They agree for l <= 3 and first differ at depth 4, e.g.
  (2,2,2,2,2): 5 vs 6.
- Pole order from arcs (`arc/deep_laces.py`): with the COMPLETE depth-3 stratification
  (Smith A3, Smith A1, entrywise A2), the number of strata with codim = pC grows as
  (p+1)(p+2)/2 when theta = 3, and stays 1 when theta = 1. That is p^(theta-1): arcs
  read the pole order at depth 3, consistent with Aoyagi. Monomial (lace-with-exponent)
  strata miss non-monomial minimizers already at depth 3: (2,2,2,2) gives 3p instead of
  the triangular numbers. So they cannot settle theta at depth 4, where
  (2,2,2,2,2) monomial counts are 6, 20, 42.
- Depth >= 4: open. Pairwise non-adjacent matrices (e.g. A_4 and A_2) can be put in
  Smith form simultaneously, leaving alternating chains D X D Y with generic X, Y. The
  next step needs classifying a matrix arc relative to a lattice t^alpha (parahoric /
  Iwahori double cosets: monomial normal forms = permutation times t-powers). Then
  check whether the resulting LP is still integral.

### 4i. Step 3: the complex formula fails; real arcs do not (`arc/real_vs_complex.py`)
RLCT_R(a) >= lct_C(a). lct_C depends on the chosen representation; RLCT_R does not.
- K = (x^2+y^2+z^2)^2: a = (x^2+y^2+z^2) gives lct_C/2 = 1/2 (minimizing divisor is the
  cone hypersurface, whose only real point is 0). a = m^2 gives 3/4. Monte Carlo
  gives 0.753. So section 2's "rlct = lct_C(a)/2" is FALSE as stated. Correct version:
  rlct = RLCT_R(a)/2 (Lin), which equals lct_C(a)/2 when a minimizing divisor has
  Zariski-dense real points. Real contact loci give 3/2 directly; the original "real
  contact conjecture" is the robust formulation.
- Symmetric factorization W W^T = M* (W n x k; same shape as TMS's W^T W): reduction
  gives V in R^{m x h}, m = n-r0, h = k-r0, with
  lambda_sym = (r0 n - r0(r0-1)/2)/2 + mh/4. Monte Carlo agrees on 6 cells
  (|err| <= 0.03). The real Z is smooth, so the gap mh/2 is genuine quartic contact
  (DDS-type), not tangent excess.
  Over C, V V^T = 0 has totally isotropic components with no real points except 0, and
  codim_C = mh - mq - q(h-q) + q(q+1)/2 (Jacobian-verified). This is < mh/2 for
  (m,h) = (1,>=3), (2,4), (2,6), (3,4), (3,6), (4,8), so lct_C/2 < lambda there.
- Two gap mechanisms, then: (i) tangent excess at intersecting components (RRR), and
  (ii) higher-order transversal contact at smooth real Z with non-real complex
  structure (symmetric). Real arc counting handles both.
- Prior art for the /4 pattern: Drton, Gross, Kosta, Leykin, McCormack, Sullivant,
  Windisch, "Singular Learning Theory for Factor Analysis", arXiv:2511.15419. For
  diagonal truth, k <= p-1, they get lambda = p(k+2)/4 = p/2 (Psi) + pk/4 (Lambda quartic).
  Same count as mh/4. Do not claim lambda_sym as new; the real-vs-complex point is the
  contribution. (A review flags an error in their k = p case.)

### 4j. Step 5: TMS calibrates, no gap (exact Hessian of `tms.population_loss`)
| k-gon | rank H | rank/2 | Chen et al. lambda | notes |
|---|---|---|---|---|
| 5 | 14 | 7.0 | 7.0 | dead column + negative bias: 3 exactly flat dims |
| 5+ | 17 | 8.5 | 8.5 | positive vestigial bias makes w_6 enter linearly |
| 6 | 17 | 8.5 | 8.5 | only the rotation is null |
| 4+, 4++ | 13, 14 | | | 2 negative eigenvalues: saddles (consistent with 4 -> 4+ -> 5) |
| 4 | 8 | | | on ReLU kinks (W_i.W_j = 0 exactly); Hessian meaningless |
Spectral gaps are clean (smallest kept >= 6e-3 relative). Morse-Bott at every analytic
minimum, as predicted. The W^T W quartic mechanism (4i) is present at 5+ (output 6 sees
|w_6|^2), but it is masked by the linear cross terms w_6.w_i. A TMS point with a
genuine gap would need a vestigial column whose cross terms are all inactive while its
own output is active. Not constructed (open).

### 4k. Step 4: Design E, numerical real contact loci (`arc/contact_gn.py`), PARTIAL
Land on Cont^{>=p} by proximal Gauss-Newton on the jet-coefficient map (jet,
factorial_scaled=False), walk a few 1e-2 steps on the locus, then local PCA of
re-projected 1e-4 perturbations gives the real dimension. Three fixes were each
necessary, and each is a real feature of non-reduced systems:
1. Plain GN is not a projection: on (c0^2, 2c0c1) it halves c1 along with c0, so
   tangent directions collapse. Fix: proximal term mu ||c - anchor||^2, mu -> 0.
2. GN drops r * Hess(r), so near a non-reduced zero the anchor pull overshoots into a
   c -> -c two-cycle. Fix: backtracking on the proximal objective.
3. High-contact near-solutions (x^2, p = 3: c0 = -c1^2/(2c2) leaves residual ~ c1^3)
   make the linearized step useless. Fix: predictor + one GN corrector step.
Also: threshold PCA against eps*sqrt(n), not sv[0], and walk off intersections first.

Result (true / estimate): correct on 10/15 cases, namely x^4, x^2+y^2, x^2+y^2+z^2,
(b-a^2)^2, (x^2+y^2+z^2)^2 with f = x^2+y^2+z^2 (0.75, where the complex formula gives
0.5), sym (1,3,0), sym (2,4,1), and RRR (3,2,2,0), (4,2,2,1), (5,2,3,1), (3,3,3,1).
Wrong on (ab)^2 0.375/0.5, a^2+b^6 0.600/0.667, cusp 0.250/0.417, sym (2,2,1) 1.167/1.25.
The failures are one-sided: a dimension overcounted by ~1 at some p >= 2 near a
singular locus drags the min down. Projections onto those loci also often stall at
residual ~1e-12 (cusp p = 2). Diagnosis: deciding the real dimension of a non-reduced
jet scheme near a singular point at finite precision is the 4e problem again. GN
tuning will not fix it; it needs numerical-AG machinery (real witness sets / homotopy
continuation: Bertini, HomotopyContinuation.jl) or exact computation. Runtime is also
minutes per small case, so this is a validation tool, not an LLC estimator for nets.

---

## 5. Next steps (in priority order)

Status of the original list: 0 DONE (4g). 1 DONE: formula derived and verified (4f),
write-up remaining. 2 DONE (4h). 3 DONE (4i): fails for (x^2+y^2+z^2) and symmetric
factorization. 4 PARTIAL (4k). 5 DONE (4j): Morse-Bott, no gap.

1. **Write up RRR.** Orbit derivation of Aoyagi-Watanabe including parity; lct = codim of
   the variety of complexes (why: rational singularities?, check De Concini-Strickland /
   Kempf and whether lct = codim is known there); tangent-excess explanation of the
   repo's Hessian / DDS / VPD gaps (min-norm point = intersection of components).
2. **Practical consequence for the repo.** The min-codim Hessian estimator is exact on RRR
   when restarts are unregularized and land on generic true points. But it can miss
   the largest component (4g caveat), and ridge or small init pushes runs to the
   singular point. Worth an experiment: estimator accuracy vs number of restarts, ridge,
   and init scale on the anchor grid.
3. **Real vs complex, theory.** State the correct principle: rlct = RLCT_R(a)/2 (Lin);
   = lct_C(a)/2 iff some divisor computing lct_C has real points of full dimension in a
   real log resolution. Find the literature statement (Saito 0707.2308 for functions;
   ideals?). Symmetric factorization is the natural ML example.
4. **Design E rescue, if wanted.** Replace GN+PCA with HomotopyContinuation.jl real
   witness sets on the jet schemes, or exact Groebner dimension for complex dims plus a
   real-points check. Target the 4 failing cases first.
5. **TMS with a genuine gap** (open, 4j): look for vestigial structures where the
   W^T W quartic mechanism is not masked, e.g. r > 2 with a vestigial column orthogonal
   to the k-gon plane. A quick analysis suggests these are saddles, not minima; check.

---

## 6. DDS correspondence (for writeup)

A dead direction of KL order k (Shirodkar, arXiv:2606.05957, Thm 3) is a linear arc in
Cont^{>=2k} minus Cont^{>=2k+1}. Under Thm 3's hypotheses DDS is the linear-arc
restriction of the contact formula; its additivity 1/(2k) + m/2 is the Newton-polyhedron
sum of weights. The arc / ideal formulation supplies what Thm 3 defers: multiplicity
(number of minimizing components), branch selection (min over components), and the
Jacobian exponents h_i Shirodkar takes as a prior choice. DDS breaks at Newton-degenerate
points and crossing points. CORRECTED (4f/4g): on RRR, unused capacity is NOT
higher-order contact. It is tangent excess at a crossing point (the min-norm point),
exactly where DDS breaks. Higher-order (quartic) contact is the symmetric-factorization
mechanism (4i), where DDS's additive 1/4-per-direction count is right.

---

## 7. Files (paths relative to `arc/`)

| file | purpose | status |
|---|---|---|
| `early/arc_hierarchy.py` | Designs A and B | B is a negative result (4b) |
| `early/x4_check.py` | exact demo of Design B failure | done |
| `early/jet_test.py` | pins the `jet` convention | done |
| `early/design_C_demo.py` | jet lifting on (b - a^2)^2 | done (4d) |
| `early/real_contact_check.py` | exact order-arithmetic table | done |
| `early/contact_dim.py` | numeric C + D estimator | broken (4e); replaced by `contact_gn.py` |
| `rrr_orbit.py` | RRR lct by orbit counting vs Aoyagi-Watanabe | done (4f) |
| `step0_tangent_excess.py` | tangent-excess test on the 14 anchor cells | done (4g) |
| `real_vs_complex.py` | complex fiber-ideal failures, symmetric lambda | done (4i) |
| `deep_linear_orbit.py` | PLR QIP vs ours; depth-3 arc proof | done (4h'') |
| `aoyagi_compare.py` | Aoyagi's deep-linear formula vs PLR | done (4h'') |
| `deep_laces.py` | monomial strata; depth-3 pole order | done (4h'') |
| `contact_gn.py` | Design E: numerical real contact loci | partial (4k) |

---

## 8. References

- Mustață, "Singularities of pairs via jet schemes", arXiv:math/0102201
- Ein, Lazarsfeld, Mustață, "Contact loci in arc spaces", arXiv:math/0303268
- Docampo, "Arcs on determinantal varieties", Trans. AMS 365 (2013), arXiv:1011.1930
- Yuen, "Jet schemes of determinantal varieties", arXiv:math/0608632
- S. Lin, "Ideal-Theoretic Strategies for Asymptotic Approximation of Marginal
  Likelihood Integrals", J. Algebraic Statistics 8 (2017), arXiv:1003.5338
  (from his thesis, "Algebraic Methods for Evaluating Integrals in Bayesian Statistics",
  UC Berkeley, 2011)
- M. Saito, "On real log canonical thresholds", arXiv:0707.2308
- Aoyagi, Watanabe, "Stochastic complexities of reduced rank regression in Bayesian
  estimation", Neural Networks 18(7), 2005
- M. Aoyagi, "Consideration on the learning efficiency of multiple-layered neural
  networks with linear units", Neural Networks 172 (2024) 106132
- M. Aoyagi, "Singular leaning coefficients and efficiency in learning theory",
  arXiv:2501.12747 (restates the 2024 deep-linear formula)
- S. Pepin Lehalleur, R. Rimanyi, "Geometry of fibers of the multiplication map of deep
  linear neural networks", arXiv:2411.19920
- M. Drton, E. Gross, D. Kosta, A. Leykin, A. McCormack, S. Sullivant, D. Windisch,
  "Singular Learning Theory for Factor Analysis", arXiv:2511.15419
- Shirodkar, "Dead Directions: Geometric Singular Learning", arXiv:2606.05957
- T. P. Shirodkar, P. J. Narayanan, "Dead-Direction Signatures: A Cheap Spectral Reading
  of Singular Complexity", arXiv:2606.21158
- T. P. Shirodkar, "Measuring Dead Directions: Decomposing and Classifying Singular
  Structure off Canonical Alignment", arXiv:2607.00603
- Chen et al., Toy Model of Superposition phase transitions, arXiv:2310.06301
- Lau et al., "The Local Learning Coefficient", arXiv:2308.12108
- Watanabe, "A Widely Applicable Bayesian Information Criterion", arXiv:1208.6338
- Treisman, "Rationally generated arcs", arXiv:math/0610134
- Bettencourt, Johnson, Duvenaud, Taylor-mode AD in JAX
