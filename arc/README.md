# arc/: contact-locus (arc-space) reading of the RLCT

Working notes and full results: [`../arc_llc_contact_hierarchy_context.md`](../arc_llc_contact_hierarchy_context.md)
(sections 4f-4k). Run everything with the repo venv (`.venv/Scripts/python` on Windows).

| script | what it shows | runtime |
|---|---|---|
| `rrr_orbit.py` | RRR lambda from GL(C[[t]]) orbit counting: `lct = r0(N+M-r0) + min_k g(k)`. Matches the full Aoyagi-Watanabe theorem on 2754 cells; jet-scheme Jacobian check of every stratum | seconds |
| `step0_tangent_excess.py` | The RRR gap 2*lambda - rank H is tangent excess at the min-norm point: rank H at generic true points (GD restarts, or projected perturbations) recovers 2*lambda on all 14 anchor cells | ~6 min |
| `real_vs_complex.py` | `rlct = lct_C(fiber ideal)/2` fails when minimizing divisors lack real points: (x^2+y^2+z^2)^2, and symmetric factorization W W^T (Monte Carlo check of lambda_sym) | ~20 s |
| `contact_gn.py` | Design E: real contact-locus codimension by proximal Gauss-Newton on jets (`jax.experimental.jet`) + local PCA. Validated on toy cases, RRR, symmetric; known failures listed in 4k | ~5 min |

Two mechanisms make 2*lambda exceed the Hessian rank:
1. **Tangent excess** (RRR). Z is singular where its components meet. lambda = codim of the
   largest component / 2. Hessians at generic true points see it.
2. **Higher-order transversal contact** (symmetric WW^T). Z is smooth over R, K is quartic
   across it, and the complex fiber ideal undercounts.
