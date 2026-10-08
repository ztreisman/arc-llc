"""
Exact check:  does  min_p  codim_R Cont^{>=p}(K) / p  =  rlct(K) ?

Mustata's theorem gives lct over an ALGEBRAICALLY CLOSED field.  SLT wants the rlct.
These differ (e.g. lct_C(x^2+y^2+z^2) = 1 but rlct = 3/2).  The question is whether
taking the REAL contact locus repairs the formula.

For every case below, K is a sum/product of EVEN powers, so no cancellation is possible
over R and ord_gamma(K) is determined by the coordinate orders alone.  That makes
codim_R Cont^{>=p} exactly computable by order arithmetic -- no optimiser, no sampling.

Notation: gamma = (gamma_1,...,gamma_d), q_i = ord(gamma_i).
Each condition "q_i >= m" costs m linear conditions on the jet coefficients.
"""
from math import ceil
from fractions import Fraction as F

def codim_x4(p):
    # ord(x^4) = 4 q.  Need 4q >= p  =>  q >= ceil(p/4).
    return ceil(p / 4)

def codim_sumsq(p, d):
    # K = |x|^2, ord = 2 q  (no cancellation over R).  q >= ceil(p/2), cost d per unit.
    return d * ceil(p / 2)

def codim_a2b2(p):
    # ord = 2 qa + 2 qb.  Need qa + qb >= ceil(p/2); cheapest is to load one variable.
    return ceil(p / 2)

def codim_a2b6(p):
    # K = a^2 + b^6, both even => no cancellation => ord = min(2 qa, 6 qb).
    return ceil(p / 2) + ceil(p / 6)

CASES = [
    ("x^4",         F(1, 4), codim_x4,                      range(1, 13)),
    ("x^2+y^2",     F(1),    lambda p: codim_sumsq(p, 2),   range(1, 13)),
    ("x^2+y^2+z^2", F(3, 2), lambda p: codim_sumsq(p, 3),   range(1, 13)),
    ("a^2 b^2",     F(1, 2), codim_a2b2,                    range(1, 13)),
    ("a^2 + b^6",   F(2, 3), codim_a2b6,                    range(1, 25)),
]

print(f"{'case':>12s} {'rlct':>7s} {'min_p codim/p':>15s} {'argmin p':>9s}  {'match':>5s}   codim/p for p=1..8")
print("-" * 100)
for name, rlct, cod, ps in CASES:
    vals = [(F(cod(p), p), p) for p in ps]
    best, argmin = min(vals)
    head = "  ".join(f"{str(F(cod(p), p)):>5s}" for p in list(ps)[:8])
    ok = "yes" if best == rlct else "NO"
    print(f"{name:>12s} {str(rlct):>7s} {str(best):>15s} {argmin:>9d}  {ok:>5s}   {head}")

print()
print("Note: lct_C(x^2+y^2+z^2) = 1, but the REAL contact codimension gives 3/2 = rlct.")
print("The real contact locus is strictly smaller than the complex one, and that is")
print("exactly the correction needed to turn Mustata's formula into an rlct formula.")
