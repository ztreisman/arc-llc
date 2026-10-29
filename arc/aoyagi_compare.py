"""Aoyagi's deep-linear learning coefficient (as restated in arXiv:2501.12747, Def. 4 /
Thm 4; proved in Aoyagi, Neural Networks 172 (2024)) vs Pepin Lehalleur-Rimanyi's
codimension C (QIP, Thm 6.1 of arXiv:2411.19920) and vs our arc counts.

Aoyagi, no biases: dims H^(1) (output), ..., H^(L+1) (input), true rank r,
M^(s) = H^(s) - r. Choose M_set = the l+1 smallest M's satisfying
    sum_{M_set} M >= l * M^(s) for s in M_set,   sum_{M_set} M < l * M^(s) for s not in it.
With S = sum_{M_set} M, Mi = ceil(S / l), a = S - (Mi - 1) l:
    lambda = r(H^(1) + H^(L+1) - r)/2 + a(l-a)/(4l) + S^2/(4l) - sum_{M_set} M^2 / 4,
    theta  = a(l - a) + 1     (order of the pole: the power of log log n).
If some M^(s) = 0, lambda is just the first term and theta = 1 (Remark 1).

Checks: 2*(lambda - smooth part) == C(M vector) (PLR QIP), all depths L = 1..5;
and Aoyagi's theta (pole order) vs PLR's theta (number of top components).
"""
import itertools
import math
from fractions import Fraction

from deep_linear_orbit import plr_qip


def aoyagi(H, r):
    """H = (H^(1), ..., H^(L+1)), output first. Returns (lambda, theta)."""
    smooth = Fraction(r * (H[0] + H[-1] - r), 2)
    M = [h - r for h in H]
    if min(M) == 0:
        return smooth, 1
    Ms = sorted(M)
    for k in range(2, len(M) + 1):
        l = k - 1
        S = sum(Ms[:k])
        inside = all(S >= l * m for m in Ms[:k])
        outside = all(S < l * m for m in Ms[k:])
        if inside and outside:
            Mi = -(-S // l)
            a = S - (Mi - 1) * l
            lam = smooth + Fraction(a * (l - a), 4 * l) + Fraction(S * S, 4 * l) \
                - Fraction(sum(m * m for m in Ms[:k]), 4)
            return lam, a * (l - a) + 1
    raise ValueError(f"no admissible M_set for {H}, r={r}")


def main():
    print("Aoyagi lambda vs PLR codim C (r = 0 and r > 0), and theta (pole order) vs")
    print("PLR theta (number of top components)")
    for L, top in [(1, 7), (2, 7), (3, 5), (4, 4), (5, 3)]:
        n = bad = theta_diff = 0
        examples = []
        for H in itertools.product(range(1, top + 1), repeat=L + 1):
            for r in range(0, min(H) + 1):
                lam, th_a = aoyagi(H, r)
                Mvec = tuple(h - r for h in H)
                smooth = Fraction(r * (H[0] + H[-1] - r), 2)
                if min(Mvec) == 0:
                    C, th_p = 0, 1
                else:
                    C, th_p = plr_qip(Mvec)
                n += 1
                if 2 * (lam - smooth) != C:
                    bad += 1
                if th_a != th_p:
                    theta_diff += 1
                    if len(examples) < 4:
                        examples.append((Mvec, th_a, th_p))
        print(f"  L={L}: {n} cases, {bad} lambda mismatches, "
              f"{theta_diff} where pole order != component count  {examples}")


if __name__ == "__main__":
    main()
