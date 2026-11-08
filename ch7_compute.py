"""Chapter 7 (Step 5): Phi, HIQ, the norming band, the profile by stratum, weights,
the ceiling fraction, and the benchmark bands of Appendix D (benchmarks.py)."""
import numpy as np
from benchmarks import band_name, secure_band

M, S, N_REF = 0.150, 0.040, 20                      # reference group: mean, SD, members
HIQ_SE = 8.7                                        # from the cluster bootstrap (Chapter 8)
STRATA = {"fast": {"P": .2600, "Q": .2000, "R": .2425},     # Table 7.4
          "seasonal": {"P": .1400, "Q": .1800, "R": .1850},
          "new": {"P": .0775, "Q": .1425, "R": .1350}}
SHARES = {"fast": .5, "seasonal": .3, "new": .2}

def hiq(phi, m=M, s=S):
    return 100 + 15 * (phi - m) / s                 # Equation (7.2)

def weighted_phi(system, weights):
    return sum(weights[k] * STRATA[k][system] for k in STRATA)

def ceiling_fraction(phi, ceiling):
    return phi / ceiling                            # Equation (7.7)

if __name__ == "__main__":
    norm_se = 15 / np.sqrt(N_REF)
    print(f"Norming band of the level: +/-{1.96 * norm_se:.1f} HIQ points (n = {N_REF})\n")
    equal = {k: 1 / 3 for k in STRATA}
    print("            shares of questions            equal weights")
    for a in "PQR":
        q1, q2 = hiq(weighted_phi(a, SHARES)), hiq(weighted_phi(a, equal))
        print(f"  {a}: Phi {weighted_phi(a, SHARES):.5f}  HIQ {q1:6.2f}  {band_name(q1):<9} "
              f"secure {secure_band(q1, HIQ_SE):<9} | HIQ {q2:6.2f}  {band_name(q2)}")
    print("\nProfile (HIQ by stratum):")
    for k in STRATA:
        print(f"  {k:<9}" + "".join(f"  {a} {hiq(STRATA[k][a]):6.2f} ({band_name(hiq(STRATA[k][a]))})"
                                    for a in "PQR"))
    print("\nCeiling fraction of Phi (ceilings 0.40 and 0.32):")
    for a in "PQR":
        p = weighted_phi(a, SHARES)
        print(f"  {a}: {ceiling_fraction(p, .40):.2f} and {ceiling_fraction(p, .32):.2f}")
