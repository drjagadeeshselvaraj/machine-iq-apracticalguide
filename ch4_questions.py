"""Chapter 4 (Step 2): how many questions, with clusters, and what label noise does to the ceiling.
Appendix D (benchmarks.py) is used to ask whether the planned size can support a band scheme."""
import math
from benchmarks import sigma_for_accuracy, SCHEMES

Z = 1.96 + 0.84                                                   # 5% level, 80% power

def questions_needed(delta, sd_diff):
    return math.ceil(Z**2 * sd_diff**2 / delta**2)               # Equation (4.3)

def detectable(n, sd_diff):
    return Z * sd_diff / math.sqrt(n)

def deff(m, rho):
    return 1 + (m - 1) * rho                                      # Equation (4.4)

def ceiling_from_noise(eta):
    """Best skill of a perfect forecaster when a share eta of the labels is wrong (base 0.5)."""
    return 1 - eta * (1 - eta) / 0.25

def questions_for_bands(scheme, se_now=8.7, n_now=3000):
    se = sigma_for_accuracy(0.80, SCHEMES[scheme])
    return se, math.ceil(n_now * (se_now / se) ** 2)

if __name__ == "__main__":
    n = questions_needed(0.0125, 0.10)
    print(f"Questions per horizon for a difference of 0.0125 (sd 0.10): {n}")
    print(f"Planned 600 detect {detectable(600, 0.10):.4f}")
    d = deff(4, 0.25)
    print(f"Clusters of 4, correlation 0.25: DEFF {d:.2f}, questions {math.ceil(n * d)}, "
          f"effective sample of {n}: {n / d:.0f}")
    print(f"Audit 9 of 300 disagree (eta 3%): ceiling skill {ceiling_from_noise(0.03):.3f}")
    print("\nWhat the plan buys in HIQ bands (HIQ standard error 8.7 at 3,000 questions):")
    for s in ("five bands", "three bands", "two bands"):
        se, q = questions_for_bands(s)
        print(f"  {s:<12} needs SE <= {se:.1f}: about {q:,} questions")
