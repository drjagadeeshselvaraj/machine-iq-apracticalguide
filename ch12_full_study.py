"""Chapter 12: the Meridian study from skills to a report line, and the size of a study.
Uses the bands, the secure band and the accuracy search of benchmarks.py (Appendix D)."""
import numpy as np
from benchmarks import band_name, secure_band, sigma_for_accuracy, SCHEMES

H = np.array([1, 2, 4, 8, 16])
SKILL = {"P": [.36, .28, .18, .09, .04],       # Table 1.3
         "Q": [.22, .21, .19, .16, .12],
         "R": [.30, .26, .21, .15, .09]}
REF_MEAN, REF_SD = 0.150, 0.040                # reference group of 20 assistants
HIQ_SE, N_STUDY = 8.7, 3000                    # standard error of the HIQ, questions
ASSISTANTS, DRAWS, PRICE = 3, 3, 0.03          # calls = questions x assistants x draws

def phi(skill):
    """Trapezoid average of skill over log2 of the horizon."""
    x = np.log2(H)
    s = np.asarray(skill)
    return np.sum(np.diff(x) * (s[1:] + s[:-1]) / 2) / (x[-1] - x[0])

def hiq(p):
    return 100 + 15 * (p - REF_MEAN) / REF_SD

def questions_for(se_target, se_now=HIQ_SE, n_now=N_STUDY):
    """The standard error falls with the square root of the number of questions."""
    return int(np.ceil(n_now * (se_now / se_target) ** 2))

def report_line(name, skill):
    p = phi(skill); q = hiq(p)
    lo, hi = q - 1.96 * HIQ_SE, q + 1.96 * HIQ_SE
    return (f"{name}: Phi {p:.4f}  HIQ {q:6.2f} ({lo:5.1f} to {hi:5.1f})\n"
            f"   band {band_name(q)}, secure band {secure_band(q, HIQ_SE)}")

if __name__ == "__main__":
    print("Report lines (reference group mean 0.150, SD 0.040, HIQ standard error 8.7):")
    for a, s in SKILL.items():
        print(" ", report_line(a, s))
    print("\nHow many questions would each scheme need for 80% accuracy?")
    for name, cuts in SCHEMES.items():
        se = sigma_for_accuracy(0.80, cuts)
        n = questions_for(se)
        cost = n * ASSISTANTS * DRAWS * PRICE
        print(f"  {name:<12} SE <= {se:3.1f}: {n:6,} questions, about ${cost:6,.0f} in all")
    se_diff = 4.9                                   # HIQ points, paired, from Chapter 12
    print(f"\nSmallest difference found with 80% power at 3,000 questions: "
          f"{2.8*se_diff:.1f} HIQ points")
