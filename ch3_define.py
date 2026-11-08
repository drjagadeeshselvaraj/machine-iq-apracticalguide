"""Chapter 3 (Step 1): window, grid, reference group.  Phi on three windows (Example 3.3),
the size of the reference group (Equation 3.2), and the same Phi in two groups (Example 3.6)."""
import math
import numpy as np

H = np.array([1, 2, 4, 8, 16])
SKILL = {"P": [.36, .28, .18, .09, .04], "Q": [.22, .21, .19, .16, .12],
         "R": [.30, .26, .21, .15, .09]}                        # Table 1.3

def doublings(a, b):
    return math.log2(b / a)                                      # Equation (3.1)

def phi_window(skill, a, b):
    """Trapezoid average of skill over log2(h) on the grid points inside [a, b]."""
    keep = (H >= a) & (H <= b)
    x, s = np.log2(H[keep]), np.asarray(skill)[keep]
    return np.sum(np.diff(x) * (s[1:] + s[:-1]) / 2) / (x[-1] - x[0])

def group_size(half_width):
    return math.ceil((1.96 * 15 / half_width) ** 2)              # Equation (3.2)

def norming_band(n):
    return 1.96 * 15 / math.sqrt(n)

def hiq(phi, mean, sd):
    return 100 + 15 * (phi - mean) / sd

if __name__ == "__main__":
    print(f"Window [1, 16] has {doublings(1, 16):.0f} doublings; a hospital window [1, 28] has "
          f"{doublings(1, 28):.2f}")
    print("\nPhi on three windows (a different winner on each):")
    for a, b in ((1, 4), (4, 16), (1, 16)):
        v = {k: phi_window(s, a, b) for k, s in SKILL.items()}
        best = max(v, key=v.get)
        row = "  ".join(f"{k} {x:.4f}" for k, x in v.items())
        print(f"  [{a:>2}, {b:>2}]  {row}   best: {best}")
    print(f"\nReference group for a band of +/-5 HIQ points: {group_size(5)} systems")
    for n in (12, 20, 35):
        print(f"  n = {n:2d}: norming band +/-{norming_band(n):.1f}")
    print("\nThe same Phi = 0.20 in two reference groups:")
    print(f"  group A (0.12, 0.05): HIQ {hiq(.20, .12, .05):.0f}")
    print(f"  group B (0.18, 0.03): HIQ {hiq(.20, .18, .03):.0f}")
