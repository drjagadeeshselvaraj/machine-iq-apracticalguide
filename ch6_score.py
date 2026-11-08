"""Chapter 6 (Step 4): Brier score and skill, their standard errors, and the fitted curve
S(h) = (2/pi) arcsin(w rho^(2h)) by weighted least squares (Table 6.5)."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

H = np.array([1, 2, 4, 8, 16.0])
SKILL = {"P": [.36, .28, .18, .09, .04], "Q": [.22, .21, .19, .16, .12], "R": [.30, .26, .21, .15, .09]}
SE_S = 0.20 / (0.25 * np.sqrt(600))                     # Equation (6.3): s 0.20, 600 questions

def skill(p, o):
    """Brier skill of forecasts p against outcomes o (Equation 6.2)."""
    p, o = np.asarray(p, float), np.asarray(o, float)
    bs = np.mean((p - o) ** 2)
    base = o.mean() * (1 - o.mean())
    return 1 - bs / base, bs, base

def curve(h, w, rho):
    return 2 / np.pi * np.arcsin(w * rho ** (2 * h))

def fit(s):
    """Weighted least squares, with starting values read from the first and last points."""
    rho0 = np.exp(-np.log(s[0] / s[-1]) / (2 * (H[-1] - H[0])))
    w0 = s[0] * (np.pi / 2) / rho0**2
    (w, rho), _ = curve_fit(curve, H, s, p0=[w0, rho0], sigma=np.full(5, SE_S),
                            bounds=([0, .5], [1, 1]))
    x2 = np.sum(((s - curve(H, w, rho)) / SE_S) ** 2)
    return w, rho, x2

def phi(h, s):
    x = np.log2(h)
    return np.sum(np.diff(x) * (s[1:] + s[:-1]) / 2) / (x[-1] - x[0])

def reach(w, rho, level=0.10):
    g = np.linspace(1, 40, 4000)
    below = np.nonzero(curve(g, w, rho) < level)[0]
    return g[below[0]] if below.size else np.inf

if __name__ == "__main__":
    p = [.9, .7, .2, .6, .1, .8, .4, .3]; o = [1, 1, 0, 0, 0, 1, 1, 0]       # Table 6.1
    s, bs, base = skill(p, o)
    print(f"Table 6.1: BS {bs:.3f}, baseline {base:.2f}, skill {s:.2f}")
    se_diff = 0.10 / (0.25 * np.sqrt(600))
    print(f"SE of a skill {SE_S:.4f}, interval +/-{1.96*SE_S:.3f}; SE of a difference {se_diff:.4f}")
    grid = np.linspace(1, 16, 400)
    print("\n   w      rho    chi2  (p)    Phi points  Phi curve  reach (days)")
    for a, v in SKILL.items():
        v = np.array(v); w, rho, x2 = fit(v)
        r = reach(w, rho)
        print(f" {a} {w:.3f}  {rho:.3f}  {x2:5.2f} ({chi2.sf(x2, 3):.2f})  {phi(H, v):.4f}     "
              f"{phi(grid, curve(grid, w, rho)):.4f}    {r:5.1f}{'*' if r > 16 else ''}")
    print("* outside the window of 1 to 16 days: an extrapolation")
