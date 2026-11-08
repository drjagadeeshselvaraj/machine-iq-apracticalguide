"""Chapter 9: fit the exponential success curve, test it, compute Phi and HIQ for tasks,
and bootstrap the half-success length in threads. Bands come from benchmarks.py (Appendix D)."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, binom
from benchmarks import band_name, secure_band

N = 150
LENGTHS = np.array([2, 4, 8, 16, 32, 64])
DONE = {"P": [142, 137, 123, 105, 68, 29],        # Table 9.5
        "Q": [147, 143, 135, 119, 94, 90],
        "R": [146, 144, 135, 129, 121, 84]}
W = np.array([.1, .2, .2, .2, .2, .1])            # trapezoid weights, doubling grid
MEAN, SD = 0.70, 0.08                             # reference group of 16 agents

def fit_lambda(k, n=N):
    nll = lambda lam: -binom.logpmf(k, n, np.exp(-lam * LENGTHS)).sum()
    return minimize_scalar(nll, bounds=(1e-4, 0.5), method="bounded").x

def pearson(k, lam, n=N):
    p = np.exp(-lam * LENGTHS)
    x2 = np.sum((k - n * p) ** 2 / (n * p * (1 - p)))
    return x2, chi2.sf(x2, len(LENGTHS) - 1)

def phi_points(k, n=N):
    p = np.asarray(k) / n
    return W @ p, np.sqrt(np.sum(W**2 * p * (1 - p) / n))

def _boot(args):                                   # one thread = one block of resamples
    seed, reps, lam = args
    rng = np.random.default_rng(seed)
    p = np.exp(-lam * LENGTHS)
    out = np.empty(reps)
    for i in range(reps):
        out[i] = np.log(2) / fit_lambda(rng.binomial(N, p))
    return out

def half_success(lam, reps=800, threads=8):
    seeds = np.random.SeedSequence(9).spawn(threads)
    with ThreadPoolExecutor(threads) as ex:
        res = np.concatenate(list(ex.map(_boot, [(s, reps // threads, lam) for s in seeds])))
    return np.percentile(res, [2.5, 97.5])

if __name__ == "__main__":
    for a, k in DONE.items():
        k = np.array(k)
        lam = fit_lambda(k); x2, p = pearson(k, lam)
        phi, se = phi_points(k)
        hiq, hse = 100 + 15 * (phi - MEAN) / SD, 15 * se / SD
        lo, hi = half_success(lam)
        print(f"{a}: lambda {lam:.4f}  half-success {np.log(2)/lam:5.1f} ({lo:.1f} to {hi:.1f})  "
              f"chi2 {x2:5.2f}  p {p:.3f}")
        print(f"   Phi {phi:.4f}  HIQ {hiq:5.1f} +/-{1.96*hse:.1f}  "
              f"band {band_name(hiq)}, secure band {secure_band(hiq, hse)}")
