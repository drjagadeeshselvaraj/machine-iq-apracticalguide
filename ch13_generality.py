"""Chapter 13: a generality profile for three systems on constructed data (five domains),
its uncertainty by Monte Carlo in threads, and a test for a common factor by parallel analysis.
Bands come from benchmarks.py (Appendix D).  All data are constructed."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks import band_name, secure_band

M, S = 0.150, 0.040                                   # norms, taken as the same in every domain
DOMAINS = ["Operations", "Health", "Energy", "Finance", "Events"]
QUESTIONS = np.array([3000, 1500, 1500, 800, 800])    # questions per domain
SE = 8.7 * np.sqrt(3000 / QUESTIONS)                  # HIQ standard error in each domain
PHI = {"A": [.185, .190, .175, .180, .170],           # balanced
       "B": [.290, .150, .140, .130, .120],           # a specialist
       "C": [.220, .230, .200, .215, .210]}           # broad and strong
SKILL = {"A": [.30, .27, .22, .17, .12], "B": [.35, .25, .15, .08, .04],
         "C": [.34, .31, .27, .22, .18]}              # skill at 1, 2, 4, 8, 16 days
H = np.array([1, 2, 4, 8, 16.0])

def hiq(phi):
    return 100 + 15 * (np.asarray(phi) - M) / S

def phi_window(s, lo, hi):
    keep = (H >= lo) & (H <= hi); x, v = np.log2(H[keep]), np.asarray(s)[keep]
    return np.sum(np.diff(x) * (v[1:] + v[:-1]) / 2) / (x[-1] - x[0])

def retention(s):
    """Share of the skill on the short window [1, 4] that is kept on the long window [4, 16]."""
    return phi_window(s, 4, 16) / phi_window(s, 1, 4)

def _mc(args):                                         # breadth and floor with measurement error
    seed, reps, est = args
    r = np.random.default_rng(seed)
    x = est + r.normal(0, SE, (reps, len(est)))
    return x.mean(axis=1), x.min(axis=1)

def breadth_floor(est, reps=200_000, threads=8):
    seeds = np.random.SeedSequence(13).spawn(threads)
    with ThreadPoolExecutor(threads) as ex:
        res = list(ex.map(_mc, [(s, reps // threads, est) for s in seeds]))
    return np.concatenate([a for a, _ in res]), np.concatenate([b for _, b in res])

def _eig(args):                      # first eigenvalues of simulated tables
    seed, reps, n, d, load = args
    r = np.random.default_rng(seed); out = np.empty(reps)
    for k in range(reps):
        g = r.normal(size=(n, 1))
        x = load * g + np.sqrt(1 - load**2) * r.normal(size=(n, d))
        out[k] = np.linalg.eigvalsh(np.corrcoef(x.T))[-1]
    return out

def first_eigen(load, n=20, d=5, reps=4000, threads=8):
    seeds = np.random.SeedSequence(int(14 + 100 * load)).spawn(threads)
    with ThreadPoolExecutor(threads) as ex:
        return np.concatenate(list(ex.map(_eig, [(s, reps // threads, n, d, load) for s in seeds])))

def factor_power(load, **kw):
    """Chance that the first eigenvalue beats the 95th percentile of noise (parallel analysis)."""
    cut = np.percentile(first_eigen(0.0, **kw), 95)
    return cut, np.mean(first_eigen(load, **kw) > cut)

if __name__ == "__main__":
    print("HIQ by domain, with the SE of each (questions 3000/1500/1500/800/800):")
    print("  SE        " + "  ".join(f"{v:6.1f}" for v in SE))
    for a, p in PHI.items():
        q = hiq(p); lo = q - 1.96 * SE
        prof = " ".join(f"{band_name(v)[:3]}" for v in q)
        print(f"  {a}  " + "  ".join(f"{v:6.2f}" for v in q) + f"   bands {prof}")
        b, f = breadth_floor(q)
        se_b = np.sqrt(np.sum(SE**2)) / len(q)
        print(f"     breadth {q.mean():6.2f} (SE {se_b:.1f}, secure {secure_band(q.mean(), se_b)})")
        print(f"     floor {q.min():6.1f}; mean of the noisy floor {f.mean():6.1f} "
              f"(bias {f.mean()-q.min():+.1f})")
        print(f"     horizon retention {retention(SKILL[a]):.2f}")
    cut, p = factor_power(0.6)
    print(f"\nCommon factor, 20 systems x 5 domains: noise 95th percentile {cut:.2f}")
    print(f"  chance of detecting a general factor with loading 0.6: {p:.0%}")
    for load in (0.3, 0.45):
        print(f"  loading {load}: {factor_power(load)[1]:.0%}")
