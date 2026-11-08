#!/usr/bin/env python3
"""Machine IQ benchmark bands (Practical Guide, Section 7.12-7.13), with threads.

Bands on the HIQ scale (mean 100, SD 15):
  low <90 | moderate 90-109.9 | high 110-119.9 | excellent 120-129.9 | outstanding 130+

Usage:  python3 benchmarks.py [--draws 400000] [--threads 8] [--sigma 8.7]
Needs:  numpy, scipy  (pip install numpy scipy)

Threads: numpy releases the GIL in its array kernels, so a ThreadPoolExecutor
gives a real speed-up here. Each thread has its own random generator spawned
from one SeedSequence, so results are reproducible and the streams independent.
"""
import argparse, bisect, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import norm

MEAN, SD = 100.0, 15.0
SCHEMES = {                      # name -> cut points (ascending)
    "five bands":  [90, 110, 120, 130],
    "four bands":  [90, 110, 120],
    "three bands": [90, 110],
    "two bands":   [100],
}
NAMES = ["Low", "Moderate", "High", "Excellent", "Outstanding"]
CUTS5 = SCHEMES["five bands"]


# ---------- deterministic part ----------
def band_index(hiq, cuts=CUTS5):
    """Band number 0..len(cuts); a value equal to a cut point goes up."""
    return int(np.searchsorted(cuts, hiq, side="right"))

def band_name(hiq):
    return NAMES[band_index(hiq)]

def secure_band(hiq, se, level=0.95):
    """Band of the one-sided lower limit (Theorem 7.2)."""
    z = norm.ppf(1 - (1 - level) / 2)          # two-sided 95% -> 1.96
    return band_name(hiq - z * se)

def shares(cuts=CUTS5):
    """Share of a normal reference group in each band (exact)."""
    edges = [-np.inf] + list(cuts) + [np.inf]
    p = norm.cdf((np.array(edges[1:]) - MEAN) / SD) - norm.cdf((np.array(edges[:-1]) - MEAN) / SD)
    return p

def p_empty(p, n):                              # Theorem 7.4
    return (1 - p) ** n

def n_for_five(p):                              # rule of thumb n >= 5/p
    return int(np.ceil(5 / p))


# ---------- simulation part (run in threads) ----------
def _chunk(args):
    seed, n, sigma, cuts_list = args
    rng = np.random.default_rng(seed)
    true_sd = np.sqrt(SD**2 - sigma**2)         # observed SD stays 15
    true = rng.normal(MEAN, true_sd, n)
    meas = true + rng.normal(0, sigma, n)
    out = {}
    for name, cuts in cuts_list.items():
        bt, bm = np.searchsorted(cuts, true, "right"), np.searchsorted(cuts, meas, "right")
        out[name] = int(np.sum(bt == bm))
    lower = meas - 1.96 * sigma                 # secure band test, five bands
    bl = np.searchsorted(CUTS5, lower, "right")
    bt5 = np.searchsorted(CUTS5, true, "right")
    out["secure_ok"] = int(np.sum(bl <= bt5))
    return n, out

def simulate(sigma, draws=400_000, threads=8, seed=2026):
    """Accuracy per scheme and secure-band coverage, split across threads."""
    per = draws // threads
    seeds = np.random.SeedSequence(seed).spawn(threads)
    jobs = [(s, per, sigma, SCHEMES) for s in seeds]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(_chunk, jobs))
    total = sum(n for n, _ in res)
    agg = {k: sum(r[k] for _, r in res) / total for k in list(SCHEMES) + ["secure_ok"]}
    return agg

def _empty_chunk(args):
    seed, trials, n, cuts = args
    rng = np.random.default_rng(seed)
    x = rng.normal(MEAN, SD, (trials, n))
    b = np.searchsorted(cuts, x, "right")
    empty = np.zeros(len(cuts) + 1)
    for k in range(len(cuts) + 1):
        empty[k] = np.sum(~(b == k).any(axis=1))
    return trials, empty

def simulate_empty(n=20, trials=200_000, threads=8, seed=7):
    per = trials // threads
    seeds = np.random.SeedSequence(seed).spawn(threads)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(_empty_chunk, [(s, per, n, CUTS5) for s in seeds]))
    t = sum(r[0] for r in res)
    return sum(r[1] for r in res) / t

def sigma_for_accuracy(target, cuts, threads=8, draws=200_000):
    """Largest standard error with accuracy >= target (bisection; each step threaded)."""
    lo, hi = 0.5, 14.5
    for _ in range(14):
        mid = (lo + hi) / 2
        per = draws // threads
        seeds = np.random.SeedSequence(11).spawn(threads)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            res = list(ex.map(_chunk, [(s, per, mid, {"x": cuts}) for s in seeds]))
        acc = sum(r["x"] for _, r in res) / sum(n for n, _ in res)
        lo, hi = (mid, hi) if acc >= target else (lo, mid)
    return (lo + hi) / 2


# ---------- report ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=400_000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=8.7, help="standard error of the HIQ")
    a = ap.parse_args()
    t0 = time.time()

    print("1. Band shares of a normal reference group (exact)")
    for nm, p in zip(NAMES, shares()):
        print(f"   {nm:<11} {100*p:6.2f}%   empty-band chance, n=20: {p_empty(p,20):.3f}   n for 5 expected: {n_for_five(p)}")

    print(f"\n2. Example: HIQ 114.1 with SE {a.sigma}:  band = {band_name(114.1)},  secure band = {secure_band(114.1, a.sigma)}")

    print(f"\n3. Simulation, {a.draws:,} systems, {a.threads} threads")
    print(f"   {'SE':>5} " + " ".join(f"{k:>12}" for k in SCHEMES) + "   secure<=true")
    for s in (8.7, 6.2, 4.4):
        r = simulate(s, a.draws, a.threads)
        print(f"   {s:5.1f} " + " ".join(f"{100*r[k]:11.1f}%" for k in SCHEMES) + f"   {100*r['secure_ok']:6.1f}%")
    print("   (Theorem 7.2 bound for the secure band: at least 97.5%)")

    print("\n4. Chance that a group of 20 has an empty band (simulated)")
    e = simulate_empty(20, 200_000, a.threads)
    for nm, v in zip(NAMES, e):
        print(f"   {nm:<11} {v:.3f}")

    print("\n5. Largest SE that still gives 80% accuracy")
    for k, c in SCHEMES.items():
        print(f"   {k:<12} SE <= {sigma_for_accuracy(0.80, c, a.threads):.1f}")

    print(f"\nDone in {time.time()-t0:.1f} s")

if __name__ == "__main__":
    main()
