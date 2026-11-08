"""Chapter 8 (Step 6): the cluster bootstrap, run in threads.  The squared errors below are a
constructed set shaped like Meridian's: 600 pairs, 5 horizons, 3 systems, errors linked within
a pair (correlation about 0.35 between horizons).  Its numbers resemble Table 8.2, not equal it."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks import band_name, secure_band

H = 5; PAIRS = 600; M, S = 0.150, 0.040
SKILL = np.array([[.36, .28, .18, .09, .04], [.22, .21, .19, .16, .12], [.30, .26, .21, .15, .09]])
W = np.array([.125, .25, .25, .25, .125])               # trapezoid weights of the doubling grid

def make_data(seed=20261031):
    """Squared errors e[system, horizon, pair] with a pair effect shared by all of them."""
    r = np.random.default_rng(seed)
    pair = r.normal(size=PAIRS)                          # difficulty of the pair
    hz = r.normal(size=(H, PAIRS))                       # shared by the systems at one horizon
    sp = r.normal(size=(3, PAIRS))                       # a system's own difficulty with a pair
    e = np.empty((3, H, PAIRS))
    for s in range(3):
        for h in range(H):
            z = (np.sqrt(.29) * pair + np.sqrt(.585) * hz[h] + np.sqrt(.059) * sp[s]
                 + np.sqrt(.066) * r.normal(size=PAIRS))
            x = np.clip(0.25 * (1 - SKILL[s, h]) + 0.20 * z, 0, 1)
            e[s, h] = x + (0.25 * (1 - SKILL[s, h]) - x.mean())       # keep the horizon mean exact
    return e

def hiq_of(e, idx):
    skill = 1 - e[:, :, idx].mean(axis=2) / 0.25            # systems x horizons
    return 100 + 15 * (skill @ W - M) / S

def _block(args):
    seed, reps, e, cluster = args
    r = np.random.default_rng(seed)
    out = np.empty((reps, 3))
    for i in range(reps):
        if cluster:
            idx = r.integers(0, PAIRS, PAIRS)                        # one draw of pairs for everything
            out[i] = hiq_of(e, idx)
        else:                                                        # questions resampled alone
            sk = np.empty((3, H))
            for h in range(H):
                idx = r.integers(0, PAIRS, PAIRS)
                sk[:, h] = 1 - e[:, h, idx].mean(axis=1) / 0.25
            out[i] = 100 + 15 * (sk @ W - M) / S
    return out

def bootstrap(e, reps=2000, threads=8, cluster=True, seed=8):
    seeds = np.random.SeedSequence(seed).spawn(threads)
    with ThreadPoolExecutor(threads) as ex:
        return np.vstack(list(ex.map(_block, [(s, reps // threads, e, cluster) for s in seeds])))

if __name__ == "__main__":
    e = make_data(); point = hiq_of(e, np.arange(PAIRS))
    cl, ind = bootstrap(e), bootstrap(e, cluster=False)
    print("            HIQ   SE(pairs)  SE(questions)  95% interval (pairs)   band / secure")
    for i, a in enumerate("PQR"):
        lo, hi = np.percentile(cl[:, i], [2.5, 97.5])
        print(f"  {a}     {point[i]:6.1f}   {cl[:, i].std():6.1f}     {ind[:, i].std():6.1f}        "
              f"{lo:6.1f} to {hi:6.1f}   {band_name(point[i])} / "
              f"{secure_band(point[i], cl[:, i].std())}")
    for a, b, na in ((2, 1, "R - Q"), (2, 0, "R - P"), (0, 1, "P - Q")):
        d = cl[:, a] - cl[:, b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"  {na}  {point[a]-point[b]:5.1f}  SE {d.std():.1f}   {lo:5.1f} to {hi:5.1f}   "
              f"above 0 in {np.mean(d > 0):.1%}")
    first = np.bincount(cl.argmax(axis=1), minlength=3) / len(cl)
    print("  First place in resamples:", "  ".join(f"{a} {f:.1%}" for a, f in zip("PQR", first)))
