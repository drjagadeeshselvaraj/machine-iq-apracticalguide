"""Chapter 10: control chart and CUSUM on the 24 monthly values of Assistant R (Table 10.4),
and the average delay of both detectors by simulation in threads.
Monthly HIQ is labelled with the bands of benchmarks.py (Appendix D)."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from benchmarks import band_name

PHI = [.2445, .1975, .2248, .1478, .2158, .2347, .2132, .2284, .1667, .2129, .2003, .2036,
       .1015, .0751, .1722, .1900, .0977, .1564, .1900, .1458, .1587, .1539, .1060, .1151]
MU0, SIGMA = 0.204, 0.033            # baseline and monthly standard error (study value)
REF_MEAN, REF_SD = 0.150, 0.040      # Meridian reference group, for the HIQ label
K_CHART, DELTA, H = 2.33, 1.5, 3.1

def chart_alarm(x):                   # x in standard errors below baseline
    hits = np.nonzero(x > K_CHART)[0]
    return hits[0] + 1 if hits.size else None

def cusum_alarm(x, delta=DELTA, h=H):
    w = 0.0
    for t, v in enumerate(x, 1):
        w = max(0.0, w + delta * v - delta**2 / 2)
        if w > h:
            return t
    return None

def _delay(args):
    seed, n, fall = args
    rng = np.random.default_rng(seed)
    d = {"chart": [], "cusum": []}
    for _ in range(n):
        x = rng.normal(0, 1, 400) + fall                # months after the change
        for name, f in (("chart", chart_alarm), ("cusum", cusum_alarm)):
            t = f(x)
            d[name].append(t if t else 400)
    return d

def mean_delay(fall, series=3000, threads=8):
    seeds = np.random.SeedSequence(10).spawn(threads)
    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(_delay, [(s, series // threads, fall) for s in seeds]))
    return {k: np.mean(sum((p[k] for p in parts), [])) for k in ("chart", "cusum")}

if __name__ == "__main__":
    x = (MU0 - np.array(PHI)) / SIGMA
    print("Run on Table 10.4: control chart alarm at month", chart_alarm(x),
          "| CUSUM alarm at month", cusum_alarm(x))
    for m in (12, 13, 24):
        hiq = 100 + 15 * (PHI[m - 1] - REF_MEAN) / REF_SD
        print(f"  month {m:2d}: Phi {PHI[m-1]:.4f}  HIQ {hiq:5.1f}  band {band_name(hiq)}")
    r0 = mean_delay(0.0, 800)
    print(f"Months between false alarms: chart {r0['chart']:.0f}, CUSUM {r0['cusum']:.0f}")
    print("Average months to alarm after a fall of Phi (3,000 series, 8 threads):")
    for fall in (0.03, 0.06, 0.10):
        s = fall / SIGMA; r = mean_delay(s)
        print(f"  fall {fall:.2f} = {s:.2f} SE: chart {r['chart']:.1f}  CUSUM {r['cusum']:.1f}"
              f"  formula {2*np.log(100)/s**2:.1f}")
