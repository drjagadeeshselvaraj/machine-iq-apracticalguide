"""Chapter 11: break-even lead, the chance that a lead pays for itself, the size of the
acceptance test (checked by simulation in threads), and the band of each candidate.
Bands come from benchmarks.py (Appendix D): a purchase is judged on the secure band."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.stats import norm
from benchmarks import band_name, secure_band

VALUE_PER_PHI = 45_000                # dollars a month for one unit of Phi (constructed)
CAND = {"P": (114.06, 2600), "Q": (112.19, 1900), "R": (120.16, 3200)}   # HIQ, $/month
HIQ_SE = 8.7

def break_even(extra_price):
    return extra_price / VALUE_PER_PHI

def prob_pays(lead, se, extra_price):
    return 1 - norm.cdf((break_even(extra_price) - lead) / se)

def test_size(true_phi, phi_min, n_study=3000, se_study=0.0234):
    m = true_phi - phi_min
    return int(np.ceil(n_study * (1.645 * se_study / m) ** 2))

def _pass(args):                       # one thread = one block of simulated acceptance tests
    seed, trials, n, true_phi, phi_min, se_study, n_study = args
    rng = np.random.default_rng(seed)
    se = se_study * np.sqrt(n_study / n)
    est = rng.normal(true_phi, se, trials)
    return int(np.sum(est - 1.645 * se >= phi_min))

def pass_rate(n, true_phi, phi_min, trials=200_000, threads=8, se_study=0.0234, n_study=3000):
    seeds = np.random.SeedSequence(11).spawn(threads)
    jobs = [(s, trials // threads, n, true_phi, phi_min, se_study, n_study) for s in seeds]
    with ThreadPoolExecutor(threads) as ex:
        return sum(ex.map(_pass, jobs)) / trials

if __name__ == "__main__":
    print(f"Break-even lead of R over Q: {break_even(1300):.4f} Phi "
          f"({break_even(1300)*15/0.04:.1f} HIQ points)")
    print(f"Chance that R's lead (0.0213, SE 0.0131) covers it: {prob_pays(0.0213, 0.0131, 1300):.0%}")
    print(f"At twice the value of skill: break-even {break_even(1300)/2:.4f}, "
          f"chance {1 - norm.cdf((break_even(1300)/2 - 0.0213)/0.0131):.0%}")
    n1, n2 = test_size(0.2038, 0.170), test_size(0.190, 0.170)
    print(f"Acceptance test: {n1:,} questions if true Phi is 0.2038 (cost ${n1*3*0.03:,.0f}); "
          f"{n2:,} if true Phi is 0.190")
    print("Simulated chance that a system of true Phi 0.2038 passes the test (threads):")
    for n in (n1, 8_890, 15_560):
        print(f"  {n:6,} questions: {pass_rate(n, 0.2038, 0.170):.2f}   (cost ${n*3*0.03:,.0f})")
    print("Candidates (HIQ standard error 8.7):")
    for a, (hiq, price) in CAND.items():
        print(f"  {a}: HIQ {hiq:6.2f}  band {band_name(hiq):<9}"
              f" secure band {secure_band(hiq, HIQ_SE):<9} ${price:,}")
