"""Chapter 5 (Step 3): collect forecasts with threads, retries, a log and a seal.
The forecaster here is a stand-in (a noisy draw about the system's true belief); replace
ask() by the call to your own system.  Each call's randomness comes from its own key
(assistant, question, draw), so the result does not depend on how the threads interleave."""
import hashlib, json, time, argparse
from concurrent.futures import ThreadPoolExecutor
import numpy as np

SIGMA_DRAW, FAIL, PRICE, MAX_TRY = 0.10, 0.015, 0.03, 4

def belief(q):                                          # the true probability of question q
    return float(np.random.default_rng([7, q]).beta(2, 2))

def ask(a, q, d, attempt, latency):
    """One call.  Fails with probability FAIL, as a rate-limited service would."""
    rng = np.random.default_rng([a, q, d, attempt])
    if latency: time.sleep(latency)
    if rng.random() < FAIL:
        raise TimeoutError("no answer")
    return float(np.clip(belief(q) + rng.normal(0, SIGMA_DRAW), 0.01, 0.99))

def one(job, latency):
    a, q, d = job
    for attempt in range(MAX_TRY):
        try:
            p = round(ask(a, q, d, attempt, latency), 4)
            return {"a": a, "q": q, "d": d, "p": p, "tries": attempt + 1}
        except TimeoutError:
            continue
    return {"a": a, "q": q, "d": d, "p": None, "tries": MAX_TRY}   # logged as missing, never replaced

def collect(n_q=3000, systems=3, k=3, threads=8, latency=0.0):
    jobs = [(a, q, d) for a in range(systems) for q in range(n_q) for d in range(k)]
    with ThreadPoolExecutor(threads) as ex:
        log = list(ex.map(lambda j: one(j, latency), jobs))
    return log

def seal(log):
    rows = sorted(log, key=lambda r: (r["a"], r["q"], r["d"]))
    text = "\n".join(json.dumps(r, sort_keys=True) for r in rows)
    return hashlib.sha256(text.encode()).hexdigest()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--latency", type=float, default=0.0); a = ap.parse_args()
    t0 = time.time()
    log = collect(threads=a.threads, latency=a.latency)
    calls = sum(r["tries"] for r in log)
    print(f"{len(log):,} forecasts, {calls:,} calls ({calls/len(log)-1:.2%} retries), "
          f"{sum(r['p'] is None for r in log)} missing,")
    print(f"cost ${calls*PRICE:,.2f}, {time.time()-t0:.1f} s")
    plan = 3000 * 3 * 3 * 1.015
    print(f"Equation (5.4): 3,000 x 3 x 3 x 1.015 = {plan:,.0f} calls, ${plan*PRICE:,.2f}")
    print("Seal (sha256):", seal(log)[:32], "...  same on every run and every thread count")
    # sampling penalty: mean of k draws against the true belief
    sys0 = [r for r in log if r["a"] == 0 and r["p"] is not None]
    by_q = {}
    for r in sys0: by_q.setdefault(r["q"], []).append(r["p"])
    for k in (1, 3):
        pen = np.mean([(np.mean(v[:k]) - belief(q)) ** 2 for q, v in by_q.items() if len(v) >= k])
        print(f"k = {k}: sampling penalty {pen:.4f}   (sigma^2/k = {SIGMA_DRAW**2/k:.4f})")
