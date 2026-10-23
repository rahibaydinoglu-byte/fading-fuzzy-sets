"""Empirical study for 'Fading fuzzy sets' (Section 10 of the manuscript).

Data: SNAP Bitcoin OTC and Bitcoin Alpha trust networks (Kumar et al., 2016).
Each row: rater, ratee, rating in [-10, 10] \\ {0}, unix time.
Task: before each rating that a user receives, predict whether it will be
positive (membership in the fuzzy set 'trustworthy user'), using only the
ratings that user received earlier. Every method keeps one grade per user.

Every method outputs a grade; a common two-parameter logistic calibration
q = sigmoid(a + b*logit(grade)) is fitted on the training period for each method
(profiled inside the parameter fit), so that methods are compared on equal terms.
Methods (all parameters global, fitted on the training period by minimizing log-loss):
  STATIC   running mean of evidence with a prior pseudo-count (no aging)
  EWMA     event-count forgetting factor (no calendar-time aging)
  TWM      exponentially time-weighted mean with prior = beta-reputation
           forgetting with graded evidence (state = evidence masses, not grade-memoryless)
  FADE-LIN linear fading: h=id aging between events + linear pooling update
  FADE-LOG logarithmic fading: h=logit aging + logarithmic pooling update
Run:  python experiment.py   (writes exp_results.json and figures)
"""
import json, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy import stats

EPS = 1e-4
DAY = 86400.0

def load(path):
    d = pd.read_csv(path, header=None, names=["src", "dst", "rating", "time"])
    d = d.sort_values("time", kind="mergesort").reset_index(drop=True)
    d["t"] = (d["time"] - d["time"].min()) / DAY          # days
    d["y"] = (d["rating"] > 0).astype(int)
    d["e"] = 0.5 + 0.475 * d["rating"] / 10.0              # evidence grade in [0.025, 0.975]
    _, d["uid"] = np.unique(d["dst"].values, return_inverse=True)
    return d

# ---------------------------------------------------------------- predictors
def run(method, th, uid, t, e, nusers):
    """Return the grade predicted for each event (before its own update)."""
    n = len(uid); p = np.empty(n); hist = np.zeros(n, bool)
    if method == "STATIC":
        a, n0 = th; S = np.zeros(nusers); C = np.zeros(nusers)
        for k in range(n):
            j = uid[k]; p[k] = (S[j] + n0 * a) / (C[j] + n0); hist[k] = C[j] > 0
            S[j] += e[k]; C[j] += 1
    elif method == "EWMA":
        a, lam = th; U = np.full(nusers, a); C = np.zeros(nusers)
        for k in range(n):
            j = uid[k]; p[k] = U[j]; hist[k] = C[j] > 0
            U[j] = lam * U[j] + (1 - lam) * e[k]; C[j] += 1
    elif method == "TWM":
        a, W, c = th; R = np.zeros(nusers); Sx = np.zeros(nusers); L = np.zeros(nusers); C = np.zeros(nusers)
        for k in range(n):
            j = uid[k]; f = np.exp(-c * (t[k] - L[j])); R[j] *= f; Sx[j] *= f; L[j] = t[k]
            p[k] = (R[j] + W * a) / (R[j] + Sx[j] + W); hist[k] = C[j] > 0
            R[j] += e[k]; Sx[j] += 1 - e[k]; C[j] += 1
    elif method in ("FADE-LIN", "FADE-LOG"):
        pi, d, eta = th
        h, hi = (lambda u: u, lambda z: z) if method == "FADE-LIN" else (logit, expit)
        c = h(pi); Z = np.full(nusers, c); L = np.zeros(nusers); C = np.zeros(nusers); he = h(e)
        for k in range(n):
            j = uid[k]; Z[j] = c + (Z[j] - c) * np.exp(-d * (t[k] - L[j])); L[j] = t[k]   # aging (Theorem 2, form R)
            p[k] = hi(Z[j]); hist[k] = C[j] > 0
            Z[j] = (1 - eta) * Z[j] + eta * he[k]; C[j] += 1                              # h-pooling update
    else:
        raise ValueError(method)
    return np.clip(p, EPS, 1 - EPS), hist

# parameter transforms: unconstrained -> admissible
TRANS = {
    "STATIC":   (lambda x: (expit(x[0]), np.exp(x[1])), [2.0, 0.0]),
    "EWMA":     (lambda x: (expit(x[0]), expit(x[1])), [2.0, 1.0]),
    "TWM":      (lambda x: (expit(x[0]), np.exp(x[1]), np.exp(x[2])), [2.0, 0.0, -5.0]),
    "FADE-LIN": (lambda x: (expit(x[0]), np.exp(x[1]), expit(x[2])), [2.0, -5.0, -1.0]),
    "FADE-LOG": (lambda x: (expit(x[0]), np.exp(x[1]), expit(x[2])), [2.0, -5.0, -1.0]),
}

def calib_fit(p, y, iters=25):
    """Logistic calibration q = sigmoid(a + b*logit(p)); returns (a, b). Newton-Raphson."""
    x = logit(p); a, b = 0.0, 1.0
    for _ in range(iters):
        q = expit(a + b * x); w = q * (1 - q) + 1e-12; g0 = np.sum(q - y); g1 = np.sum((q - y) * x)
        H00 = np.sum(w); H01 = np.sum(w * x); H11 = np.sum(w * x * x) + 1e-9
        det = H00 * H11 - H01 * H01
        da = (H11 * g0 - H01 * g1) / det; db = (-H01 * g0 + H00 * g1) / det
        a -= da; b -= db
        if abs(da) + abs(db) < 1e-10: break
    return a, b

def calib_apply(p, ab): return np.clip(expit(ab[0] + ab[1] * logit(p)), EPS, 1 - EPS)

def logloss(p, y): return -(y * np.log(p) + (1 - y) * np.log(1 - p))
def brier(p, y): return (p - y) ** 2
def auc(p, y):
    r = stats.rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def fit(method, D, train, fixed=None):
    tr, x0 = TRANS[method]
    uid, t, e, y = D["uid"].values, D["t"].values, D["e"].values, D["y"].values; nu = uid.max() + 1
    def obj(x):
        th = list(tr(x))
        if fixed:
            for i, v in fixed.items(): th[i] = v
        p, hist = run(method, th, uid, t, e, nu); m = train & hist
        ab = calib_fit(p[m], y[m])
        return logloss(calib_apply(p[m], ab), y[m]).mean()
    best = None
    for start in (x0, [v + 1.5 for v in x0], [v - 1.5 for v in x0]):
        r = minimize(obj, start, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": 600})
        if best is None or r.fun < best.fun: best = r
    th = list(tr(best.x))
    if fixed:
        for i, v in fixed.items(): th[i] = v
    return th, best.fun

def evaluate(method, th, D, test):
    uid, t, e, y = D["uid"].values, D["t"].values, D["e"].values, D["y"].values
    p, hist = run(method, th, uid, t, e, uid.max() + 1)
    tr = (~test) & hist; ab = calib_fit(p[tr], y[tr])
    return calib_apply(p, ab), test & hist

def holm(pvals):
    k = len(pvals); order = np.argsort(pvals); adj = np.empty(k); run_max = 0
    for rank, i in enumerate(order):
        run_max = max(run_max, (k - rank) * pvals[i]); adj[i] = min(1.0, run_max)
    return adj

def study(name, path, split=0.6, min_events=5, seed=20261023):
    D = load(path); y = D["y"].values
    cut = D["t"].quantile(split); train = (D["t"] <= cut).values; test = ~train
    out = {"dataset": name, "n_events": len(D), "n_users_rated": int(D["uid"].max() + 1),
           "share_negative": float(1 - y.mean()), "split_day": float(cut)}
    methods = ["STATIC", "EWMA", "TWM", "FADE-LIN", "FADE-LOG"]
    P, M, params = {}, None, {}
    for meth in methods:
        th, trloss = fit(meth, D, train); p, m = evaluate(meth, th, D, test)
        P[meth] = p; M = m; params[meth] = {"theta": [float(v) for v in th], "train_logloss": float(trloss)}
    yt = y[M]
    res = {}
    for meth in methods:
        p = P[meth][M]
        res[meth] = {"logloss": float(logloss(p, yt).mean()), "brier": float(brier(p, yt).mean()), "auc": float(auc(p, yt))}
    out["n_test_events"] = int(M.sum()); out["params"] = params; out["test"] = res
    # cluster bootstrap CI for AUC difference FADE-LOG minus each
    rng = np.random.default_rng(seed); users = D["uid"].values[M]; uu = np.unique(users)
    idx_by_user = {u: np.where(users == u)[0] for u in uu}
    boots = {m: [] for m in methods if m != "FADE-LOG"}
    for _ in range(1000):
        samp = rng.choice(uu, len(uu), replace=True); ii = np.concatenate([idx_by_user[u] for u in samp])
        ys = yt[ii]
        if ys.min() == ys.max(): continue
        a_log = auc(P["FADE-LOG"][M][ii], ys)
        for m in boots: boots[m].append(a_log - auc(P[m][M][ii], ys))
    out["auc_diff_vs_FADE-LOG_CI95"] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boots.items()}
    # per-user blocks for Friedman / Wilcoxon on log-loss
    dfb = pd.DataFrame({"u": users})
    for meth in methods: dfb[meth] = logloss(P[meth][M], yt)
    g = dfb.groupby("u"); cnt = g.size(); keep = cnt[cnt >= min_events].index
    B = g.mean().loc[keep, methods]
    fr = stats.friedmanchisquare(*[B[m].values for m in methods])
    ranks = B.rank(axis=1).mean()
    wil = {}
    others = [m for m in methods if m != "FADE-LOG"]
    pv = []
    for m in others:
        w = stats.wilcoxon(B["FADE-LOG"], B[m], alternative="two-sided"); pv.append(w.pvalue)
        wil[m] = {"median_diff": float(np.median(B["FADE-LOG"] - B[m])), "p_raw": float(w.pvalue)}
    for m, pa in zip(others, holm(np.array(pv))): wil[m]["p_holm"] = float(pa)
    out["blocks"] = {"n_users": int(len(B)), "min_events": min_events, "friedman_chi2": float(fr.statistic),
                     "friedman_p": float(fr.pvalue), "mean_rank_logloss": {m: float(ranks[m]) for m in methods},
                     "wilcoxon_FADE-LOG_vs": wil}
    # ablation (FADE-LOG): no aging; self-dual prior pi = 1/2; linear rule is FADE-LIN above
    abl = {}
    for lab, fixed in (("no_aging_delta0", {1: 0.0}), ("selfdual_prior_0.5", {0: 0.5})):
        th, _ = fit("FADE-LOG", D, train, fixed=fixed); p, m = evaluate("FADE-LOG", th, D, test)
        abl[lab] = {"theta": [float(v) for v in th], "logloss": float(logloss(p[m], y[m]).mean()),
                    "brier": float(brier(p[m], y[m]).mean()), "auc": float(auc(p[m], y[m]))}
    out["ablation_FADE-LOG"] = abl
    # sensitivity of FADE-LOG to delta and eta (others fixed at fitted values)
    th0 = params["FADE-LOG"]["theta"]; sens = {"delta": {}, "eta": {}}
    for f in (0.25, 0.5, 1, 2, 4):
        th = [th0[0], th0[1] * f, th0[2]]; p, m = evaluate("FADE-LOG", th, D, test)
        sens["delta"][str(f)] = {"logloss": float(logloss(p[m], y[m]).mean()), "auc": float(auc(p[m], y[m]))}
        eta = float(np.clip(th0[2] * f, 1e-3, 0.999)); th = [th0[0], th0[1], eta]; p, m = evaluate("FADE-LOG", th, D, test)
        sens["eta"][str(f)] = {"eta": eta, "logloss": float(logloss(p[m], y[m]).mean()), "auc": float(auc(p[m], y[m]))}
    out["sensitivity_FADE-LOG"] = sens
    # choice of pooling rule by out-of-sample likelihood (M3.4)
    out["pooling_rule_choice"] = {"delta_test_logloss_LIN_minus_LOG": res["FADE-LIN"]["logloss"] - res["FADE-LOG"]["logloss"]}
    return out

if __name__ == "__main__":
    R = [study("Bitcoin OTC", "data/soc-sign-bitcoinotc.csv"), study("Bitcoin Alpha", "data/soc-sign-bitcoinalpha.csv")]
    json.dump(R, open("exp_results.json", "w"), indent=1)
    print(json.dumps(R, indent=1))
