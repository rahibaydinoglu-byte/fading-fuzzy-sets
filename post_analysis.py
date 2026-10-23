"""Post-hoc analysis reusing fitted parameters from exp_results.json (no refitting)."""
import json, numpy as np, pandas as pd
from scipy import stats
import experiment as E
R = json.load(open("exp_results.json")); out = {}
methods = ["STATIC", "EWMA", "TWM", "FADE-LIN", "FADE-LOG"]
for r, path in zip(R, ["data/soc-sign-bitcoinotc.csv", "data/soc-sign-bitcoinalpha.csv"]):
    D = E.load(path); y = D["y"].values; cut = r["split_day"]; test = (D["t"] > cut).values
    P = {}; M = None
    for m in methods:
        p, M = E.evaluate(m, r["params"][m]["theta"], D, test); P[m] = p
    yt = y[M]; users = D["uid"].values[M]; res = {}
    # per-user blocks, FADE-LIN as reference
    dfb = pd.DataFrame({"u": users}); 
    for m in methods: dfb[m] = E.logloss(P[m][M], yt)
    g = dfb.groupby("u"); cnt = g.size(); keep = cnt[cnt >= 5].index; B = g.mean().loc[keep, methods]
    others = [m for m in methods if m != "FADE-LIN"]; pv = []; wil = {}
    for m in others:
        w = stats.wilcoxon(B["FADE-LIN"], B[m]); pv.append(w.pvalue)
        wil[m] = {"median_diff": float(np.median(B["FADE-LIN"] - B[m])), "share_users_better": float(np.mean(B["FADE-LIN"] < B[m])), "p_raw": float(w.pvalue)}
    for m, pa in zip(others, E.holm(np.array(pv))): wil[m]["p_holm"] = float(pa)
    res["wilcoxon_FADE-LIN_vs"] = wil
    # stratification by user activity in the test period (all users with >=1 test event)
    allu = g.mean()[methods]; c = cnt
    strata = {"1-4": (1, 4), "5-19": (5, 19), "20-99": (20, 99), ">=100": (100, 10**9)}
    st = {}
    for lab, (lo, hi) in strata.items():
        idx = c[(c >= lo) & (c <= hi)].index
        ev = dfb[dfb.u.isin(idx)]
        st[lab] = {"users": int(len(idx)), "events": int(len(ev)),
                   "mean_ll": {m: float(ev[m].mean()) for m in methods}}
    res["strata_by_test_events"] = st
    # bootstrap AUC difference FADE-LIN minus others (cluster by user)
    rng = np.random.default_rng(20261023); uu = np.unique(users); idx_by = {u: np.where(users == u)[0] for u in uu}; boots = {m: [] for m in others}
    for _ in range(1000):
        ii = np.concatenate([idx_by[u] for u in rng.choice(uu, len(uu), replace=True)]); ys = yt[ii]
        a = E.auc(P["FADE-LIN"][M][ii], ys)
        for m in others: boots[m].append(a - E.auc(P[m][M][ii], ys))
    res["auc_diff_FADE-LIN_minus_CI95"] = {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boots.items()}
    # half-lives in days
    res["half_life_days"] = {m: float(np.log(2) / r["params"][m]["theta"][1]) for m in ["FADE-LIN", "FADE-LOG"]}
    res["half_life_days"]["TWM"] = float(np.log(2) / r["params"]["TWM"]["theta"][2])
    out[r["dataset"]] = res
json.dump(out, open("post_analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
