"""Supplementary code for 'Fading fuzzy sets' manuscript.
Reproduces every number, table and figure in the paper. Seeds fixed.
Run: python fading_supplement.py   (numpy, scipy, mpmath, matplotlib)"""
import json, numpy as np, mpmath as mp
from scipy.special import expit, logit
from scipy.optimize import brentq
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
mp.mp.dps = 50
OUT = {}

# ---------- aging laws ----------
def lin(g, p, d, t):  return p + (g - p) * np.exp(-d * t)
def lgt(g, p, d, t):  return expit(logit(p) + (logit(g) - logit(p)) * np.exp(-d * t))
def t_hit(g, p, d, a, h):   # time at which grade reaches level a (Corollary: time-to-threshold)
    return np.log((h(g) - h(p)) / (h(a) - h(p))) / d
ident = lambda u: u

# ---------- Section 9: illustrative example ----------
X = {"x1": (.95, .05, .10), "x2": (.70, .30, .02), "x3": (.40, .35, .01),
     "x4": (.60, .05, .05), "x5": (.85, .05, .03)}
T = [0, 24, 72]
tab = {}
for k, (g, p, d) in X.items():
    row = {"g": g, "pi": p, "delta": d,
           "lin": [round(float(lin(g, p, d, t)), 3) for t in T],
           "log": [round(float(lgt(g, p, d, t)), 3) for t in T]}
    if g > .5 > p:
        row["t05_lin"] = round(float(t_hit(g, p, d, .5, ident)), 2)
        row["t05_log"] = round(float(t_hit(g, p, d, .5, logit)), 2)
    tab[k] = row
OUT["example_table"] = tab
for rule, f in (("lin", lin), ("log", lgt)):
    OUT[f"ranking_{rule}"] = {t: sorted(X, key=lambda k: -f(*X[k], t)) for t in T}

def crossings(fa, fb, S=2000.0, n=400001):
    """roots of fa-fb on (0,S]; grid in t, refined by brentq; differences evaluated
    in h-space-safe form (no exact-zero underflow because pi_A != pi_B here)."""
    t = np.linspace(1e-9, S, n); d = fa(t) - fb(t)
    idx = np.where(np.sign(d[1:]) * np.sign(d[:-1]) < 0)[0]
    return [round(brentq(lambda s: fa(s) - fb(s), t[i], t[i + 1]), 4) for i in idx]
pairs = {}
for i, a in enumerate(X):
    for b in list(X)[i + 1:]:
        for rule, f in (("lin", lin), ("log", lgt)):
            pairs[f"{a}-{b}-{rule}"] = crossings(lambda s: f(*X[a], s), lambda s: f(*X[b], s))
OUT["example_crossings"] = {k: v for k, v in pairs.items() if v}

# ---------- tightness examples ----------
A2, B2 = (.95, .30, 5.0), (.90, .20, .5)
OUT["tight_same_rule_lin"] = crossings(lambda s: lin(*A2, s), lambda s: lin(*B2, s), S=50, n=500001)

pA, gA, a, b, k, c = map(mp.mpf, ('0.010468029221414033', '0.4768616672759329', '0.08834682664035878',
                                   '0.03421571812878666', '4.492157733419383', '-4.59'))
sig = lambda z: 1 / (1 + mp.e ** (-z)); lg = lambda u: mp.log(u / (1 - u))
pB, gB = sig(c), sig(c + k)
DA = lambda s: pA + (gA - pA) * mp.e ** (-a * s); DB = lambda s: sig(c + k * mp.e ** (-b * s))
D = lambda s: DA(s) - DB(s)
grid = [mp.mpf(i) / 50 for i in range(0, 50 * 1500 + 1)]
vals = [D(x) for x in grid]
roots4 = [mp.findroot(D, (grid[i], grid[i + 1]), solver='anderson') for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0]
OUT["four_crossing"] = {"A_linear": [float(gA), float(pA), float(a)], "B_log": [float(gB), float(pB), float(b)],
                        "roots": [float(mp.nstr(r, 10)) for r in roots4], "D0": float(D(0)), "Dinf": float(pA - pB)}

# ---------- Monte Carlo checks (Appendix A) ----------
rng = np.random.default_rng(20261023)
s = np.concatenate([np.geomspace(1e-6, 1, 3000), np.linspace(1, 300, 3000)[1:]])
def count(D):
    sg = np.sign(D); return np.sum(sg[:, 1:] * sg[:, :-1] < 0, axis=1)
N, B = 100000, 5000
mx = {"RR": 0, "RR_eq": 0, "DR": 0, "DD": 0, "lin_vs_log": 0}
for _ in range(N // B):
    zA, zB, cA, cB = rng.normal(0, 2, (4, B, 1)); dA, dB = np.exp(rng.uniform(np.log(1e-2), np.log(1e2), (2, B, 1)))
    bA, bB = rng.normal(0, 1, (2, B, 1))
    RA = cA + (zA - cA) * np.exp(-dA * s); RB = cB + (zB - cB) * np.exp(-dB * s)
    mx["RR"] = max(mx["RR"], count(RA - RB).max())
    mx["RR_eq"] = max(mx["RR_eq"], count(RA - (cB + (zB - cB) * np.exp(-dA * s))).max())
    mx["DR"] = max(mx["DR"], count(zA + bA * s - RB).max())
    mx["DD"] = max(mx["DD"], count((zA - zB) + (bA - bB) * s).max())
    g1, p1, g2, p2 = rng.uniform(1e-3, 1 - 1e-3, (4, B, 1))
    mx["lin_vs_log"] = max(mx["lin_vs_log"], count(lin(g1, p1, dA, s) - lgt(g2, p2, dB, s)).max())
OUT["monte_carlo"] = {"pairs_per_case": N, "seed": 20261023, "max_crossings": {k: int(v) for k, v in mx.items()}}

# ---------- Remark: beta-reputation aging is not a grade-only semigroup ----------
E = lambda r, s_: (r + 1) / (r + s_ + 2)
OUT["beta_reputation"] = {"before": [E(1, 0), E(3, 1)], "after_halving": [E(.5, 0), E(1.5, .5)]}

# ---------- Section 7: convexity counterexample ----------
x = np.linspace(-5, 15, 20001); f = .5 * np.exp(-x ** 2) + .5 * np.exp(-(x - 10) ** 2)
OUT["convexity_counterexample"] = {"f(0)": float(f[np.argmin(abs(x))]), "f(10)": float(f[np.argmin(abs(x - 10))]),
                                   "f(5)": float(f[np.argmin(abs(x - 5))])}
json.dump(OUT, open("results.json", "w"), indent=1, default=str)

# ---------- figures ----------
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63"
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
tt = np.linspace(0, 96, 961)
fig, ax = plt.subplots(figsize=(5.2, 3.0))
from matplotlib.lines import Line2D
handles, labels = [], []
for key, ls in (("x1", "-"), ("x4", "--")):
    for rule, f, col in (("linear", lin, BLUE), ("logarithmic", lgt, ORANGE)):
        ax.plot(tt, f(*X[key], tt), color=col, lw=1.6, ls=ls)
        handles.append(Line2D([], [], color=col, lw=1.6, ls=ls)); labels.append(f"{key}, {rule} pooling")
ax.axhline(.5, color=MUTED, lw=.6, ls=":"); ax.text(95, .515, "review threshold 0.5", fontsize=7, color=MUTED, ha="right")
ax.set_xlabel("hours since the transaction"); ax.set_ylabel("membership grade"); ax.set_xlim(0, 96); ax.set_ylim(0, 1)
ax.legend(handles, labels, frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1, .95))
fig.tight_layout(); fig.savefig("fig1_example.png", dpi=300); fig.savefig("fig1_example.eps")

ss = np.geomspace(1e-2, 1500, 4000)
Dn = lin(float(gA), float(pA), float(a), ss) - lgt(float(gB), float(pB), float(b), ss)
fig, ax = plt.subplots(figsize=(5.2, 2.8))
ax.plot(ss, Dn * 1e3, color=BLUE, lw=1.6); ax.axhline(0, color=MUTED, lw=.7)
for r in OUT["four_crossing"]["roots"]:
    ax.plot([r], [0], "o", ms=5, mfc="white", mec=INK, mew=1.2)
ax.set_xscale("log"); ax.set_xlabel("elapsed time s (log scale)"); ax.set_ylabel(r"$u_A(s)-u_B(s)$  ($\times 10^{-3}$)")
fig.tight_layout(); fig.savefig("fig2_four_crossings.png", dpi=300); fig.savefig("fig2_four_crossings.eps")
print(json.dumps(OUT, indent=1, default=str))
