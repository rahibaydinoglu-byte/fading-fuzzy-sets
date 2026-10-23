"""Figures 1-3 in the format required by Fuzzy Optimization and Decision Making:
119 mm wide, sans-serif lettering (Arial-compatible Liberation Sans) of 8-10 pt,
lines >= 0.3 pt, information readable in black and white, EPS with embedded fonts."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import expit, logit
W = 119 / 25.4
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "ps.fonttype": 42, "pdf.fonttype": 42})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#6b6a63"
lin = lambda g, p, d, t: p + (g - p) * np.exp(-d * t)
lgt = lambda g, p, d, t: expit(logit(p) + (logit(g) - logit(p)) * np.exp(-d * t))
X = {"x1": (.95, .05, .10), "x4": (.60, .05, .05)}
def save(fig, name):
    fig.savefig(name + ".eps"); fig.savefig(name + ".png", dpi=1200); fig.savefig(name + ".pdf")
# Fig 1
t = np.linspace(0, 96, 961); fig, ax = plt.subplots(figsize=(W, 0.62 * W))
for key, mk, col in (("x1", None, BLUE), ("x4", "o", ORANGE)):
    for rule, f, ls in (("linear", lin, "-"), ("logarithmic", lgt, "--")):
        ax.plot(t, f(*X[key], t), color=col, lw=1.4, ls=ls, marker=mk, markevery=60, ms=4, mfc="white",
                label=f"$x_{key[1]}$, {rule} pooling")
ax.axhline(.5, color=GREY, lw=.6, ls=":"); ax.text(95, .52, "review threshold 0.5", fontsize=8, color="#333333", ha="right")
ax.set_xlabel("Hours since the transaction"); ax.set_ylabel("Membership grade"); ax.set_xlim(0, 96); ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, .95)); fig.tight_layout(); save(fig, "Fig1")
# Fig 2
gA, pA, a = 0.4768616672759329, 0.010468029221414033, 0.08834682664035878
c, k, b = -4.59, 4.492157733419383, 0.03421571812878666
ss = np.geomspace(1e-2, 1500, 4000); D = (pA + (gA - pA) * np.exp(-a * ss)) - expit(c + k * np.exp(-b * ss))
fig, ax = plt.subplots(figsize=(W, 0.55 * W)); ax.plot(ss, D * 1e3, color=BLUE, lw=1.4); ax.axhline(0, color=GREY, lw=.6)
for r in (0.5574448, 3.936024, 21.35628, 137.0187): ax.plot([r], [0], "o", ms=5, mfc="white", mec="black", mew=1.0)
ax.set_xscale("log"); ax.set_xlabel("Elapsed time $s$ (logarithmic scale)"); ax.set_ylabel(r"$u_A(s)-u_B(s)$  ($\times10^{-3}$)")
fig.tight_layout(); save(fig, "Fig2")
# Fig 3
R = json.load(open("exp_results.json")); f = [0.25, 0.5, 1, 2, 4]
fig, axs = plt.subplots(1, 2, figsize=(W, 0.5 * W))
for ax, par, lab, tag in zip(axs, ["delta", "eta"], [r"Rate $\delta$ (multiple of fitted value)", r"Update weight $\eta$ (multiple)"], ["(a)", "(b)"]):
    for r, col, ls, mk in zip(R, [BLUE, ORANGE], ["-", "--"], ["o", "s"]):
        v = [r["sensitivity_FADE-LOG"][par][str(x)]["logloss"] for x in f]
        ax.plot(f, v, color=col, lw=1.4, ls=ls, marker=mk, ms=4, mfc="white", label=r["dataset"])
    ax.set_xscale("log", base=2); ax.set_xticks(f); ax.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
    ax.set_xlabel(lab); ax.axvline(1, color=GREY, lw=.6, ls=":"); ax.text(-0.22, 1.04, tag, transform=ax.transAxes, va="bottom", fontsize=9, fontweight="bold")
axs[0].set_ylabel("Test log-loss"); axs[1].legend(frameon=False, loc="upper center")
fig.tight_layout(); save(fig, "Fig3")
print("ok")
