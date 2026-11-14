"""The citation-model effects made readable, and corrected for testing 58 of them at once.

The ridge association model of 04_citation_features.ipynb (all modelled papers, features standardised, journals
coded against ApJ, penalty chosen by cross-validation) is refitted here with 2,000 bootstrap refits, so that every
effect has a 95% interval and a two-sided bootstrap p-value (twice the share of refits on the other side of zero,
at least 1/2,000). The p-values are corrected for the number of effects with the Benjamini-Hochberg false-discovery
rate (q < 0.05).

Readable effects, in percentile points (places out of 100 among papers posted the same month):
  yes/no features   the gap between papers with and without the feature, everything else fixed:
                    effect per standard deviation / sqrt(p (1 - p)), with p the share of papers that have it
  other features    the effect of one standard deviation, and the feature's standard deviation in its own units
Effects per natural unit (the alternative Figure 2): yes/no features as above; features stored as log10 per doubling
(effect per sd x log10(2) / sd); counts and years per a stated step (UNITS); any other feature stays per sd.
Writes general/outputs/effects_table.csv, and general/plots/effects_fdr.png (a, how many of 58 truly zero effects
pass p < 0.05 by chance, simulated; b, the 58 p-values against the Benjamini-Hochberg line) and
general/plots/effects_vif.png (the variance inflation factor of every feature: how much of it the other 57 predict).
Run from the repository root (after step 4):
    python general/08_effects.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, GEN_PLOTS, data  # noqa: E402
from common.figstyle import CAT, MUTED, SECONDARY, apply_style  # noqa: E402

N_BOOT = 2000
Q = 0.05
RIDGE_ALPHAS = np.logspace(-1, 4, 30)
# Natural unit of each continuous feature: (label, step in the feature's own units). Log10 features: one doubling.
DOUBLING = ("per doubling", np.log10(2))
UNITS = {"log_n_authors": DOUBLING, "log_references": DOUBLING, "log_pages": DOUBLING, "fa_log_prior_papers": DOUBLING,
         "abs_log_words": DOUBLING, "log_topic_papers_before": DOUBLING,
         "n_programmes": ("per programme", 1), "n_countries": ("per country", 1), "n_regions": ("per continent", 1),
         "abs_claims": ("per claim", 1), "fa_career_years": ("per 10 years", 10), "title_words": ("per 5 words", 5)}

F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
meta = json.load(open(data("features_ads_metadata.json")))
apjs = F["bibcode"].map(lambda b: "Astrophysical Journal Supplement" in ((meta.get(b) or {}).get("pub") or ""))
X = F[FEATURES].drop(columns=["refereed", "journal_apj"]).assign(preprint=1 - F["refereed"],
                                                                 journal_apjs=apjs.astype(float))
y = F["pct"].to_numpy()
raw = SimpleImputer(strategy="median").fit_transform(X)
mean, sd = raw.mean(0), raw.std(0)
Z = (raw - mean) / np.where(sd > 0, sd, 1)
alpha = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      RidgeCV(alphas=RIDGE_ALPHAS)).fit(X, y)[-1].alpha_
coef = Ridge(alpha=alpha).fit(Z, y).coef_
rng = np.random.default_rng(0)
boot = np.array([Ridge(alpha=alpha).fit(Z[i], y[i]).coef_ for i in (rng.integers(0, len(y), len(y))
                                                                    for _ in range(N_BOOT))])
p = np.maximum(2 * np.minimum((boot <= 0).mean(0), (boot >= 0).mean(0)), 1 / N_BOOT)

# Benjamini-Hochberg: sort the p-values, compare the i-th smallest with q i / m, keep all up to the last that passes
order = np.argsort(p)
m = len(p)
passed = p[order] <= Q * np.arange(1, m + 1) / m
k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
survives = np.zeros(m, bool)
survives[order[:k]] = True
q_adj = np.empty(m)
q_adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1].clip(max=1)

binary = np.array([set(np.unique(X[c].dropna())) <= {0.0, 1.0} for c in X.columns])
share = np.where(binary, raw.mean(0), np.nan)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
scale = np.where(binary, np.sqrt(share * (1 - share)), 1.0)
# per natural unit: the coefficient is per sd, so multiply by (step / sd); yes/no features: with vs. without
unit = [("with vs. without" if b else UNITS.get(c, ("per sd", None))[0]) for c, b in zip(X.columns, binary)]
step = np.array([np.nan if b else (UNITS.get(c, (None, np.nan))[1]) for c, b in zip(X.columns, binary)], float)
uscale = np.where(binary, scale, np.where(np.isnan(step), 1.0, sd / np.where(np.isnan(step), 1, step)))
df = pd.DataFrame({
    "feature": X.columns, "yes/no": binary, "share with it": share, "sd (own units)": np.where(binary, np.nan, sd),
    "effect per sd": coef, "low": lo, "high": hi,
    "readable effect (places)": coef / scale, "readable low": lo / scale, "readable high": hi / scale,
    "unit": unit, "effect per unit": coef / uscale, "unit low": lo / uscale, "unit high": hi / uscale,
    "p (bootstrap)": p, "q (BH)": q_adj, "survives FDR 5%": survives,
}).sort_values("effect per sd", key=np.abs, ascending=False).reset_index(drop=True)
pd.set_option("display.width", 220)
print(f"ridge alpha {alpha:.1f}; {len(y)} papers; {m} effects; {N_BOOT} bootstrap refits")
print(f"95% interval excludes 0: {int(((lo > 0) | (hi < 0)).sum())}; survive the false-discovery-rate correction "
      f"(q < {Q}): {int(survives.sum())}")
show = ["feature", "yes/no", "share with it", "effect per sd", "readable effect (places)", "readable low",
        "readable high", "p (bootstrap)", "q (BH)", "survives FDR 5%"]
print(df[show].head(30).round(3).to_string(index=False))
df.round(4).to_csv(os.path.join(GEN_OUT, "effects_table.csv"), index=False)
print("-> general/outputs/effects_table.csv")

# ── figure: why the correction, and where the Benjamini-Hochberg line cuts ──
apply_style()
ps = p[order]
rank = np.arange(1, m + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.3})
ax = axes[0]
null = np.sort(np.random.default_rng(1).uniform(size=m))   # m effects that are all truly zero
ax.scatter(rank, null, s=14, color=[CAT[1] if v < Q else MUTED for v in null], zorder=3)
ax.axhline(Q, color=CAT[1], lw=1, ls="--")
ax.text(m, Q + 0.02, f"p = {Q}", ha="right", color=CAT[1], fontsize=8.5)
ax.set_xlabel("Effect, sorted by p-value")
ax.set_ylabel("p-value")
ax.set_title(f"a  {m} effects that are all truly zero (simulated):\n{int((null < Q).sum())} pass p < {Q} by chance alone",
             loc="left", fontsize=10, fontweight="bold")
ax = axes[1]
ax.plot(rank, Q * rank / m, color=CAT[0], lw=1.5, label=f"Benjamini–Hochberg line: {Q} × rank / {m}")
ax.axhline(Q, color=CAT[1], lw=1, ls="--", label=f"p = {Q}, no correction")
ax.scatter(rank, ps, s=16, color=[CAT[0] if j < k else MUTED for j in range(m)], zorder=3)
ax.axvline(k + 0.5, color=SECONDARY, lw=0.8, ls=":")
ax.text(k + 1.2, 0.33, f"{k} survive", color=CAT[0], fontsize=9, fontweight="bold")
ax.set_xlim(0, 40)
ax.set_ylim(0, 0.4)
ax.set_xlabel(f"Effect, sorted by p-value (first 40 of {m})")
ax.set_ylabel("Bootstrap p-value")
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_title(f"b  The {m} ridge effects: {int((p < Q).sum())} have p < {Q},\n{k} survive the false-discovery-rate correction",
             loc="left", fontsize=10, fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.98, top=0.83, bottom=0.13)
fig.savefig(os.path.join(GEN_PLOTS, "effects_fdr.png"), dpi=150)
print("-> general/plots/effects_fdr.png")

# ── figure: variance inflation factors of the 58 features (the collinearity check of notebook 04, test 9) ──
# VIF of a feature = 1 / (1 - R^2), with R^2 of the regression of that feature on the 57 others: how much the variance
# of its coefficient grows because the others carry the same information (the interval widens by sqrt(VIF)).
r2 = np.empty(m)
for j in range(m):
    A = np.c_[np.ones(len(Z)), np.delete(Z, j, axis=1)]
    res = Z[:, j] - A @ np.linalg.lstsq(A, Z[:, j], rcond=None)[0]
    r2[j] = 1 - res.var() / Z[:, j].var()
vif = 1 / np.maximum(1 - r2, 1e-9)
vo = np.argsort(-vif)
print(f"VIF: median {np.median(vif):.2f}, largest {X.columns[vo[0]]} {vif[vo[0]]:.2f}, {X.columns[vo[1]]} {vif[vo[1]]:.2f}; "
      f"{int((vif > 5).sum())} above 5")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"wspace": 0.55, "width_ratios": [1, 1.15]})
ax = axes[0]
rr = np.linspace(0, 0.95, 200)
ax.plot(rr, 1 / (1 - rr), color=CAT[0], lw=1.8)
for t in (5, 10):
    ax.axhline(t, color=MUTED, lw=0.8, ls="--")
    ax.text(0.01, t + 0.3, f"VIF = {t}", color=MUTED, fontsize=8)
ax.scatter(r2, vif, s=12, color=[CAT[1] if v > 5 else MUTED for v in vif], zorder=3)
j = vo[0]
ax.annotate(f"{X.columns[j]}: R² = {r2[j]:.2f}, VIF = {vif[j]:.1f}\n(interval {np.sqrt(vif[j]):.1f} times wider)",
            (r2[j], vif[j]), xytext=(0.05, 14), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=SECONDARY, lw=0.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 20)
ax.set_xlabel("R², the feature predicted from the other 57")
ax.set_ylabel("VIF = 1 / (1 − R²)")
ax.set_title("a  How much the other features already\nknow about a feature (dots: the 58 features)", loc="left", fontsize=10,
             fontweight="bold")
ax = axes[1]
top = vo[:20][::-1]
ax.barh(range(len(top)), vif[top], color=[CAT[1] if vif[t] > 5 else CAT[0] for t in top], height=0.65)
ax.set_yticks(range(len(top)), [X.columns[t] for t in top], fontsize=8)
for t in (1, 5):
    ax.axvline(t, color=MUTED, lw=0.8, ls="--")
ax.set_xlabel("Variance inflation factor")
ax.set_title(f"b  The 20 largest (median of all {m}: {np.median(vif):.1f})", loc="left", fontsize=10, fontweight="bold")
fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.12)
fig.savefig(os.path.join(GEN_PLOTS, "effects_vif.png"), dpi=150)
print("-> general/plots/effects_vif.png")
