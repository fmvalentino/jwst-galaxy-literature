"""05 · Rank stability: does an early citation rank last, and does the model predict the lasting rank?

The time split of 04 predicts papers the model never saw, but their citations are young (3-18 months), so
it tests prediction of the EARLY citation rank. Here the papers old enough to have a lasting rank (posted
at least 24 months before the citation snapshot) are followed back in time: their citations are dated by
the citing paper's arXiv posting month (its publication month when it has no preprint), and their
within-month percentile is rebuilt as it stood 6 and 12 months after posting. Three questions:

1. How well does the rank at 6 or 12 months predict the rank today?
2. Of the top 10% at 6 or 12 months, how many are still top 10% today?
3. The model of 04 (same features, same trees), trained on the 6-month rank with whole months held out
   (5-fold), scored against the 6-month rank and against today's rank: does it predict lasting impact?

Citation lists: data/<freeze>/features_citing_authors.json (pulled by 04, test 11). Dates of citing papers:
data/<freeze>/citing_dates.json, built from the caches and one ADS bigquery for the rest (needs ADS_TOKEN
only when the cache is incomplete). Run from the repository root after 04:
    python general/05_rank_stability.py
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.ads import bigquery, load_json, save_json  # noqa: E402
from common.config import CITES_AS_OF, GEN_OUT, GEN_PLOTS, data  # noqa: E402
from common.figstyle import ACCENT, NEUTRAL, apply_style  # noqa: E402

MIN_AGE_LASTING = 24            # months between posting and the snapshot for a paper to have a "lasting" rank
WINDOWS = (6, 12)               # early ranks, months after posting
SNAP = pd.Period(CITES_AS_OF, "M")

# ── 1 · dates of the citing papers ─────────────────────────────────────────────────────────────────
cited = json.load(open(data("features_citing_authors.json")))["cited"]
citing = sorted({b for v in cited.values() for b in (v.get("citation") or [])})


def month_of(doc):
    """Posting month: the arXiv identifier's yymm if the paper has a preprint, else its publication month."""
    for i in doc.get("identifier") or []:
        m = re.match(r"(?:arXiv:)?(\d{2})(\d{2})\.\d{4,5}", i)
        if m:
            return f"20{m.group(1)}-{m.group(2)}"
    pd_ = doc.get("pubdate") or ""
    return pd_[:7] if re.match(r"\d{4}-\d{2}", pd_) and pd_[5:7] != "00" else None


DATES = data("citing_dates.json")
dates = load_json(DATES, {})
if len(dates) < len(citing):
    for src in (json.load(open(data("features_ads_metadata.json"))).values(),
                json.load(open(data("raw_candidate_search.json")))["docs"]):
        for doc in src:
            m = month_of(doc)
            if m and doc["bibcode"] not in dates:
                dates[doc["bibcode"]] = m
    for b in citing:                                   # arXiv-only bibcodes carry the month: 2023arXiv230113210
        m = re.match(r"\d{4}arXiv(\d{2})(\d{2})", b)
        if m and b not in dates:
            dates[b] = f"20{m.group(1)}-{m.group(2)}"
    missing = [b for b in citing if b not in dates]
    if missing:
        print(f"Pulling dates of {len(missing):,} citing papers from ADS ...")
        for doc in bigquery("*:*", missing, fl="bibcode,identifier,pubdate"):
            m = month_of(doc)
            if m:
                dates[doc["bibcode"]] = m
        save_json(DATES, dates)
    else:
        save_json(DATES, dates)
n_dated = sum(b in dates for b in citing)
print(f"Citing papers: {len(citing):,}; dated: {n_dated:,} ({100 * n_dated / len(citing):.1f}%)")

# ── 2 · early and current ranks ────────────────────────────────────────────────────────────────────
F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
F = F[F["pct"].notna() & F["bibcode"].isin(cited)].reset_index(drop=True)
posted = F["month"].map(lambda m: pd.Period(m, "M"))
F["age_months"] = [(SNAP - p).n for p in posted]
L = F[F["age_months"] >= MIN_AGE_LASTING].reset_index(drop=True)
posted = L["month"].map(lambda m: pd.Period(m, "M"))

n_links = n_undated = 0
rows = []
for b, p in zip(L["bibcode"], posted):
    cites = cited[b].get("citation") or []
    months = [pd.Period(dates[c], "M") for c in cites if c in dates]
    n_links += len(cites)
    n_undated += len(cites) - len(months)
    r = {"now": len(cites)}
    for k in WINDOWS:
        r[f"at_{k}m"] = sum(m <= p + k for m in months)
    rows.append(r)
C = pd.DataFrame(rows)
L = pd.concat([L, C], axis=1)
print(f"Lasting set: {len(L):,} papers posted {L['month'].min()} to {L['month'].max()} "
      f"(at least {MIN_AGE_LASTING} months before {CITES_AS_OF}); {n_links:,} citations, "
      f"{100 * n_undated / max(n_links, 1):.1f}% undated (counted today, not in the early windows)")


def pct(col):
    """Within-cohort percentile, as in 04: average rank of ties, scaled to 0-100."""
    return L.groupby("cohort")[col].rank(pct=True, method="average").mul(100)


L["pct_now"] = pct("now")
for k in WINDOWS:
    L[f"pct_{k}m"] = pct(f"at_{k}m")

# ── 3 · does the early rank last? ─────────────────────────────────────────────────────────────────
summary = []
for k in WINDOWS:
    e, n = L[f"pct_{k}m"], L["pct_now"]
    top_e, top_n = e >= 90, n >= 90
    summary.append({"comparison": f"rank at {k} months vs. today", "papers": len(L),
                    "spearman_rho": spearmanr(e, n).statistic,
                    "top10_early_still_top10_pct": 100 * (top_e & top_n).sum() / top_e.sum(),
                    "top10_today_already_top10_pct": 100 * (top_e & top_n).sum() / top_n.sum()})

# ── 4 · does the model predict the lasting rank? ─────────────────────────────────────────────────
# The trees of 04, trained on the 6-month rank, five folds that hold out whole posting months. Each
# paper's prediction comes from a model that never saw its month.
X = L[FEATURES]
pred = {}
for target in ("pct_6m", "pct_now"):
    p = np.zeros(len(L))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=L["cohort"]):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                          min_samples_leaf=30, l2_regularization=1.0, random_state=0)
        m.fit(X.iloc[tr], L[target].iloc[tr])
        p[te] = m.predict(X.iloc[te])
    pred[target] = p
L["pred_from_6m"] = pred["pct_6m"]
L["pred_from_now"] = pred["pct_now"]
for name, p, against in [("model trained on 6-month rank, scored on 6-month rank", pred["pct_6m"], "pct_6m"),
                         ("model trained on 6-month rank, scored on today's rank", pred["pct_6m"], "pct_now"),
                         ("model trained on today's rank, scored on today's rank", pred["pct_now"], "pct_now")]:
    summary.append({"comparison": name, "papers": len(L), "spearman_rho": spearmanr(p, L[against]).statistic})

S = pd.DataFrame(summary)
pd.set_option("display.width", 160)
print(S.round(3).to_string(index=False))

L[["bibcode", "month", "cohort", "age_months", "now", *[f"at_{k}m" for k in WINDOWS], "pct_now",
   *[f"pct_{k}m" for k in WINDOWS], "pred_from_6m", "pred_from_now"]].to_csv(
    os.path.join(GEN_OUT, "rank_stability.csv"), index=False)
S.to_csv(os.path.join(GEN_PLOTS, "rank_stability_summary.csv"), index=False)

apply_style()
# ── 5 · figure: early rank against today's rank ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
for ax, (col, lab) in zip(axes, [("pct_6m", "percentile at 6 months"), ("pct_12m", "percentile at 12 months"),
                                 ("pred_from_6m", "model prediction (features at posting)")]):
    ax.scatter(L[col], L["pct_now"], s=6, alpha=0.35, color=NEUTRAL, edgecolor="none")
    bins = pd.qcut(L[col], 10, duplicates="drop")
    g = L.groupby(bins, observed=True).agg(x=(col, "mean"), y=("pct_now", "mean"))
    ax.plot(g["x"], g["y"], color=ACCENT, lw=2, marker="o", ms=4)
    rho = spearmanr(L[col], L["pct_now"]).statistic
    ax.set_title(f"{lab}\nρ = {rho:.2f} with today's percentile", fontsize=10)
    ax.set_xlabel(lab)
    ax.plot([0, 100], [0, 100], color="#cccccc", lw=1, ls="--")
axes[0].set_ylabel("percentile today")
fig.suptitle(f"Rank stability: {len(L):,} papers posted at least {MIN_AGE_LASTING} months before {CITES_AS_OF} "
             "(blue: mean in ten bins)", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(GEN_PLOTS, "rank_stability.png"), dpi=200)
print("-> general/outputs/rank_stability.csv, general/plots/rank_stability_summary.csv, rank_stability.png")
