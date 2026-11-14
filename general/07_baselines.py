"""Naive baselines next to the citation model: what do the 58 features add?

The models of 04_citation_features.ipynb (ridge and boosted trees, the same settings and time split: trained on
cohorts before 2025-04, tested on the later ones) are refitted on small sets of features that a reader could guess
alone: the journal, the team size, the topics, the first author's record. For each, on the test papers:
  rho         Spearman rank correlation with the citation percentile
  AUC         for picking the top 10% of each month's papers
  top 10% hit share of the model's top 10% (by predicted percentile) that are in the actual top 10%, and the same
              divided by the share of actual top-10% papers among the test papers ("times chance")
The rank of the papers 6 months after posting, as a predictor of today's rank, comes from 05_rank_stability.py (on
papers at least 24 months old, so not the same test papers): it is the reference for "early citations".
Writes general/outputs/baselines.csv. Run from the repository root (after steps 2-5b):
    python general/07_baselines.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, GEN_PLOTS  # noqa: E402

TEST_FROM = "2025-04"      # as in 04_citation_features.ipynb
RIDGE_ALPHAS = np.logspace(-1, 4, 30)

F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
SETS = {
    "all features": FEATURES,
    "journal only": ["refereed", "letter"] + [c for c in FEATURES if c.startswith("journal_")],
    "team size only": ["log_n_authors", "large_collab"],
    "topics only": [c for c in FEATURES if c.startswith("topic_") and c not in ("topic_rank_earliest",)],
    "first author's record only": ["fa_log_prior_papers", "fa_career_years", "fa_shared_name"],
    "team size + first author + topics": ["log_n_authors", "large_collab", "fa_log_prior_papers", "fa_career_years",
                                          "fa_shared_name"] + [c for c in FEATURES if c.startswith("topic_")
                                                               and c != "topic_rank_earliest"],
}


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=RIDGE_ALPHAS))


def trees():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


train = (F["cohort"] < TEST_FROM).to_numpy()
test = ~train
y = F["pct"].to_numpy()
top = F.loc[test, "top10"].astype(bool).to_numpy()
base_rate = top.mean()
rows = []
for name, cols in SETS.items():
    for mname, model in (("ridge", ridge()), ("trees", trees())):
        p = model.fit(F.loc[train, cols], y[train]).predict(F.loc[test, cols])
        k = int(round(0.10 * test.sum()))
        picked = np.argsort(-p)[:k]
        hit = top[picked].mean()
        rows.append({"features": name, "n features": len(cols), "model": mname,
                     "test rho": spearmanr(p, y[test]).statistic, "AUC (top 10%)": roc_auc_score(top, p),
                     "top 10% hit rate": hit, "times chance": hit / base_rate})
df = pd.DataFrame(rows)
stab = pd.read_csv(os.path.join(GEN_PLOTS, "rank_stability_summary.csv"))
ref = stab[stab["comparison"].str.startswith("rank at 6 months")].iloc[0]
df = pd.concat([df, pd.DataFrame([{"features": "rank 6 months after posting (05_rank_stability, older papers)",
                                   "n features": 1, "model": "-", "test rho": ref["spearman_rho"],
                                   "top 10% hit rate": ref["top10_early_still_top10_pct"] / 100,
                                   "times chance": ref["top10_early_still_top10_pct"] / 10}])], ignore_index=True)
pd.set_option("display.width", 200)
print(f"{train.sum()} training, {test.sum()} test papers; actual top 10% among test papers: {100 * base_rate:.1f}%")
print(df.round(3).to_string(index=False))
df.round(4).to_csv(os.path.join(GEN_OUT, "baselines.csv"), index=False)
print("-> general/outputs/baselines.csv")
