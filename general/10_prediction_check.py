"""How well a paper's features predict its citation rank: the numbers behind the paper's Fig. 3.

The boosted trees of 04_citation_features.ipynb (same settings, same time split: trained on cohorts before 2025-04,
tested on the later ones), scored three ways on papers the model never saw:
  test        the time split: predicted percentile of every test paper
  5-fold CV   cross-validation holding out whole posting months (GroupKFold on the cohort): every paper predicted by
              a model that never saw its month
  forward     for every quarter from 2024 on with at least 40 papers, a model trained on all earlier papers predicts
              that quarter; 95% interval of rho from 500 bootstrap resamples of the quarter's papers
This repeats section 8 of the notebook (its Figure 4) and saves the numbers, so the paper's figure is drawn from
files. Writes general/outputs/prediction_check.csv (one row per modelled paper) and prediction_forward.csv (one row
per quarter). Run from the repository root (after step 5):
    python general/10_prediction_check.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT  # noqa: E402

TEST_FROM = "2025-04"      # as in 04_citation_features.ipynb

F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
X, y, top = F[FEATURES], F["pct"].to_numpy(), F["top10"].to_numpy()
train = (F["cohort"] < TEST_FROM).to_numpy()
test = ~train


def trees():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


# ── the time split ──
pred_test = np.full(len(y), np.nan)
pred_test[test] = trees().fit(X[train], y[train]).predict(X[test])

# ── 5-fold cross-validation, whole posting months held out ──
pred_cv, folds = np.full(len(y), np.nan), []
for a, b in GroupKFold(n_splits=5).split(X, y, groups=F["cohort"]):
    pred_cv[b] = trees().fit(X.iloc[a], y[a]).predict(X.iloc[b])
    folds.append(spearmanr(pred_cv[b], y[b]).statistic)

# ── forward in time, quarter by quarter ──
quarter = pd.PeriodIndex(F["month"], freq="Q")
rng = np.random.default_rng(0)
fwd = []
for q in sorted(set(quarter[quarter >= pd.Period("2024Q1")])):
    tr, te = (quarter < q), (quarter == q)
    if te.sum() < 40:
        continue
    pq, yq = trees().fit(X[tr], y[tr]).predict(X[te]), y[te]
    bs = [spearmanr(pq[i], yq[i]).statistic for i in (rng.integers(0, te.sum(), te.sum()) for _ in range(500))]
    fwd.append({"quarter": str(q), "papers": int(te.sum()), "rho": spearmanr(pq, yq).statistic,
                "low": np.percentile(bs, 2.5), "high": np.percentile(bs, 97.5), "AUC": roc_auc_score(top[te], pq)})
fwd = pd.DataFrame(fwd)


def found_in_top(pred, is_top, share=0.2):
    """Share of the actual top-10% papers among the papers the model ranks in its top `share`."""
    order = np.argsort(-pred)
    k = int(round(share * len(pred)))
    return is_top[order[:k]].sum() / is_top.sum()


print(f"test ({test.sum()} papers): rho {spearmanr(pred_test[test], y[test]).statistic:.3f}, "
      f"AUC {roc_auc_score(top[test], pred_test[test]):.3f}, "
      f"top 20% of predictions hold {found_in_top(pred_test[test], top[test]):.0%} of the top-10% papers")
print(f"5-fold CV by month ({len(y)} papers): rho {np.mean(folds):.3f} ± {np.std(folds):.3f} (mean ± s.d. of folds), "
      f"AUC {roc_auc_score(top, pred_cv):.3f}, top 20% hold {found_in_top(pred_cv, top):.0%}")
print(f"forward, {len(fwd)} quarters: rho {fwd['rho'].min():.2f}-{fwd['rho'].max():.2f} (mean {fwd['rho'].mean():.2f})")
print(fwd.round(3).to_string(index=False))
F[["bibcode", "month", "cohort", "pct", "top10"]].assign(
    split=np.where(train, "train", "test"), predicted_test=pred_test.round(3), predicted_cv=pred_cv.round(3)).to_csv(
    os.path.join(GEN_OUT, "prediction_check.csv"), index=False)
fwd.round(4).to_csv(os.path.join(GEN_OUT, "prediction_forward.csv"), index=False)
print("-> general/outputs/prediction_check.csv, prediction_forward.csv")
