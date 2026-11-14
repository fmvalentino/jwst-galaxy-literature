"""Sensitivity of the citation results to the sample definition: curated papers only.

The sample rule (common/jwst_use.py, step 2) accepts two kinds of data-use evidence: ADS's curated JWST
bibliography (bibgroup:JWST, checked by STScI) and explicit data use in the text. The text evidence is the less
certain of the two. This script refits the models of 04_citation_features.ipynb on the curated papers only, and,
as a reference for how much any such cut moves the results, on 50 random subsets of the same size. Percentiles are
recomputed within cohorts on the papers kept.

Reported: test Spearman rho and top-10% AUC (ridge and trees, the time split of 04), the correlation of the ridge
association coefficients with the whole-sample ones, the share of significant whole-sample effects that keep their
sign, and the main effects one by one. Writes general/outputs/sample_sensitivity_models.csv and
sample_sensitivity_effects.csv. Run from the repository root (after steps 2-5):
    python general/06_sample_sensitivity.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402

TEST_FROM = "2025-04"      # as in 04_citation_features.ipynb
N_RANDOM = 50
N_BOOT = 300
RIDGE_ALPHAS = np.logspace(-1, 4, 30)

F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
evidence = pd.read_csv(os.path.join(GEN_OUT, "sample_table.csv")).set_index("bibcode")["evidence"]
meta = json.load(open(data("features_ads_metadata.json")))
F["evidence"] = F["bibcode"].map(evidence)
F["apjs"] = F["bibcode"].map(lambda b: "Astrophysical Journal Supplement" in ((meta.get(b) or {}).get("pub") or ""))


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=RIDGE_ALPHAS))


def trees():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


def recompute_pct(D):
    D = D.reset_index(drop=True).copy()
    D["pct"] = D.groupby("cohort")["citations"].rank(pct=True, method="average").mul(100)
    D["top10"] = D["pct"] >= 90
    return D


def fit(D, boot=0):
    """Test rho / AUC of both models; ridge association coefficients (journals coded against ApJ, as in 04)."""
    train = (D["cohort"] < TEST_FROM).to_numpy()
    X, y = D[FEATURES], D["pct"].to_numpy()
    res = {}
    for name, model in (("ridge", ridge()), ("trees", trees())):
        pr = model.fit(X[train], y[train]).predict(X[~train])
        res[f"{name} rho"] = spearmanr(pr, y[~train]).statistic
        res[f"{name} AUC"] = roc_auc_score(D.loc[~train, "top10"], pr)
    XA = X.drop(columns=["refereed", "journal_apj"]).assign(preprint=1 - D["refereed"], journal_apjs=D["apjs"].astype(float))
    Z = SimpleImputer(strategy="median").fit_transform(XA)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    alpha = ridge().fit(XA, y)[-1].alpha_
    coef = pd.Series(Ridge(alpha=alpha).fit(Z, y).coef_, index=XA.columns)
    ci = None
    if boot:
        rng = np.random.default_rng(0)
        bs = np.array([Ridge(alpha=alpha).fit(Z[i], y[i]).coef_ for i in (rng.integers(0, len(y), len(y))
                                                                          for _ in range(boot))])
        ci = pd.DataFrame(np.percentile(bs, [2.5, 97.5], axis=0).T, index=XA.columns, columns=["low", "high"])
    return res, coef, ci


base_res, base_coef, base_ci = fit(F, boot=N_BOOT)
sig = (base_ci["low"] > 0) | (base_ci["high"] < 0)
cur = recompute_pct(F[F["evidence"] == "curated"])
cur_res, cur_coef, cur_ci = fit(cur, boot=N_BOOT)
rng = np.random.default_rng(1)
rand = []
for _ in range(N_RANDOM):
    res, coef, _ = fit(recompute_pct(F.iloc[np.sort(rng.choice(len(F), len(cur), replace=False))]))
    rand.append({**res, "coefficient correlation": np.corrcoef(coef, base_coef)[0, 1],
                 "significant effects, same sign": (np.sign(coef[sig]) == np.sign(base_coef[sig])).mean(),
                 **{f"coef {k}": v for k, v in coef.items()}})
rand = pd.DataFrame(rand)

rows = [{"variant": "whole sample", "papers": len(F), **base_res, "coefficient correlation": 1.0,
         "significant effects, same sign": 1.0},
        {"variant": "curated papers only", "papers": len(cur), **cur_res,
         "coefficient correlation": np.corrcoef(cur_coef, base_coef)[0, 1],
         "significant effects, same sign": (np.sign(cur_coef[sig]) == np.sign(base_coef[sig])).mean()}]
for q, label in ((2.5, "random subsets, same size (2.5th pct.)"), (97.5, "random subsets, same size (97.5th pct.)")):
    rows.append({"variant": label, "papers": len(cur),
                 **{c: np.percentile(rand[c], q) for c in rand.columns if not c.startswith("coef ")}})
models = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"{len(F)} modelled papers, {len(cur)} with curated evidence; {sig.sum()} whole-sample effects with a 95% "
      "interval excluding 0")
print(models.round(3).to_string(index=False))
models.round(4).to_csv(os.path.join(GEN_OUT, "sample_sensitivity_models.csv"), index=False)

show = list(base_coef.abs().sort_values(ascending=False).index[:15])
show += [c for c in base_coef.index if (c.startswith("topic_") or c.startswith("title_")) and c not in show]
effects = pd.DataFrame({"feature": show, "whole sample": base_coef[show].to_numpy(),
                        "low": base_ci.loc[show, "low"].to_numpy(), "high": base_ci.loc[show, "high"].to_numpy(),
                        "curated only": cur_coef[show].to_numpy(),
                        "curated low": cur_ci.loc[show, "low"].to_numpy(),
                        "curated high": cur_ci.loc[show, "high"].to_numpy(),
                        "random 2.5%": [np.percentile(rand[f"coef {f}"], 2.5) for f in show],
                        "random 97.5%": [np.percentile(rand[f"coef {f}"], 97.5) for f in show]})
print(effects.round(2).to_string(index=False))
effects.round(3).to_csv(os.path.join(GEN_OUT, "sample_sensitivity_effects.csv"), index=False)
print("-> general/outputs/sample_sensitivity_models.csv, sample_sensitivity_effects.csv")
