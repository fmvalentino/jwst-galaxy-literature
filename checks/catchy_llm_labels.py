"""The catchy-title results with the labels of a large language model (LLM) instead of the author's.

The LLM labelled every title of the sample with the written rules of 03_topics_sample_titles.ipynb, section 5
(data/<freeze>/catchy_title_labels_llm.csv: catchy / borderline / not; borderline counts as not catchy, as for the
author's labels). For each set of labels:
  - the share of catchy titles in 2023 and in 2026 (sample papers);
  - the ridge effect of a catchy title on the citation rank and its two-sided bootstrap p-value, as in
    general/08_effects.py (same features, penalty, 2,000 refits, seed);
  - the test rho of the boosted trees of step 4 (same settings and time split).
With the author's labels the last two reproduce general/outputs/effects_table.csv and the test rho of step 10.
Run from the repository root (after steps 3, 4 and 8): python checks/catchy_llm_labels.py
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402

N_BOOT, TEST_FROM = 2000, "2025-04"
sample = pd.read_csv(os.path.join(GEN_OUT, "sample_table.csv"))
llm = pd.read_csv(data("catchy_title_labels_llm.csv")).set_index("bibcode")["label"]
assert set(sample["bibcode"]) <= set(llm.index), "an LLM label is missing for a sample paper"
labels = {"author": sample.set_index("bibcode")["catchy_label"].eq("catchy"), "LLM": llm.eq("catchy")}

F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
meta = json.load(open(data("features_ads_metadata.json")))
apjs = F["bibcode"].map(lambda b: "Astrophysical Journal Supplement" in ((meta.get(b) or {}).get("pub") or ""))
y = F["pct"].to_numpy()
rows = []
for name, catchy in labels.items():
    year = sample.set_index("bibcode")["year"]
    share = {y_: catchy[year[year == y_].index].mean() for y_ in (2023, 2026)}
    G = F.assign(title_catchy=F["bibcode"].map(catchy).astype(float))
    # ridge effect, as general/08_effects.py
    X = G[FEATURES].drop(columns=["refereed", "journal_apj"]).assign(preprint=1 - G["refereed"],
                                                                     journal_apjs=apjs.astype(float))
    raw = SimpleImputer(strategy="median").fit_transform(X)
    sd = raw.std(0)
    Z = (raw - raw.mean(0)) / np.where(sd > 0, sd, 1)
    alpha = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-1, 4, 30))).fit(X, y)[-1].alpha_
    k = list(X.columns).index("title_catchy")
    coef = Ridge(alpha=alpha).fit(Z, y).coef_[k]
    rng = np.random.default_rng(0)
    boot = np.array([Ridge(alpha=alpha).fit(Z[i], y[i]).coef_[k] for i in (rng.integers(0, len(y), len(y))
                                                                             for _ in range(N_BOOT))])
    p = max(2 * min((boot <= 0).mean(), (boot >= 0).mean()), 1 / N_BOOT)
    s = raw[:, k].mean()
    # boosted trees, as step 4
    tr, te = G["cohort"] < TEST_FROM, G["cohort"] >= TEST_FROM
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                      l2_regularization=1.0, random_state=0).fit(G.loc[tr, FEATURES], G.loc[tr, "pct"])
    rho = spearmanr(m.predict(G.loc[te, FEATURES]), G.loc[te, "pct"]).statistic
    rows.append({"labels": name, "catchy papers": int(catchy[sample["bibcode"]].sum()),
                 "share 2023 %": round(100 * share[2023], 1), "share 2026 %": round(100 * share[2026], 1),
                 "effect (places)": round(coef / np.sqrt(s * (1 - s)), 2), "p (bootstrap)": round(p, 3),
                 "test rho (trees)": round(rho, 3)})
t = pd.DataFrame(rows)
print(t.to_string(index=False))
agree = (labels["author"][sample["bibcode"]].to_numpy() == labels["LLM"][sample["bibcode"]].to_numpy()).mean()
print(f"agreement, catchy or not: {agree:.1%} of {len(sample):,} titles")
t.to_csv(os.path.join(ROOT, "checks", "catchy_llm_labels.csv"), index=False)
print("-> checks/catchy_llm_labels.csv")
