"""Self-citations matched by ORCID: how much of the team-size effect is a team citing itself?

04_citation_features.ipynb (test 11) counts a citation as a self-citation when the citing paper shares an author
with the cited paper, matched by surname and first initial. That overcounts: namesakes with the same initial are
merged. Here the same citations (data/<freeze>/features_citing_authors.json, the citing papers and their authors as
pulled for the freeze) are matched three ways, author by author:
  A. name           surname + first initial (the rule of 04)
  B. ORCID, else name     two authors who both have an ORCID are the same person only if the ORCIDs match;
                          otherwise surname + first initial
  C. ORCID, else full name   as B, but the fallback needs the same surname and the same first given name in full
                          (an initial-only name falls back to the initial)
ORCIDs: the three ADS fields (orcid_pub, orcid_user, orcid_other, first non-empty per author position), fetched once
for the cited and citing papers (data/<freeze>/citation_orcids.json, with its fetch date). ORCID claims accumulate
over time; the citations themselves stay those of the freeze.

For each rule: the self-citation share (all citations, and by team size), then the models of 04 refitted on
citations by other authors only (percentile recomputed within cohorts): test rho of the trees and the ridge effects of
team size, countries and the first author's record. Writes general/outputs/self_citations.csv and
self_citations_effects.csv (effects per standard deviation, and per natural unit as in 08_effects.py). Run from the repository root (needs ADS_TOKEN the first time):
    python general/09_self_citations.py
"""
import datetime
import json
import os
import sys
import unicodedata

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
from common.ads import bigquery, load_json, save_json  # noqa: E402
from common.config import GEN_OUT, data  # noqa: E402

TEST_FROM = "2025-04"
RIDGE_ALPHAS = np.logspace(-1, 4, 30)
ORCID_JSON = data("citation_orcids.json")
EFFECTS = ["log_n_authors", "large_collab", "n_countries", "fa_log_prior_papers"]

sc = json.load(open(data("features_citing_authors.json")))
F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
F = F[F["pct"].notna()].reset_index(drop=True)
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
cited = [b for b in F["bibcode"] if b in sc["cited"]]
citing = sorted({c for b in cited for c in sc["cited"][b]["citation"]})

# ── ORCIDs of every author position, cited and citing papers (cached) ──
cache = load_json(ORCID_JSON, None) or {"fetched": None, "orcid": {}}
missing = sorted(set(cited + citing) - set(cache["orcid"]))
for i in range(0, len(missing), 2000):
    part = missing[i:i + 2000]
    for d in bigquery("*:*", part, fl="bibcode,author,orcid_pub,orcid_user,orcid_other"):
        n = len(d.get("author") or [])
        cols = [d.get(f) or ["-"] * n for f in ("orcid_pub", "orcid_user", "orcid_other")]
        cache["orcid"][d["bibcode"]] = [next((c[k] for c in cols if k < len(c) and c[k] not in ("-", "")), None)
                                        for k in range(n)]
    for b in part:
        cache["orcid"].setdefault(b, [])
if missing:
    cache["fetched"] = datetime.date.today().isoformat()
    save_json(ORCID_JSON, cache)
    print(f"ORCIDs fetched for {len(missing):,} papers")
meta = json.load(open(data("features_ads_metadata.json")))


def deaccent(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def keys(name):
    """(surname|initial, surname|first given name or initial)."""
    sur, _, giv = deaccent(name).partition(",")
    toks = [t for t in giv.replace(".", " ").split() if t]
    first = toks[0].lower() if toks else ""
    initial_only = len(first.replace("-", "")) <= 1
    return f"{sur.strip().lower()}|{first[:1]}", f"{sur.strip().lower()}|{first[:1] if initial_only else first}"


def people(authors, orcids):
    return [(*keys(a), (orcids[i] if i < len(orcids) else None)) for i, a in enumerate(authors)]


def same(p, q, rule):
    """Are two author entries (initial key, full key, orcid) the same person under a rule?"""
    if rule != "A" and p[2] and q[2]:
        return p[2] == q[2]
    if rule == "C":
        # full first name on both sides when both have it, else the initial
        full_p, full_q = p[1].split("|")[1], q[1].split("|")[1]
        if len(full_p) > 1 and len(full_q) > 1:
            return p[1] == q[1]
    return p[0] == q[0]


RULES = {"A": "name (surname + initial)", "B": "ORCID, else name", "C": "ORCID, else full name"}
counts = {r: {} for r in RULES}
for b in cited:
    mine = people(sc["cited"][b]["author"], cache["orcid"].get(b, []))
    by_sur = {}
    for p in mine:
        by_sur.setdefault(p[0].split("|")[0], []).append(p)
    for r in RULES:
        n_self = 0
        for c in sc["cited"][b]["citation"]:
            theirs = people(sc["citing_authors"].get(c, []), cache["orcid"].get(c, []))
            if any(same(p, q, r) for q in theirs for p in by_sur.get(q[0].split("|")[0], [])):
                n_self += 1
        counts[r][b] = n_self
n_all = pd.Series({b: len(sc["cited"][b]["citation"]) for b in cited})
team = pd.Series({b: len((meta.get(b) or {}).get("author") or sc["cited"][b]["author"]) for b in cited})
team_bin = pd.cut(team, [0, 5, 15, 50, 10_000], labels=["1-5 authors", "6-15", "16-50", "> 50"])
orcid_cov = np.mean([o is not None for b in citing for o in cache["orcid"].get(b, [])])
print(f"{len(cited):,} modelled papers, {int(n_all.sum()):,} citations from {len(citing):,} citing papers; "
      f"author positions with an ORCID (citing papers): {orcid_cov:.0%}; ORCIDs fetched {cache['fetched']}")


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         RidgeCV(alphas=RIDGE_ALPHAS))


def trees():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                         min_samples_leaf=30, l2_regularization=1.0, random_state=0)


apjs = F["bibcode"].map(lambda b: "Astrophysical Journal Supplement" in ((meta.get(b) or {}).get("pub") or ""))


def fit(cites):
    """Test rho (trees, ridge) and ridge effects for a citation count per paper (percentile within cohort)."""
    D = F.assign(c=F["bibcode"].map(cites)).dropna(subset=["c"]).reset_index(drop=True)
    D["pct"] = D.groupby("cohort")["c"].rank(pct=True, method="average").mul(100)
    D["top10"] = D["pct"] >= 90
    train = (D["cohort"] < TEST_FROM).to_numpy()
    X, y = D[FEATURES], D["pct"].to_numpy()
    out = {}
    for name, model in (("trees", trees()), ("ridge", ridge())):
        p = model.fit(X[train], y[train]).predict(X[~train])
        out[f"test rho ({name})"] = spearmanr(p, y[~train]).statistic
        out[f"AUC ({name})"] = roc_auc_score(D.loc[~train, "top10"], p)
    XA = X.drop(columns=["refereed", "journal_apj"]).assign(preprint=1 - D["refereed"],
                                                           journal_apjs=apjs[D.index].astype(float).to_numpy())
    Z = SimpleImputer(strategy="median").fit_transform(XA)
    sd = pd.Series(Z.std(0), index=XA.columns)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    alpha = ridge().fit(XA, y)[-1].alpha_
    coef = pd.Series(Ridge(alpha=alpha).fit(Z, y).coef_, index=XA.columns)
    rng = np.random.default_rng(0)
    boot = np.array([Ridge(alpha=alpha).fit(Z[i], y[i]).coef_ for i in (rng.integers(0, len(y), len(y))
                                                                        for _ in range(300))])
    ci = pd.DataFrame(np.percentile(boot, [2.5, 97.5], axis=0).T, index=XA.columns, columns=["low", "high"])
    # per natural unit (UNITS of 08_effects.py): the coefficient is per sd, so multiply by step / sd
    ci = ci.assign(**{"per unit": coef * pd.Series(STEP) / sd, "unit low": ci["low"] * pd.Series(STEP) / sd,
                      "unit high": ci["high"] * pd.Series(STEP) / sd})
    return out, coef, ci


UNIT = {"log_n_authors": "per doubling", "large_collab": "with vs. without", "n_countries": "per country",
        "fa_log_prior_papers": "per doubling"}
_share = F["large_collab"].mean()
STEP = {"log_n_authors": np.log10(2), "n_countries": 1, "fa_log_prior_papers": np.log10(2)}


def unit_cols(f, coef, ci):
    """Effect per natural unit; the yes/no flag as the gap with vs. without (sd of a 0/1 flag)."""
    if f == "large_collab":
        return {"unit": UNIT[f], **{k: ci.loc[f, c] / np.sqrt(_share * (1 - _share)) if c != "effect" else
                                    coef[f] / np.sqrt(_share * (1 - _share))
                                    for k, c in (("effect per unit", "effect"), ("unit low", "low"),
                                                 ("unit high", "high"))}}
    return {"unit": UNIT[f], "effect per unit": ci.loc[f, "per unit"], "unit low": ci.loc[f, "unit low"],
            "unit high": ci.loc[f, "unit high"]}


rows, eff = [], []
res, coef, ci = fit(n_all)
rows.append({"citations": "all", "self share": 0.0, **res})
eff += [{"citations": "all", "feature": f, "effect": coef[f], "low": ci.loc[f, "low"], "high": ci.loc[f, "high"],
         **unit_cols(f, coef, ci)} for f in EFFECTS]
for r, label in RULES.items():
    n_self = pd.Series(counts[r])
    by_team = (n_self.groupby(team_bin, observed=True).sum() / n_all.groupby(team_bin, observed=True).sum())
    res, coef, ci = fit(n_all - n_self)
    rows.append({"citations": f"by other authors, rule {r}: {label}", "self share": n_self.sum() / n_all.sum(),
                 **{f"self share, {k}": v for k, v in by_team.items()}, **res})
    eff += [{"citations": f"rule {r}", "feature": f, "effect": coef[f], "low": ci.loc[f, "low"],
             "high": ci.loc[f, "high"], **unit_cols(f, coef, ci)} for f in EFFECTS]
summary, effects = pd.DataFrame(rows), pd.DataFrame(eff)
pd.set_option("display.width", 220)
print(summary.round(3).to_string(index=False))
print(effects.pivot(index="feature", columns="citations", values="effect").round(2).to_string())
summary.round(4).to_csv(os.path.join(GEN_OUT, "self_citations.csv"), index=False)
effects.round(4).to_csv(os.path.join(GEN_OUT, "self_citations_effects.csv"), index=False)
print("-> general/outputs/self_citations.csv, self_citations_effects.csv")
