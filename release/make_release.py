"""Data release: the files behind every number of the paper, ready for Zenodo.

Writes release/<freeze>/:
  sample.csv               the 2,227 papers: bibcode, title, year, document type, data-use evidence and score, galaxy
                           screen result, citations (ADS, freeze date), the nine topic flags, catchy-title label
  candidates_rejected.csv  every other candidate, with the step of the sample rule it failed
  citation_features.csv    the 58 features and the target (citation percentile within the posting month)
  predictions.csv          the predicted percentile of every sample paper from the ridge and boosted-tree models
                           trained on all modelled papers (an archive for the 12-month check of the paper plan)
  hand_galaxy_verdicts.csv, hand_catchy_title_labels.csv, hand_validation.csv   the hand inputs, as used
  topics.json              the candidate query, the data-use threshold, the topic list (phrases and redshift
                           patterns) and the topic exclusion
  README.md, checksums.txt
Not released: ADS full-text passages and abstracts (publisher and ADS terms), the ADS caches. Run from the repository
root (after run_all.sh):
    python release/make_release.py
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import FREEZE, GEN_OUT, data  # noqa: E402
from common.jwst_use import CANDIDATE_Q, DATA_USE_MIN_SCORE  # noqa: E402
from common.trending_shared import TOPICS, TOPIC_EXCLUDES  # noqa: E402

OUT = os.path.join(ROOT, "release", FREEZE)
os.makedirs(OUT, exist_ok=True)
w = lambda df, name: df.to_csv(os.path.join(OUT, name), index=False)   # noqa: E731

# ── the sample and the rejected candidates ──
sample = pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))
table = pd.read_csv(os.path.join(GEN_OUT, "sample_table.csv"))
topic_cols = [c for c in table.columns if c.startswith("topic: ")]
s = sample[["bibcode", "title", "year", "doctype", "evidence", "jwst_score", "jwst_bibgroup", "galaxy_screen",
            "galaxy_check", "n_authors", "identifiers"]].merge(
    table[["bibcode", "citations", "catchy_label"] + topic_cols], on="bibcode", how="left")
s = s.rename(columns={c: "topic_" + re.sub(r"\W+", "_", c[7:].lower()).strip("_") for c in topic_cols})
w(s, "sample.csv")
rej = pd.read_csv(os.path.join(GEN_OUT, "rejected.csv"))
w(rej[["bibcode", "title", "year", "doctype", "evidence", "jwst_score", "jwst_bibgroup", "status"]],
  "candidates_rejected.csv")

# ── features and predictions ──
F = pd.read_csv(os.path.join(GEN_OUT, "citation_features.csv"))
w(F, "citation_features.csv")
FEATURES = list(F.columns[F.columns.get_loc("top10") + 1:])
M = F["pct"].notna()
ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                      RidgeCV(alphas=np.logspace(-1, 4, 30))).fit(F.loc[M, FEATURES], F.loc[M, "pct"])
trees = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30,
                                      l2_regularization=1.0, random_state=0).fit(F.loc[M, FEATURES], F.loc[M, "pct"])
pred = F[["bibcode", "year", "month", "cohort", "citations", "pct"]].assign(
    modelled=M, predicted_pct_ridge=ridge.predict(F[FEATURES]).round(2),
    predicted_pct_trees=trees.predict(F[FEATURES]).round(2), predicted_on=FREEZE)
w(pred, "predictions.csv")

# ── hand inputs ──
w(pd.read_csv(data("galaxy_verdicts.csv")), "hand_galaxy_verdicts.csv")
w(pd.read_csv(data("catchy_title_labels.csv")), "hand_catchy_title_labels.csv")
VDIR = data("validation")
key = pd.concat([pd.read_csv(p).assign(round=int(re.search(r"round(\d+)_key", p).group(1)))
                 for p in sorted(glob.glob(os.path.join(VDIR, "round*_key.csv")))], ignore_index=True)


def latest(pattern):
    out = {}
    for p in sorted(glob.glob(os.path.join(VDIR, pattern))):
        saved = json.load(open(p))
        for d in saved["decisions"]:
            t = d.get("decided_at") or saved["saved_at"]
            if d.get("decision") and (d["paper_id"] not in out or t >= out[d["paper_id"]][0]):
                out[d["paper_id"]] = (t, d["decision"])
    return {k: v for k, (_, v) in out.items()}


key["verdict"] = key["paper_id"].map(latest("validation_decisions_*.json"))
key["first_pass_verdict"] = key["paper_id"].map(latest("validation_round1_decisions_*.json"))
w(key[["paper_id", "bibcode", "round", "group", "stratum", "stratum_N", "stratum_n", "verdict", "first_pass_verdict"]],
  "hand_validation.csv")

# ── definitions ──
json.dump({"freeze": FREEZE, "candidate_query": CANDIDATE_Q, "data_use_min_score": DATA_USE_MIN_SCORE,
           "topics (regular expressions, case-insensitive, whole words, title + abstract)": TOPICS,
           "topic exclusions": TOPIC_EXCLUDES},
          open(os.path.join(OUT, "topics.json"), "w"), indent=1, ensure_ascii=False)

README = f"""# JWST galaxy literature, 2023–2026: data release

NASA ADS snapshot of {FREEZE}. Code: the repository `jwst-galaxy-literature` at the commit of this release
(`./run_all.sh` rebuilds everything from the ADS caches). The rules are stated in `AGENTS.md`, section 4, of that
repository.

| File | Content |
| --- | --- |
| `sample.csv` | The {len(s):,} papers of the sample: data-use evidence (`curated` = in ADS's JWST bibliography; `explicit text` = one passage of the full text scoring at least {DATA_USE_MIN_SCORE}), galaxy-screen terms and result, citations on {FREEZE}, topic flags, catchy-title label |
| `candidates_rejected.csv` | The other {len(rej):,} candidates, with the step of the rule they failed (`status`) |
| `citation_features.csv` | The 58 features of the citation model and the target (`pct`: citation percentile among papers posted the same month; `cohort`: the month group) |
| `predictions.csv` | Predicted percentile of every sample paper, models trained on the modelled papers (`modelled`); dated {FREEZE}, for a later check against citations |
| `hand_galaxy_verdicts.csv` | Every hand verdict on scope (galaxy / not_galaxy / unsure), with its source and date; read only for papers the galaxy screen flags |
| `hand_catchy_title_labels.csv` | Labels of the titles (catchy / borderline / not), assigned by a large language model following the author's rules; `reviewed_by_user` marks those the author checked; borderline counts as not catchy |
| `hand_validation.csv` | The blind validation: 400 random papers in three rounds, their stratum and verdict (in: `yes_analysis`, `yes_plot`; out: `no`, `not_galaxy`; `unsure`); round 1 judged twice |
| `topics.json` | The candidate query, the data-use threshold, the topic phrases and redshift patterns, and the topic exclusion (Black holes & AGN leaves out the Little Red Dot papers) |
| `checksums.txt` | SHA-256 of every file |

Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The code is under the MIT licence.

Not included: ADS full-text passages and abstracts (copyright of the publishers; available from ADS by bibcode).
All hand inputs are the work of one rater (the author); the rater's agreement with themself over two blind passes
of 135 papers is Cohen's kappa = 0.84. In an independent check, a large language model applying the written rules
agreed with 87.5% of 200 galaxy verdicts (kappa = 0.75) and, relabelling all 2,227 sample titles, with 94.8% of the
catchy / not labels. The disagreements are mostly edge cases; the check was stricter on titles (91
catchy instead of 186). With its labels the share of catchy titles rises from 2.7% (2023) to 4.8% (2026) instead of
3.7% to 10.5%, and catchy titles still have no significant effect on the citation rank.
"""
open(os.path.join(OUT, "README.md"), "w").write(README)
with open(os.path.join(OUT, "checksums.txt"), "w") as f:
    for name in sorted(os.listdir(OUT)):
        if name != "checksums.txt":
            f.write(f"{hashlib.sha256(open(os.path.join(OUT, name), 'rb').read()).hexdigest()}  {name}\n")
for name in sorted(os.listdir(OUT)):
    print(f"{os.path.getsize(os.path.join(OUT, name)) / 1e6:6.2f} MB  {name}")
print(f"-> release/{FREEZE}/")
