"""Purity and completeness of the sample rule, from the author's blind review.

One definition (fixed before review): a paper belongs in the sample when it is a galaxy paper and it is about JWST
data or, if it does not use them, shows JWST data in its figures (whatever the kind of paper). Verdicts "in: analyses JWST data" and
"in: JWST data in figures" count as in; "out: no JWST data" and "out: not a galaxy paper" as out; "unsure" papers
are left out of the estimate and bound it (all in, all out).

Each reviewed paper is placed where the current rule puts it (cell): the sample ("accepted"), or, among papers
rejected at the data-use step (articles or preprints, primary arXiv class not SR/EP), "weak text", "score 3-4" or
"score 0-2". Papers rejected at another step (document type, galaxy screen, arXiv class) are not part of either
estimate. Cell sizes N are the current pools; the share of "in" papers in a cell, p, comes from the reviewed papers
in it (a random sample of a stratum, restricted to a cell, is a random sample of that cell).

  purity       = p_accepted, from the accepted papers of all three rounds
  completeness = N_acc p_acc / (N_acc p_acc + sum over rejected cells of N_c p_c), the rejected cells from all
                 three rounds
95% intervals: 4,000 bootstrap resamples within each cell. Also reported: the agreement of the author's two blind
passes over round 1 (percentage, Cohen's kappa). Writes general/outputs/validation_summary.csv, validation_cells.csv,
validation_agreement.csv. Run from the repository root:
    python general/tools/score_validation.py
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.ads import is_paper  # noqa: E402
from common.config import GEN_OUT, data  # noqa: E402
from common.jwst_use import NON_GALAXY_PRIMARY  # noqa: E402

VDIR = data("validation")
IN = {"yes_analysis": 1, "yes_plot": 1, "yes": 1, "no": 0, "not_galaxy": 0, "unsure": None}
REJECTED_CELLS = ("weak text", "score 3-4", "score 0-2")
N_BOOT = 4000


def read_decisions(pattern):
    """{paper_id: decision}, latest decision per paper over all files matching pattern."""
    latest = {}
    for p in sorted(glob.glob(os.path.join(VDIR, pattern))):
        saved = json.load(open(p))
        for d in saved["decisions"]:
            if d.get("decision"):
                t = d.get("decided_at") or saved["saved_at"]
                if d["paper_id"] not in latest or t >= latest[d["paper_id"]][0]:
                    latest[d["paper_id"]] = (t, d["decision"])
    return {i: v for i, (_, v) in latest.items()}


key = pd.concat([pd.read_csv(p).assign(round=int(re.search(r"round(\d+)_key", p).group(1)))
                 for p in sorted(glob.glob(os.path.join(VDIR, "round*_key.csv")))], ignore_index=True)
key["decision"] = key["paper_id"].map(read_decisions("validation_decisions_*.json"))
key["first_pass"] = key["paper_id"].map(read_decisions("validation_round1_decisions_*.json"))

# where the current rule puts every paper, and the cell sizes
arxiv = json.load(open(data("arxiv_class.json")))
acc = pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))
rej = pd.read_csv(os.path.join(GEN_OUT, "rejected.csv"))
rej = rej[rej["status"].str.startswith("no JWST data-use evidence") & rej["doctype"].map(is_paper)]
rej = rej[[not ((arxiv.get(b) or [None])[0] in NON_GALAXY_PRIMARY) for b in rej["bibcode"]]]
rej["cell"] = np.where(rej["evidence"] == "weak text", "weak text",
                       np.where(rej["jwst_score"] >= 3, "score 3-4", "score 0-2"))
cell_of = {**dict.fromkeys(acc["bibcode"], "accepted"), **dict(zip(rej["bibcode"], rej["cell"]))}
N = {"accepted": len(acc), **rej["cell"].value_counts().to_dict()}
key["cell"] = key["bibcode"].map(cell_of).fillna("rejected at another step")
key["value"] = key["decision"].map(IN)

decided = key[key["decision"].notna()]
print(f"{len(key)} papers drawn in {key['round'].nunique()} rounds, {len(decided)} decided")
print(pd.crosstab(decided["cell"], decided["decision"]).to_string())
print("Pool sizes:", N)


def share(df, undetermined=None):
    v = df["value"]
    v = v.dropna() if undetermined is None else v.fillna(undetermined)
    return v.astype(float).mean() if len(v) else np.nan


def estimates(acc_rows, rej_rows, undetermined=None):
    """(purity, completeness, {cell: p})."""
    p = {"accepted": share(acc_rows, undetermined)}
    p.update({c: share(rej_rows[rej_rows["cell"] == c], undetermined) for c in REJECTED_CELLS})
    users_acc = N["accepted"] * p["accepted"]
    users_rej = sum(N[c] * p[c] for c in REJECTED_CELLS)
    return p["accepted"], users_acc / (users_acc + users_rej), p


def report(label, acc_rows, rej_rows, rng):
    pur, comp, p = estimates(acc_rows, rej_rows)
    groups = [acc_rows] + [rej_rows[rej_rows["cell"] == c] for c in REJECTED_CELLS]
    boot = []
    for _ in range(N_BOOT):
        g = [x.iloc[rng.integers(0, len(x), len(x))] if len(x) else x for x in groups]
        boot.append(estimates(g[0], pd.concat(g[1:]))[:2])
    lo, hi = np.nanpercentile(np.array(boot), [2.5, 97.5], axis=0)
    b0, b1 = estimates(acc_rows, rej_rows, 0), estimates(acc_rows, rej_rows, 1)
    n_uns = int(pd.concat([acc_rows, rej_rows])["value"].isna().sum())
    rows = [{"estimate": label, "quantity": q, "value": v, "low": l, "high": h, "unsure as out": x0,
             "unsure as in": x1, "unsure papers": n_uns, "accepted papers reviewed": len(acc_rows)}
            for q, v, l, h, x0, x1 in (("purity", pur, lo[0], hi[0], b0[0], b1[0]),
                                       ("completeness", comp, lo[1], hi[1], b0[1], b1[1]))]
    cells = [{"estimate": label, "cell": c, "pool size N": N.get(c, 0),
              "reviewed": len(acc_rows) if c == "accepted" else int((rej_rows["cell"] == c).sum()),
              "share in": p[c]} for c in p]
    return rows, cells


rng = np.random.default_rng(0)
d = decided
rej_rows = d[d["cell"].isin(REJECTED_CELLS)]
rows, cells = report("all rounds", d[d["cell"] == "accepted"], rej_rows, rng)
summary, cells = pd.DataFrame(rows), pd.DataFrame(cells)
pd.set_option("display.width", 200)
print(cells.round(3).to_string(index=False))
print(summary.round(3).to_string(index=False))
summary.to_csv(os.path.join(GEN_OUT, "validation_summary.csv"), index=False)
cells.to_csv(os.path.join(GEN_OUT, "validation_cells.csv"), index=False)

# Among papers judged "in", how many only show JWST data in their figures (descriptive, per cell)
ins = d[d["value"] == 1]
print("\nIn papers that only show JWST data in figures, per cell:",
      (ins.groupby("cell")["decision"].apply(lambda x: f"{(x == 'yes_plot').sum()} of {len(x)}")).to_dict())

# Agreement between the author's two blind passes over round 1 (in / out / unsure; first-pass verdicts were
# yes / no / unsure, "not a galaxy paper" did not exist yet and is left out)
both = key[key["first_pass"].notna() & key["decision"].notna() & (key["decision"] != "not_galaxy")]
if len(both):
    lab = {1: "in", 0: "out", None: "unsure"}
    a = both["first_pass"].map(IN).map(lambda v: lab[None if pd.isna(v) else int(v)])
    b = both["value"].map(lambda v: lab[None if pd.isna(v) else int(v)])
    po = (a == b).mean()
    pe = sum((a == x).mean() * (b == x).mean() for x in set(a) | set(b))
    kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
    print(f"\nRound 1, first vs. second blind pass: {len(both)} papers; agreement {100 * po:.0f}%, Cohen's kappa {kappa:.2f}")
    print(pd.crosstab(a.rename("first pass"), b.rename("second pass")).to_string())
    pd.DataFrame([{"papers judged twice": len(both), "agreement": po, "kappa": kappa}]).to_csv(
        os.path.join(GEN_OUT, "validation_agreement.csv"), index=False)
print("-> general/outputs/validation_summary.csv, validation_cells.csv, validation_agreement.csv")
