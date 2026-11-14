"""Topic attention in the sample by half-year: the numbers behind the paper's Fig. 1.

For the 2,227 papers of the sample (general/outputs/sample_table.csv, topic flags of 03_topics_sample_titles.ipynb),
binned by publication month (ADS pubdate, topic_texts.json) into half-years from 2023 H1; the last half-year is
partial (up to the freeze date). Per topic and half-year: the number of papers on the topic, and its share of all
sample papers of that half-year. Papers whose publication date has no month are left out (counted below).
Writes general/outputs/sample_topics_by_half_year.csv. Run from the repository root (after step 3):
    python general/11_sample_topics_by_half_year.py
"""
import json
import os
import sys
from datetime import date

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import FREEZE, GEN_OUT, data  # noqa: E402

T = pd.read_csv(os.path.join(GEN_OUT, "sample_table.csv"))
texts = json.load(open(data("topic_texts.json")))
freeze = date.fromisoformat(FREEZE)
topics = [c for c in T.columns if c.startswith("topic: ")]


def half_year(b):
    p = (texts.get(b) or {}).get("pubdate") or ""
    y, m = int(p[:4] or 0), int(p[5:7] or 0)
    return f"{y} H{1 if m <= 6 else 2}" if m else None


T["half_year"] = T["bibcode"].map(half_year)
print(f"{len(T)} sample papers; {T['half_year'].isna().sum()} without a publication month (left out)")
# half-years from 2023 H1 to the freeze date's half-year; papers dated after the freeze (journal issue dates) go
# into the last half-year
last = f"{freeze.year} H{1 if freeze.month <= 6 else 2}"
T.loc[T["half_year"] > last, "half_year"] = last
halves = [f"{y} H{h}" for y in range(2023, freeze.year + 1) for h in (1, 2) if f"{y} H{h}" <= last]
T = T[T["half_year"].isin(halves)]
den = T.groupby("half_year").size().reindex(halves)
rows = []
for c in topics:
    n = T[T[c].astype(bool)].groupby("half_year").size().reindex(halves, fill_value=0)
    for h in halves:
        rows.append({"topic": c[7:], "half_year": h, "partial": h == last, "count": int(n[h]),
                     "papers in half-year": int(den[h]), "share_pct": round(100 * n[h] / den[h], 3)})
out = pd.DataFrame(rows)
print("papers per half-year:", den.to_dict())
print(out.pivot(index="topic", columns="half_year", values="share_pct").round(1).to_string())
out.to_csv(os.path.join(GEN_OUT, "sample_topics_by_half_year.csv"), index=False)
print("-> general/outputs/sample_topics_by_half_year.csv")
