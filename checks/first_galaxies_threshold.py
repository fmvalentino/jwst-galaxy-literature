"""Robustness of the First galaxies trend to its redshift threshold.

The topic "First galaxies" (common/trending_shared.py) matches its phrases (first galaxies, earliest galaxies,
cosmic dawn) or a redshift of 10 or more named in the title or abstract. This check recomputes the topic's share of papers
per publication year with the phrases alone, the redshift alone (z >= 8, 9, 10), and phrases or redshift, on the
sample, with the same frozen texts as the notebooks. Writes checks/first_galaxies_threshold.csv.
Run from the repository root:
    python checks/first_galaxies_threshold.py
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402
from common.trending_shared import clean_text, topic_texts  # noqa: E402

YEARS = [2023, 2024, 2025, 2026]
OP = r"(?:[>≳~∼≈=]|\\gtrsim|\\geq|\\ge|\\sim|\\approx|\\simeq)"
NUMS = {8: r"(?:[89]|1[0-9]|[2-9][0-9])", 9: r"(?:9|1[0-9]|[2-9][0-9])", 10: r"(?:1[0-9]|[2-9][0-9])"}


def z_at_least(lo):
    n = NUMS[lo]
    return (rf"z\s*\$?\s*{OP}\s*\$?\s*{n}(?:\.\d+)?(?![\d])"
            rf"|redshifts?\s+(?:above|beyond|of|greater than|>)\s*{n}(?![\d])")


PHRASES = r"first galaxies|earliest galaxies|cosmic dawn"
VARIANTS = {"phrases only": PHRASES, "z >= 10 only": z_at_least(10), "z >= 9 only": z_at_least(9),
            "z >= 8 only": z_at_least(8), "phrases or z >= 10 (the topic)": PHRASES + "|" + z_at_least(10),
            "phrases or z >= 9": PHRASES + "|" + z_at_least(9), "phrases or z >= 8": PHRASES + "|" + z_at_least(8)}


def papers(fname, keep):
    docs = json.load(open(data(fname)))["corpus_docs"]
    docs = [d for d in docs if d["bibcode"] in keep and str(d.get("year", "")).isdigit()]
    texts = topic_texts([d["bibcode"] for d in docs])
    return [(int(d["year"]), clean_text(texts[d["bibcode"]]["title"]) + " " + clean_text(texts[d["bibcode"]]["abstract"]))
            for d in docs]


sample = set(pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))["bibcode"])
rows = []
for scope, docs in (("sample", papers("topics_sample_pull.json", sample)),):
    n = {y: sum(1 for yy, _ in docs if yy == y) for y in YEARS}
    for name, pat in VARIANTS.items():
        rx = re.compile(r"\b(?:" + pat + r")\b", re.I)
        hits = {y: sum(1 for yy, x in docs if yy == y and rx.search(x)) for y in YEARS}
        row = {"scope": scope, "variant": name}
        for y in YEARS:
            row[f"{y} papers"], row[f"{y} %"] = hits[y], 100 * hits[y] / n[y]
        row["change in share"] = row["2026 %"] / row["2023 %"] - 1
        rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
df.round(4).to_csv(os.path.join(ROOT, "checks", "first_galaxies_threshold.csv"), index=False)
print("-> checks/first_galaxies_threshold.csv")
