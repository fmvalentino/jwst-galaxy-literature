"""Sensitivity of the topic trends to the phrase lists.

Each topic of common/trending_shared.py is matched three ways: a narrow phrase list (the core phrases), the current
list, and a broad list (the current one plus acronyms and synonyms). For each, the share of the sample's papers in
2023 and 2026 (frozen texts) and the direction (gaining: share up more than 20%; losing: down more than 20%; steady in between). A topic result holds if
the direction is the same under the three lists. Black holes & AGN always leaves out the Little Red Dot papers of the
same list. Writes checks/topic_sensitivity.csv. Run from the repository root:
    python checks/topic_sensitivity.py
"""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402
from common.trending_shared import TOPICS, clean_text, topic_texts  # noqa: E402

NARROW = {
    "First galaxies": [r"first galaxies", r"earliest galaxies", r"cosmic dawn"],
    "Little Red Dots": [r"little red dots?"],
    "Black holes & AGN": [r"AGNs?", r"active galactic nucle(?:us|i)", r"black holes?"],
    "Population III": [r"Population III", r"Pop[ -]?III"],
    "Reionization": [r"reioni[sz]ation"],
    "Quenching": [r"quenched", r"quenching"],
    "Dust": [r"dust (?:emission|continuum|mass(?:es)?|attenuation|extinction)"],
    "Chemical abundances": [r"chemical abundances?", r"chemical enrichment", r"metallicit(?:y|ies)"],
    "Lensing": [r"lensing", r"lensed"],
}
EXTRA = {   # added to the current list for the broad variant
    "First galaxies": [r"high-redshift frontier", r"z\s*\$?\s*(?:[>≳~∼≈=]|\\gtrsim|\\sim)\s*\$?\s*(?:[89])(?:\.\d+)?(?![\d])"],
    "Little Red Dots": [r"LRDs?"],
    "Black holes & AGN": [r"SMBHs?", r"supermassive", r"broad[- ]line", r"accreting black holes?"],
    "Population III": [r"metal-free", r"zero[- ]metallicity", r"pristine gas"],
    "Reionization": [r"reioni[sz](?:ed|ing)", r"EoR", r"Lyman[- ]continuum", r"LyC"],
    "Quenching": [r"red and dead", r"quiescence", r"suppressed star formation"],
    "Dust": [],                                  # the current list is already specific; generic words left out
    "Chemical abundances": [r"abundances?", r"metal[- ](?:poor|rich)", r"metal enrichment"],
    "Lensing": [r"magnified", r"magnification", r"cluster lens(?:es|ing)?"],
}
VARIANTS = {"narrow": NARROW, "current": TOPICS, "broad": {t: TOPICS[t] + EXTRA[t] for t in TOPICS}}


def compile_(lists):
    return {t: re.compile(r"\b(?:" + "|".join(ps) + r")\b", re.I) for t, ps in lists.items()}


def topics(text, rx):
    hit = {t for t, r in rx.items() if r.search(text)}
    if "Little Red Dots" in hit:
        hit.discard("Black holes & AGN")
    return hit


def load(fname, keep):
    docs = json.load(open(data(fname)))["corpus_docs"]
    docs = [d for d in docs if d["bibcode"] in keep and str(d.get("year", "")) in ("2023", "2026")]
    tx = topic_texts([d["bibcode"] for d in docs])
    return [(int(d["year"]), clean_text(tx[d["bibcode"]]["title"]) + " " + clean_text(tx[d["bibcode"]]["abstract"]))
            for d in docs]


def direction(a, b):
    if a == 0 or b > a * 1.2:
        return "gaining"
    return "losing" if b < a * 0.8 else "steady"


sample = set(pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))["bibcode"])
rows = []
for scope, docs in (("sample", load("topics_sample_pull.json", sample)),):
    n = {y: sum(1 for yy, _ in docs if yy == y) for y in (2023, 2026)}
    for v, lists in VARIANTS.items():
        rx = compile_(lists)
        hits = [(y, topics(x, rx)) for y, x in docs]
        for t in TOPICS:
            s23 = 100 * sum(1 for y, h in hits if y == 2023 and t in h) / n[2023]
            s26 = 100 * sum(1 for y, h in hits if y == 2026 and t in h) / n[2026]
            rows.append({"scope": scope, "topic": t, "variant": v, "2023 %": s23, "2026 %": s26,
                         "direction": direction(s23, s26)})
df = pd.DataFrame(rows)
wide = df.pivot_table(index=["scope", "topic"], columns="variant", values=["2023 %", "2026 %"]).round(1)
dirs = df.pivot_table(index=["scope", "topic"], columns="variant", values="direction", aggfunc="first")
dirs["same direction"] = dirs.nunique(axis=1) == 1
pd.set_option("display.width", 220)
print(pd.concat([wide, dirs[["narrow", "current", "broad", "same direction"]]], axis=1).to_string())
df.round(3).to_csv(os.path.join(ROOT, "checks", "topic_sensitivity.csv"), index=False)
print("-> checks/topic_sensitivity.csv")
