"""ADS's own phrase search for each topic, with the phrase lists the matcher uses, for checks/topic_matcher.py.

The topic pull of the freeze (topics_sample_pull.json) also holds an ADS phrase search per topic, but with the phrase
lists of that day; phrases were added to three lists afterwards (Population III, Chemical abundances, Black holes &
AGN), so that search cannot test the lists the paper uses. This script searches again, once per topic, among the
sample papers: =abs:(...) reads title, abstract and keywords, exact forms (no synonyms). These are fixed fields,
which may be fetched after the freeze (AGENTS.md, section 2); no citation data is read. Papers are asked by bibcode
and mapped back by identifier, so a preprint ADS has re-bibcoded since the freeze is still found.

ADS_PHRASES below is the topic list of common/trending_shared.py written as ADS phrases. A test runs first, without
any query: every ADS phrase must match its topic's local pattern, and every local pattern must be covered by an ADS
phrase, except the two that ADS cannot search (NOT_SEARCHABLE): the redshift pattern of First galaxies (that topic
is not searched) and element ratios such as N/O (found by the matcher only). Edit both lists together.
Writes data/<freeze>/topics_sample_search.json (date, queries, the sample papers found per topic); a rerun with the
same queries makes no query. Needs ADS_TOKEN the first time, about 20 bigqueries. Run from the repository root:
    ADS_TOKEN=... python checks/topic_search.py           (python checks/topic_search.py test: the test only)
"""
import datetime
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402
from common.trending_shared import RATIO, TOPICS, Z10  # noqa: E402

OUT = data("topics_sample_search.json")
ADS_PHRASES = {
    "Little Red Dots": ["little red dot", "little red dots"],
    "Black holes & AGN": ["AGN", "AGNs", "active galactic nucleus", "active galactic nuclei", "black hole",
                          "black holes", "quasar", "quasars", "QSO", "QSOs", "quasi-stellar object",
                          "quasi-stellar objects", "blazar", "blazars", "Seyfert", "Seyferts"],
    "Population III": ["Population III", "Pop III", "PopIII", "Pop-III", "first stars"],
    "Reionization": ["reionization", "reionisation"],
    "Quenching": ["quiescent", "passive", "quenched", "quenching", "post-starburst"],
    "Dust": ["dust emission", "dust continuum", "dust mass", "dust masses", "dust temperature", "dust temperatures",
             "dust attenuation", "dust extinction", "attenuation law", "attenuation curve", "attenuation curves",
             "Balmer decrement", "Balmer decrements", "UV slope", "UV slopes", "UV continuum slope",
             "UV continuum slopes", "UV-continuum slope", "UV-continuum slopes", "dust-obscured", "dust obscured",
             "dusty star-forming"],
    "Chemical abundances": ["chemical abundance", "chemical abundances", "chemical enrichment", "metallicity",
                            "metallicities", "oxygen", "carbon", "nitrogen", "iron", "magnesium", "Fe abundance",
                            "Fe abundances", "Mg abundance", "Mg abundances"],
    "Lensing": ["lensing", "weak lensing", "lensed", "gravitational lens", "gravitational lenses", "strong lens",
                "strong lenses"],
}
NOT_SEARCHABLE = {("First galaxies", Z10), ("Chemical abundances", RATIO)}


def test():
    """ADS phrases and local patterns describe the same lists (no query)."""
    problems = []
    for topic, pats in TOPICS.items():
        phrases = ADS_PHRASES.get(topic, [])
        if topic == "First galaxies":
            continue                                   # not searched in ADS (its redshift pattern cannot be)
        for ph in phrases:
            if not any(re.fullmatch(p, ph, re.I) for p in pats):
                problems.append(f"{topic}: ADS phrase {ph!r} matches no local pattern")
        for p in pats:
            if (topic, p) in NOT_SEARCHABLE:
                continue
            if not any(re.fullmatch(p, ph, re.I) for ph in phrases):
                problems.append(f"{topic}: local pattern {p!r} has no ADS phrase")
    assert set(ADS_PHRASES) == set(TOPICS) - {"First galaxies"}, "ADS_PHRASES must list every searched topic"
    for p in problems:
        print("  " + p)
    print("ADS phrases and local patterns agree" if not problems else f"{len(problems)} differences: fix ADS_PHRASES")
    return not problems


def query(topic):
    return "=abs:(" + " OR ".join(f'"{ph}"' for ph in ADS_PHRASES[topic]) + ")"


def main():
    assert test()
    if os.path.exists(OUT) and json.load(open(OUT))["queries"] == {t: query(t) for t in ADS_PHRASES}:
        print(f"{os.path.relpath(OUT, ROOT)}: same queries already searched, nothing to do")
        return
    from common.ads import bigquery, by_request
    bibs = list(pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))["bibcode"])
    found = {}
    for topic in ADS_PHRASES:
        hits = set()
        for i in range(0, len(bibs), 2000):
            part = bibs[i:i + 2000]
            hits |= set(by_request(part, bigquery(query(topic), part, fl="bibcode,identifier")))
        found[topic] = sorted(hits)
        print(f"  {topic}: {len(hits):,} sample papers")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"saved": datetime.datetime.now().isoformat(timespec="seconds"), "papers": len(bibs),
                   "queries": {t: query(t) for t in ADS_PHRASES}, "topics": found}, f, ensure_ascii=False, indent=1)
    print(f"-> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    sys.exit(0 if test() else 1) if sys.argv[1:] == ["test"] else main()
