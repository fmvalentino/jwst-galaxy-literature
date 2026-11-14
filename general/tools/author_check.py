"""Hand check of the first author's publication record, on 100 random people of the modelled papers.

The first author's earlier refereed papers and career length (04_citation_features.ipynb, section 3) come from one
ADS query per person, built by the identity rules of that section. The query can collect papers of somebody else:
an identical full name that belongs to another person, or a spelling that ADS's author synonyms attach to this
one. This check estimates how often, from people drawn at random (fixed seed) among the first authors of the
modelled papers. For each person the page shows the name and ORCID on the paper, the rule and the query used, the
spellings ADS has under "Surname, Initial" (marking those the query counts), the counted papers per year with the
career start, and links to the paper, the ADS search of the query and the ORCID record. The verdict: is every
counted paper this person's?
Two rounds. Round 1 (seed 20261019) found 19% wrong records under the first rules (career gap of 10 years, and
every earlier paper counted); the rules were then changed (gap of 7 years, papers before the career start not
counted, both checked against the round-1 verdicts), so round 2 draws 100 other people (seed 20261114) to measure
the error of the final rules on people the change was not tuned on.
    python general/tools/author_check.py page [round]    -> general/outputs/author_check[_r2].html (no ADS query, no
                                                          ADS text); round 2 by default
    python general/tools/author_check.py score [round]   reads data/<freeze>/author_check[_r2]_decisions_*.json (the
                                                          latest file wins) -> general/outputs/author_check[_r2].csv
                                                          and the share of wrong records with its 95% interval
Reads general/outputs/first_author_records.csv, prediction_check.csv and data/<freeze>/features_author_identity.json.
Run from the repository root.
"""
import glob
import json
import os
import random
import re
import sys
import unicodedata
from urllib.parse import quote

import pandas as pd
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import FREEZE, GEN_OUT, data  # noqa: E402
from review_page import NO, UNSURE, YES, render  # noqa: E402

N_PEOPLE = 100
ROUNDS = {1: {"seed": 20261019, "tag": ""}, 2: {"seed": 20261114, "tag": "_r2"}}
RULES = {"single": "nobody else shares this 'Surname, Initial' in ADS: every spelling under it is counted, and the "
                   "ORCID",
         "shared": "others share this 'Surname, Initial': only this person's full-name spellings and the ORCID are "
                   "counted (records with initials only are left out, so the count is a lower limit)",
         "shared-orcid": "the name is shared and no full given name is known: only the ORCID is counted (lower limit)",
         "unresolved": "the name is shared, with no full given name and no ORCID: no count"}


def draw(pool, rnd):
    """The person ids of a round; a later round never repeats a person of an earlier one."""
    taken = set()
    for k in sorted(ROUNDS):
        left = [p for p in pool if p not in taken]
        ids = sorted(random.Random(ROUNDS[k]["seed"]).sample(left, N_PEOPLE))
        if k == rnd:
            return ids
        taken |= set(ids)


def people(rnd):
    """The checked people: (person id, the record of their latest modelled paper, every sample paper of theirs)."""
    rec = pd.read_csv(os.path.join(GEN_OUT, "first_author_records.csv"))
    modelled = set(pd.read_csv(os.path.join(GEN_OUT, "prediction_check.csv"))["bibcode"])
    pool = sorted(rec.loc[rec["bibcode"].isin(modelled) & (rec["person"] >= 0), "person"].unique())
    chosen = draw(pool, rnd)
    out = []
    for pid in chosen:
        mine = rec[rec["person"] == pid]
        latest = mine[mine["bibcode"].isin(modelled)].sort_values(["year", "bibcode"]).iloc[-1]
        out.append((int(pid), latest, mine))
    return out, len(pool)


def page(rnd):
    ident = json.load(open(data("features_author_identity.json")))
    chosen, n_pool = people(rnd)
    seed, tag = ROUNDS[rnd]["seed"], ROUNDS[rnd]["tag"]
    cards = []
    for pid, r, mine in chosen:
        sur, _, giv = "".join(c for c in unicodedata.normalize("NFKD", r["name"]) if not unicodedata.combining(c)) \
            .partition(",")                              # as parse_name in notebook 04
        toks = [t for t in re.split(r"[\s.]+", giv.strip()) if t]
        ini = toks[0][0].upper() if toks else ""
        q = r["query"] if isinstance(r["query"], str) else None
        forms = ident.get(f"forms|{sur.strip()}, {ini}", {})
        counted = set() if q is None else (set(forms) if r["rule"] == "single" else
                                           {f for f in forms if f'=author:"{f}"' in q})
        spell = ", ".join(f"{'✓ ' if f in counted else ''}{f} ({n})"
                          for f, n in sorted(forms.items(), key=lambda kv: -kv[1])[:40]) or "none in ADS"
        years = {int(y): n for y, n in ident.get(f"years|{q}", {}).items()} if q else {}
        y0 = None if pd.isna(r["first_year"]) else int(r["first_year"])
        def why(y):
            if y >= int(r["year"]):
                return " (the paper's year or later: not counted)"
            return " (before the career start: not counted)" if y0 and y < y0 else ""
        per_year = " · ".join(f"{y}: {years[y]}{why(y)}" for y in sorted(years)) or "none"
        names = sorted({f"{n}" + (f" [{o}]" if isinstance(o, str) else "") for n, o in zip(mine["name"], mine["orcid"])})
        n_prior = "no count" if pd.isna(r["n_prior"]) else f"{int(r['n_prior'])}"
        start = "none" if pd.isna(r["first_year"]) else f"{int(r['first_year'])}"
        links = []
        if q:
            links.append({"label": "ADS search of the counted papers",
                          "url": "https://ui.adsabs.harvard.edu/search/q=" + quote(
                              f"({q}) property:refereed collection:astronomy"
                              + (f" year:{y0}-{int(r['year']) - 1}" if y0 else "")) + "&sort=date%20asc"})
            links.append({"label": "all papers under the query (context)",
                          "url": "https://ui.adsabs.harvard.edu/search/q=" + quote(
                              f"({q}) property:refereed collection:astronomy") + "&sort=date%20asc"})
        if isinstance(r["orcid"], str):
            links.append({"label": "ORCID record", "url": f"https://orcid.org/{r['orcid']}"})
        cards.append({
            "id": f"person-{pid}", "bibcode": r["bibcode"], "year": int(r["year"]),
            "title": r["name"] + (f"  ·  ORCID {r['orcid']}" if isinstance(r["orcid"], str) else "  ·  no ORCID"),
            "authors": f"rule: {r['rule']}",
            "abstract": f"{RULES.get(r['rule'], r['rule'])}. Earlier refereed papers before {int(r['year'])}: "
                        f"{n_prior}; career start: {start}. First author of {len(mine)} sample paper(s).",
            "ads": f"https://ui.adsabs.harvard.edu/abs/{r['bibcode']}/abstract", "arxiv": None, "links": links,
            "facilities": None, "data_links": None,
            "snippets": [{"field": "names on the sample papers", "text": " | ".join(names)},
                         {"field": "ADS query", "text": q or "none"},
                         {"field": "spellings under 'Surname, Initial' (✓ counted), refereed astronomy papers",
                          "text": spell},
                         {"field": "counted papers per year, all years", "text": per_year}]})
    help_html = f"""
    <p>Round {rnd}: {N_PEOPLE} people drawn at random (seed {seed}{", none of round 1" if rnd > 1 else ""}) from the {n_pool:,} first authors of the modelled papers of the
      {FREEZE} sample. Each card shows the record the citation model uses for that person: the papers counted by
      the ADS query, refereed and in astronomy, from the career start to the year <b>before</b> the paper: the
      paper's own year and later are never counted, so a young author may have only one or two counted papers.
      Open "ADS search of the counted papers" (oldest first; exactly those years) and judge whether every counted
      paper is this person's; "all papers under the query" shows the whole record, which helps to tell people
      apart. The search runs on today's ADS, so it can show a few papers
      added since the freeze; the papers-per-year line is the count of the freeze.</p>
    <dl>
      <dt style="color:{YES['color']}">Right record</dt>
      <dd>The counted papers are this person's. For a shared name some of their papers are missing by design (the
        count is a lower limit): that is not an error.</dd>
      <dt style="color:{NO['color']}">Wrong record</dt>
      <dd>Papers of somebody else are counted: an identical name of another person (often an old, unrelated
        record), or spellings ADS attaches to this person that are someone else. Say which in the note.</dd>
      <dt style="color:{UNSURE['color']}">Unsure</dt>
      <dd>Cannot tell from the ADS record and the ORCID.</dd>
    </dl>"""
    options = [{"v": "right", "label": "Right record", "short": "right", **YES},
               {"v": "wrong", "label": "Wrong record", "short": "wrong", **NO},
               {"v": "unsure", "label": "Unsure", "short": "unsure", **UNSURE}]
    out = render(cards, name="author check", title="Is the first author's record right? · author check",
                 help_html=help_html, options=options, file_prefix=f"author_check{tag}_decisions",
                 store_key=f"jwst-author-check{tag}-{FREEZE}", save_dir=f"data/{FREEZE}/",
                 out_path=os.path.join(GEN_OUT, f"author_check{tag}.html"))
    print(f"{N_PEOPLE} of {n_pool:,} people -> {os.path.relpath(out, ROOT)}")


def score(rnd):
    tag = ROUNDS[rnd]["tag"]
    prefix = f"author_check{tag}_decisions"
    files = sorted(glob.glob(data(f"{prefix}_*.json")))
    if not files:
        print(f"no {prefix}_*.json in data/{FREEZE}/: run `page {rnd}`, judge, save, then score")
        return
    decided = {}
    for path in files:                                   # the latest file wins
        for d in json.load(open(path))["decisions"]:
            if d.get("decision"):
                decided[d["paper_id"]] = d
    chosen, n_pool = people(rnd)
    rows = [{"person": f"person-{pid}", "bibcode": r["bibcode"], "name": r["name"], "rule": r["rule"],
             "decision": decided.get(f"person-{pid}", {}).get("decision"),
             "note": decided.get(f"person-{pid}", {}).get("note", "")} for pid, r, _ in chosen]
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(GEN_OUT, f"author_check{tag}.csv"), index=False)
    n_right, n_wrong = (t["decision"] == "right").sum(), (t["decision"] == "wrong").sum()
    n = n_right + n_wrong
    print(f"{t['decision'].notna().sum()} of {len(t)} people judged: {n_right} right, {n_wrong} wrong, "
          f"{(t['decision'] == 'unsure').sum()} unsure")
    if n:
        ci = binomtest(int(n_wrong), int(n)).proportion_ci(0.95, method="exact")
        print(f"wrong records: {n_wrong / n:.1%} of {n} people [95% interval {ci.low:.1%}, {ci.high:.1%}]")
        print(t.loc[t["decision"] == "wrong", ["name", "rule", "note"]].to_string(index=False))


if __name__ == "__main__":
    {"page": page, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "page"](
        int(sys.argv[2]) if len(sys.argv) > 2 else 2)
