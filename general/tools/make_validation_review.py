"""Validation of the sample rule: blind random samples for hand review (purity and completeness).

Validation only measures: it never edits the rule, the galaxy verdicts or the sample (AGENTS.md, section 4).
One question per paper: does it belong in the sample? A paper belongs when it is a galaxy paper and it is about
JWST data (its own or archival observations, reduction, analysis, catalogues) or, if it does not use them, its
figures show JWST data (whatever the kind of paper). Verdicts (one vocabulary: in / out with a reason / unsure):
  yes_analysis  in: analyses JWST data
  yes_plot      in: does not analyse JWST data, but shows JWST data in its figures
  no            out: no JWST data
  not_galaxy    out: not a galaxy paper
  unsure

Rounds (each with its own seed and key file; an existing key file is never redrawn):
  round 1  seed 20261020: 75 accepted, 45 rejected with score 3-4, 30 with score 0-2 (judged twice, blind)
  round 2  seed 20261027: 60 / 25 / 65, excluding round 1
  round 3  seed 20261029: 75 papers of the sample and 25 "weak text" papers (score 5-6, no bibgroup), excluding
           rounds 1-2
The strata are fixed in the key files; score_validation.py places every reviewed paper where the rule puts it.
Pools: the sample (general/outputs/sample.csv); the papers rejected at the data-use step (journal articles or
preprints, primary arXiv class not SR/EP) by evidence: weak text, none with score 3-4, none with score 0-2. Key files
data/<freeze>/validation/round<k>_key.csv stay out of the page, so the review is blind.

The page general/outputs/validation.html holds every round (P001-...). Decisions already saved
(validation_decisions_*.json) are built in; the filter shows what is left. Snippets: every passage of the rule's
three highlight passes plus one review-only pass (JWST, Webb and instrument names over the whole body and the
acknowledgements); ADS facilities and data links. Needs ADS_TOKEN only for papers not yet in the review caches.
Run from the repository root:
    python general/tools/make_validation_review.py
"""
import glob
import html
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.ads import ads_get, bigquery, dedupe, highlights_for, is_paper, load_json, save_json, strip_em  # noqa: E402
from common.config import FREEZE, GEN_OUT, data  # noqa: E402
from common.jwst_use import NON_GALAXY_PRIMARY, DataUseScorer  # noqa: E402
from review_page import NO, SCOPE, UNSURE, YES, render  # noqa: E402

ROUNDS = [{"round": 1, "seed": 20261020, "n": {"all accepted": 75, "score 3-4": 45, "score 0-2": 30}},
          {"round": 2, "seed": 20261027, "n": {"all accepted": 60, "score 3-4": 25, "score 0-2": 65}},
          {"round": 3, "seed": 20261029, "n": {"accepted": 75, "weak text": 25}}]
MAX_SNIPPETS, MAX_CHARS = 18, 450
FIELD_ORDER = {"title": 0, "abstract": 1, "body": 2, "ack": 3}
VDIR = data("validation")
os.makedirs(VDIR, exist_ok=True)

# ── pools under the current rule ──
raw = json.load(open(data("raw_candidate_search.json")))
docs, hl, _ = dedupe(raw["docs"], raw["highlighting"])
by_bib = {d["bibcode"]: d for d in docs}
abstracts = json.load(open(data("abstracts.json")))
arxiv = json.load(open(data("arxiv_class.json")))
acc = pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))
rej_all = pd.read_csv(os.path.join(GEN_OUT, "rejected.csv"))
rej = rej_all[rej_all["status"].str.startswith("no JWST data-use evidence") & rej_all["doctype"].map(is_paper)]
rej = rej[[not ((arxiv.get(b) or [None])[0] in NON_GALAXY_PRIMARY) for b in rej["bibcode"]]]
none = rej[rej["evidence"] == "none"]
POOLS = {"accepted": ("accepted", acc), "weak text": ("rejected", rej[rej["evidence"] == "weak text"]),
         "score 3-4": ("rejected", none[none["jwst_score"].between(3, 4)]),
         "score 0-2": ("rejected", none[none["jwst_score"] <= 2])}
POOLS["all accepted"] = POOLS["accepted"]          # name of the accepted stratum in rounds 1-2

# ── keys: load existing rounds, draw new ones (never redraw an existing key) ──
keys, next_id = [], 1
for R in ROUNDS:
    path = os.path.join(VDIR, f"round{R['round']}_key.csv")
    if os.path.exists(path):
        k = pd.read_csv(path)
    else:
        rng = np.random.default_rng(R["seed"])
        taken = set(pd.concat(keys)["bibcode"]) if keys else set()
        rows = []
        for stratum, n in R["n"].items():
            group, pool = POOLS[stratum]
            pool = pool[~pool["bibcode"].isin(taken)]
            for b in pool["bibcode"].iloc[rng.choice(len(pool), n, replace=False)]:
                rows.append({"bibcode": b, "group": group, "stratum": stratum, "stratum_N": len(POOLS[stratum][1]),
                             "stratum_n": n, "weight": len(POOLS[stratum][1]) / n})
        k = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
        k.insert(0, "paper_id", [f"P{next_id + i:03d}" for i in range(len(k))])
        k.to_csv(path, index=False)
        print(f"round {R['round']}: drew {len(k)} papers -> {os.path.relpath(path, ROOT)}")
    k["round"] = R["round"]
    keys.append(k)
    next_id = int(k["paper_id"].str[1:].astype(int).max()) + 1
key = pd.concat(keys, ignore_index=True)
bibs = list(key["bibcode"])

# ── texts: the rule's passes, plus the review-only pass and ADS metadata (cached) ──
scorer = DataUseScorer()
scorer.prepare(bibs, hl, abstracts)
REVIEW_HL, META = os.path.join(VDIR, "review_highlights.json"), os.path.join(VDIR, "review_metadata.json")
review_hl, meta = load_json(REVIEW_HL, {}) or {}, load_json(META, {}) or {}
todo = [b for b in bibs if b not in review_hl]
for i in range(0, len(todo), 50):
    part = todo[i:i + 50]
    review_hl.update(highlights_for(part, {
        "hl.fl": "body,ack", "hl.snippets": "8", "hl.fragsize": "300",
        "hl.q": 'JWST OR "James Webb" OR NIRCam OR NIRSpec OR NIRISS OR MIRI', "hl.maxAnalyzedChars": "5000000"}))
    for b in part:
        review_hl.setdefault(b, {})
if todo:
    save_json(REVIEW_HL, review_hl)
    print(f"review highlights: queried {len(todo)} papers")
todo = [b for b in bibs if b not in meta]
if todo:
    for d in bigquery("*:*", todo, fl="bibcode,facility,data"):
        meta[d["bibcode"]] = {"facility": d.get("facility") or [], "data": d.get("data") or []}
    for b in todo:
        meta.setdefault(b, {"facility": [], "data": []})
    save_json(META, meta)
    print(f"review metadata: queried {len(todo)} papers")


def all_snippets(bib):
    texts = [(f, strip_em(t)) for f, ts in hl.get(bib, {}).items() if isinstance(ts, list) for t in ts]
    for extra in (scorer.use_hl, scorer.deep_hl, review_hl):
        texts += [(f, strip_em(t)) for f, ts in extra.get(bib, {}).items() if isinstance(ts, list) for t in ts]
    out, seen = [], set()
    for f, t in texts:
        t = html.unescape(re.sub(r"\s+", " ", t)).strip()
        if t and t not in seen and f not in ("abstract", "title"):   # title and abstract are shown whole
            seen.add(t)
            out.append((f, t if len(t) <= MAX_CHARS else t[:MAX_CHARS] + " …"))
    out.sort(key=lambda x: FIELD_ORDER.get(x[0], 9))
    return out[:MAX_SNIPPETS]


all_rows = pd.concat([acc, rej_all]).drop_duplicates("bibcode").set_index("bibcode")
papers = []
for _, k in key.iterrows():
    b, d, r = k["bibcode"], by_bib.get(k["bibcode"], {}), all_rows.loc[k["bibcode"]]
    ids = d.get("identifier") or str(r.get("identifiers", "")).split(" | ")
    arx = next((re.sub(r"^arXiv:", "", i) for i in ids if re.match(r"(arXiv:)?\d{4}\.\d{4,5}", str(i))), None)
    authors = str(r.get("authors", "")).split(" | ")
    papers.append({"id": k["paper_id"], "bibcode": b, "title": str(r["title"]), "year": int(r["year"]),
                   "authors": authors[0] + (" et al." if len(authors) > 1 else ""),
                   "abstract": html.unescape(re.sub(r"<[^>]+>", "", abstracts.get(b) or "")) or "(no abstract in ADS)",
                   "ads": f"https://ui.adsabs.harvard.edu/abs/{b}/abstract",
                   "arxiv": f"https://arxiv.org/abs/{arx}" if arx else None,
                   "facilities": meta.get(b, {}).get("facility", []), "data_links": meta.get(b, {}).get("data", []),
                   "snippets": [{"field": f, "text": t} for f, t in all_snippets(b)]})

# ── decisions already saved, built into the page (latest per paper wins) ──
preload = {}
for p in sorted(glob.glob(os.path.join(VDIR, "validation_decisions_*.json"))):   # not the round-1 first pass
    saved = json.load(open(p))
    for d in saved["decisions"]:
        if d.get("decision") or d.get("note"):
            t = d.get("decided_at") or saved["saved_at"]
            if d["paper_id"] not in preload or (t or "") >= (preload[d["paper_id"]]["at"] or ""):
                preload[d["paper_id"]] = {"decision": d.get("decision"), "note": d.get("note") or "", "at": t}

starts = [k["paper_id"].iloc[0] for k in keys]
HELP = f"""
    <p>{len(papers)} papers from the JWST galaxy candidates of the {FREEZE} snapshot, in random order within each round
      (rounds start at {", ".join(starts)}). Some passed the sample rule and some did not; the page does not say which,
      so judge each paper on its own. <b>One question: does the paper belong in the sample?</b> For each paper you see the title, the facilities and data
      archives ADS lists for it, the abstract, and every passage of the full text that ADS highlighted for JWST terms
      (<span class="f">body</span>, <span class="f">ack</span> = acknowledgements). For some journals ADS returns no
      passages at all; judge those from the abstract and the facilities, or open the paper. Decisions saved from this
      page are shown; use the filter to see what is left.</p>
    <dl>
      <dt style="color:{YES['color']}">In: analyses JWST data</dt>
      <dd>The paper is about JWST data: its own or archival JWST observations, reduction, analysis, catalogues.</dd>
      <dt style="color:#4f9b77">In: shows JWST data in its figures</dt>
      <dd>Papers that do not analyse JWST data themselves but show JWST data, or measurements derived from them,
        in their figures (for comparison), whatever the kind of paper.</dd>
      <dt style="color:{NO['color']}">Out: no JWST data</dt>
      <dd>No JWST data shown: JWST mentioned only in the text (motivation, discussion, future prospects), other
        telescopes only, or theory and forecasts with no JWST data in the figures.</dd>
      <dt style="color:{SCOPE['color']}">Out: not a galaxy paper</dt>
      <dd>Whatever its data: stars, star formation, ISM or astrochemistry in the Milky Way or the Local Group
        (including the Magellanic Clouds), planets, brown dwarfs, the solar system, laboratory astrochemistry, the
        stellar distance ladder, papers not about galaxies.</dd>
      <dt style="color:{UNSURE['color']}">Unsure</dt>
      <dd>The text shown is not enough to decide. Open ADS or arXiv if you wish, or leave it unsure.</dd>
    </dl>"""
OPTIONS = [{"v": "yes_analysis", "label": "In: analyses JWST data", "short": "in: analyses", **YES},
           {"v": "yes_plot", "label": "In: JWST data in figures", "short": "in: figures", "color": "#4f9b77", "bg": "#eef6f1"},
           {"v": "no", "label": "Out: no JWST data", "short": "out: no data", **NO},
           {"v": "not_galaxy", "label": "Out: not a galaxy paper", "short": "out: not galaxy", **SCOPE},
           {"v": "unsure", "label": "Unsure", "short": "unsure", **UNSURE}]
out = render(papers, name="validation", title="Does the paper belong in the sample? · validation", help_html=HELP,
             options=OPTIONS, file_prefix="validation_decisions", store_key=f"jwst-sample-validation-v3-{FREEZE}",
             save_dir=f"data/{FREEZE}/validation/", out_path=os.path.join(GEN_OUT, "validation.html"),
             preload=preload)
n_snip = [len(p["snippets"]) for p in papers]
print(f"{len(papers)} papers in {len(keys)} rounds; {sum(1 for v in preload.values() if v['decision'])} decisions "
      f"built in; snippets per paper: median {int(np.median(n_snip))}, none for {sum(x == 0 for x in n_snip)}")
print(f"-> {os.path.relpath(out, ROOT)}")
