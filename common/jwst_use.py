"""The sample rule: which papers are JWST galaxy papers that use JWST data (AGENTS.md, section 4).

A candidate paper is in the sample when it passes four steps, in this order; the first step it fails is
the reason it is left out.

1. Candidate (ADS query CANDIDATE_Q): abstract mentions galaxies, full text mentions JWST, arXiv class
   astro-ph.GA or astro-ph.CO (or no astro-ph class), JWST in the title or abstract or JWST instrument /
   survey / data words in the full text; published 2023 or later.
2. Uses JWST data: the paper has one of two kinds of evidence (`evidence`):
     curated        it is in ADS's curated JWST bibliography (bibgroup:JWST, kept by STScI)
     explicit text  one passage of its text scores at least DATA_USE_MIN_SCORE (7) points:
                      7  STScI's JWST data acknowledgement, or a "Facilities: ... JWST" tag
                     +2  JWST data wording ("JWST data / observations / spectra ...")
                     +1  instrument name (NIRCam, NIRSpec, NIRISS, MIRI)
                     +2  survey, programme, MAST or GO identifier
                     +2  "we use" or past-tense wording
                     +2  data claimed in the title or abstract ("JWST spectra of ...")
                     -3  prediction wording (will, future, forecast, mock, simulation ...)
   Passages: three ADS highlight passes over the full text, plus each abstract sentence naming JWST, an
   instrument or a survey. `jwst_score` is the best passage's score (0-9). Papers scoring 5-6 without
   bibgroup are "weak text" and left out.
3. Journal article or preprint (ADS doctype article or eprint).
4. Galaxy paper: primary arXiv class not astro-ph.SR or astro-ph.EP; and, if the galaxy screen (GALAXY_SCREEN
   below) flags the paper, a hand verdict "galaxy" in data/<freeze>/galaxy_verdicts.csv. The screen runs on
   every paper that reaches this step; verdicts are read only for flagged papers.

No paper is added or removed by name outside these steps.
"""
import html
import os
import re
import time

import pandas as pd

from common.ads import paginated_search, ads_get, highlights_for, load_json, save_json, strip_em
from common.config import GALAXY_CLASSES, DATA_USE_MIN_SCORE, SLEEP_SEC, REFRESH, data, require_texts

ADS_ABS = "https://ui.adsabs.harvard.edu/abs/{}/abstract"

# ── Step 1: the candidate query ──
CMP_TERMS = ('NIRSpec OR NIRCam OR NIRISS OR MIRI OR "JWST data" OR "JWST observations" '
             'OR "JWST imaging" OR "JWST spectroscopy" OR "JWST spectra" OR "JWST photometry" '
             'OR CEERS OR JADES OR UNCOVER OR PRIMER OR "COSMOS-Web" OR NGDEEP OR FRESCO '
             'OR GLASS')
# galaxy papers only: galaxy arXiv subject, OR no astro-ph subject at all (e.g. not on arXiv)
GAL_CLAUSE = ("(arxiv_class:(" + " OR ".join(f'"{c}"' for c in GALAXY_CLASSES) + ") "
              "OR (*:* NOT arxiv_class:astro-ph*))")
CANDIDATE_Q = (f'abs:("galaxy" OR "galaxies") AND {GAL_CLAUSE} AND full:"JWST" '
               f'AND (abs:"JWST" OR title:"JWST" OR full:({CMP_TERMS}))')
CANDIDATE_FIELDS = ("id,bibcode,title,year,citation_count,bibgroup,identifier,author,aff,"
                    "orcid_pub,orcid_user,orcid_other")
HL_Q = CMP_TERMS + " OR JWST"          # first highlight pass, requested with the search itself

# ── Step 2: snippet scoring ──
# telescope / instrument name followed, within the same sentence, by a data noun
DATA_RE = re.compile(r"\b(?:JWST|James Webb Space Telescope|Webb|NIRCam|NIRSpec|NIRISS|MIRI)\b"
                     r"[^.;:]{0,40}?\b(?:data|observations?|imaging|images?|spectroscopy|spectra|spectrum|"
                     r"photometry|catalogs?|surveys?|programs?|mosaics?|grism|IFU|IFS|prism)\b", re.I)
# same, reversed order: "spectroscopic and photometric data from the JWST"
DATA_REV_RE = re.compile(r"\b(?:data|observations?|imaging|images?|spectroscopy|spectra|photometry)\b"
                         r"[^.;:]{0,30}?\b(?:from|with|by|of)\s+(?:the\s+)?(?:NASA/ESA/CSA\s+)?"
                         r"(?:JWST|James Webb Space Telescope|Webb)\b", re.I)
# decisive on their own: STScI's acknowledgement for papers using JWST data, the AAS facility tag
ACK_RE = re.compile(r"(?:observations (?:made )?with the NASA/ESA/CSA (?:James Webb Space Telescope|JWST)"
                    r"|(?:JWST|Webb)\W{0,3}The data were obtained from the Mikulski Archive"
                    r"|data (?:were |was )?obtained from the Mikulski Archive for Space Telescopes"
                    r"|\bFacilit(?:y|ies)\s*:[^|]{0,200}?\b(?:JWST|James Webb)\b)", re.I)
INSTR_RE = re.compile(r"\b(?:NIRCam|NIRSpec|NIRISS|MIRI)\b")
PROG_RE = re.compile(r"\b(?:CEERS|JADES|UNCOVER|PRIMER|COSMOS-Web|COSMOS-3D|NGDEEP|FRESCO|GLASS|"
                     r"EIGER|ASPIRE|SMILES|GTO|MAST|GO[\s-]?\d{4})\b")
USE_RE = re.compile(r"\b(?:observed|obtained|acquired|taken|imaged|reduced|processed|retrieved|"
                    r"downloaded|analy[sz]ed|archival|publicly available|using|based on|"
                    r"we (?:use|used|present|select|selected|measure|measured|fit|fitted))\b", re.I)
NEG_RE = re.compile(r"\b(?:will|would|could|future|upcoming|forthcoming|planned|proposed|anticipated|"
                    r"prospects?|forecasts?|predict\w*|prepar\w*|mock|simulat\w*|synthetic|expected to)\b", re.I)


def score_snippet(text, field="body"):
    """(score, has prediction wording) of one snippet; see the module docstring for the weights."""
    if ACK_RE.search(text):
        return 7, False
    fwd = bool(DATA_RE.search(text))
    datum = fwd or bool(DATA_REV_RE.search(text))
    neg = bool(NEG_RE.search(text))
    # the headline bonus needs the forward form ("JWST spectra of ..."); the reversed form is also how
    # theory papers cite results ("galaxies observed by JWST")
    sc = (2 * datum + bool(INSTR_RE.search(text)) + 2 * bool(PROG_RE.search(text))
          + 2 * bool(USE_RE.search(text)) + 2 * (fwd and not neg and field in ("title", "abstract")))
    return max(sc - 3 * neg, 0), neg


# ── extra text for scoring: two more highlight passes and the abstract sentences ──
# Second pass: sentences stating data use ("We use ... obtained by JWST") are often not among the few
# snippets of the first pass.
USE_HL_Q = ('(JWST OR NIRCam OR NIRSpec OR MIRI OR NIRISS) AND ("we use" OR "we used" OR "we present" '
            'OR "we analyze" OR obtained OR "observed with" OR "taken with" OR "data reduction" OR reduced '
            'OR pipeline OR MAST OR program OR programme OR Facilities)')
# Third pass, only for papers still without any snippet: ADS's highlighter reads only the first
# ~51,000 characters of the body unless hl.maxAnalyzedChars is raised. A plain query works best here.
DEEP_HL_Q = 'JWST OR "James Webb"'
ABS_KEEP_RE = re.compile(r"\b(?:JWST|Webb|NIRCam|NIRSpec|NIRISS|MIRI)\b|" + PROG_RE.pattern)


def fetch_use_highlights(bibcodes, cache_path, batch=50):
    """{bibcode: {field: [snippets]}} from the second highlight pass, cached per bibcode."""
    cache = {} if REFRESH else (load_json(cache_path, {}) or {})
    missing = sorted(set(bibcodes) - set(cache))
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        cache.update(highlights_for(part, {"hl.fl": "body,ack", "hl.q": USE_HL_Q, "hl.snippets": "10"}))
        for b in part:
            cache.setdefault(b, {})            # not returned by ADS: remember, do not re-query
        time.sleep(SLEEP_SEC)
    if missing:
        print(f"  data-use highlights: queried {len(missing)} papers")
        save_json(cache_path, cache)
    return {b: cache.get(b, {}) for b in bibcodes}


def fetch_deep_highlights(bibcodes, cache_path, batch=50):
    """{bibcode: {field: [snippets]}} from the whole-body pass, cached per bibcode."""
    cache = {} if REFRESH else (load_json(cache_path, {}) or {})
    missing = sorted(set(bibcodes) - set(cache))
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        cache.update(highlights_for(part, {"hl.fl": "body", "hl.q": DEEP_HL_Q, "hl.snippets": "5",
                                           "hl.maxAnalyzedChars": "5000000"}))
        for b in part:
            cache.setdefault(b, {})
        time.sleep(SLEEP_SEC)
    if missing:
        print(f"  whole-body highlights: queried {len(missing)} papers")
        save_json(cache_path, cache)
    return {b: cache.get(b, {}) for b in bibcodes}


class DataUseScorer:
    """Scores papers with the JWST data-use rule. Holds the extra texts (abstracts, second and third
    highlight passes) for the papers it has been prepared for."""

    def __init__(self):
        self.abstracts, self.use_hl, self.deep_hl = {}, {}, {}

    def abstract_sentences(self, bibcode):
        text = html.unescape(re.sub(r"<[^>]+>", "", self.abstracts.get(bibcode) or ""))
        return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text) if ABS_KEEP_RE.search(s)]

    def prepare(self, bibcodes, first_pass_hl, abstracts):
        """Fetch (or load from cache) the second and third highlight passes for `bibcodes`.
        `first_pass_hl`: {bibcode: highlights of the first pass}; `abstracts`: {bibcode: text}."""
        require_texts()
        self.abstracts.update(abstracts)
        self.use_hl.update(fetch_use_highlights(bibcodes, data("jwst_use_highlights.json")))
        no_text = [b for b in bibcodes
                   if not (first_pass_hl.get(b) or self.use_hl.get(b) or self.abstract_sentences(b))]
        self.deep_hl.update(fetch_deep_highlights(no_text, data("jwst_deep_highlights.json")))

    def score(self, doc, first_pass_hl):
        """Data-use verdict for one ADS doc: dict with score, reason, snippets, bibgroup flag."""
        bib = doc["bibcode"]
        texts = [(fld, strip_em(t)) for fld, ts in first_pass_hl.get(bib, {}).items()
                 if isinstance(ts, list) for t in ts]
        for extra in (self.use_hl, self.deep_hl):
            texts += [(fld, strip_em(t)) for fld, ts in extra.get(bib, {}).items()
                      if isinstance(ts, list) for t in ts]
        texts += [("abstract", s) for s in self.abstract_sentences(bib)]
        snips, seen = [], set()
        for fld, text in texts:
            if text in seen:
                continue
            seen.add(text)
            sc, neg = score_snippet(text, fld)
            snips.append({"field": fld, "text": text, "score": sc, "neg": neg, "hit": sc >= DATA_USE_MIN_SCORE})
        hits = [x for x in snips if x["hit"]]
        curated = "JWST" in (doc.get("bibgroup") or [])      # STScI-curated: paper uses JWST data
        score = max((x["score"] for x in snips), default=0)
        evidence = ("curated" if curated else "explicit text" if score >= DATA_USE_MIN_SCORE
                    else "weak text" if score >= 5 else "none")
        shown = hits or snips
        return {"jwst_score": score, "jwst_bibgroup": curated, "evidence": evidence,
                "uses_jwst_data": evidence in USES_DATA, "n_snippets": len(snips), "n_hits": len(hits),
                "snippets": " ||| ".join(f"[{x['field']}] {x['text']}" for x in shown)}


USES_DATA = ("curated", "explicit text")        # evidence classes that pass step 2


# ── Step 4: galaxy papers only ──
NON_GALAXY_PRIMARY = {"astro-ph.SR", "astro-ph.EP"}

# The galaxy screen flags a paper for a hand verdict when its title or abstract names an unambiguous
# non-galaxy subject (SCREEN_ALWAYS), or a likely one (SCREEN_UNLESS_EXTRAGALACTIC) with no extragalactic
# term (EXTRAGALACTIC). Out of scope: stars, star formation, the ISM and astrochemistry in the Milky Way and
# the Local Group (including the Magellanic Clouds); planets, brown dwarfs, the solar system; laboratory
# astrochemistry; the stellar distance ladder; papers not about galaxies. In scope: everything about
# galaxies beyond the Local Group, including their resolved stars, clusters and H II regions.
SCREEN_ALWAYS = re.compile(
    r"\b(protostar\w*|proto-?stellar|protoplanetary|young stellar objects?|YSOs?|proplyds?|pre-?stellar cores?|"
    r"hot corinos?|interstellar ices?|ice mantles?|complex organic molecules|astrochemi\w*|laboratory|"
    r"brown dwarfs?|exoplanets?|debris dis[ck]s?|solar system|"
    r"Magellanic|LMC|SMC|30 Dor\w*|R ?136|Orion|Taurus|Ophiuchus|Serpens|Chamaeleon|Horsehead|"
    r"photo-?dissociation regions?|PDRs4All|"
    r"Cepheids?|TRGB|JAGB|tip of the red giant branch|RR Lyrae|distance ladder|"
    r"Galactic (?:cent(?:er|re)|plane|disk|bulge|halo|supernova remnants?)|Sgr ?A|Cassiopeia A|Cas A|"
    r"(?:water|CO|CO2|H2O) ices?|ice features?)\b", re.I)
SCREEN_UNLESS_EXTRAGALACTIC = re.compile(
    r"\b(hot cores?|molecular clouds?|dark clouds?|COMs|H ?II regions?|Cygnus|Carina|globules?|pillars?|planets?|"
    r"circumstellar|Milky Way|solar neighbo(?:u)?rhood|"
    r"white dwarfs?|supernova remnants?|planetary nebulae?|Wolf-Rayet|globular clusters?|"
    r"NGC ?346|NGC ?604|M ?33|Andromeda|M ?31|Local Group|PDRs?|reflection nebulae?|"
    r"red supergiants?|AGB stars?|massive stars?|star clusters?|proper motions?)\b", re.I)
EXTRAGALACTIC = re.compile(
    r"\b(redshifts?|z ?[=~≈>≳] ?\d|high-z|quasars?|QSOs?|AGNs?|active galactic|galaxy clusters?|clusters? of galaxies|"
    r"cosmic|cosmolog\w*|reionization|epoch of|Little Red Dots?|LRDs?|lensed|lensing|early universe|"
    r"star-forming galaxies|dwarf galaxies|galaxies at|massive galaxies|nearby galaxies|"
    r"(?:spiral|elliptical|barred|disk) galax\w+)\b", re.I)
GALAXY_VERDICTS = "galaxy_verdicts.csv"     # data/<freeze>/: bibcode, verdict (galaxy / not_galaxy / unsure), source


def screen_terms(text):
    """The non-galaxy terms that make the screen flag `text` (title + abstract); empty if not flagged."""
    always = {m.lower() for m in SCREEN_ALWAYS.findall(text)}
    likely = {m.lower() for m in SCREEN_UNLESS_EXTRAGALACTIC.findall(text)}
    if always or (likely and not EXTRAGALACTIC.search(text)):
        return sorted(always | likely)
    return []


def load_galaxy_verdicts():
    """{bibcode: verdict} from data/<freeze>/galaxy_verdicts.csv (the only hand input of step 4)."""
    path = data(GALAXY_VERDICTS)
    return pd.read_csv(path).set_index("bibcode")["verdict"].to_dict() if os.path.exists(path) else {}


def galaxy_status(bibcode, arxiv_class, text, verdicts):
    """Step 4 for one paper: (passes, reason). reason is None for a paper that passes without review,
    "awaiting review" for a flagged paper with no verdict yet (it passes until judged)."""
    cls = arxiv_class or []
    if cls and cls[0] in NON_GALAXY_PRIMARY:
        return False, f"primary arXiv class {cls[0]}"
    if not screen_terms(text):
        return True, None
    v = verdicts.get(bibcode)
    if v == "not_galaxy":
        return False, "galaxy screen, judged not a galaxy paper"
    return True, ("awaiting review" if v is None else None)


def is_galaxy(arxiv_class, bibcode, text, verdicts):
    """Galaxy paper, for a paper found outside the candidate query: a galaxy arXiv class or no astro-ph
    class at all (the candidate query's condition), and step 4."""
    cls = arxiv_class or []
    if not (set(cls) & set(GALAXY_CLASSES) or not any(c.startswith("astro-ph") for c in cls)):
        return False
    return galaxy_status(bibcode, cls, text, verdicts)[0]
