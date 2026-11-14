"""Paths and settings shared by every notebook and script.

FREEZE is the date of the ADS data snapshot. Every cache, raw search and derived table lives under
data/<FREEZE>/ (inputs) or general/outputs (results). Change FREEZE only to start a new snapshot
(see AGENTS.md).
"""
import os

FREEZE = "2026-10-19"
CITES_AS_OF = FREEZE[:7]            # month of the citation snapshot, used by the citation models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", FREEZE)          # ADS pulls and caches of this freeze
GEN_OUT = os.path.join(ROOT, "general", "outputs")  # tables written by the general analysis
GEN_PLOTS = os.path.join(ROOT, "general", "plots")
for _d in (DATA, GEN_OUT, GEN_PLOTS):
    os.makedirs(_d, exist_ok=True)


def data(name):
    """Path of a file in this freeze's data folder."""
    return os.path.join(DATA, name)


# Search window and sample rules shared by every analysis.
YEAR_RANGE = "2023-"                 # ADS publication-year filter of every search
CALC_FROM_YEAR = 2023                # first publication year in all calculations
GALAXY_CLASSES = ["astro-ph.GA", "astro-ph.CO"]   # arXiv subjects that count as galaxy papers
PAPER_DOCTYPES = {"article", "eprint"}            # journal articles and preprints only
SORT = "date asc, bibcode asc"       # unique sort, so paged results neither overlap nor skip papers
SLEEP_SEC = 0.3                      # polite pause between ADS requests

# JWST data-use rule (common/jwst_use.py, step 2): a candidate uses JWST data when it is in ADS's curated
# JWST bibliography (bibgroup:JWST), or one passage of its text scores at least DATA_USE_MIN_SCORE points.
DATA_USE_MIN_SCORE = 7

REFRESH = False                      # True: ignore every cache and query ADS again


def ads_token():
    """ADS API token from the ADS_TOKEN environment variable (never stored in this repository)."""
    tok = os.environ.get("ADS_TOKEN")
    if not tok:
        raise RuntimeError("Set the ADS_TOKEN environment variable (https://ui.adsabs.harvard.edu/user/settings/token)")
    return tok


# The ADS texts (abstracts, full-text passages) are copyright of their publishers and are not redistributed: a copy
# without them has data/<FREEZE>/texts_removed.json, and every step that reads them stops until
# `python general/tools/restore_texts.py` has fetched them again from ADS.
TEXTS_REMOVED = os.path.join(DATA, "texts_removed.json")


def require_texts():
    """Stop when the ADS texts of this freeze have been removed (see TEXTS_REMOVED)."""
    if os.path.exists(TEXTS_REMOVED):
        raise RuntimeError("The ADS abstracts and full-text passages are not in this copy: run "
                           "`ADS_TOKEN=... python general/tools/restore_texts.py` first (AGENTS.md, step 0).")
