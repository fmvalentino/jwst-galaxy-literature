"""Put back the ADS texts of the freeze: abstracts and full-text passages (highlights).

They are copyright of their publishers, so a public copy of this repository carries them emptied (null), with
data/<freeze>/texts_removed.json as the marker; every step that reads them stops until they are restored
(common.config.require_texts). Run, with an ADS token (about 40 queries per 2,000 papers, 4 bigqueries):
    ADS_TOKEN=... python general/tools/restore_texts.py
Each text is fetched again with the same ADS parameters as the step that first pulled it, and written back in the
original key order and format. Papers are asked by `identifier`, not `bibcode`: ADS files many preprints of the
freeze under their published bibcode now, and each record it returns is stored under the bibcode of the freeze. The script then compares every file with the checksums of
data/<freeze>/manifest.json, written when the freeze was complete: a file that matches is exactly the frozen
one; one that does not holds a text ADS has changed since (new full text, corrected abstract). Removes the marker.

`strip(data_dir)` is the reverse, used to make the public copy of the repository.
"""
import hashlib
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.config import DATA, SLEEP_SEC, TEXTS_REMOVED  # noqa: E402

# Highlight caches {bibcode: {field: [passages]}} and the ADS parameters that filled each (01_sample.ipynb,
# common/jwst_use.py, 04_citation_features.ipynb, general/tools/make_validation_review.py). The first pass
# (raw_candidate_search.json) used paginated_search, which sends no hl.maxAnalyzedChars.
HL_Q_REVIEW = 'JWST OR "James Webb" OR NIRCam OR NIRSpec OR NIRISS OR MIRI'
HL_Q_PROG = "program OR programme OR programs OR programmes OR PID OR GO OR GTO OR ERS OR ERO OR DDT OR proposal"


def _hl_params():
    from common.jwst_use import DEEP_HL_Q, HL_Q, USE_HL_Q
    return {
        "raw_candidate_search.json": dict(batch=50, fl="body,abstract,title,ack", q=HL_Q, snippets="5"),
        "jwst_use_highlights.json": dict(batch=50, fl="body,ack", q=USE_HL_Q, snippets="10"),
        "jwst_deep_highlights.json": dict(batch=50, fl="body", q=DEEP_HL_Q, snippets="5", max_chars="5000000"),
        "features_programme_snippets.json": dict(batch=40, fl="body,ack", q=HL_Q_PROG, snippets="10",
                                                 max_chars="5000000"),
        "validation/review_highlights.json": dict(batch=50, fl="body,ack", q=HL_Q_REVIEW, snippets="8",
                                                  fragsize="300", max_chars="5000000"),
    }


HL_FILES = ["raw_candidate_search.json", "jwst_use_highlights.json", "jwst_deep_highlights.json",
            "features_programme_snippets.json", "validation/review_highlights.json"]
ABSTRACT_FILES = ["abstracts.json", "topic_texts.json", "features_ads_metadata.json"]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path, obj):                      # as common.ads.save_json and every cache writer of the repository
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _hl_map(name, obj):
    return obj["highlighting"] if name == "raw_candidate_search.json" else obj


# ── strip: texts -> null, structure and order kept ──
def strip(data_dir):
    """Empty every ADS text in data_dir (null), keep bibcodes, field names and order; write the marker."""
    removed = {}
    for name in ABSTRACT_FILES:
        path = os.path.join(data_dir, name)
        obj = _load(path)
        n = 0
        for b, v in obj.items():
            if name == "abstracts.json":
                n += v is not None
                obj[b] = None
            elif isinstance(v, dict) and "abstract" in v:
                n += bool(v["abstract"])
                v["abstract"] = None
        _save(path, obj)
        removed[name] = f"abstract emptied ({n:,} abstracts)"
    for name in HL_FILES:
        path = os.path.join(data_dir, name)
        obj = _load(path)
        hl = _hl_map(name, obj)
        n = 0
        for b, h in hl.items():
            n += bool(h)
            hl[b] = {field: None for field in h}
        _save(path, obj)
        removed[name] = f"full-text passages emptied ({n:,} papers)"
    _save(os.path.join(data_dir, os.path.basename(TEXTS_REMOVED)),
          {"note": "ADS abstracts and full-text passages are copyright of their publishers and are not "
                   "redistributed; run `ADS_TOKEN=... python general/tools/restore_texts.py` to fetch them again.",
           "files": removed})
    return removed


# ── restore ──
def fetch_abstracts(bibcodes):
    """{bibcode: abstract or None}, one ADS bigquery per 2,000 bibcodes."""
    from common.ads import bigquery, by_request
    out = {}
    for i in range(0, len(bibcodes), 2000):
        part = bibcodes[i:i + 2000]
        for b, d in by_request(part, bigquery("*:*", part, fl="bibcode,identifier,abstract")).items():
            out[b] = d.get("abstract")
    return out


def fetch_highlights(bibcodes, p):
    """{bibcode: {field: [passages]}} with the parameters p of the pass that first filled the file."""
    from common.ads import highlights_for
    out = {}
    for i in range(0, len(bibcodes), p["batch"]):
        part = bibcodes[i:i + p["batch"]]
        params = {"hl.fl": p["fl"], "hl.q": p["q"], "hl.snippets": p["snippets"]}
        if "fragsize" in p:
            params["hl.fragsize"] = p["fragsize"]
        if "max_chars" in p:
            params["hl.maxAnalyzedChars"] = p["max_chars"]
        out.update(highlights_for(part, params))
        time.sleep(SLEEP_SEC)
        print(f"  {i + len(part):,} of {len(bibcodes):,}", end="\r")
    return out


def restore(data_dir=DATA):
    marker = os.path.join(data_dir, os.path.basename(TEXTS_REMOVED))
    if not os.path.exists(marker):
        print("The texts are in place (no texts_removed.json): nothing to do.")
        return
    objs = {n: _load(os.path.join(data_dir, n)) for n in ABSTRACT_FILES}
    need = set(objs["abstracts.json"])
    for n in ABSTRACT_FILES[1:]:
        need |= {b for b, v in objs[n].items() if isinstance(v, dict) and "abstract" in v}
    print(f"abstracts: {len(need):,} papers")
    got = fetch_abstracts(sorted(need))
    for b in objs["abstracts.json"]:
        objs["abstracts.json"][b] = got.get(b)
    for b, v in objs["topic_texts.json"].items():           # as common.trending_shared.topic_texts
        if v.get("title") or v.get("pubdate"):              # returned by ADS when the cache was filled
            v["abstract"] = got.get(b) or ""
        else:
            v["abstract"] = ""
    for b, v in objs["features_ads_metadata.json"].items():    # the ADS doc as returned (04, pull 1)
        if isinstance(v, dict) and "abstract" in v:
            v["abstract"] = got.get(b)
    for n, obj in objs.items():
        _save(os.path.join(data_dir, n), obj)
    params = _hl_params()
    for name in HL_FILES:
        path = os.path.join(data_dir, name)
        obj = _load(path)
        hl = _hl_map(name, obj)
        todo = [b for b, h in hl.items() if h]             # papers that had passages; the others stay {}
        print(f"{name}: {len(todo):,} papers")
        got = fetch_highlights(todo, params[name])
        for b in todo:
            hl[b] = got.get(b, {})
        _save(path, obj)
    os.remove(marker)
    check(data_dir)


def check(data_dir=DATA):
    """Compare the text files with the checksums of the complete freeze (manifest.json)."""
    files = _load(os.path.join(data_dir, "manifest.json"))["files"]
    for name in ABSTRACT_FILES + HL_FILES:
        sha = hashlib.sha256(open(os.path.join(data_dir, name), "rb").read()).hexdigest()
        print(f"  {'identical to the freeze' if sha == files[name]['sha256'] else 'DIFFERS from the freeze':24s} {name}")


if __name__ == "__main__":
    restore()
