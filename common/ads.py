"""ADS API helpers shared by all notebooks: requests with retries, paged searches (with optional
highlights), bigquery (a search restricted to a list of bibcodes), per-bibcode field caches, raw-result
caches, and merging of preprint/published duplicates.

Every function that queries ADS reads the token from the ADS_TOKEN environment variable
(common.config.ads_token). Caches are plain JSON files in data/<FREEZE>/.
"""
import html
import json
import os
import re
import time
from datetime import datetime

import requests

from common.config import SLEEP_SEC, SORT, REFRESH, PAPER_DOCTYPES, ads_token

ADS_BASE = "https://api.adsabs.harvard.edu/v1"


def ads_get(path, params=None, retries=3, timeout=60):
    """Authenticated GET to the ADS API, with a short retry on network errors."""
    for attempt in range(retries):
        try:
            r = requests.get(ADS_BASE + path, headers={"Authorization": f"Bearer {ads_token()}"},
                             params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def paginated_search(q, fq=None, fl="bibcode", rows=200, max_rows=5000, hl=False, hl_fl=None, hl_q=None,
                     hl_snippets=4, sort=SORT):
    """Page through /search/query results; yields (docs, highlighting) per page ({} without hl).

    Never pass 'hl.fragsize': current ADS deployments fail or drop the highlighting block with it.
    The sort must be unique (date, then bibcode) so pages neither overlap nor skip papers.
    """
    start = 0
    while start < max_rows:
        params = {"q": q, "fl": fl, "rows": min(rows, max_rows - start), "start": start, "sort": sort}
        if fq:
            params["fq"] = fq
        if hl:
            params.update({"hl": "true", "hl.fl": hl_fl, "hl.q": hl_q or q, "hl.snippets": str(hl_snippets)})
        resp = ads_get("/search/query", params)
        response = resp.get("response", {})
        docs = response.get("docs", [])
        yield docs, (resp.get("highlighting", {}) if hl else {})
        start += len(docs)
        if not docs or start >= response.get("numFound", 0):
            break
        time.sleep(SLEEP_SEC)


def by_request(asked, docs):
    """{requested bibcode: ADS record}: the record filed under that bibcode, or else the one that lists it among its
    identifiers (a preprint ADS has filed under its published bibcode since it was first pulled)."""
    asked = set(asked)
    out = {d["bibcode"]: d for d in docs if d["bibcode"] in asked}
    for d in docs:
        for b in d.get("identifier") or []:
            if b in asked and b not in out:
                out[b] = d
    return out


def highlights_for(bibcodes, hl_params):
    """{requested bibcode: {field: [passages]}} for a few dozen bibcodes in one query, with the highlight parameters
    hl_params (hl.fl, hl.q, ...). Asked by `identifier`, not `bibcode`, so that a re-bibcoded paper is still found;
    papers without passages are left out."""
    resp = ads_get("/search/query", {"q": "identifier:(" + " OR ".join(f'"{b}"' for b in bibcodes) + ")",
                                     "fl": "id,bibcode,identifier", "rows": 2 * len(bibcodes), "hl": "true",
                                     **hl_params})
    hl = resp.get("highlighting", {})
    return {b: hl[str(d["id"])] for b, d in by_request(bibcodes, resp.get("response", {}).get("docs", [])).items()
            if hl.get(str(d["id"]))}


def bigquery(q, bibcodes, fl="bibcode", rows=2000, sort=None):
    """All docs matching q among a fixed list of bibcodes (POST /search/bigquery), paged.
    ADS allows 100 bigqueries per day."""
    body = "bibcode\n" + "\n".join(bibcodes)
    docs, start = [], 0
    while True:
        params = {"q": q, "fl": fl, "rows": rows, "start": start}
        if sort:
            params["sort"] = sort
        for attempt in range(3):
            try:
                r = requests.post(ADS_BASE + "/search/bigquery", params=params, data=body, timeout=180,
                                  headers={"Authorization": f"Bearer {ads_token()}", "Content-Type": "big-query/csv"})
                r.raise_for_status()
                break
            except requests.exceptions.RequestException:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        resp = r.json()["response"]
        docs.extend(resp["docs"])
        start += len(resp["docs"])
        if not resp["docs"] or start >= resp["numFound"]:
            return docs
        time.sleep(SLEEP_SEC)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def fetch_field(bibcodes, field, cache_path):
    """{bibcode: value of one ADS field}, looked up via bigquery for bibcodes not yet cached.
    Bibcodes ADS does not return (re-bibcoded since) map to None and are not asked again."""
    cache = {} if REFRESH else (load_json(cache_path, {}) or {})
    missing = sorted(set(bibcodes) - set(cache))
    if missing:
        found = {d["bibcode"]: d.get(field) for d in bigquery("*:*", missing, fl=f"bibcode,{field}")}
        cache.update({b: found.get(b) for b in missing})
        save_json(cache_path, cache)
        print(f"  {field}: looked up {len(missing)} bibcodes in ADS")
    return {b: cache.get(b) for b in bibcodes}


def is_paper(doctype):
    """Journal article or preprint. An unknown type (bibcode not found) counts as a paper: almost
    always a preprint whose bibcode was replaced by the published one."""
    return not isinstance(doctype, str) or doctype in PAPER_DOCTYPES


def strip_em(text):
    """Remove ADS highlight tags <em>...</em> and decode HTML entities."""
    return html.unescape(re.sub(r"</?em>", "", text))


def load_raw(path, signature):
    """Raw ADS results saved at `path` if they exist and came from the same query, else None."""
    raw = None if REFRESH else load_json(path)
    if raw is None:
        return None
    if raw.get("signature") != signature:
        print(f"{os.path.basename(path)}: saved with a different query -> querying ADS again")
        return None
    print(f"Loaded {os.path.basename(path)} (saved {raw['saved']}, {len(raw['docs'])} records): no ADS query")
    return raw


def save_raw(path, signature, docs, highlighting):
    save_json(path, {"signature": signature, "saved": datetime.now().isoformat(timespec="seconds"),
                     "docs": docs, "highlighting": highlighting})
    print(f"Saved raw results -> {os.path.basename(path)} ({len(docs)} records)")


def dedupe(docs, hl):
    """Merge records of the same paper (arXiv preprint + published version, or several preprint
    versions), linked through shared bibcodes / arXiv ids in `identifier` or an identical long title.
    The published record is kept (else the most cited), with the highest citation count, the union of
    `bibgroup` and of all identifiers, and the highlights of whichever record has them.
    Returns (docs, highlights, n_removed)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def keys(d):
        ids = [i.lower() for i in (d.get("identifier") or []) if re.match(r"(?i)arxiv:|\d{4}arxiv", i)]
        title = re.sub(r"\W+", " ", html.unescape((d.get("title") or [""])[0])).strip().lower()
        ks = ["id:" + d["bibcode"].lower()] + ["id:" + i for i in ids]
        if len(title) > 25:
            ks.append("t:" + title)
        return ks

    for d in docs:
        ks = keys(d)
        for k in ks[1:]:
            parent[find(ks[0])] = find(k)
    groups = {}
    for d in docs:
        groups.setdefault(find("id:" + d["bibcode"].lower()), []).append(d)
    out, out_hl = [], {}
    for g in groups.values():
        g.sort(key=lambda d: (bool(re.match(r"\d{4}arXiv", d["bibcode"])), -(d.get("citation_count") or 0)))
        kept = dict(g[0])
        kept["citation_count"] = max((d.get("citation_count") or 0) for d in g)
        kept["bibgroup"] = sorted({b for d in g for b in (d.get("bibgroup") or [])})
        kept["identifier"] = sorted({i for d in g for i in [d["bibcode"]] + list(d.get("identifier") or [])})
        out.append(kept)
        for d in g:
            if d["bibcode"] in hl:
                out_hl[kept["bibcode"]] = hl[d["bibcode"]]
                break
    return out, out_hl, len(docs) - len(out)
