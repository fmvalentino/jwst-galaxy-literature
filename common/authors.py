"""Author helpers: team size, first author, and resolution of name variants into one identity per
person (ORCID first, then exact names linked to an ORCID, then compatible full names within a surname).
Copied unchanged from the earlier working folder of this project (AGENTS.md, section 1)."""
import re
import unicodedata


# Number of authors on an ADS record.
def n_authors(doc):
    return len(doc.get("author") or [])


# First author's name as ADS spells it ('Surname, Given').
def first_author(doc):
    au = doc.get("author") or []
    return au[0] if au else None


# Name helpers for resolve_authors: 'Müller' and 'Muller' compare equal; surname and given-name
# tokens are taken from 'Surname, Given Middle'.
def _strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _surname(name):
    last, _, _ = name.partition(",")
    return _strip_accents(last).strip().lower()


def _given_tokens(name):
    ''"'J.-S.' -> ['j','s']; 'Jiasheng' -> ['jiasheng']; 'Katriona M. L.' -> ['katriona','m','l'].''"
    _, _, rest = name.partition(",")
    toks = re.split(r"[\s\-]+", _strip_accents(rest).strip())
    return [t.strip(".").lower() for t in toks if t.strip(".")]


def _names_compatible(name_a, name_b):
    '''
    True if two names sharing a surname could plausibly be the same person: every
    given/middle-name token that is spelled out (not a bare initial) in BOTH names must
    match exactly; a token that is only an initial on one side just needs to agree on
    its first letter. This is what tells 'Ellis, R.' / 'Ellis, Richard' apart from
    'Zhang, Junkai' / 'Zhang, Junyu' -- same surname, but neither given name is a bare
    initial and they disagree once spelled out. First-initial-only matching (an earlier
    version of this function) merged those wrongly; this is the fix.
    '''
    ta, tb = _given_tokens(name_a), _given_tokens(name_b)
    if not ta or not tb:
        return False
    for x, y in zip(ta, tb):
        if len(x) > 1 and len(y) > 1:
            if x != y:
                return False
        elif x[0] != y[0]:
            return False
    return True


def resolve_authors(docs):
    '''
    Assigns each (bibcode, position) author slot an identity key, merging occurrences
    that are very likely the same person:

      1. Same ORCID (orcid_pub, else orcid_user, else orcid_other) -> same identity.
         High confidence -- this is what ORCID is for.
      2. No ORCID on this occurrence, but the exact same name string has a known ORCID
         somewhere else in the dataset -> linked to that identity. Still high confidence:
         it takes an exact full "Last, First Middle" match, not just a surname.
      3. Never seen with an ORCID anywhere -> clustered within each surname by
         _names_compatible (see its docstring). A heuristic, and the only tier that can
         mis-merge or mis-split -- e.g. two people who share a surname, initial-only
         given names on both records, and are in fact different ('Huang, J.' could be
         almost anyone) still collide here. Rare in practice; spot-check if in doubt.

    Returns (identities, orcid_coverage): identities is {(bibcode, position): key};
    orcid_coverage is the fraction of author-occurrences carrying some ORCID. `docs` must
    be raw ADS doc dicts (author/orcid_pub/orcid_user/orcid_other arrays), not row dicts
    built by results_to_df/cmp_row.
    '''
    # ORCID of the i-th author on a record, from whichever of ADS's three ORCID fields has it ('-'
    # means none).
    def orcid_of(doc, i):
        for field in ("orcid_pub", "orcid_user", "orcid_other"):
            vals = doc.get(field) or []
            if i < len(vals) and vals[i] and vals[i] != "-":
                return vals[i]
        return None

    # Every author slot of every paper, plus the ORCID seen for each exact name string (tiers 1 and
    # 2).
    occurrences = []
    name_to_orcid = {}
    for doc in docs:
        for i, name in enumerate(doc.get("author") or []):
            orcid = orcid_of(doc, i)
            occurrences.append((doc["bibcode"], i, name, orcid))
            if orcid:
                name_to_orcid.setdefault(name, orcid)

    # tier 3: cluster the remaining names within each surname by mutual compatibility
    # (union-find), instead of a flat "surname + first initial" key that can't tell
    # apart two different given names sharing a first letter.
    remaining = {name for _, _, name, orcid in occurrences if not orcid and name not in name_to_orcid}
    by_surname = {}
    for name in remaining:
        by_surname.setdefault(_surname(name), []).append(name)

    name_cluster = {}
    for surname, names in by_surname.items():
        parent = {n: n for n in names}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                if _names_compatible(names[a], names[b]):
                    ra, rb = find(names[a]), find(names[b])
                    if ra != rb:
                        parent[ra] = rb
        for n in names:
            name_cluster[n] = (surname, find(n))

    # Final key per author slot: the ORCID when known (directly or through the name), else the name
    # cluster.
    identities = {}
    n_with_orcid = 0
    for bib, i, name, orcid in occurrences:
        if orcid:
            key = ("orcid", orcid)
            n_with_orcid += 1
        elif name in name_to_orcid:
            key = ("orcid", name_to_orcid[name])
        else:
            key = ("name",) + name_cluster[name]
        identities[(bib, i)] = key

    coverage = n_with_orcid / len(occurrences) if occurrences else 0.0
    return identities, coverage
