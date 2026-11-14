"""Galaxy screen (sample rule, step 4): the hand verdicts for the papers the screen flags.

Every paper that reaches step 4 (general/01_sample.ipynb) is run through the screen of common/jwst_use.py; a
flagged paper needs a hand verdict in data/<freeze>/galaxy_verdicts.csv (galaxy / not_galaxy / unsure). Until it
has one it stays in the sample, marked "awaiting review" in sample.csv (column galaxy_check).

    python general/tools/galaxy_screen.py page    review page of the flagged papers without a verdict
                                                  -> general/outputs/galaxy_screen.html
    python general/tools/galaxy_screen.py merge   add the verdicts saved from that page
                                                  (data/<freeze>/galaxy_screen_decisions_<date>.json) to
                                                  galaxy_verdicts.csv; then rerun steps 2-12

galaxy_verdicts.csv keeps every verdict ever given (source and date per row; the latest wins). The rule reads it
only for flagged papers. Run from the repository root.
"""
import datetime
import glob
import html
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.config import FREEZE, GEN_OUT, data, require_texts  # noqa: E402
from common.jwst_use import GALAXY_VERDICTS  # noqa: E402
from review_page import NO, UNSURE, YES, render  # noqa: E402

VERDICTS = data(GALAXY_VERDICTS)
PREFIX = "galaxy_screen_decisions"


def page():
    sample = pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))
    todo = sample[sample["galaxy_check"] == "awaiting review"]
    require_texts()
    abstracts = json.load(open(data("abstracts.json")))
    papers = []
    for _, r in todo.iterrows():
        authors = str(r.get("authors", "")).split(" | ")
        papers.append({"id": r["bibcode"], "bibcode": r["bibcode"], "title": str(r["title"]), "year": int(r["year"]),
                       "authors": authors[0] + (" et al." if len(authors) > 1 else ""),
                       "abstract": html.unescape(re.sub(r"<[^>]+>", "", abstracts.get(r["bibcode"]) or ""))
                       or "(no abstract in ADS)",
                       "ads": f"https://ui.adsabs.harvard.edu/abs/{r['bibcode']}/abstract", "arxiv": None,
                       "facilities": None, "data_links": None,
                       "snippets": [{"field": "screen terms", "text": r["galaxy_screen"]}]})
    help_html = f"""
    <p>{len(papers)} papers of the {FREEZE} sample that the galaxy screen flags and nobody has judged yet. Judge
      each from its title and abstract.</p>
    <dl>
      <dt style="color:{YES['color']}">Galaxy paper</dt>
      <dd>About galaxies beyond the Local Group, including their resolved stars, star clusters and H II regions,
        and galaxy-scale theory.</dd>
      <dt style="color:{NO['color']}">Not a galaxy paper</dt>
      <dd>Stars, star formation, the ISM or astrochemistry in the Milky Way or the Local Group (including the
        Magellanic Clouds); planets, brown dwarfs, the solar system; laboratory astrochemistry; the stellar
        distance ladder; papers not about galaxies.</dd>
      <dt style="color:{UNSURE['color']}">Unsure</dt>
      <dd>Cannot tell from the abstract. Unsure papers stay in the sample.</dd>
    </dl>"""
    options = [{"v": "galaxy", "label": "Galaxy paper", "short": "galaxy", **YES},
               {"v": "not_galaxy", "label": "Not a galaxy paper", "short": "not galaxy", **NO},
               {"v": "unsure", "label": "Unsure", "short": "unsure", **UNSURE}]
    out = render(papers, name="galaxy screen", title="Is it a galaxy paper? · galaxy screen", help_html=help_html,
                 options=options, file_prefix=PREFIX, store_key=f"jwst-galaxy-screen-v2-{FREEZE}",
                 save_dir=f"data/{FREEZE}/", out_path=os.path.join(GEN_OUT, "galaxy_screen.html"))
    print(f"{len(papers)} flagged papers without a verdict -> {os.path.relpath(out, ROOT)}")


def merge():
    v = pd.read_csv(VERDICTS)
    titles = pd.read_csv(os.path.join(GEN_OUT, "sample.csv")).set_index("bibcode")["title"]
    new = []
    for path in sorted(glob.glob(data(f"{PREFIX}_*.json"))):
        saved = json.load(open(path))
        for d in saved["decisions"]:
            # the page uses bibcodes as ids; decisions on papers no longer in the sample are skipped
            if d.get("decision") and d["paper_id"] in titles.index:
                new.append({"bibcode": d["paper_id"], "verdict": d["decision"], "source": "screen",
                            "date": (d.get("decided_at") or saved["saved_at"])[:10], "title": titles[d["paper_id"]]})
    if not new:
        print("no new verdicts")
        return
    allv = pd.concat([v, pd.DataFrame(new)], ignore_index=True)
    allv = allv.sort_values("date", kind="stable").drop_duplicates("bibcode", keep="last")
    allv.sort_values(["source", "bibcode"]).to_csv(VERDICTS, index=False)
    print(f"{len(new)} verdicts read; galaxy_verdicts.csv now {len(allv)} papers "
          f"({(allv['verdict'] == 'not_galaxy').sum()} not galaxy) on {datetime.date.today()}")


if __name__ == "__main__":
    {"page": page, "merge": merge}[sys.argv[1] if len(sys.argv) > 1 else "page"]()
