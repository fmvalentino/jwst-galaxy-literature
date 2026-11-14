"""Shared builder of the local hand-review pages (validation of the data-use rule, galaxy screening).

One page per review: a card per paper (title, ADS facilities and data links, abstract, highlighted passages),
decision buttons with keyboard shortcuts, browser autosave, and a "Save decisions" button that downloads
<file_prefix>_<date>.json for the reviewer to put in `save_dir`. The page never shows any verdict of the
pipeline. Template: review_page_template.html (same folder).
"""
import json
import os

from common.config import FREEZE

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_page_template.html")

# Decision colours: green = in, red = out, amber = unsure, purple = out of scope
YES = {"color": "#1a8a5a", "bg": "#e6f4ec"}
NO = {"color": "#c23b33", "bg": "#fbe9e7"}
UNSURE = {"color": "#a86b00", "bg": "#fdf3dc"}
SCOPE = {"color": "#6b4bb5", "bg": "#efe9fa"}


def render(papers, *, name, title, help_html, options, file_prefix, store_key, save_dir, out_path, preload=None):
    """papers: list of dicts with id, bibcode, title, year, authors, abstract, ads, arxiv, facilities,
    data_links, snippets ([{field, text}]). options: list of {v, label, short, color, bg, button}; button False =
    a verdict shown and filterable but not offered as a button (e.g. an older verdict to be refined).
    preload: {paper_id: {decision, note, at}} built into the page; decisions made in the browser override it."""
    config = {"name": name, "options": options, "file_prefix": file_prefix, "store_key": store_key,
              "save_dir": save_dir, "preload": preload or {}}
    page = open(TEMPLATE, encoding="utf-8").read()
    for k, v in {"__PAPERS__": json.dumps(papers, ensure_ascii=False).replace("</", "<\\/"),
                 "__CONFIG__": json.dumps(config, ensure_ascii=False).replace("</", "<\\/"),
                 "__PAGE_TITLE__": title, "__HELP_HTML__": help_html, "__SAVE_DIR__": save_dir,
                 "__FREEZE__": FREEZE}.items():
        page = page.replace(k, v)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(page)
    return out_path
