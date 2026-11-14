"""The topic list, its matcher, and the per-year attention table and figures of
general/03_topics_sample_titles.ipynb. The topic list is also read by the citation model (04) and the checks:
rerun steps 4-13 after editing this file.
"""
import html as _html
import re
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common.figstyle import STYLE_RC
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# ── THE TOPIC LIST: one list for every table, figure and model feature ──
# Each topic is a set of phrases (regular expressions, case-insensitive, whole words) matched in the title and
# abstract of a paper, locally, on the frozen texts (topic_texts below). A paper counts in every topic it matches,
# except that Black holes & AGN leaves out the Little Red Dot papers (89% of which also mention black holes or
# AGN, so without this the topic would carry most of the Little Red Dot boom). The phrases are chosen by the
# author, from the title phrases that rose most in the sample (notebook 03, section 2).
Z10 = (r"z\s*\$?\s*(?:[>≳~∼≈=]|\\gtrsim|\\geq|\\ge|\\sim|\\approx|\\simeq)\s*\$?\s*(?:1[0-9]|[2-9][0-9])(?:\.\d+)?"
       r"|redshifts?\s+(?:above|beyond|of|greater than|>)\s*(?:1[0-9])")
# Element abundance ratios: C/O, N/O, O/H, [Fe/H], [Mg/Fe] ... (S/N, signal-to-noise, excluded)
_EL = r"(?:H|He|C|N|O|Ne|Mg|Si|S|Ar|Fe)"
RATIO = rf"(?!S\s*/\s*N\b)\[?{_EL}\s*/\s*{_EL}\]?"   # not S/N (signal-to-noise)
TOPICS = {
    "First galaxies":      [r"first galaxies", r"earliest galaxies", r"cosmic dawn", Z10],
    "Little Red Dots":     [r"little red dots?"],
    "Black holes & AGN":   [r"AGNs?", r"active galactic nucle(?:us|i)", r"black holes?", r"quasars?", r"QSOs?",
                            r"quasi-stellar objects?", r"blazars?", r"Seyferts?"],
    "Population III":      [r"Population III", r"Pop[ -]?III", r"first stars"],
    "Reionization":        [r"reioni[sz]ation"],
    "Quenching":           [r"quiescent", r"passive", r"quenched", r"quenching", r"post-starburst"],
    # dust as a subject (emission, attenuation, obscuration), not every paper that corrects for it
    "Dust":                [r"dust (?:emission|continuum|mass(?:es)?|temperatures?|attenuation|extinction)",
                            r"attenuation (?:law|curves?)", r"Balmer decrements?", r"UV(?:[- ]continuum)? slopes?",
                            r"dust[- ]obscured", r"dusty star-forming"],
    "Chemical abundances": [r"chemical abundances?", r"chemical enrichment", r"metallicit(?:y|ies)", r"oxygen", r"carbon",
                            r"nitrogen", r"iron", r"magnesium", r"(?:Fe|Mg) abundances?", RATIO],
    "Lensing":             [r"lensing", r"weak lensing", r"lensed", r"gravitational lens(?:es)?", r"strong lens(?:es)?"],
}
TOPIC_EXCLUDES = {"Black holes & AGN": "Little Red Dots"}
_TOPIC_RE = {t: re.compile(r"\b(?:" + "|".join(ps) + r")\b", re.I) for t, ps in TOPICS.items()}


def clean_text(t):
    """Title or abstract as plain text: HTML tags and entities removed."""
    return _html.unescape(re.sub(r"<[^>]+>", " ", t or ""))


def topics_of(text):
    """The topics a title + abstract matches, in TOPICS order."""
    hit = {t for t, rx in _TOPIC_RE.items() if rx.search(text)}
    hit -= {t for t, ex in TOPIC_EXCLUDES.items() if ex in hit}
    return [t for t in TOPICS if t in hit]


def topic_lists(docs, texts):
    """{topic: [docs]} for a list of ADS docs; texts: {bibcode: {"title", "abstract"}}."""
    out = {t: [] for t in TOPICS}
    for d in docs:
        x = texts.get(d["bibcode"]) or {}
        for t in topics_of(clean_text(x.get("title")) + " " + clean_text(x.get("abstract"))):
            out[t].append(d)
    return out



def plot_topic_members(topic_docs):
    """{topic: {bibcode: doc}} from {topic: [docs]} (each paper once per topic)."""
    return {t: {d["bibcode"]: d for d in topic_docs[t]} for t in TOPICS if t in topic_docs}

# ── dataviz reference palette (white surface) ──
# Text and surface colours, a 7-step sequential blue (for the heatmap) and the 8-hue categorical
# palette (one fixed hue per topic).
INK, SECONDARY, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8c8b85", "#ffffff", "#e4e3de"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # sequential blue
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
       "#8a5a2b", "#1fa3b8"]   # ten hues, one per topic


# ── Fixed texts of the papers (title, abstract, publication date) ──
from common.config import data as _data, require_texts
from common.ads import bigquery as _bigquery, load_json as _load_json, save_json as _save_json
TOPIC_TEXTS_JSON = _data("topic_texts.json")   # cache: {bibcode: {title, abstract, pubdate}} (fixed fields)


def topic_texts(bibcodes, batch=2000):
    """{bibcode: {"title", "abstract", "pubdate"}}: fixed fields, fetched once per bibcode (ADS bigquery), cached."""
    require_texts()
    cache = _load_json(TOPIC_TEXTS_JSON, None) or {}
    missing = sorted(set(bibcodes) - set(cache))
    for i in range(0, len(missing), batch):
        part = missing[i:i + batch]
        for d in _bigquery("*:*", part, fl="bibcode,title,abstract,pubdate"):
            cache[d["bibcode"]] = {"title": (d.get("title") or [""])[0], "abstract": d.get("abstract") or "",
                                   "pubdate": d.get("pubdate") or ""}
        for b in part:
            cache.setdefault(b, {"title": "", "abstract": "", "pubdate": ""})   # not returned: remember
    if missing:
        _save_json(TOPIC_TEXTS_JSON, cache)
        print(f"  topic texts: fetched {len(missing)} papers")
    return {b: cache[b] for b in bibcodes if b in cache}


def _wrapped_text(fig, text, x0, x1, y, fontsize, **kw):
    """Bottom-anchored text spanning x0..x1 (figure coords), words wrapped by measured width."""
    # Measure each candidate line with the actual renderer and break before the word that would
    # overflow the available width.
    r = fig.canvas.get_renderer()
    width = (x1 - x0) * fig.bbox.width
    probe = fig.text(0, 0, "", fontsize=fontsize)
    lines = []
    for par in text.split("\n"):
        cur = ""
        for word in par.split(" "):
            trial = (cur + " " + word).strip()
            probe.set_text(trial)
            if cur and probe.get_window_extent(r).width > width:
                lines.append(cur)
                cur = word
            else:
                cur = trial
        lines.append(cur)
    probe.remove()
    return fig.text(x0, y, "\n".join(lines), fontsize=fontsize, va="bottom", ha="left", **kw)


# Publication year of an ADS doc as an int (None if missing or malformed).
def _year(d):
    try:
        return int(d.get("year"))
    except (TypeError, ValueError):
        return None


def attention_table(topic_docs, corpus_by_year, corpus_cites_by_year, years):
    """Per plotted topic and publication year: % of that year's papers on the topic, and % of the
    citations to that year's papers that go to the topic. topic_docs: {searched topic: [docs with
    bibcode, year, citation_count]}. Returns (table sorted by change in paper share, {topic: bibcodes})."""
    rows, bibs = [], {}
    for name, docs in plot_topic_members(topic_docs).items():
        bibs[name] = set(docs)
        # Papers and citations of this topic per publication year; citations are those received so
        # far by that year's papers.
        papers = {y: 0 for y in years}
        cites = {y: 0 for y in years}
        for d in docs.values():
            y = _year(d)
            if y in papers:
                papers[y] += 1
                cites[y] += d.get("citation_count", 0) or 0
        row = {"topic": name, "papers": sum(papers.values()), "citations": sum(cites.values())}
        for y in years:
            row[f"papers {y} %"] = 100 * papers[y] / corpus_by_year[y] if corpus_by_year.get(y) else np.nan
            row[f"citations {y} %"] = (100 * cites[y] / corpus_cites_by_year[y]
                                       if corpus_cites_by_year.get(y) else np.nan)
        # Headline trend: paper share in the two most recent years minus the share in the first two
        # (percentage points); used to order the rows.
        early, recent = years[:2], years[-2:]
        share = lambda ys: 100 * sum(papers[y] for y in ys) / sum(corpus_by_year[y] for y in ys)
        row["change in paper share (pp)"] = share(recent) - share(early)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("change in paper share (pp)", ascending=False).reset_index(drop=True)
    return df, bibs


def _overlap_note(bibs):
    """Footnote text flagging the largest containment: LRDs inside Black holes & AGN, etc."""
    notes = []
    for a, b in combinations(bibs, 2):
        for small, big in ((a, b), (b, a)):
            f = len(bibs[small] & bibs[big]) / len(bibs[small]) if bibs[small] else 0
            if f >= 0.5:
                notes.append(f"{f:.0%} of {small} papers also count in {big}")
    return "; ".join(notes)


def attention_figures(df, bibs, years, today, sample_note, title, out_prefix):
    """Two figures from attention_table(): a heatmap and a line chart, each with a papers panel
    and a citations panel. Saved as <out_prefix>.png and <out_prefix>_lines.png (+ <out_prefix>.csv)."""
    plt.rcParams["font.family"] = STYLE_RC["font.family"]   # shared house font (common/figstyle.py)
    topics = list(df["topic"])
    # Matrices topic x year of paper shares (P) and citation shares (C), the rows in the table's
    # order.
    P = df[[f"papers {y} %" for y in years]].to_numpy()
    C = df[[f"citations {y} %" for y in years]].to_numpy()
    xlabels = [str(y) if y < today.year else f"{y}\n(to date)" for y in years]
    overlap = _overlap_note(bibs)
    # Topics named in the overlap note get an asterisk in the labels.
    flagged = {t for t in topics if overlap and t in overlap}
    ylabels = [f"{t}*" if t in flagged else t for t in topics]
    foot = (f"{sample_note}\nTopics overlap (except LRDs, removed from Black holes & AGN); a paper counts in "
            "every topic it mentions."
            + (f" * {overlap[0].upper() + overlap[1:]}." if overlap else "")
            + f"  Citations as of {today:%b %Y}; {years[-1]} papers are still accruing them.")
    panels = [("Papers", "% of that year's papers that mention the topic", P),
              ("Citations", "% of citations to that year's papers that go to the topic", C)]

    # ── heatmap ──
    # One colour scale for both heatmap panels, so a shade means the same share on the left and on
    # the right.
    cmap = LinearSegmentedColormap.from_list("blue_ramp", RAMP)
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=np.nanmax([P, C]))   # square root: small topics stay readable
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(topics) + 2.9), sharey=True, facecolor=SURFACE)
    for ax, (name, sub, M) in zip(axes, panels):
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        # Write each cell's value; white text on dark cells, dark text on light ones.
        for (r, c), v in np.ndenumerate(M):
            ax.text(c, r, f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", ha="center", va="center", fontsize=8.5,
                    color="white" if norm(v) > 0.55 else INK)
        ax.set_xticks(np.arange(0.5, M.shape[1] - 1), minor=True)   # 2px surface gaps, inner edges only
        ax.set_yticks(np.arange(0.5, M.shape[0] - 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0)
        ax.set_xticks(range(len(years)), xlabels, fontsize=8.5, color=SECONDARY)
        ax.xaxis.tick_top()
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(topics)), ylabels, fontsize=9.5, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    fig.text(0.01, 0.005, foot + "\nRows ordered by change in paper share, "
             f"{years[0]}-{str(years[1])[2:]} to {years[-2]}-{str(years[-1])[2:]}.",
             fontsize=7.8, color=MUTED, va="bottom", ha="left", linespacing=1.6)
    fig_h = fig.get_figheight()
    fig.subplots_adjust(left=0.17, right=0.99, top=1 - 1.45 / fig_h, bottom=0.95 / fig_h, wspace=0.04)
    # panel headings in inches above each heatmap, clear of the two-line year labels
    for ax, (name, sub, _) in zip(axes, panels):
        box = ax.get_position()
        fig.text(box.x0, box.y1 + 0.72 / fig_h, name, fontsize=11.5, fontweight="bold", color=INK, va="bottom")
        fig.text(box.x0, box.y1 + 0.50 / fig_h, sub, fontsize=8.5, color=SECONDARY, va="bottom")
    fig.savefig(f"{out_prefix}.png", dpi=400, facecolor=SURFACE)
    plt.show()

    # ── lines ──
    # color follows the topic, not its rank: fixed by TOPICS order, same in both notebooks
    styles = {t: CAT[i] for i, t in enumerate(TOPICS)}
    x = np.array(years)
    partial = years[-1] >= today.year          # the last year is still running
    ymax = np.nanmax([P, C]) * 1.08            # one y axis for both panels, so they compare directly
    fig, axes = plt.subplots(1, 2, figsize=(9, 5.6), facecolor=SURFACE, sharey=True)
    for ax, (name, sub, M) in zip(axes, panels):
        for r, t in enumerate(topics):
            done = slice(0, len(x) - 1 if partial else len(x))
            ax.plot(x[done], M[r, done], color=styles[t], lw=2, solid_capstyle="round", zorder=3)
            if partial:   # partial year: dashed segment, hollow end marker
                ax.plot(x[-2:], M[r, -2:], color=styles[t], lw=2, ls=(0, (3, 2)), zorder=3)
                ax.plot(x[-1], M[r, -1], "o", mfc=SURFACE, mec=styles[t], mew=1.8, ms=6.5, zorder=4)
            else:
                ax.plot(x[-1], M[r, -1], "o", color=styles[t], ms=6.5, mec=SURFACE, mew=1.5, zorder=4)
        ax.set_ylim(0, ymax)
        ax.set_xlim(x[0] - 0.15, x[-1] + 0.15)
        ax.tick_params(axis="y", labelleft=True)   # y labels on both panels, so both look the same size
        ax.set_xticks(x, xlabels, fontsize=8.5, color=SECONDARY)
        ax.tick_params(axis="y", labelsize=8.5, colors=SECONDARY, length=0)
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.text(0, 1.10, name, transform=ax.transAxes, fontsize=11.5, fontweight="bold", color=INK, va="bottom")
        ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=8.5, color=SECONDARY, va="bottom")
    handles = [plt.Line2D([], [], color=styles[t], lw=2) for t in topics]
    fig.subplots_adjust(left=0.075, right=0.98, top=0.83, bottom=0.30, wspace=0.2)
    # legend and footnote span both panels: from the left panel's left edge to the right panel's right edge
    x0, x1 = axes[0].get_position().x0, axes[1].get_position().x1
    fig.legend(handles, ylabels, loc="lower left", ncol=4, fontsize=9, frameon=False, mode="expand",
               bbox_to_anchor=(x0, 0.115, x1 - x0, 0.05), borderaxespad=0, handlelength=1.8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.985)
    note = foot + (f" Dashed segment and open circle: {years[-1]} so far." if partial else "")
    _wrapped_text(fig, note, x0, x1, 0.005, fontsize=7.6, color=MUTED, linespacing=1.55)
    fig.savefig(f"{out_prefix}_lines.png", dpi=400, facecolor=SURFACE)
    fig.savefig(f"{out_prefix}_lines.pdf", facecolor=SURFACE)   # vector copy of the line chart
    plt.show()
    df.round(2).to_csv(f"{out_prefix}.csv", index=False)
    print(f"Saved -> {out_prefix}.png, {out_prefix}_lines.png, {out_prefix}_lines.pdf, {out_prefix}.csv")

