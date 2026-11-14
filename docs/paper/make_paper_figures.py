"""Figures of the paper, in the journal's layout: no titles inside the figures (the
legends carry them), bold lowercase panel letters, double-column width (183 mm).

Reads general/outputs and general/plots only; writes PNGs to docs/paper/figures/. Run from the repository
root after run_all.sh:
    python docs/paper/make_paper_figures.py
"""
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, GEN_PLOTS  # noqa: E402
from common.figstyle import CAT, GRID, INK, MUTED, NEUTRAL, SECONDARY, apply_style, style_axes  # noqa: E402

OUT = os.path.join(ROOT, "docs", "paper", "figures")
os.makedirs(OUT, exist_ok=True)
apply_style()
plt.rcParams.update({"savefig.dpi": 300})
W2 = 183 / 25.4          # double column, inches
BLUE, ORANGE, GREEN, RED = CAT[0], CAT[1], CAT[2], CAT[7]


def panel(ax, letter, x=-0.12, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK, va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("->", os.path.join("docs", "paper", "figures", name))


# ── Figure 1: topics of the 2,227 sample papers by half-year (general/11_sample_topics_by_half_year.py); a, shares;
# b, number of papers, so that a falling share of a growing topic reads as such. Each panel has its own time axis:
# a runs to the partial last half-year (dashed, open circle), b stops at the last complete one. The other topics are
# thin light context lines; the three topics of the text are highlighted, with a dot per half-year. Colours: Little
# Red Dots crimson, first galaxies gold (first light), reionization deep blue; checked distinct under red-green
# colour blindness and against the grey context lines (OKLab difference >= 16). ──
att = pd.read_csv(os.path.join(GEN_OUT, "sample_topics_by_half_year.csv"))
halves = list(dict.fromkeys(att["half_year"]))
x = np.arange(len(halves))
HL = {"Little Red Dots": "#D62839", "First galaxies": "#E8A100", "Reionization": "#1F5FAE"}
CONTEXT, CONTEXT_TEXT = "#c9c8c3", "#8c8b85"
DODGE = {"First galaxies": -0.22, "Reionization": -0.1}         # half-years, in the partial half-year of a only
fig, (axa, axb) = plt.subplots(2, 1, figsize=(W2, 6.4), gridspec_kw={"hspace": 0.16})
for ax, col_, ylabel, letter in ((axa, "share_pct", "Share of the sample's papers", "a"),
                                 (axb, "count", "Number of papers", "b")):
    last = len(x) - 1 if col_ == "share_pct" else len(x) - 2
    hi = att[col_].max() * 1.06
    ends = []
    for topic, g in att.groupby("topic", sort=False):
        hl = topic in HL
        col, lw, z = (HL[topic], 2.0, 3) if hl else (CONTEXT, 0.9, 2)
        y = g[col_].to_numpy()
        ax.plot(x[:-1], y[:-1], color=col, lw=lw, zorder=z, marker="o" if hl else None, ms=2.8, mec="white", mew=0.5)
        if col_ == "share_pct":
            # first galaxies and reionization end at the same share (14.0%): their last points are set apart
            dx = DODGE.get(topic, 0)
            ax.plot([x[-2], x[-1] + dx], y[-2:], color=col, lw=lw, ls=(0, (3, 2)), zorder=z)
            ax.plot(x[-1] + dx, y[-1], "o", mfc="white", mec=col, mew=1.1, ms=3.6, zorder=z + 2)
        ends.append((y[last], topic, hl))
    # end labels at their values, pushed down from the top to stay at least `gap` apart
    gap, prev = hi * (0.052 if col_ == "share_pct" else 0.058), 1e9
    for v, name, hl in sorted(ends, reverse=True):
        ypos = min(v, prev - gap)
        prev = ypos
        lab = f"{name} {v:.0f}%" if col_ == "share_pct" else f"{name} {v:.0f}"
        ax.text(x[last] + (0.15 if col_ == "share_pct" else 0.3), ypos, lab, va="center", fontsize=8 if hl else 7,
                color=HL[name] if hl else CONTEXT_TEXT, fontweight="bold" if hl else "normal")
    ax.set_ylim(0, hi)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    ax.grid(axis="y", color="#eeede9", lw=0.6)
    panel(ax, letter, x=-0.07, y=1.0)
    ax.set_xticks(x[:last + 1], [h.replace(" ", "\n") for h in halves[:last + 1]])
    ax.set_xlim(-0.3, last + 2.0)
axa.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
save(fig, "fig1_topics.png")

# ── Figure 2: what goes with a high citation rank (effects that survive the false-discovery rate) ──
eff = pd.read_csv(os.path.join(GEN_OUT, "effects_table.csv"))
eff = eff[eff["survives FDR 5%"]]
NAMES = {"topic_little_red_dots": "Little Red Dot topic", "journal_nature_science": "Nature or Science",
         "abs_nirspec": "NIRSpec in abstract", "abs_very_high_z": "Very high redshift in abstract",
         "abs_prism": "Prism in abstract", "fa_europe": "First author in Europe",
         "topic_chemical_abundances": "Chemical-abundance topic", "fa_n_america": "First author in N. America",
         "abs_alma": "ALMA in abstract", "abs_hst": "HST in abstract", "abs_ifu": "IFU in abstract",
         "fa_log_prior_papers": "First author's earlier papers (log)", "log_references": "References (log)",
         "log_n_authors": "Authors (log)", "n_programmes": "JWST programmes named", "n_countries": "Countries in team",
         "log_pages": "Pages (log)", "fa_career_years": "First author's career length",
         "abs_log_words": "Abstract length (log)", "abs_claims": "Claims of firsts/surprises",
         "title_words": "Title length"}
yes = eff[eff["yes/no"]].sort_values("readable effect (places)")
# panel b in natural units: doublings for the features stored as log10, a stated step for counts and years
# (08_effects.py, UNITS)
NAMES_U = {"fa_log_prior_papers": "First author's earlier papers", "log_references": "References",
           "log_n_authors": "Authors", "log_pages": "Pages", "abs_log_words": "Abstract length",
           "n_programmes": "JWST programmes named", "n_countries": "Countries in team",
           "fa_career_years": "First author's career length", "abs_claims": "Claims of firsts/surprises",
           "title_words": "Title length"}
cont_u = eff[~eff["yes/no"]].sort_values("effect per unit")
fig, axes = plt.subplots(1, 2, figsize=(W2, 3.9), gridspec_kw={"width_ratios": [1, 1], "wspace": 1.15, "right": 0.95})
for ax, df, col, lo, hi, xlabel, letter, names in (
        (axes[0], yes, "readable effect (places)", "readable low", "readable high",
         "Papers with it vs. without (places out of 100)", "a", lambda r: NAMES.get(r.feature, r.feature)),
        (axes[1], cont_u, "effect per unit", "unit low", "unit high", "Places out of 100, per stated change", "b",
         lambda r: f"{NAMES_U.get(r.feature, r.feature)}, {r.unit}")):
    yy = np.arange(len(df))
    c = [BLUE if v > 0 else ORANGE for v in df[col]]
    ax.hlines(yy, df[lo], df[hi], color=c, lw=2, alpha=0.45)
    ax.scatter(df[col], yy, color=c, s=22, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(yy, [names(r) for r in df.itertuples()], fontsize=8)
    if letter == "b":
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(xlabel)
    style_axes(ax, grid="x")
    panel(ax, letter, x=-0.02, y=1.02)
save(fig, "fig2_effects_units.png")

# ── Figure 3: how well a paper's features predict its rank (general/10_prediction_check.py) ──
pc = pd.read_csv(os.path.join(GEN_OUT, "prediction_check.csv"))
fq = pd.read_csv(os.path.join(GEN_OUT, "prediction_forward.csv"))
tst = pc[pc["split"] == "test"]
CV_GREY = SECONDARY
fig, axes = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw={"width_ratios": [1, 1, 1.35], "wspace": 0.55, "top": 0.86, "bottom": 0.2})
axes[0].set_box_aspect(1)
axes[1].set_box_aspect(1)
axes[2].set_box_aspect(1 / 1.35)          # same height as a and b


def binned(pred, true, n=10):
    """Mean predicted and mean actual percentile in n equal-count bins of the prediction, and the 95% interval."""
    idx = np.array_split(np.argsort(pred), n)
    return np.array([(pred[i].mean(), true[i].mean(), 1.96 * true[i].std(ddof=1) / np.sqrt(len(i))) for i in idx])


def gains(pred, is_top):
    order = np.argsort(-pred)
    found = np.cumsum(is_top[order]) / is_top.sum()
    return np.r_[0, np.arange(1, len(pred) + 1) / len(pred)] * 100, np.r_[0, found] * 100


ax = axes[0]
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.axhline(50, color=GRID, lw=0.8)
for pred, true, col, ms, lab in ((pc["predicted_cv"].to_numpy(), pc["pct"].to_numpy(), CV_GREY, 3,
                                  "All papers, month held out"),
                                 (tst["predicted_test"].to_numpy(), tst["pct"].to_numpy(), BLUE, 3.5,
                                  "Later papers (test)")):
    b = binned(pred, true)
    ax.errorbar(b[:, 0], b[:, 1], yerr=b[:, 2], fmt="o-", color=col, ms=ms, lw=1.3, capsize=0, mec="white", mew=0.6,
                label=lab)
ax.text(91, 93, "Perfect", ha="center", va="bottom", fontsize=7, color=MUTED, rotation=45, rotation_mode="anchor")
ax.text(97, 48, "No skill", ha="right", va="top", fontsize=7, color=MUTED)
h, l = ax.get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper left", bbox_to_anchor=(0.08, 1.0), ncol=2, fontsize=7, frameon=False,
           handlelength=1.4)
ax.set_xlim(0, 100), ax.set_ylim(0, 100)
ax.set_xlabel("Predicted rank (percentile)")
ax.set_ylabel("Actual rank (percentile)")
style_axes(ax)
panel(ax, "a", x=-0.3)

ax = axes[1]
ax.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls=(0, (3, 2)))
for pred, is_top, col in ((pc["predicted_cv"].to_numpy(), pc["top10"].to_numpy(), CV_GREY),
                          (tst["predicted_test"].to_numpy(), tst["top10"].to_numpy(), BLUE)):
    gx, gy = gains(pred, is_top)
    ax.plot(gx, gy, color=col, lw=1.5)
gx, gy = gains(tst["predicted_test"].to_numpy(), tst["top10"].to_numpy())
g20 = np.interp(20, gx, gy)
ax.plot([20, 20, 0], [0, g20, g20], color=MUTED, lw=0.7, ls=":")
ax.text(24, g20 - 7, f"{g20:.0f}% found in\nthe top 20%", fontsize=7, color=INK, va="top")
ax.text(82, 79, "Chance", ha="center", va="top", fontsize=7, color=MUTED, rotation=45, rotation_mode="anchor")
ax.set_xlim(0, 100), ax.set_ylim(0, 100)
ax.set_xlabel("Papers ranked highest by the model (%)")
ax.set_ylabel("Actual top 10% found (%)")
style_axes(ax)
panel(ax, "b", x=-0.3)

ax = axes[2]
xq = np.arange(len(fq))
ax.vlines(xq, fq["low"], fq["high"], color=BLUE, lw=2.2, alpha=0.4)
ax.plot(xq, fq["rho"], "o", color=BLUE, ms=4, mec="white", mew=0.6, zorder=3)
ax.axhline(0, color=MUTED, lw=0.8)
from scipy.stats import spearmanr  # noqa: E402
rho_test = spearmanr(tst["predicted_test"], tst["pct"]).statistic
ax.axhline(rho_test, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(-0.4, 0.06, f"Dashed: main test, ρ = {rho_test:.2f}", ha="left", va="bottom", fontsize=6.5,
        color=SECONDARY)
ax.set_xticks(xq, [q[-2:] + ("\n" + q[:4] if q.endswith("Q1") else "") for q in fq["quarter"]], fontsize=7)
ax.set_ylim(-0.05, 0.85)
ax.set_ylabel("Rank correlation, ρ")
ax.set_xlabel("Quarter predicted")
style_axes(ax, grid="y")
panel(ax, "c", x=-0.2)
save(fig, "fig3_prediction.png")

# ── Figure 4: how predictable, and how lasting ──
base = pd.read_csv(os.path.join(GEN_OUT, "baselines.csv"))
b = base[base["model"].isin(["ridge", "-"])].copy()
ORDER = ["journal only", "first author's record only", "team size only", "topics only",
         "team size + first author + topics", "all features"]
LBL = {"journal only": "Journal", "first author's record only": "First author's record", "team size only": "Team size",
       "topics only": "Topics", "team size + first author + topics": "Team + first author + topics",
       "all features": "All 58 features"}
b = b.set_index("features").loc[ORDER]
stab = pd.read_csv(os.path.join(GEN_PLOTS, "rank_stability_summary.csv")).set_index("comparison")
fig, axes = plt.subplots(1, 2, figsize=(W2, 3.2), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.75})
ax = axes[0]
yy = np.arange(len(b))
ax.barh(yy, b["test rho"], color=[NEUTRAL] * (len(b) - 1) + [BLUE], height=0.6)
for y_, (r, t) in enumerate(zip(b["test rho"], b["times chance"])):
    ax.text(max(r, 0) + 0.015, y_, f"ρ = {r:.2f}  ·  top 10%: {t:.1f}× chance", va="center", fontsize=7.5, color=INK)
ax.set_yticks(yy, [LBL[i] for i in b.index])
ax.set_xlim(-0.05, 1.0)
ax.set_xlabel("Rank correlation with the citation percentile (test papers)")
style_axes(ax, grid="x")
panel(ax, "a", x=-0.45)
ax = axes[1]
rows = [("Citation rank\nat 6 months", stab.loc["rank at 6 months vs. today", "spearman_rho"], NEUTRAL),
        ("Citation rank\nat 12 months", stab.loc["rank at 12 months vs. today", "spearman_rho"], NEUTRAL),
        ("Features at posting\n(model)", stab.loc["model trained on 6-month rank, scored on today's rank", "spearman_rho"], BLUE)]
yy = np.arange(len(rows))
ax.barh(yy, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
for y_, r in zip(yy, rows):
    ax.text(r[1] + 0.015, y_, f"{r[1]:.2f}", va="center", fontsize=8, color=INK)
ax.set_yticks(yy, [r[0] for r in rows])
ax.set_xlim(0, 1.05)
ax.set_xlabel("Rank correlation with the\nrank at the snapshot date")
ax.invert_yaxis()
style_axes(ax, grid="x")
panel(ax, "b", x=-0.62)
save(fig, "fig4_predictability.png")

# ── Extended Data figures of the paper (journal layout: no titles inside, sentence-case labels) ──
import json  # noqa: E402
from common.config import data  # noqa: E402

# Extended Data Fig. 1: the sample rule, step by step
records = len(json.load(open(data("raw_candidate_search.json")))["docs"])
sample = pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))
rej = pd.read_csv(os.path.join(GEN_OUT, "rejected.csv"))
papers = len(sample) + len(rej)
st = rej["status"].str.replace(r"\s*[:(].*", "", regex=True).str.strip()
n_nouse = int((st == "no JWST data-use evidence").sum())
n_type = int(st.str.startswith("not a journal article").sum())
n_gal = int(st.str.startswith("not a galaxy paper").sum())
stages = [("Candidate records (ADS query)", records, ""),
          ("Papers, with preprints merged with their published versions", papers, ""),
          ("Uses JWST data: in the curated JWST bibliography, or explicit data use in the text", papers - n_nouse,
           f"−{n_nouse:,}: no evidence of JWST data use"),
          ("Journal articles and preprints", papers - n_nouse - n_type,
           f"−{n_type:,}: conference abstracts, proceedings, proposals, datasets, and others"),
          ("Galaxy papers: the sample", papers - n_nouse - n_type - n_gal,
           f"−{n_gal:,}: not galaxy papers (stellar or planetary arXiv class, or labelled after the galaxy screen)")]
assert stages[-1][1] == len(sample)
fig, ax = plt.subplots(figsize=(W2, 3.4))
for i, (name, n, cut) in enumerate(stages):
    ax.barh(i, n, color=BLUE if i == len(stages) - 1 else NEUTRAL, height=0.55)
    ax.text(n + 120, i - 0.08, f"{n:,}", fontweight="bold", color=INK, va="center", fontsize=8.5)
    ax.text(n + 120, i + 0.22, name, color=SECONDARY, va="center", fontsize=8)
    if cut:
        ax.text(n + 120, i - 0.42, cut, color=SECONDARY, va="center", fontsize=7.5)
ax.set_ylim(len(stages) - 0.4, -0.7)
ax.set_xlim(0, records * 1.9)
ax.axis("off")
save(fig, "ed1_sample_funnel.png")

# Extended Data Fig. 2: robustness of the prediction (boosted trees)
rob = pd.read_csv(os.path.join(GEN_PLOTS, "robustness_summary.csv"))
PICK = [("1 Tree settings", "all 225 settings (median, 5-95% range)", "Model", "225 tree settings (median, 5–95%)"),
        ("1 Tree settings", "min 60 papers per leaf", "Model", "At least 60 papers per leaf"),
        ("2 Number of trees", "settings chosen without the test period", "Model", "Settings chosen without the test period"),
        ("4 Tree family", "random forest, min 5 papers per leaf", "Model", "Random forest, 5 papers per leaf"),
        ("4 Tree family", "random forest, min 50 papers per leaf", "Model", "Random forest, 50 papers per leaf"),
        ("3 Chance", "training papers resampled (mean ± 2 s.d.)", "Chance", "Training papers resampled (±2 s.d.)"),
        ("5 Null test", "citation ranks shuffled (mean ± 2 s.d.)", "Null test", "Citation ranks shuffled (±2 s.d.)"),
        ("6 Posting time", "without journal, refereed, pages, references", "Known at posting",
         "No journal, refereed status, pages, or references"),
        ("7 Data quality", "refereed papers only", "Data", "Refereed papers only"),
        ("7 Data quality", "test-period preprints only", "Data", "Test-period preprints only"),
        ("7 Data quality", "first author's name not shared", "Data", "First author's name not shared"),
        ("7 Data quality", "without first-author history", "Data", "No first-author history"),
        ("8 Target and design", "quarterly cohorts", "Target", "Quarterly cohorts"),
        ("8 Target and design", "minimum age 6 months", "Target", "Papers at least 6 months old"),
        ("8 Target and design", "minimum age 12 months", "Target", "Papers at least 12 months old"),
        ("8 Target and design", "test from 2024-10", "Target", "Test period from October 2024"),
        ("8 Target and design", "target: log citations vs. cohort mean", "Target", "Log citations relative to the cohort mean"),
        ("10 Self-citations", "without self-citations", "Self-citations", "Self-citations removed")]
ref = rob[(rob["test"] == "3 Chance") & rob["variant"].str.startswith("test papers")].iloc[0]
fig, ax = plt.subplots(figsize=(W2, 5.0))
ax.axvspan(ref["low"], ref["high"], color=GRID, alpha=0.6, zorder=0)
ax.axvline(ref["rho"], color=MUTED, ls=(0, (3, 2)), lw=0.9)
ax.text(ref["low"] - 0.01, -1.0, f"Main test, ρ = {ref['rho']:.2f}", ha="right", fontsize=7.5, color=SECONDARY)
for i, (test, variant, grp, label) in enumerate(PICK):
    r = rob[(rob["test"] == test) & (rob["variant"] == variant)].iloc[0]
    col = ORANGE if grp == "Null test" else BLUE
    if pd.notna(r["low"]):
        ax.plot([r["low"], r["high"]], [i, i], color=col, lw=1.8, alpha=0.6)
    ax.scatter([r["rho"]], [i], s=22, color=col, zorder=3)
    ax.text(-0.03, i, label, ha="right", va="center", fontsize=7.5, color=INK, transform=ax.get_yaxis_transform())
    ax.text(-1.05, i, grp, ha="left", va="center", fontsize=7.5, color=SECONDARY, transform=ax.get_yaxis_transform())
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_xlim(-0.25, 0.75)
ax.set_yticks([])
ax.set_ylim(len(PICK) - 0.4, -1.5)
style_axes(ax, grid="x")
ax.set_xlabel(f"Rank correlation, ρ ({int(ref['papers']):,} test papers unless stated)")
fig.subplots_adjust(left=0.5, right=0.97)
save(fig, "ed2_robustness.png")

# Extended Data Fig. 3: catchy titles by year (share, and median citations of catchy and other titles)
tab = pd.read_csv(os.path.join(GEN_OUT, "sample_table.csv"))
tab = tab[tab["catchy_label"].notna()]
tab["catchy"] = tab["catchy_label"].eq("catchy")       # borderline counts as not catchy
rows = [(int(yr), len(g), int(g["catchy"].sum()), g.loc[g["catchy"], "citations"].median(),
         g.loc[~g["catchy"], "citations"].median()) for yr, g in tab[tab["year"].between(2023, 2026)].groupby("year")]
fig, ax = plt.subplots(figsize=(W2, 2.6))
for i, (yr, n, nc, mc, mo) in enumerate(rows):
    ax.plot([mo, mc], [i, i], color=NEUTRAL, lw=1.8, zorder=1)
    ax.scatter([mo], [i], s=34, color=NEUTRAL, zorder=2)
    ax.scatter([mc], [i], s=34, color=BLUE, zorder=3)
    lo, hi = (mo, mc) if mo <= mc else (mc, mo)
    ax.text(lo / 1.12, i, f"{(mo if lo == mo else mc):g}", ha="right", va="center", fontsize=7.5,
            color=SECONDARY if lo == mo else BLUE)
    ax.text(hi * 1.12, i, f"{(mc if hi == mc else mo):g}", ha="left", va="center", fontsize=7.5,
            color=BLUE if hi == mc else SECONDARY)
    ax.text(-0.03, i, f"{yr}: {nc} of {n:,} titles catchy ({100 * nc / n:.1f}%)", ha="right", va="center",
            transform=ax.get_yaxis_transform(), color=INK, fontsize=8)
ax.set_xscale("log")
ax.set_xlim(1, 100)
ax.set_xticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
ax.minorticks_off()
ax.set_yticks([])
ax.set_ylim(len(rows) - 0.4, -0.9)
style_axes(ax, grid="x")
ax.set_xlabel("Median citations (log scale)")
ax.scatter([], [], color=BLUE, s=34, label="Catchy titles")
ax.scatter([], [], color=NEUTRAL, s=34, label="Other titles")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.04), ncol=2, frameon=False, fontsize=7.5)
fig.subplots_adjust(left=0.36, right=0.97, bottom=0.2)
save(fig, "ed3_catchy_titles.png")
