"""Check of the local topic matcher (common/trending_shared.py) against ADS's own phrase search.

The topic pull of the freeze (data/<freeze>/topics_sample_pull.json) holds, per topic, the sample papers ADS
returned for the topic's phrases with =abs: (title + abstract + keywords). The local matcher reads title
and abstract only. For every topic searched in ADS, on the same papers, this reports the papers found by both, by
ADS only (mostly a phrase in the keywords only) and by the matcher only, and the agreement (Jaccard index).
Phrases added to a list after that search (Population III, Chemical abundances, Black holes & AGN) show up as
papers found by the matcher only; First galaxies was not searched in ADS.
Writes checks/topic_matcher.csv and checks/topic_matcher.png (a, the Jaccard index on the topic with the lowest
agreement; b, papers found by both, by ADS only and by the matcher only, per topic). Run from the repository root:
    python checks/topic_matcher.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from common.config import GEN_OUT, data  # noqa: E402
from common.figstyle import CAT, INK, apply_style  # noqa: E402
from common.trending_shared import clean_text, topic_texts, topics_of  # noqa: E402

# ADS searches of the freeze -> topic of the list they correspond to
ADS_TO_LIST = {"Little Red Dots": "Little Red Dots", "AGN": "Black holes & AGN", "Black holes": "Black holes & AGN",
               "Quenching": "Quenching", "Dust": "Dust", "Reionization": "Reionization",
               "Chemical abundances": "Chemical abundances", "Lensing": "Lensing", "Population III": "Population III"}
rows = []
for scope, fname in (("sample", "topics_sample_pull.json"),):
    pull = json.load(open(data(fname)))
    sample = set(pd.read_csv(os.path.join(GEN_OUT, "sample.csv"))["bibcode"])
    bibs = [d["bibcode"] for d in pull["corpus_docs"] if d["bibcode"] in sample]
    texts = topic_texts(bibs)
    local = {b: set(topics_of(clean_text(texts.get(b, {}).get("title")) + " "
                              + clean_text(texts.get(b, {}).get("abstract")))) for b in bibs}
    lrd_ads = {d["bibcode"] for subs in [pull["topics"]["Little Red Dots"]] for docs in subs.values() for d in docs}
    for topic in sorted(set(ADS_TO_LIST.values()), key=list(ADS_TO_LIST.values()).index):
        ads = {d["bibcode"] for t, subs in pull["topics"].items() if ADS_TO_LIST.get(t) == topic
               for docs in subs.values() for d in docs} & set(bibs)
        if topic == "Black holes & AGN":
            ads -= lrd_ads
        mine = {b for b in bibs if topic in local[b]}
        both = len(ads & mine)
        rows.append({"scope": scope, "topic": topic, "ADS": len(ads), "local": len(mine), "both": both,
                     "ADS only": len(ads - mine), "local only": len(mine - ads),
                     "agreement (Jaccard)": both / len(ads | mine) if ads | mine else 1.0})
    no_text = sum(not (texts.get(b, {}).get("abstract")) for b in bibs)
    print(f"{scope}: {len(bibs)} papers, {no_text} without an abstract in ADS")
df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(df.round(3).to_string(index=False))
df.round(4).to_csv(os.path.join(ROOT, "checks", "topic_matcher.csv"), index=False)
print("-> checks/topic_matcher.csv")

# ── figure: what the Jaccard index measures (a), and the agreement per topic (b) ──
apply_style()
d = df.sort_values("agreement (Jaccard)").reset_index(drop=True)
fig = plt.figure(figsize=(11, 4.6))
ax = fig.add_axes([0.02, 0.08, 0.36, 0.84])
ax.set_aspect("equal")
ax.axis("off")
r = d.iloc[0]
ax.add_patch(Circle((0.38, 0.5), 0.3, color=CAT[0], alpha=0.35))
ax.add_patch(Circle((0.62, 0.5), 0.3, color=CAT[1], alpha=0.35))
ax.text(0.22, 0.5, f"ADS only\n{int(r['ADS only'])}", ha="center", va="center", fontsize=10)
ax.text(0.5, 0.5, f"Both\n{int(r['both'])}", ha="center", va="center", fontsize=11, fontweight="bold")
ax.text(0.78, 0.5, f"Matcher only\n{int(r['local only'])}", ha="center", va="center", fontsize=10)
ax.text(0.38, 0.86, "ADS phrase search", ha="center", color=CAT[0], fontsize=9.5)
ax.text(0.62, 0.12, "Local matcher", ha="center", color=CAT[1], fontsize=9.5)
ax.text(0.5, 0.0, f"Jaccard = both / either = {int(r['both'])} / ({int(r['both'])} + {int(r['ADS only'])} + "
        f"{int(r['local only'])}) = {r['agreement (Jaccard)']:.2f}", ha="center", fontsize=9.5, color=INK)
ax.text(0.0, 1.0, f"a  {r['topic']} (lowest agreement)", fontsize=10.5, fontweight="bold", va="top")
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1.02)
ax = fig.add_axes([0.52, 0.14, 0.45, 0.74])
y = list(range(len(d)))
ax.barh(y, d["both"], color="#7f7f7f", label="Both")
ax.barh(y, d["ADS only"], left=d["both"], color=CAT[0], label="ADS only")
ax.barh(y, d["local only"], left=d["both"] + d["ADS only"], color=CAT[1], label="Matcher only")
for i, r in d.iterrows():
    ax.text(r["both"] + r["ADS only"] + r["local only"] + 8, i, f"J = {r['agreement (Jaccard)']:.2f}", va="center",
            fontsize=8.5)
ax.set_yticks(y, d["topic"])
ax.set_xlabel("Sample papers on the topic")
ax.set_xlim(0, (d["both"] + d["ADS only"] + d["local only"]).max() * 1.2)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
ax.set_title(f"b  The {len(d)} topics searched in ADS ({len(bibs):,} sample papers)", loc="left", fontsize=10.5,
             fontweight="bold")
fig.savefig(os.path.join(ROOT, "checks", "topic_matcher.png"), dpi=150)
print("-> checks/topic_matcher.png")
