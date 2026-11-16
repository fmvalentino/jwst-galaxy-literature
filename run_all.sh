#!/usr/bin/env bash
# Rebuild everything, in order. Needs ADS_TOKEN in the environment (a rerun with the caches of
# data/<freeze>/ in place makes almost no ADS query). Stops at the first error.
# Needs the ADS texts: run `python general/tools/restore_texts.py` once first (AGENTS.md, step 0).
# Without a token, ./run_paper.sh rebuilds the paper's numbers and figures from the frozen tables.
set -euo pipefail
: "${ADS_TOKEN:?Set ADS_TOKEN (https://ui.adsabs.harvard.edu/user/settings/token)}"
cd "$(dirname "$0")"
run() { (cd "$(dirname "$1")" && jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.startup_timeout=300 "$(basename "$1")" >/dev/null) && echo "ok  $1"; }

# the analysis
run general/01_sample.ipynb
run general/03_topics_sample_titles.ipynb
run general/04_citation_features.ipynb
python general/05_rank_stability.py
python general/06_sample_sensitivity.py
python general/07_baselines.py
python general/08_effects.py
python general/09_self_citations.py   # ORCIDs cached in data/<freeze>/citation_orcids.json
python general/10_prediction_check.py
python general/11_sample_topics_by_half_year.py
python general/tools/score_validation.py   # purity / completeness from the blind review (measures only)

# checks
python checks/topic_search.py          # ADS phrase search with the matcher's lists (cached after the first run)
python checks/topic_matcher.py         # local topic matcher vs. ADS phrase search
python checks/first_galaxies_threshold.py   # First galaxies trend for z >= 8, 9, 10
python checks/topic_sensitivity.py     # topic trends with narrow, current and broad phrase lists
python checks/catchy_llm_labels.py     # catchy-title results with the LLM's labels
python checks/manifest.py              # checksums of the frozen data

# the paper's figures and the data release
python docs/paper/make_paper_figures.py   # figures of the paper
python release/make_release.py         # the data release (release/<freeze>/)
