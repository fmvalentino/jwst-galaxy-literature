#!/usr/bin/env bash
# Rebuild every number and figure of the paper from the frozen tables (general/outputs/sample.csv, rejected.csv,
# sample_table.csv, citation_features.csv and the caches of data/<freeze>/). No ADS token and no ADS text needed.
# Ends with checks/verify_outputs.py, which exits with an error unless every output holds what the committed one
# holds (same pixels, same numbers to 0.001); the bytes of a PNG can differ with the version of the image libraries.
# The full rebuild, from the ADS pulls on, is ./run_all.sh (AGENTS.md, step 16).
set -euo pipefail
cd "$(dirname "$0")"
python general/05_rank_stability.py
python general/06_sample_sensitivity.py
python general/07_baselines.py
python general/08_effects.py
python general/09_self_citations.py
python general/10_prediction_check.py
python general/11_sample_topics_by_half_year.py
python general/tools/score_validation.py
python docs/paper/make_paper_figures.py
python checks/verify_outputs.py
