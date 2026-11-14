# JWST galaxy literature, 2023–2026: data release

NASA ADS snapshot of 2026-10-19. Code: the repository `jwst-galaxy-literature` at the commit of this release
(`./run_all.sh` rebuilds everything from the ADS caches). The rules are stated in `AGENTS.md`, section 4, of that
repository.

| File | Content |
| --- | --- |
| `sample.csv` | The 2,227 papers of the sample: data-use evidence (`curated` = in ADS's JWST bibliography; `explicit text` = one passage of the full text scoring at least 7), galaxy-screen terms and result, citations on 2026-10-19, topic flags, catchy-title label |
| `candidates_rejected.csv` | The other 6,732 candidates, with the step of the rule they failed (`status`) |
| `citation_features.csv` | The 58 features of the citation model and the target (`pct`: citation percentile among papers posted the same month; `cohort`: the month group) |
| `predictions.csv` | Predicted percentile of every sample paper, models trained on the modelled papers (`modelled`); dated 2026-10-19, for a later check against citations |
| `hand_galaxy_verdicts.csv` | Every hand verdict on scope (galaxy / not_galaxy / unsure), with its source and date; read only for papers the galaxy screen flags |
| `hand_catchy_title_labels.csv` | Labels of the titles (catchy / borderline / not), assigned by a large language model following the author's rules; `reviewed_by_user` marks those the author checked; borderline counts as not catchy |
| `hand_validation.csv` | The blind validation: 400 random papers in three rounds, their stratum and verdict (in: `yes_analysis`, `yes_plot`; out: `no`, `not_galaxy`; `unsure`); round 1 judged twice |
| `topics.json` | The candidate query, the data-use threshold, the topic phrases and redshift patterns, and the topic exclusion (Black holes & AGN leaves out the Little Red Dot papers) |
| `checksums.txt` | SHA-256 of every file |

Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The code is under the MIT licence.

Not included: ADS full-text passages and abstracts (copyright of the publishers; available from ADS by bibcode).
All hand inputs are the work of one rater (the author); the rater's agreement with themself over two blind passes
of 135 papers is Cohen's kappa = 0.84. In an independent check, a large language model applying the written rules
agreed with 87.5% of 200 galaxy verdicts (kappa = 0.75) and, relabelling all 2,227 sample titles, with 94.8% of the
catchy / not labels. The disagreements are mostly edge cases; the check was stricter on titles (91
catchy instead of 186). With its labels the share of catchy titles rises from 2.7% (2023) to 4.8% (2026) instead of
3.7% to 10.5%, and catchy titles still have no significant effect on the citation rank.
