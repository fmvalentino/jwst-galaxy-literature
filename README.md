# jwst-galaxy-literature

The analysis behind the paper on the JWST galaxy papers of 2023-2026 (link at publication), from one frozen
snapshot of NASA ADS taken on 2026-10-19: which galaxy papers use JWST data (2,227 papers, selected by one explicit
rule and validated by a blind review), how their topics moved, and what goes with a paper being highly cited.

## Reproduce

```
git clone https://github.com/fmvalentino/jwst-galaxy-literature.git && cd jwst-galaxy-literature
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
./run_paper.sh
```
`run_paper.sh` rebuilds every number and figure of the paper from the frozen tables in this repository, with no
ADS account, in a few minutes. It ends with `checks/verify_outputs.py`, which fails unless every figure has the same
pixels and every table the same numbers (to 0.001) as the committed ones; `git status` can still list changed
files, because the bytes of a PNG depend on the version of the image libraries. `AGENTS.md` section 6 gives the file
each number and figure comes from, and sections 4-5 every rule and step.

**The full pipeline**, from the ADS pulls on (the sample selection, the topic matching, the features), reads the
abstracts and full-text passages of the papers. These are copyright of their publishers and are not in this
repository: every field is kept, with the text emptied. With a free ADS token
(https://ui.adsabs.harvard.edu/user/settings/token):
```
export ADS_TOKEN=...
python general/tools/restore_texts.py     # fetches the texts again, then compares them with the freeze
./run_all.sh                              # every step in order, from the cached pulls
```
ADS serves its current version of each text, and no longer returns some passages, so `restore_texts.py` reports
that the files differ from the frozen ones. A few papers then enter or leave the sample, `run_all.sh` asks ADS for
the texts it has not cached, and the results move slightly (AGENTS.md, step 0). The citation counts stay those of
the freeze: nothing time-dependent is pulled again. The committed outputs are the reference. Do not commit the
restored texts.

## Layout

`common/` shared code (the sample rule, the topic list, ADS access) · `data/2026-10-19/` the ADS pulls of the
snapshot and the author's labels · `general/` notebooks 01, 03 and 04, scripts 05-11, review and scoring tools,
outputs and plots · `checks/` checks of key results · `docs/paper/` the paper's figures · `release/` the data release (also on
Zenodo, DOI at publication).

Code: MIT licence. Data release: CC BY 4.0. Uses NASA's Astrophysics Data System.
