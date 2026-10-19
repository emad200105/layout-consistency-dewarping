# Paper

**Layout-Consistency Loss: Free Geometric Supervision for Document Image Dewarping**,
a draft in IEEE conference format (8 pages). The author block is left empty for
now.

| file | |
|---|---|
| `main.pdf` | the compiled draft |
| `main.tex` | source (IEEEtran) |
| `refs.bib` | bibliography; every entry was verified against a primary source (DOI record, CVF, arXiv, publisher) |
| `refs_notes.md` | for each reference: where it was verified, what it does relative to this work, and the published DocUNet numbers with their protocols and inconsistencies |
| `figs/` | figures (PDF in the paper, PNG for viewing) |
| `scripts/` | one script per figure, plus `subset_paired.py` (Table I, paired rows) and `export_subset_boxes.py` |

Build:

```bash
cd paper && latexmk -pdf main.tex     # TeX Live + IEEEtran (Ubuntu: texlive-publishers)
```

Regenerate the figures: see [../docs/REPRODUCE.md](../docs/REPRODUCE.md) §10.
`scripts/paths.py` says where each script looks for its inputs.

Every number in the paper is traceable to a file in `../results/`; see
[../docs/RESULTS.md](../docs/RESULTS.md).
