# Result files

Everything the paper's numbers are computed from. A guide with every number and
its interpretation is in [../docs/RESULTS.md](../docs/RESULTS.md).

| folder | contents | produced by |
|---|---|---|
| `synthetic_val/` | `final_{layout,ablation}.epe.json`: full-validation EPE and layout metric per profile. `identity_epe.json`: the do-nothing baseline. `COMPARISON.txt`: the two-arm table. `session1_best_*.epe.json`: intermediate check. `subset_paired.{json,txt}`: per-page paired statistics on 334 pages | `training/inference.py --full_val`, `training/compare_arms.py`, `paper/scripts/subset_paired.py` |
| `ocr_synthetic/` | `OCR_RESULT.txt` (summary + paired tests) and `per_page.csv` (ED/CER per page for input, both arms, true flow) | `training/ocr_eval.py` |
| `docunet/` | per-image MS-SSIM/LD (`*_msssim_ld.csv`), per-image OCR ED/CER (`ocr_per_image_tesseract{501,534}.csv`, no text), paired analyses (`paired_analysis_tesseract{501,534}.json`, differences = ours − ablation) | `evaluation/docunet/run_all.sh` (outputs renamed: `<m>_msld.csv` → `<m>_msssim_ld.csv`, `ours` = layout arm); per-image OCR CSVs by `analyse.py --per_image_csv`; the 5.3.4 files by the same commands with the system `tesseract` and separate caches |
| `training_logs/` | `*.config.json`: all flags of each Kaggle session. `*.log.jsonl.gz`: training rows every 20 steps plus all validations | `training/train.py` |
| `dataset_build/` | dataset and layout-store metadata, verification summary, early reconnaissance reports, and `source_pages.txt.gz`, the exact list of 47,958 PubLayNet pages used | `build_dataset.py`, `verify_dataset.py`, `prepare_layout.py`, `check_annotations.py`, `tools/inspect_dataset.py`; the page list was exported from the built store's `clean_index.jsonl` (234 pages without layout marked) |

Naming: "layout" = our arm (λ = 0.2) and "ablation" = λ = 0, in every file.

**Two versions of the layout metric.** In the `.epe.json` files,
`layout_metric.loss/horizontal/vertical` are pooled over mini-batches of 16
pages, while `layout_metric.by_profile` are per-page means. The paper uses
per-page values; `COMPARISON.txt` shows both.
