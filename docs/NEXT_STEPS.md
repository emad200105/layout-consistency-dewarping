# Next steps — how to continue this work

Ordered roughly by how much each would strengthen the paper per unit of effort.
GPU costs assume one free Kaggle T4, where 30 epochs takes about 17 GPU-hours per
arm.

---

## 1. Make the main result robust

**More seeds.** This is the paper's biggest weakness: one training run per arm.
All confidence intervals describe variation over *pages*, not over *training
runs*.

- **What to do:** train 2–3 more seeds of each arm with
  `--seed 1235, 1236, …` and everything else unchanged. Report the mean and
  spread of EPE, 𝒞 and synthetic-OCR CER across seeds, plus a paired test per
  seed.
- **Cost:** about 34 GPU-hours per extra seed pair, roughly one week of free
  quota each.
- **Cheaper partial version:** shorter runs (e.g. 10 epochs with
  `TOTAL_STEPS` scaled to match) for several seeds, to show that the 𝒞 gap
  appears consistently.

**λ sweep.** Only λ = 0.2 was tried. Try λ ∈ {0.05, 0.1, 0.5, 1.0}, ideally at
reduced epochs first. Expected: the 𝒞 gap grows with λ until EPE starts to
suffer. Finding that point is a useful figure.

**Loss variants already in the code:**

- `--layout_mode per_region` (bands per region instead of per page, about 12×
  slower).
- A different number of bands (`layout_bands` in `DewarpLoss`).
- Including table and figure regions (`layout_categories` in
  `DewarpTrainingSet`).

## 2. Close the real-photo gap

On DocUNet both arms improve clearly on the raw photos but trail the state of
the art: MS-SSIM 0.32 vs about 0.51, LD 17 vs about 7.5. The paper does not
claim to know why. Candidate causes, each testable:

| hypothesis | test |
|---|---|
| page-boundary errors on cluttered backgrounds dominate MS-SSIM | crop each photo to its page with an off-the-shelf segmenter (as DocTr does) and re-run `evaluation/docunet/run_all.sh` |
| real creases and folds are sharper than our smooth spline creases | add higher-frequency folds or crumples to `warp_augment.py`, e.g. piecewise-linear folds or a denser control grid |
| born-digital English articles are too narrow a content domain | mix in other page types (forms, receipts, magazines, non-Latin scripts); this needs layout boxes for them, see §3 |
| capacity or resolution | `--base 48` (about 2.2× parameters) or a finer flow grid; this needs a dataset rebuild for the grid |
| 2D synthetic warps vs 3D-rendered data | train the same network on Doc3D, with and without `L_lay` (§3) |

Report any of these as a separate experiment, and keep the λ = 0.2 vs 0
comparison intact inside each one.

## 3. Test the loss where it matters most

The cleanest way to show the loss is useful on real photos: **add `L_lay` to a
strong baseline trained on Doc3D**. Doc3D has no layout boxes, but you can
obtain them:

- run a layout detector, e.g. one trained on PubLayNet, on Doc3D's flat albedo
  images; or
- OCR the flat image and group lines into blocks.

The loss only needs boxes in flat-page coordinates plus the ground-truth
backward map, both of which Doc3D-style data provides. Then compare
baseline vs baseline + `L_lay` on DocUNet, DIR300 and UVDoc.

## 4. Compare against the closest losses

ForCenNet (ICCV 2025, text-line curvature consistency) and MataDoc
(equal-y text-line constraint) are the nearest prior losses. Implement each as
an alternative to `L_lay` in `augmentation/layout_loss.py` and train with the
identical setup. A three-way comparison (none / line-curvature / region-band
variance) would answer the obvious reviewer question directly.

## 5. More evaluation

- **Line-straightness metrics** that do not come from our own loss: UVDoc's
  horizontal and vertical straightness, the AAD metric of Axis-Aligned Document
  Dewarping, and DocScanner's Li-D.
- **More benchmarks:** DIR300 (DocGeoNet), the UVDoc benchmark, DocReal.
- **Synthetic OCR against true text:** PubLayNet pages come from PMC articles,
  so the text could be extracted from the source PDFs instead of using
  Tesseract's reading of the flat page as the reference.

## 6. The paper

- **Submitted to Array (Elsevier) in November 2026.** The submitted version (CAS
  template, 7 authors, required declarations) is in
  `paper/array_submission/`.
- `paper/` here is the earlier IEEE-format draft. The results are identical;
  only the template, the abstract length and small wording points differ.
- Work to do while the paper is under review: the second seed pair, DIR300
  and UVDoc evaluation, and a λ sweep (§1, §5). If these are added, update
  Table 1, Fig. 3 and the abstract, and remove "one seed per arm" from the
  limitations.
- `paper/refs_notes.md` explains what every cited paper does relative to ours,
  and which published numbers can be compared with which.

---

## Small known issues in the code

- `dataset_format.py`'s docstring still describes `manifest.parquet` and both it and
  `build_dataset.py` say "~18 GB"; the built store uses `manifest.jsonl` /
  `clean_index.jsonl` and is about 25 GB.
- `train.py` prints "identity warp scores ~35 px" as a sanity hint; on the final
  dataset the identity EPE is 39.88 px (per page, full validation) and the
  step-0 validation printout is about 37.6 px (batch-pooled, 480-sample subset).
- `tools/prepare_upload.py` labels the Kaggle datasets `CC0-1.0`; data derived from
  PubLayNet should carry CDLA-Permissive-1.0 if published.
- `inference.py --out` and `extract_ocr_subset.py --out` do not create missing parent
  folders (create `outputs/` first).
- `training/inference.py --image` resamples with `BORDER_REPLICATE`, which
  smears the photo's edge into off-page areas. `evaluation/docunet/rectify.py`
  uses white fill instead; port it if you need clean outputs.
- `layout_metric` in `inference.py` reports the batch-pooled 𝒞 as `loss` and
  the per-page 𝒞 only per profile. A per-page overall mean (`compare_arms.py`
  computes it) would be cleaner to return directly.
- `train.py`'s docstring examples predate the Kaggle `auto` path resolution;
  the notebook shows the commands that were actually used.
- Synthetic OCR (`ocr_eval.py`) used Tesseract 5.3.4, while the DocUNet
  evaluation used 5.0.1. Each comparison is internally consistent, but do not
  compare CERs across the two.
