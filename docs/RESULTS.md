# Results — every number and where it comes from

The two arms:

- **ours**: λ = 0.2, `weights/dewarpnet_layout_lambda0.2.pt`.
- **ablation**: λ = 0, `weights/dewarpnet_ablation_lambda0.pt`.

Both were trained for 30 epochs (step 531,990) with identical data, seed (1234),
schedule and architecture.

**Conventions used throughout:**

- *Reduction* = (ablation − ours) / ablation, so positive means ours is better.
- Paired differences say which way round they are taken.
- 𝒞 is the layout distortion metric (layout loss as a metric, flow-grid pixels;
  lower means straighter text regions).
- p-values come from two-sided Wilcoxon signed-rank tests, and CIs from 95 %
  percentile bootstraps with 10,000 resamples.
- **All tests measure variation over pages, not over training runs: there is one
  seed per arm.**

---

## 1. Synthetic validation set (2,737 samples)

Source: `results/synthetic_val/final_{layout,ablation}.epe.json`, written by
`python training/inference.py --ckpt <ckpt> --root dataset --full_val`.
Summarized by `training/compare_arms.py` → `results/synthetic_val/COMPARISON.txt`.

| | ours | ablation | reduction |
|---|---|---|---|
| EPE (448×576 page px) | 3.662 | 3.689 | 0.7 % |
| 𝒞, per-page mean (**paper**) | 0.4398 | 0.4714 | 6.7 % |
| 𝒞, batch-pooled (as in the training log) | 0.5274 | 0.5631 | 6.3 % |
| 𝒞_h, batch-pooled | 0.2492 | 0.2726 | 8.6 % |
| 𝒞_v, batch-pooled | 0.2782 | 0.2905 | 4.2 % |

Identity (zero flow) EPE is 39.88 px (`results/synthetic_val/identity_epe.json`,
from `training/identity_epe.py`), so ours removes 90.8 % of it and the ablation 90.7 %.
Older notes and the training-log printouts quote 35.13 px; that value came from the
early pilot dataset and is not the identity EPE of the final validation set.

The two 𝒞 rows are **different statistics**. Batch-pooled pools the band
variances of 16 pages before the square root; per-page computes 𝒞 for each page
and averages. Do not mix them.

**Per warp profile** (per-page means):

| profile | n | EPE ours | EPE abl. | red. | 𝒞 ours | 𝒞 abl. | red. |
|---|---|---|---|---|---|---|---|
| flat | 34 | 2.389 | 2.511 | 4.8 % | 0.2684 | 0.2791 | 3.8 % |
| cylinder | 901 | 2.274 | 2.256 | −0.8 % | 0.2742 | 0.2923 | 6.2 % |
| perspective | 901 | 3.241 | 3.288 | 1.4 % | 0.3563 | 0.3819 | 6.7 % |
| combined | 901 | 5.519 | 5.568 | 0.9 % | 0.6953 | 0.7471 | 6.9 % |

## 2. Paired per-page test (334-page subset)

Source: `results/synthetic_val/subset_paired.json` (written by
`paper/scripts/subset_paired.py`) and `subset_paired.txt` (its printed output). The subset is the OCR subset (§4), fixed by a
seeded hash before any evaluation. We did not run a paired test on the full
validation set because per-page outputs were not saved for it.

| metric | ours | ablation | reduction [95 % CI] | pages better / worse | p |
|---|---|---|---|---|---|
| EPE | 3.4376 | 3.5036 | 1.9 % [−1.6, 5.6] | 160 / 174 | 0.59 |
| 𝒞 | 0.4276 | 0.4555 | **6.1 %** [4.1, 8.2] | 214 / 120 | **1.5·10⁻⁹** |
| 𝒞_h | 0.2132 | 0.2326 | **8.3 %** [5.6, 11.1] | 219 / 114 | **3.5·10⁻¹⁰** |
| 𝒞_v | 0.2144 | 0.2229 | 3.8 % [0.8, 6.8] | 187 / 147 | 4.9·10⁻³ |

Per-profile rows are in the source file. 𝒞 is significantly lower for cylinder
(p = 8.5·10⁻⁵) and combined (p = 1.1·10⁻⁵), and borderline for perspective
(p = 0.056). Re-running the script with 1 vs 2 torch threads changes some
p-values in the third digit, for example 3.46 vs 3.47·10⁻¹⁰.

## 3. Training dynamics

Source: `results/training_logs/*.log.jsonl.gz`, figure `paper/figs/curves.png`.

- Validation during training used a fixed 480-sample subset every 1,000 steps.
  On it, the final EPE was 3.756 (ours) vs 3.903 (ablation), a gap that settles
  around 4 %. On the full set it is 0.7 %, so **we do not claim an EPE gain**.
- **Intermediate full-validation check.** We scored the validation-selected
  `best.pt` of each arm's *first* session: ours at step 341,000 (19.2 epochs)
  and the ablation at step 326,000 (18.4 epochs).
  - EPE was 3.909 vs 4.067, a 3.9 % gap.
  - 𝒞 per page was 0.4718 vs 0.5116, a 7.8 % gap.
  - The EPE gap almost closed during the final learning-rate anneal; the 𝒞 gap
    did not.
  - Source: `results/synthetic_val/session1_best_*.epe.json`. These checkpoints
    are 0.85 epochs apart and validation-selected, so treat this only as an
    indication.

## 4. OCR on synthetic pages (332 pages)

Source: `results/ocr_synthetic/OCR_RESULT.txt` and `per_page.csv`, from
`training/ocr_eval.py` (Tesseract 5.3.4 LSTM, 3× re-synthesis; protocol in
[METHOD.md](METHOD.md) §6).

| CER (%) | mean | corpus | median | combined (n = 100) |
|---|---|---|---|---|
| input | 63.05 | 62.82 | 71.17 | 77.01 |
| ablation | 30.37 | 24.55 | 21.69 | 34.47 |
| **ours** | **28.61** | **22.82** | **20.52** | **29.78** |
| true flow (upper bound) | 13.70 | 9.92 | 7.03 | 18.19 |

**Paired, ablation minus ours:**

- **All pages:** +1.76 points [0.78, 2.79], 203 better / 126 worse / 3 tied,
  p = 4.6·10⁻⁶. That is a 5.8 % relative reduction.
- **Combined:** +4.69 points [2.95, 6.62], 74 / 25, p = 4.9·10⁻⁷, a 13.6 %
  reduction.
- **The other three profiles together:** +0.50 points, p = 0.056, not
  significant.
- **Share of the possible gain realized** (from input down to true flow):
  69.8 % for ours, 66.2 % for the ablation.

`OCR_RESULT.txt` also prints a "relative" figure of +6.2 % (+15.8 % on
combined). That is the same difference expressed relative to *ours*, i.e.
"ablation makes 6.2 % more errors". The paper uses the reduction relative to the
ablation (5.8 %, 13.6 %).

## 5. Real photos: DocUNet benchmark (130 photos, corrected)

Source: `results/docunet/`. It holds per-image MS-SSIM/LD (`*_msssim_ld.csv`),
per-image OCR ED/CER (`ocr_per_image_*.csv`, no text) and the paired analyses
(`paired_analysis_tesseract{501,534}.json`, differences = ours − ablation, CIs
resample documents). The pipeline is described in
[../evaluation/docunet/README.md](../evaluation/docunet/README.md).

**Tesseract 5.0.1 (protocol-matched, used in the paper):**

| | MS-SSIM ↑ | LD ↓ | ED S1 ↓ | CER S1 ↓ | ED S2 ↓ | CER S2 ↓ |
|---|---|---|---|---|---|---|
| distorted, published | 0.246 | 20.51 | 1552.2 | 0.509 | 2111.6 | 0.535 |
| distorted, our reproduction | 0.249 | 20.52 | 1546.5 | 0.509 | 2092.8 | 0.531 |
| ablation | 0.322 | 16.85 | 1081.1 | 0.390 | 1513.8 | 0.395 |
| ours | 0.324 | 16.98 | 1061.4 | 0.363 | 1555.9 | 0.406 |

**Paired, ours minus ablation:** none of the differences is significant.

| metric | difference [95 % CI] | p |
|---|---|---|
| MS-SSIM | +0.002 [−0.005, 0.009] | 0.55 |
| LD | +0.13 [−0.47, 0.75] | 0.66 |
| CER S1 | −0.027 [−0.072, 0.011] | 0.19 |
| CER S2 | +0.011 [−0.041, 0.059] | 0.67 |

**With Tesseract 5.3.4:**

- Distorted CER S1 is 0.484.
- Ours 0.344 vs ablation 0.384 on S1: −0.040 [−0.084, −0.005], p = 0.015.
- S2: −0.015, p = 0.44.

The conclusion therefore depends on the OCR engine, and **no real-photo OCR gain
is claimed**.

Both arms improve MS-SSIM over the raw photo on 78 % (ours) and 82 %
(ablation) of the images.

**Published methods on the same protocol**, from DocGeoNet / DocScanner tables
(sources and caveats in `paper/refs_notes.md` §6):

| method | MS-SSIM | LD | ED S1 | CER S1 | ED S2 | CER S2 |
|---|---|---|---|---|---|---|
| DocUNet (CVPR'18) | 0.410 | 14.19 | 1259.8 | 0.397 | 1933.7 | 0.463 |
| DewarpNet (ICCV'19) | 0.474 | 8.39 | 525.5 | 0.210 | 885.9 | 0.237 |
| DocTr (MM'21) | 0.511 | 7.76 | 464.8 | 0.175 | 724.8 | 0.183 |
| DocGeoNet (ECCV'22) | 0.504 | 7.71 | 379.0 | 0.151 | 713.9 | 0.182 |
| PaperEdge (SIGGRAPH'22) | 0.472 | 7.99 | 375.6 | 0.154 | 777.8 | 0.201 |
| DocScanner-L (IJCV'25) | 0.518 | 7.45 | 390.4 | 0.149 | 632.3 | 0.165 |

Our models beat the original DocUNet on ED and CER but not on MS-SSIM or LD,
and they are far behind the methods trained on Doc3D or real photos.

## 6. Dataset build reports

Source: `results/dataset_build/`.

| file | what it records |
|---|---|
| `dataset_meta.json` | 145,313 samples from 47,958 pages, seed 42, flat fraction 0.03 |
| `verify_summary.json` | `verify_dataset.py` passed on 60 checked samples |
| `layout_meta.json`, `layout_summary.json` | 465,596 regions on 47,724 pages; 234 landscape pages dropped |
| `annotation_check_report.txt` | annotations line up with the local images |
| `inspect_report.txt`, `pilot_verify.txt` | early reconnaissance and pilot build checks |

## 7. Training configuration and logs

`results/training_logs/`:

- `{layout,ablation}_session{1,2}.config.json`: every flag of each run.
- `*.log.jsonl.gz`: every 20th step plus all validations. Read with
  `gzip.open(f, 'rt')`; one JSON object per line, and validation rows carry a
  `split` key.
- Kaggle dataset owner names in the paths are replaced by `<kaggle-user>`.
