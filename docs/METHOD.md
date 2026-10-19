# Method notes

This is the technical reference: the conventions, the exact settings, and the
reasons behind them. The paper ([../paper/main.pdf](../paper/main.pdf)) has the
same material in compressed form. The code docstrings are longer still and
record the measurements behind each decision.

---

## 1. Conventions — read before touching any flow code

### Flow direction
The stored and predicted flow `f` is a **backward** displacement field defined
on the **warped** (input) image:

```
warped[y, x] == clean[y + f_y(y, x),  x + f_x(y, x)]
```

So `p + f(p)` is the flat-page position of warped pixel `p`. Because `f` lives
on the input grid, every training loss is computed on that grid and **no flow
inversion is needed during training**.

Rectifying a photo needs the inverse map `g`, defined on the flat-page grid:
`clean(q) ≈ warped(q + g(q))`. `dataset_format.invert_flow` computes it by a
**damped fixed-point iteration**, `g ← g + β(−f(q + g) − g)`, with 60 iterations
and β = 0.5. The undamped version (β = 1) diverges on strongly compressive warps.
Measured over 40 samples, the mean inversion residual inside the page is
0.0019 px (table in `dataset_format.py`; 0.97 px without damping).

### Units
| quantity | unit |
|---|---|
| stored and predicted flow | **normalized** displacement, `grid_sample` convention: x and y in [−1, 1] across the image, so 1 unit = (W−1)/2 px horizontally and (H−1)/2 px vertically |
| EPE, flow L1 | pixels of the **448×576** training page |
| layout loss / layout metric 𝒞 | pixels of the **112×144 flow grid** (1 grid px = 4 page px) |

Normalized units make a prediction resolution-independent: predict on a
448×576 copy, upsample the flow bilinearly to any size, and it applies
unchanged.

### Grids
| grid | size (W×H) | what lives there |
|---|---|---|
| training page | 448 × 576 | network input. Keeps PubLayNet's US-Letter aspect (612×792) within 0.7 %; both sides divisible by 32 |
| flow grid | 112 × 144 | stored ground-truth flow and mask; network output; all losses |
| inference | any | predicted flow upsampled, inverted, applied once |

The flow is stored at quarter resolution because that is where it is accurate.
Storage error at 112×144 is 0.018 px mean and 0.373 px p99 inside the page
(table in `dataset_format.py`), and decoding further would invent detail no
label can check.

---

## 2. Network (`training/model.py`)

`DewarpNet(base=32)`, **6.45 M parameters**:

- **Stem:** one 3×3 conv (16 ch) at full resolution.
- **Encoder:** five encoder stages of 2 × (conv3×3 → BatchNorm → SiLU), with
  32, 64, 128, 256 and 384 channels, each after a 2× max-pool. The bottleneck is
  at 14×18.
- **GlobalContext:** global average pool → MLP → broadcast and merge. Page
  curvature at one point depends on geometry hundreds of pixels away.
- **Decoder:** 3 up-stages (bilinear upsample + skip + conv block), ending at
  **112×144** (1/4). It stops there because the labels live there.
- **Head:** a 3×3 conv to 2 channels, **zero-initialized**, so the untrained net
  predicts the identity warp.
- **Deep supervision:** auxiliary heads at 1/8 and 1/16, training only.

---

## 3. Losses (`training/losses.py`, `augmentation/layout_loss.py`)

```
total = 1.0·L_flow + 0.5·L_photo + 0.2·L_aux + λ·L_lay        λ = 0.2 (ours) or 0 (ablation)
```

**L_flow:** masked L1 between predicted and true flow, in 448×576 page pixels,
averaged over both components and over page pixels (mask > 0). Background
pixels carry no meaningful flow and are excluded.

**L_aux:** the same loss on the two deep-supervision heads, upsampled.

**L_photo:** resample the clean page with the predicted flow and compare it with
the warped input after a **high-pass filter**: grey minus a Gaussian blur, σ = 8.
Without the filter the term mostly tries to explain the synthetic shading with
geometry. The raw difference between a perfectly resampled page and the shaded
input is 45.3 grey levels, almost all of it lighting.

**L_lay — the contribution.** Let `d = f_pred − f_true` (on the flow grid, in
grid pixels) and `c*(p) = p + f_true(p)`, the true flat-page position.

1. Keep pixels whose `c*` falls inside a PubLayNet **text, title or list** box
   (categories 1–3; at most 32 boxes per page, largest first) and inside the
   page mask.
2. Cut the flat page into **24 horizontal bands** by `c*_y` and **24 vertical
   bands** by `c*_x`. Bands span the whole page, not each region (mode
   `global`), so two text columns at the same height are constrained together.
3. `C_h = sqrt( mean over bands of Var(d_y within the horizontal band) )`, and
   `C_v` likewise with `d_x` and the vertical bands. Bands with less than 16
   pixels of mask weight are skipped.
4. `L_lay = C_h + C_v`.

Why this measures bending: all pixels of a straight text line share one `c*_y`.
After rectification the line sits at `c*_y + d_y`, so it stays straight **if and
only if `d_y` is constant along it**. A constant `d_y` is a rigid shift, which
the flow loss already handles.

`L_lay = 0` exactly when `d_y` is constant within each horizontal band and `d_x`
within each vertical band. `augmentation/test_layout_loss.py` checks that a
rigid shift costs 0 while a bend with the same L1 costs a lot.

Implementation details that matter when comparing numbers:

- **Training** pools the band statistics of the whole mini-batch (each page has
  its own bands) before the square root. **Evaluation as a metric 𝒞** is done
  per page and averaged.
  - `inference.layout_metric` reports both: the batch-pooled value as `loss` /
    `horizontal` / `vertical`, and per-page means per profile as `by_profile`.
  - The paper uses the per-page version (0.4398 vs 0.4714 on full validation).
    The batch-pooled numbers (0.5274 vs 0.5631) are a different statistic and
    must not be mixed with it.
- Each band variance is three `scatter_add`s, so after the K box tests the cost
  does not depend on the number of regions.
- Region membership is a box test on `c*`, which comes straight from the stored
  flow. No inversion is needed.

---

## 4. Synthetic data (`augmentation/warp_augment.py`, `build_dataset.py`)

**Source pages.** 47,958 PubLayNet training pages (about one archive shard),
resized to 448×576. Page sizes vary (297 distinct sizes in this set, almost all
close to US Letter, e.g. 596×791 and 612×792); `prepare_layout.py` drops 234
landscape pages, leaving 47,724 with layout boxes.

**Profiles.** Each page yields one sample per profile. Every warp contributes
only a flow field; flows are **composed**, and the page is **resampled once**.
Cascaded resampling destroyed about 85 % of fine detail and desynchronized image
and label; `test_warps.py` guards against both.

| profile | what | parameters |
|---|---|---|
| cylinder | page bent around a cylinder, pinhole camera | curl U[0.15, 0.85] rad, camera distance U[1.6, 4.0] page widths, spine position edge-biased, horizontal spine 25 % |
| perspective | random homography | each corner moved by up to m of the page size, m ~ U[0.04, 0.13] |
| combined | cylinder + perspective + smooth creases, all always applied | crease = random displacement of a k×k spline control grid, k ∈ [4, 8], amplitude up to U[0.01, 0.04] |
| flat | no page deformation (camera framing only) | for 3 % of source pages (1,439 samples) |

**Camera framing**, composed with every sample, so that page coverage stops
revealing warp strength:

- 25 %: tight crop, scale U[1.02, 1.30], roll ±5°
- 25 %: exact framing
- 50 %: pulled back, scale U[0.70, 0.98], roll ±7°
- off-centre translation within the slack

Before framing was added, page coverage correlated with mean |flow| at
r = −0.79 over 240 samples (−0.825 in `test_warps.py`'s own sample); with framing
the test measures about −0.13.

**Appearance.**

- The page is composited on a synthetic surface, 22 % of which are near-white
  (the hard case), with a soft contact shadow.
- A smooth shading gradient is added with probability 0.75.

**Storage** (`dataset_format.py`, format version 2):

- warped pages: JPEG q90, concatenated in `pages-*.bin`
- clean pages: once per source, in `clean-*.bin`
- flow: `flows.f16`, memmap (N, 144, 112, 2) float16
- mask: `masks.u8`, (N, 144, 112) uint8
- index files: `manifest.jsonl` (one row per sample, with byte offsets and every
  random parameter), `clean_index.jsonl` and `meta.json`

145,313 samples, about 25 GB in total.

**Determinism.** Every sample is seeded from (global seed 42, source index,
variant index), so the dataset is bit-identical with any number of workers. Any
sample can be regenerated from its manifest row; `verify_dataset.py` checks this
byte for byte.

**Layout store** (`prepare_layout.py`, `layout_format.py`): bbox, category and
polygon of every PubLayNet region of the kept pages, as uint16 normalized to the
page. 465,596 regions over 47,724 pages, 57 MB. It is read by stem, so it is
independent of the image build.

**Split.** By **page**, never by sample: `md5(stem) < 0.02` → validation. The
variants of one page share a clean image and must not straddle the split. The 707
samples of the 234 landscape pages have no layout and are dropped by the loader
(`require_layout=True`). Result: **141,869 train / 2,737 validation**
(901 cylinder, 901 perspective, 901 combined, 34 flat).

---

## 5. Training (`training/train.py`, `notebooks/dewarp_kaggle.ipynb`)

| setting | value |
|---|---|
| optimizer | AdamW, lr 3e-4, weight decay 1e-4 |
| schedule | 500 warm-up steps, cosine decay over **540,000** steps |
| batch | 8 |
| epochs | 30 = **531,990 steps** (17,733 steps per epoch) |
| precision | AMP (fp16), one T4 |
| seed | 1234 for both arms |
| validation | every 1,000 steps on a fixed 60-batch (480-sample) subset; also at every epoch end |
| checkpoints | every 500 steps; last two + `best.pt` + `last.pt` kept |
| speed | 8.63 steps/s (ours), 9.15 steps/s (ablation, no layout term) |
| sessions | 2 per arm (Kaggle 12-hour limit): ours 11.0 h + 6.6 h (`--max_hours 11`), ablation 10.0 h + 6.1 h (`--max_hours 10`); the session boundary therefore fell at step 341,833 vs 329,277, which exact resume makes immaterial |

**Resuming restores the exact state**: weights, optimizer moments, AMP scaler,
schedule position, epoch, step, position within the epoch and all RNG states. In a
40-step CPU test (20 steps + resume vs 40 uninterrupted) the weight and moment
differences were exactly 0. On GPU, cuDNN/AMP kernels are not deterministic, so a
resumed GPU run equals an uninterrupted one only up to that nondeterminism.

**Never change `--total_steps` between sessions**, or the cosine schedule
restarts. To train longer, raise `--epochs` and `--total_steps` together, for
both arms.

---

## 6. Evaluation protocols

**EPE:** mean end-point error over page pixels (mask > 0.5), in 448×576 page
pixels, computed per page and averaged. The zero-flow (identity) prediction scores
39.88 px on the validation set (`training/identity_epe.py`).

**Layout distortion 𝒞:** Eq. (4) of the paper, i.e. L_lay evaluated per page as
a metric (see the batch-vs-page note above). It is computed offline from
checkpoints for both arms, because `DewarpLoss` skips the layout branch when
λ = 0 and the ablation therefore never logs it.

**Synthetic OCR** (`training/ocr_eval.py`):

- A fixed, stratified validation subset, chosen by a seeded hash of the page ID
  before any OCR was run (`extract_ocr_subset.py`): 100 each of cylinder,
  perspective and combined, plus all 34 flat. 334 pages, of which 2 are dropped
  for having a reference under 200 characters, leaving 332.
- At 448×576 the text is too small to read: even the true flow gives 66–71 %
  CER. So each page is **re-synthesized at 3×**:
  - The input is the 3× clean page warped with its ground-truth flow.
  - Each model predicts its flow from the **stored 448×576 photo**, i.e. its
    training input.
  - That flow is upsampled, inverted and applied to the 3× input.
  - The true flow, applied the same way, gives the upper bound.
- OCR: Tesseract 5 LSTM (`--oem 1 --psm 3`, `OMP_THREAD_LIMIT=1`), with
  whitespace normalized.
- The reference is Tesseract's reading of the flat 3× page.
- Statistics: paired bootstrap (10,000 resamples) and Wilcoxon signed-rank.

**DocUNet real photos** (`evaluation/docunet/`):

- **Images:** the 130 benchmark photos with document 64 rotated 180° (the
  "corrected" benchmark). No fine-tuning, cropping or background removal.
- **Rectification:** each photo is rectified at full resolution with white
  fill.
- **MS-SSIM / LD:** DocUNet-style MS-SSIM and SIFT-flow Local Distortion, using
  the official MATLAB code run in Octave with a small patch.
- **OCR:** ED / CER with default `pytesseract` on Setting 1 (60 photos) and
  Setting 2 (50 photos), exactly as DocTr's `OCR_eval.py` does it.
- **Tesseract version:** use **5.0.1**, the version behind published numbers.
  The version changes CER by about 0.025 on the raw photos.

---

## 7. Things that went wrong and are now guarded

Each of these cost time once; the fix lives in the code.

| problem | symptom | fix / guard |
|---|---|---|
| cascaded resampling | label and image out of sync, ~85 % detail lost | compose flows, resample once (`test_warps.py` #4, #6) |
| old cylinder had `flow_y ≡ 0` | text lines never bowed | real 3D cylinder + pinhole model (`test_warps.py` #2, #3) |
| page coverage revealed the warp | shortcut r = −0.79 | random camera framing |
| `BORDER_REPLICATE` margins | smeared streaks; no honest page boundary | composite on synthetic backgrounds |
| naive photometric loss | geometry explaining shading | high-pass filtered photometric term |
| undamped flow inversion | diverged on compressive warps | damped fixed point, β = 0.5 |
| PubLayNet `val.json` nested `"annotations"` | naive streaming parser read 19 instead of 120,761 | `coco_stream.py` finds top-level keys only |
| landscape pages squeezed by the portrait resize | 707 samples out of distribution, no layout | dropped by stem |
| Kaggle mount paths vary | `FileNotFoundError` with the data attached | `--root auto` etc. search by marker file |
| Kaggle pins dataset versions | silently old code | notebook checks for a marker function |
| a 12-hour kill saves nothing | lost session | `--max_hours` clean exit + exact resume |
| Windows DataLoader cannot pickle memmaps | crash in evaluation | `num_workers=0` in evaluation |
| `layout_metric` default batch | batch-pooled vs per-page 𝒞 differ | both reported; paper uses per-page |
| torch thread count | a few border pixels differ between runs | `rectify.py` defaults to 1 thread (bit-exact) |
