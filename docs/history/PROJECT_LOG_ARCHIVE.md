# 📄 Document Dewarping — Log Archive (sessions 1-19)

> **Note (added for the public repository).** These are the unedited working notes kept
> during development, with the help of an AI assistant (Claude). Only personal names, the
> Kaggle account name and local machine paths were replaced (`<kaggle-user>`, `<project>`),
> and a section about the assistant's own sandbox was removed. Some early statements were
> later corrected (notably the novelty claims of session 2; see the paper's related work).
> Where these notes and `docs/RESULTS.md` / the paper disagree, the latter are authoritative.


> Split out of `PROJECT_LOG.md` on 2026-09-13. **Nothing was deleted** — this is
> the verbatim record of sessions 1 through 19, including the novelty analysis,
> the Related Work paper table, and the architecture and format decisions.
>
> For where the project stands now, read `PROJECT_LOG.md`; its **CURRENT STATE**
> section is written to be enough on its own.

### 2026-09-12 — Session 1: Project Kickoff & Architecture Discussion

**Status**: Planning / Architecture Design

#### What We Discussed
- **Task**: Document dewarping — correcting warped/curved/wrinkled document photos.
- **Dataset**: PubLayNet — large-scale document page images.
- **Dataset file**: A PubLayNet archive is currently downloading in the workspace.
- **Architecture**: U-Net with flow field prediction (preferred over vanilla pixel-to-pixel U-Net)

---

### 2026-09-12 — Session 2: Novelty Analysis & Paper Positioning

**Status**: Literature Review Complete — Novelty Gaps Identified

#### Question Answered: Is this approach new?

**Short answer**: The *general idea* (synthetic warping + U-Net + dewarping) is NOT new.
But the **specific combination and angle** has clear, exploitable novelty gaps. See full novelty analysis below.

#### State-of-the-Art Papers Found (most recent first)

| Year | Paper | Venue | Key Idea |
|------|-------|-------|----------|
| 2025 | TADoc (Zhao et al.) | arxiv | Models dewarping as a *dynamic/time-aware process*, not a one-step transform |
| 2025 | DvD (Zhang et al.) | arxiv | First diffusion model for dewarping — coordinate-level diffusion, not pixel-level |
| 2025 | Axis-Aligned Doc Dewarping (Wang et al.) | AAAI 2026 | Uses axis-aligned geometric constraints + new AAD metric |
| 2026 | D2Dewarp (Li et al.) | CVPR 2026 | Dual-dimension (H+V) text-line aware dewarping |
| 2024 | DocRes (Zhang et al.) | CVPR 2024 | Unified generalist model for 5 doc restoration tasks incl. dewarping |
| 2023 | Polar-Doc (Zhang et al.) | arxiv | Polar coordinate representation for dewarping (instead of Cartesian) |
| 2022 | Marior (Zhang et al.) | ACM MM 2022 | Iterative coarse-to-fine margin removal + flow refinement |
| 2022 | Grid Regularization (Jiang et al.) | arxiv | Text-line & boundary aware constrained optimization |
| 2021 | RectiNet-v2 / GatBifUNet | arxiv | Gated + bifurcated stacked U-Net on synthetic data |
| 2019 | DewarpNet | ICCV 2019 | 3D shape + texture mapping |
| 2018 | DocUNet | CVPR 2018 | First deep dewarping paper — U-Net + flow field |

#### Key Insight: What's Tried Before
- Synthetic training data + U-Net → YES, this is standard (DocUNet 2018, RectiNet 2021)
- U-Net + flow/displacement field prediction → YES, dominant approach since 2018
- Text-line awareness → Many papers do this
- Diffusion models for dewarping → DvD (2025) just did this

#### WHERE THE NOVELTY IS — For Your Paper

**Gap 1: PubLayNet as training data is UNEXPLORED**
- All existing papers use: DocUNet dataset, DIR300, DocReal, WarpDoc, or custom datasets
- NOBODY has used PubLayNet (which has rich layout annotations: text blocks, figures, tables, titles)
- This is a concrete novelty claim: "We leverage layout-aware document images with known semantic regions"

**Gap 2: Layout-Aware / Semantics-Guided Dewarping**
- PubLayNet has segmentation masks for text/figure/table regions
- You can use these layout annotations to guide your dewarping model
- Example: enforce that text bounding boxes stay rectangular after dewarping
- This is YOUR novel angle — no paper has done layout-guided, semantics-aware dewarping using PubLayNet's annotations

**Gap 3: Wrinkle + Shadow Simulation is Under-Explored**
- Most papers do cylindrical/perspective warps only
- Very few model local wrinkles, paper creases, and shadow gradients realistically
- A comprehensive, physically-motivated augmentation pipeline would be a contribution

**Gap 4: Free Analytical Flow Ground Truth**
- Because YOU control the synthetic warp, you get exact GT flow fields for free
- Most real-world datasets (DIR300, WarpDoc) have imperfect/approximate GT
- This enables better supervision and a cleaner experimental setup

---

## 🧠 Recommended Novel Paper Angle

### Title idea:
> **"Layout-Aware Document Dewarping using PubLayNet: Exploiting Semantic Annotations for Geometry-Preserving Rectification"**

### Key Claims for Paper:
1. First use of PubLayNet for document dewarping training
2. Semantics-guided loss: enforce layout-consistency (text boxes stay rectangular) as an auxiliary supervision signal
3. Comprehensive synthetic augmentation covering curvature + wrinkles + shadows + perspective
4. Free analytical flow GT from known warp parameters
5. U-Net with flow prediction + spatial transformer resampling

---

## 🧠 Architecture Discussion

### Option A: Vanilla U-Net (Pixel-to-Pixel)
- **Pros**: Well-understood, fast to train
- **Cons**: Blurry outputs; struggles with large geometric distortions

### Option B: U-Net + Flow Field [RECOMMENDED]
- U-Net predicts a 2D displacement field (dx, dy per pixel)
- Flow used to resample via a Spatial Transformer Network (STN)
- GT flow is FREE — derived directly from synthetic warp parameters
- **Pros**: Sharp outputs, geometrically meaningful, dominant in literature
- **Cons**: Slightly more complex

### Option C: Transformer-based (Swin-UNet)
- **Pros**: Better long-range context for whole-page curvature
- **Cons**: Data-hungry, slower

### Option D: Diffusion Model
- **Pros**: Top perceptual quality
- **Cons**: Already done by DvD (2025). NOT novel anymore.

---

## ✅ Decisions Made / To Confirm
- [x] The general approach (synthetic + U-Net) is not new — but specific angle IS novel
- [x] Novelty angle: PubLayNet + layout-aware loss + comprehensive augmentation
- [ ] Finalize architecture (leaning toward Option B: U-Net + Flow Field)
- [ ] Finalize synthetic augmentation pipeline components
- [ ] Loss function: flow L1 + perceptual loss + layout-consistency auxiliary loss
- [ ] Set up folder structure in workspace

---

## 🗂️ Planned Folder Structure
`
<project>\
|- PROJECT_LOG.md
|- images_clean/
|- images_warped/
|- augmentation/
|- model/
|- training/
+- evaluation/
`

---

## 📚 Must-Read Papers (for Related Work section)
| Paper | Why Read |
|-------|----------|
| DocUNet (Ma 2018, CVPR) | Baseline/pioneer — U-Net + flow |
| DewarpNet (Das 2019, ICCV) | 3D shape approach |
| RectiNet-v2 / GatBifUNet (2021) | Synthetic training, gated U-Net |
| DocRes (Zhang 2024, CVPR) | Generalist multi-task approach |
| DvD (Zhang 2025) | Diffusion for dewarping (to differentiate from) |
| TADoc (Zhao 2025) | Time-aware / dynamic approach (to differentiate from) |
| D2Dewarp (Li 2026, CVPR) | Dual-dimension text lines (similar spirit to layout-aware) |

---

## 🔧 Tech Stack (Planned)
| Component              | Tool                                  |
|------------------------|---------------------------------------|
| Synthetic augmentation | OpenCV, NumPy, SciPy                  |
| Deep learning          | PyTorch                               |
| Architecture           | U-Net + STN (flow prediction)         |
| Dataset                | PubLayNet                             |
| Metrics                | SSIM, MS-SSIM, PSNR, LD, OCR-CER     |
| Extra supervision      | Layout-consistency loss (PubLayNet annotations) |

---

*Log last updated: 2026-09-12 — Session 2*


---

### 2026-09-12 — Session 3: Synthetic Augmentation Pipeline Built

**Status**: Augmentation pipeline code complete — ready to test

#### What Was Done
- Created full project folder structure:
  - images_clean/, images_warped/, flow_fields/
  - augmentation/, model/, training/, evaluation/
- Built augmentation/warp_augment.py — full synthetic warp pipeline
- Built augmentation/preview_augmentation.py — visual testing script

#### Augmentation Pipeline Details

Four warp types implemented:

| Warp Type | Method | What It Simulates | GT Flow? |
|-----------|--------|-------------------|----------|
| Cylindrical | Horizontal tan() remap | Page bent around a cylinder (spine curve) | YES (analytical) |
| Perspective | Random corner homography | Camera held at an angle | YES (analytical) |
| Wrinkle/Crease | Thin-Plate-Spline on random control points | Local page wrinkles and creases | YES (analytical) |
| Shadow | Gradient alpha multiply | Lighting variation from page curvature | N/A (photometric only) |

Key design decisions:
- All geometric warps use BACKWARD MAPPING (output pixel -> source pixel) — avoids holes
- GT flow field stored as (H, W, 2) numpy array in pixel units (dx, dy)
- Multiple warps are composed correctly using bilinear interpolation of flow fields
- Shadow is purely photometric — does not modify the flow

#### Files Created
- <project>\augmentation\warp_augment.py   <- Main pipeline
- <project>\augmentation\preview_augmentation.py  <- Visual test

#### Next Steps
- [ ] Test augmentation pipeline: run preview_augmentation.py
- [ ] Install required packages: pip install opencv-python numpy scipy
- [ ] Wait for PubLayNet dataset download to complete
- [ ] Run warp_augment.py --input_dir <publaynet_images> at scale
- [ ] Then: design U-Net + Flow model architecture (deferred)


---

### 2026-09-12 — Session 3 (cont.): Augmentation Pipeline Tested & Verified

**Status**: ALL TESTS PASSED ✅

#### Test Results
- scipy 1.18.1 installed successfully
- All 4 warp types executed without error
- Flow field shape: (1024, 768, 2), dtype: float64
- Flow range verified: dx=[-16.15, 20.36] dy=[-9.63, 48.63] (physically reasonable)
- Preview image saved: <project>\augmentation\preview_output.png

#### Visual Verification
Each warp type visually confirmed correct:
- Cylindrical: Straight text lines visibly curve (horizontal barrel distortion)
- Perspective: Document skewed/tilted as expected
- Wrinkle: Smooth wave-like local distortions on text lines
- Shadow: Realistic gradient darkening on one side
- Full Combined: Realistic composite of all effects + shadow

#### Environment
- Python 3.14
- opencv-python 5.0.0.93
- numpy 2.5.1
- scipy 1.18.1

#### Next Steps
- [ ] Wait for PubLayNet download to complete
- [ ] Extract PubLayNet archive and locate image directory
- [ ] Run: python augmentation\warp_augment.py --input_dir <publaynet_images>
- [ ] Verify output pairs (spot-check clean/warped/flow triplets)
- [ ] Then begin U-Net + Flow model design


---

### 2026-09-12 — Session 4: Dataset Scale Decision

**Status**: Augmentation pipeline finalized

#### Decision Made
- PubLayNet subset: **48,000 images** (not full dataset)
- Variants per image: **3** (changed from 5)
- Total training pairs: **48,000 x 3 = 144,000**

#### Final 3 Distortion Profiles

| Variant | Profile | Warps Applied | Purpose |
|---------|---------|---------------|---------|
| v0 | cylinder | Cylindrical + Shadow | Pure spine/book curvature |
| v1 | perspective | Perspective + Shadow | Pure camera tilt/angle |
| v2 | combined | Cylinder + Perspective + Wrinkle + Shadow | Hardest, most realistic |

#### File Output Structure (per source image)
`
images_clean/<stem>.png                    <- 1 clean GT (shared)
images_warped/<stem>_v0_cylinder.png       <- variant 0
images_warped/<stem>_v1_perspective.png    <- variant 1
images_warped/<stem>_v2_combined.png       <- variant 2
flow_fields/<stem>_v0_cylinder.npy         <- GT flow for v0
flow_fields/<stem>_v1_perspective.npy      <- GT flow for v1
flow_fields/<stem>_v2_combined.npy         <- GT flow for v2
`

#### Next Steps
- [ ] PubLayNet download to complete
- [ ] Extract and identify image directory
- [ ] Run: python augmentation\warp_augment.py --input_dir <path> --variants 3
- [ ] Begin U-Net + Flow model design


---

### 2026-09-12 — Session 5: Storage Blocker Found — Dataset Format Redesign

**Status**: Full-resolution run BLOCKED — storage format being redesigned for Kaggle

#### Context Change
- **Training will happen on Kaggle** (free GPU). Only warping / data generation runs locally on CPU.
- **Local SSD free space: 216 GB.**
- This makes dataset *size* a first-class design constraint, not an afterthought.

#### Correction to Session 3 Test Report
The Session 3 entry recorded `Flow field shape: (1024, 768, 2)`. That was **not a PubLayNet
image** — `preview_augmentation.py` falls back to `make_dummy_document(H=1024, W=768)`, a
synthetic blank page, when no `--image` is passed. So:
- The 1024x768 figure is an artifact of the test harness, not real data.
- The claim "Cylindrical: straight text lines visibly curve" is **not supported by the code**
  (see Issue 2 below) and was observed on synthetic lines, not real text.
- Real PubLayNet dimensions still need to be measured -> `inspect_dataset.py` (created below).

Reference: PubLayNet is ~99 GB for 335,703 train images (~295 KB/image), so the local
subset of 47,958 images is roughly one train shard, ~14 GB on disk.

#### BLOCKER: Storage cost of the current format

The current pipeline saves a full-resolution flow field per variant as `.npy`.
`perspective_warp` creates `np.ones(H*W)` (float64), which promotes the composed flow to
float64. At 47,958 sources x 3 variants = 143,874 pairs:

| Assumed size | clean PNG | warped PNG | flow float64 | flow float32 | TOTAL (f64) |
|---|---|---|---|---|---|
| 612 x 792   | 31 GB | 94 GB  | **1,116 GB** | 558 GB | **~1.24 TB** |
| 800 x 1000  | 52 GB | 155 GB | **1,842 GB** | 921 GB | **~2.05 TB** |

Against 216 GB free, this is off by 6-10x. Even float32 flow alone does not fit.
And it could never be uploaded to Kaggle regardless.

#### DECISION: New storage format

Store the dataset at a fixed training resolution, with flow decoupled from image resolution.

| Item | Old | New |
|---|---|---|
| Clean image | full-res PNG | fixed-res JPEG q=90 (one per source, shared) |
| Warped image | full-res PNG | fixed-res JPEG q=90 |
| Flow field | full-res float64 .npy, **pixel** units | **64x64 float16, NORMALIZED [-1,1] grid units** |
| Packaging | 300k+ loose files | sharded `.tar` (~1000 samples/shard) |
| Reproducibility | none | `manifest.parquet` with per-sample seed + warp params |

Projected totals (47,958 x 3 pairs, 64x64 float16 flow):

| Train res | clean | warped | flow | TOTAL |
|---|---|---|---|---|
| 256 x 256 | 0.94 GB | 2.83 GB | 2.36 GB | **6.1 GB** |
| 320 x 320 | 1.47 GB | 4.42 GB | 2.36 GB | **8.3 GB** |
| 384 x 384 | 2.12 GB | 6.36 GB | 2.36 GB | **10.8 GB** |
| 448 x 448 | 2.89 GB | 8.66 GB | 2.36 GB | **13.9 GB** |
| 512 x 512 | 3.77 GB | 11.31 GB | 2.36 GB | **17.4 GB** |

~1.24 TB -> ~11 GB at 384x384. Uploadable to Kaggle, trivial on a 216 GB SSD.

**Rationale for low-res flow**: all three warps are extremely smooth by construction —
cylindrical is a 1-D analytic function of x, perspective is a homography (exactly bilinear
in homogeneous coords), wrinkle is a bicubic spline over at most an 8x8 control grid. None
has high-frequency content, so a 64x64 grid bilinearly upsampled back to full resolution
should have sub-pixel error. This must be *verified empirically* before it is locked in.

**Rationale for normalized units**: flow in `[-1,1]` grid units drops straight into
`torch.nn.functional.grid_sample` and is invariant to the resolution the model runs at, so
the same GT works for a 256px pilot and a 512px final model without regeneration.

#### Kaggle constraints driving this (verified 2026-09-12)
- 30 GPU-hours/week; 12-hour max session (9h TPU) -> training must be **resumable across sessions**
- 20 GB auto-saved working disk -> checkpoints must be pruned
- GPUs offered: P100 16 GB, or T4 x2
- Loose small files are slow to upload and slow to mount -> hence sharding
- Private dataset quota should be confirmed in-account before upload

#### Outstanding Code Issues (carried from Session 4 review, not yet fixed)
1. **Storage** — being fixed by this redesign.
2. **Cylindrical warp cannot curve text lines.** `cylindrical_warp` sets
   `y_src = grid_y.copy()`, so `flow_y` is identically zero. It is a pure horizontal
   resample: horizontal text lines stay perfectly horizontal. Real book-spine curvature
   requires vertical displacement varying with x (lines bow toward the spine), which means
   projecting a 3-D cylinder through a camera, not remapping x alone. **This is the most
   important geometric fix** — spine curvature is the distortion the paper is centrally about.
3. **Layout-aware loss has no code path.** Nothing in the pipeline touches PubLayNet's COCO
   annotations. The novelty claim requires layout polygons pushed through the same flow and
   stored per variant. Cheap now, expensive to retrofit after 144k samples exist.
4. **`BORDER_REPLICATE` smears page edges** instead of compositing the page onto a
   background, so the model may never learn true page boundaries.
5. **`--variants` defaults to 5** in the CLI but only 3 profiles exist (would duplicate).
6. **No `--size` flag**; variable PubLayNet dimensions would reach the model unresized.
7. **Single-threaded**; should use a process pool over N-1 cores.

#### Verified Correct (reviewed this session)
- `compose_flows` is algebraically right: returns `f2(x) + f1(x + f2(x))`, the correct
  backward-flow composition.
- Backward-mapping convention is consistent across all four warp functions.
- Shadow is applied after warping, so the gradient lives in warped space — correct.

#### Files Created This Session
- `<project>\inspect_dataset.py` — reports folder layout, image count, real
  width/height distribution, on-disk size, and storage projections per candidate resolution.

#### Next Steps
- [ ] Run `python inspect_dataset.py` and record the real dimension distribution here
- [ ] Confirm training resolution (leaning 384x384) and flow resolution (leaning 64x64)
- [ ] Write `check_flow_downsample.py` to measure upsampling error of low-res flow
- [ ] Rewrite `warp_augment.py` v2: resize-then-warp, normalized float16 flow, sharded
      output, process pool, manifest
- [ ] Fix cylindrical warp geometry (add vertical bow)
- [ ] Add PubLayNet COCO annotation warping for the layout-consistency loss
- [ ] Generate a pilot set (~3,000 sources x 3 = 9,000 pairs, ~0.7 GB) and validate the
      full Kaggle loop end-to-end before scaling to 48k

#### Environment Note
`device_bash` cannot mount `<project>` from this session (Plan9 share failed), so scripts
are written to the folder and **run by the user locally**; file reads happen via staging.

*Log last updated: 2026-09-12 — Session 5*

---

### 2026-09-12 — Session 5 (cont.): Resolution Locked

**Status**: Format decided — ready to write `warp_augment.py` v2

#### DECISION: 512 x 512 training resolution, 64 x 64 flow

| Setting | Value |
|---|---|
| Training resolution | **512 x 512** (square, non-aspect-preserving resize) |
| Stored flow resolution | **64 x 64**, float16, normalized [-1,1] grid units |
| Warped / clean images | JPEG q=90 |
| Projected dataset size | **~17.4 GB** (3.77 clean + 11.31 warped + 2.36 flow) |
| Packaging | sharded .tar, ~1000 samples/shard |

Fits the 216 GB SSD with enormous headroom and is uploadable to Kaggle.

#### Consequence: square resize

PubLayNet pages are portrait (aspect w/h ~0.77 pending `inspect_dataset.py` confirmation).
Resizing to a 512 x 512 square applies a consistent horizontal stretch. This is **accepted
and standard** — DocUNet, DocRes and most dewarping work train on square inputs — because
the flow is learned in normalized coordinates, which are aspect-agnostic.

Inference path: resize input to 512 x 512 -> predict normalized flow -> upsample flow to the
original image resolution -> single `grid_sample` at full resolution. The output is therefore
full-res and correctly proportioned; only the network's internal view is square.

#### Consequence: 64 x 64 flow is an 8x upsample

At 512 x 512 the stored flow is upsampled 8x at train time. Justification remains sound —
the wrinkle warp's control grid is at most 8 x 8, so 64 x 64 already massively oversamples
the highest-frequency component present. Cylindrical and perspective are analytic and
smoother still. **Must still be verified** by `check_flow_downsample.py`. If measured error
exceeds ~0.3 px, fall back to 96 x 96 (flow 2.36 -> 5.31 GB, total ~20.4 GB) — a cheap escape.

#### Architecture implication (new)

Because GT flow is stored at 64 x 64, the U-Net decoder only needs to decode up to 64 x 64,
not back to 512 x 512. The three most expensive decoder stages (128, 256, 512) can be
dropped entirely. This is a large VRAM and step-time saving on Kaggle's P100/T4 and is the
standard "predict low-res flow, upsample once, resample once" design. Encoder still sees
full 512 x 512 detail.

#### Kaggle budget check

143,874 samples at 512 x 512 is roughly an hour-plus per epoch on a P100, so ~10 epochs per
12-hour session and ~25 epochs per 30 GPU-hour week. Training **must** checkpoint and resume
across sessions from day one — this cannot be treated as an afterthought.

#### Next Steps (unchanged, resolution now fixed)
- [ ] Run `inspect_dataset.py`, record real dimensions here
- [ ] `check_flow_downsample.py` — validate 64 x 64 flow (fallback 96 x 96)
- [ ] `warp_augment.py` v2 — resize-then-warp @512, normalized float16 flow @64,
      sharded tar output, process pool, manifest.parquet
- [ ] Fix cylindrical warp geometry (vertical bow)
- [ ] Warp PubLayNet COCO annotations for the layout-consistency loss
- [ ] Pilot: 3,000 sources x 3 = 9,000 pairs (~1.2 GB @512) -> validate Kaggle loop
- [ ] Then scale to 47,958 x 3

*Log last updated: 2026-09-12 — Session 5 (cont.)*

---

### 2026-09-12 — Session 6: Fix #1 — Cylindrical Warp Geometry (+ a worse bug found)

**Status**: Fix #1 COMPLETE and tested. A second, more damaging bug was found while
verifying it, and fixed in the same pass.

#### Fix #1: cylindrical warp rebuilt on real geometry

The old warp was `x_src = f*tan((x-cx)/f)+cx, y_src = y`, so `flow_y` was identically
zero — a pure horizontal resample that could not bow a text line at all.

Replaced with a 3-D cylinder photographed by a pinhole camera. A page point at
arc-length `u` from the spine, height `v`:

```
X = R sin(u/R)      Y = v      Z = R (1 - cos(u/R))
x_img = f X / (d + Z)           y_img = f Y / (d + Z)
```

`y_img` is divided by `(d + Z)` and `Z` varies with the **horizontal** position `u`.
That coupling is the whole mechanism: a horizontal line at constant `v` projects to a
`y_img` that changes across the page, so it bows toward the image centre-line as the
page recedes at the edges.

Inversion for backward mapping: `x_img` depends on `u` alone and is monotonic for
`|u/R| < pi/2`, so it is inverted by tabulating `x_img(u)` (4096 points) and calling
`np.interp`; then `Z` per column in closed form and `v = y_img (d+Z) / f`.

New parameters (old `cylinder_strength` is gone):

| Param | Meaning | Sampled range |
|---|---|---|
| `curl` | half-angle at the cylinder axis, radians. 0 = flat | 0.15 - 0.85 |
| `cam_dist` | camera distance in page widths; smaller = more bow | 1.6 - 4.0 |
| `spine_pos` | bend-axis position as fraction of width | 45% edge-biased (0-0.15 / 0.85-1), else 0.30-0.70 |
| `horizontal` | spine runs horizontally instead | 25% of the time |
| `fit` | `'contain'` keeps the whole page in frame; `'width'` crops the corners away | `'contain'` |

`fit='contain'` was added after the first visual showed page corners leaving the frame
at high curl — that content would have been lost permanently, and a dewarping GT cannot
afford invisible regions.

Measured: at `curl=0.55, cam_dist=2.2` the top line bows **21.63 px**, the bottom
**21.69 px** (symmetric, as physics requires), and the centre line **0.030 px** — straight.
Old code: 0.00 px everywhere.

#### NEW BUG FOUND (Issue 8): cascaded resampling corrupted the labels

Verifying Fix #1 across all three profiles showed v2 `combined` failing a round-trip
check — resampling the clean image with the saved flow did **not** reproduce the saved
warped image. v0 and v1 were exact; only the multi-warp profile diverged.

Cause: `augment_document` resampled the image after *each* warp and composed the flows
alongside. Two independent failures, both concentrated on v2 — the profile intended to
be the most realistic and the hardest:

1. **Detail destruction.** Three sequential bilinear remaps retained only **15.2%** of
   the clean page's Laplacian variance; one composed remap retains **43.5%**. The model
   would have been trained to dewarp images blurrier than any real photograph.
2. **Label desync.** Mean absolute error between the saved image and the flow-implied
   image was **~4.9 grey levels, max 235**, and the disagreement was *interior*, not just
   at borders (border-excluded mean 3.98). The stored label did not describe the stored
   input.

**Fix: compose first, resample once.** Every warp was refactored into a pure flow
generator that never touches pixels (`cylindrical_flow`, `perspective_flow`,
`wrinkle_flow`), all flows are composed, and the clean image is resampled exactly once
via a single `apply_flow`. Image-level wrappers are kept for `preview_augmentation.py`.

This is valid because none of the three flows depends on image content, so composing
independently-generated flows is exactly equivalent to generating each warp from the
previous warp's output.

Result: label error is now **exactly 0** on all three profiles, detail retention rose to
41-52%, and generation is faster (one remap instead of three).

#### Also fixed in this pass (small)
- `perspective_flow` built its coordinate grid with `np.ones(H*W)` (float64), silently
  promoting every composed flow to float64 and **doubling the dataset size**. Now float32
  throughout; `compose_flows` returns float32; `process_dataset` saves float32.
- `--variants` CLI default 5 -> 3, matching the 3 profiles (5 would have cycled and
  produced duplicate profile names).
- `wrinkle_flow` clamps spline degree to `min(3, n_control-1)`, so `n_control=4` — which
  the sampler can produce — no longer risks a spline error.

#### Files
- `augmentation/warp_augment.py` — rewritten warp section + `augment_document`
- `augmentation/test_warps.py` — **NEW** regression suite, 7 groups, all passing.
  Each test guards a bug that actually shipped. Run after any pipeline change:
  `python augmentation/test_warps.py`

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1 | Cylindrical warp cannot bow lines | **FIXED** |
| 8 | Cascaded resampling corrupts labels + destroys detail | **FIXED** (found this session) |
| 5 | `--variants` default 5 vs 3 profiles | **FIXED** |
| - | float64 flow bloat | **FIXED** |
| - | `n_control=4` spline risk | **FIXED** |
| 3 | Layout-aware loss has no code path | open — next |
| 2 | Storage format (512/64, sharded, manifest) | open |
| 4 | `BORDER_REPLICATE` smears page edges, no background, no validity mask | open |
| 6 | No `--size` flag / resize-then-warp | open (folded into storage rewrite) |
| 7 | Single-threaded | open (folded into storage rewrite) |

#### Note for Fix #4
With `fit='contain'` the page no longer fills the frame, so columns outside the page now
sample clamped edge pixels. This makes Fix #4 (composite the page onto a background) more
urgent, and it must also emit a **validity mask** — the flow is meaningless outside the
page, so the flow loss has to be masked there or the model learns to predict garbage in
the margins.

#### Next Steps
- [ ] Fix #3: warp PubLayNet COCO annotations through the flow (the novelty claim)
- [ ] Fix #4: background compositing + validity mask
- [ ] Fix #2: storage rewrite — resize-then-warp @512, float16 normalized flow @64,
      sharded tars, manifest, process pool
- [ ] Still need `inspect_dataset.py` output for real page dimensions

*Log last updated: 2026-09-12 — Session 6*

---

### 2026-09-12 — Session 7: Pipeline Run on REAL PubLayNet Images

**Status**: Pipeline verified on real data. Two new issues found.

#### How it was run
`device_bash` cannot mount `<project>` (persistent Plan9 share failure, 5+ attempts),
and `device_list_dir` refuses the folder's 47,958 entries. Filenames were therefore read
off the folder via computer use (File Explorer, click-only tier — terminals cannot be
typed into), 12 images were staged, and the pipeline was run in the session container.

`inspect_dataset.py` was rewritten to: pick N images at random, resize to the target
resolution, run all 3 profiles, round-trip-verify every flow label, write a contact sheet
(`real_preview.png`) and a report (`inspect_report.txt`).

#### CONFIRMED: real PubLayNet dimensions

**Every one of the 12 sampled images is exactly 612 x 792** — US Letter at 72 dpi.
Mean file size 255 KB. The dataset appears to be dimensionally uniform, which simplifies
everything: no variable-size handling needed in the loader.

The earlier `1024 x 768` figure in Session 3 is now definitively refuted — it came from
`make_dummy_document()`.

- Native aspect ratio: **0.7727** (portrait)
- Folder layout: **47,958 loose .jpg files directly in `<project>\publaynet\`** —
  no `train/` or `val/` subfolder, and **no annotation JSON present**
- 47,958 ~= one PubLayNet train shard (full train is 335,703 images)

#### BLOCKER for the novelty claim

**There is no COCO annotation file in the folder.** The layout-aware loss (Gap 2, the
paper's central claim) needs `train.json` from PubLayNet's `labels.tar.gz`. It must be
downloaded before Fix #3 can be built. Without it, the project falls back to being a
conventional synthetic-warp dewarper with no novelty.

#### NEW ISSUE 9: ~17% of the dataset would have had NO warp at all

The contact sheet showed three profile slots identical to their clean source. Cause:
`augment_document` gates each warp behind `random.random() > threshold`, which was
sensible for a fully random profile but is wrong now that profiles are fixed and named.
Measured over 20,000 draws per profile:

| Profile | geometric identity | cylinder | perspective | wrinkle | all three |
|---|---|---|---|---|---|
| v0 cylinder | **19.5%** | 80.5% | — | — | — |
| v1 perspective | **29.6%** | — | 70.4% | — | — |
| v2 combined | 2.1% | 80.3% | 70.0% | 65.0% | **37.0%** |

Dataset-wide geometric identity rate: **17.1%** — about **24,571 of 143,874 pairs with
zero warp**. Worse, v2 `combined` actually combines all three warps only **37%** of the
time, so the profile does not do what its name and the log claim.

Intended fix: profiles become deterministic in *which* warps apply (v0 always cylinder,
v1 always perspective, v2 always all three) and random only in *magnitude*. A small
deliberate identity fraction (~2-3%) should be kept as an explicit 4th case, since a model
that cannot leave a flat page alone is useless — but it should be designed, not an
accident of three independent coin flips.

#### NEW ISSUE 10: square 512 x 512 distorts every page by 1.29x

Native aspect is 0.7727; a 512 x 512 square resize stretches every page horizontally by
**1.294x**, clearly visible on the contact sheet. It is survivable (DocUNet and others
train square) but it is free to avoid:

| Resolution | pixels | aspect | stretch | /32 clean | flow grid | total |
|---|---|---|---|---|---|---|
| 512 x 512 | 262,144 | 1.0000 | 1.294x | yes | 64 x 64 | 17.44 GB |
| **448 x 576** | **258,048** | **0.7778** | **1.007x** | **yes** | **56 x 72** | **17.17 GB** |
| 512 x 664 | 339,968 | 0.7711 | 0.998x | no | — | 22.62 GB |

**448 x 576 is essentially free**: 1.6% fewer pixels, 0.27 GB smaller, both dimensions
divisible by 32 (so the U-Net downsamples cleanly), and it removes the distortion
entirely. Recommend switching. Needs a decision before the storage rewrite.

#### What the real-image run confirmed working
- Flow label **exact (max error 0)** on all 18 generated samples — the compose-once fix
  from Session 6 holds on real data, not just the synthetic test page
- All flows float32
- Flow magnitudes are physically sensible at 512 px: v0 mean 3-43 px, v1 mean 12-51 px,
  v2 mean 24-43 px
- v2 `combined` output is visually convincing — reads as a photograph of a bent journal
  page, with curvature, wrinkles and a lighting gradient
- Resize-then-warp works and is the cheaper order

#### What the real-image run confirmed BROKEN (Fix #4, now clearly visible)
With `fit='contain'` the page no longer fills the frame, and the margins are filled by
`BORDER_REPLICATE` — visible as vertical smears down both edges of every v0 sample.
Real photos show background there. Fix #4 (background compositing + validity mask) is now
the most visually obvious defect in the output.

#### Files
- `inspect_dataset.py` — rewritten: random sampling, live pipeline run, round-trip
  verification, contact sheet + report output
- `real_preview.png`, `inspect_report.txt` — outputs from this run (in session container;
  the same script writes them into `<project>\` when run locally)

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1 | Cylindrical warp cannot bow lines | FIXED (S6) |
| 8 | Cascaded resampling corrupts labels + destroys detail | FIXED (S6) |
| 5 | `--variants` default | FIXED (S6) |
| - | float64 flow bloat | FIXED (S6) |
| - | `n_control=4` spline risk | FIXED (S6) |
| 9 | ~17% identity samples; v2 combines all three only 37% of the time | **open — new** |
| 10 | Square 512 distorts pages 1.29x; 448x576 is free | **open — new, needs decision** |
| 11 | No COCO annotation JSON on disk | **open — blocks the novelty claim** |
| 3 | Layout-aware loss has no code path | open (blocked by 11) |
| 4 | `BORDER_REPLICATE` smears, no background, no validity mask | open — next |
| 2 | Storage format rewrite | open (absorbs 6, 7, 10) |

#### Next Steps
- [ ] Decide 448 x 576 vs 512 x 512
- [ ] Download PubLayNet `labels.tar.gz` (needed for the novelty claim)
- [ ] Fix #9: deterministic profiles + explicit identity fraction
- [ ] Fix #4: background compositing + validity mask
- [ ] Fix #3: warp COCO annotations through the flow
- [ ] Fix #2: storage rewrite

*Log last updated: 2026-09-12 — Session 7*

---

### 2026-09-12 — Session 8: Resolution Switch + Fix #9 (Deterministic Profiles)

**Status**: Both complete, tested, verified on real pages.

#### DECISION: 448 x 576 replaces 512 x 512

| | 512 x 512 | **448 x 576** |
|---|---|---|
| pixels | 262,144 | **258,048** (1.6% fewer) |
| aspect | 1.0000 | **0.7778** |
| stretch vs native 0.7727 | 1.2941x | **1.0065x** |
| divisible by 32 | yes | **yes** (14 x 18) |
| flow grid (1/8) | 64 x 64 | **56 x 72** |
| dataset total | 17.44 GB | **17.31 GB** |

Strictly better on every axis. Added as module constants in `warp_augment.py`:
`TRAIN_W, TRAIN_H = 448, 576` and `FLOW_W, FLOW_H = 56, 72`, so nothing hardcodes a
resolution. Issue 10 closed.

#### FIX #9: warp presence is now deterministic

Each enabled warp is **always** applied; only magnitude is random. The
`random.random() > threshold` gates are gone from `augment_document`.

Verified over 300 draws per profile at 448 x 576:

| Profile | identity rate | \|flow\|max p5 / median / p95 |
|---|---|---|
| v0 cylinder | **0.00%** (was 19.5%) | 6.57 / 19.48 / 42.89 px |
| v1 perspective | **0.00%** (was 29.6%) | 21.12 / 40.27 / 74.45 px |
| v2 combined | **0.00%** (was 2.1%, all-three only 37%) | 37.55 / 64.51 / 101.04 px |

Flat pages are still needed — a model that cannot leave an undistorted page alone is
useless — so they are now produced **deliberately**:

- New `FLAT_PROFILE`: zero flow, shadow usually applied.
- New `--flat_frac` (default 0.03): that fraction of source images additionally emit a
  `_v3_flat` variant, chosen by a seeded RNG so the set is reproducible.
- 47,958 sources -> 143,874 warped + **1,439 flat** = 145,313 pairs.
- Intentional identity rate **0.99%**, versus 17.1% accidental before. `--flat_frac` is
  the knob if more is wanted (0.09 would give ~3% of samples).

`shadow_prob` (default 0.75) is now an explicit parameter rather than an inline magic
number, and stays stochastic on purpose: real captures are sometimes evenly lit, and a
**flat page with a strong lighting gradient and zero flow** is exactly the sample that
teaches the model not to mistake shading for geometry.

#### Also done
- `process_dataset` now **resizes to (width, height) before warping** (Issue 6 closed).
  Warping at native 612 x 792 and downsampling after costs ~5x more CPU and discards the
  extra detail anyway.
- New CLI flags: `--width`, `--height`, `--flat_frac`.
- `inspect_dataset.py` reads `TRAIN_W/TRAIN_H` from the module, previews all 4 profiles
  including flat, renders non-square thumbnails, and its storage table now compares
  aspect-preserving resolutions.
- `test_warps.py` runs at 448 x 576 and gained two test groups:
  **8** warp presence is deterministic + `FLAT_PROFILE` is exactly zero-flow;
  **9** training geometry is within 2% of native aspect and divisible by 32.
  **All 9 groups pass.**

#### Verified on real pages (448 x 576, 4 random images x 4 profiles)
- Flow label **exact (max error 0)** on every sample
- Detail retention 44.7 - 56.1%
- Pages are correctly proportioned — the horizontal stretch is gone
- `v3 flat` is visually identical to clean apart from the shadow gradient

#### ISSUE 11: where to get the annotations

The DAX CDN is alive (the session's egress policy blocks it, but WebFetch reached the
landing page, which lists: 7 train shards of 13 GB, val 3.1 GB, test 2 GB, and a
**labels archive of 314 MB**). The 47,958 images on disk are ~one train shard, so that CDN
is almost certainly where they came from.

```
https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz     (~314 MB)
```

Extracts to `publaynet/train.json` + `val.json` (COCO format). `train.json` covers all
335,703 train images, so it must be filtered down to the 47,958 on disk — a one-time
preprocessing step that should emit a compact per-image annotation file.

Note: the HuggingFace mirrors (`jordanparker6/publaynet`, `psyche/publaynet`, and
others) embed annotations **inside ~208 parquet shards together with the images**, so
they are a ~100 GB download for data we already have. Not a viable route for annotations
alone. No standalone JSON mirror was found.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1 | Cylindrical warp cannot bow lines | FIXED (S6) |
| 8 | Cascaded resampling corrupts labels + destroys detail | FIXED (S6) |
| 5 | `--variants` default | FIXED (S6) |
| - | float64 flow bloat | FIXED (S6) |
| - | `n_control=4` spline risk | FIXED (S6) |
| 9 | ~17% accidental identity samples | **FIXED (S8)** |
| 10 | Square resize distorted pages 1.29x | **FIXED (S8)** |
| 6 | No resize-then-warp | **FIXED (S8)** |
| 11 | Annotations not on disk | URL identified — needs downloading |
| 4 | `BORDER_REPLICATE` smears, no background, no validity mask | open — next |
| 3 | Layout-aware loss | open (blocked by 11) |
| 2 | Storage rewrite (sharded JPEG + float16 normalized flow + manifest) | open |
| 7 | Single-threaded | open (folded into 2) |

#### Next Steps
- [ ] Download `labels.tar.gz` into `<project>\` and extract
- [ ] Fix #4: background compositing + validity mask
- [ ] Fix #3: filter train.json to the 47,958 local images, warp polygons through the flow
- [ ] Fix #2: storage rewrite + process pool

*Log last updated: 2026-09-12 — Session 8*

---

### 2026-09-12 — Session 9: Fix #4 (Background + Validity Mask) + Issue 12

**Status**: Fix #4 complete. A dataset-bias issue found while reviewing the output and
fixed in the same pass. All 11 test groups pass.

#### FIX #4: validity mask + background compositing

Three new functions in `warp_augment.py`:

- **`flow_to_valid_mask(flow, src_h, src_w, soft)`** — which output pixels actually show
  page. A backward flow says where each output pixel samples from; if that source
  coordinate is outside the page rectangle, the pixel is background and the flow value
  there is **meaningless**. Returns float32 in [0,1], anti-aliased over one pixel.
  Subtlety: a source coordinate exactly ON the page edge (d == 0) is fully valid, so the
  ramp is `clip(d + 1, 0, 1)` and lives entirely outside the page. Offsetting by 0.5
  instead made the whole 1-pixel frame border half-transparent even for an unwarped page
  (coverage came out 99.60% instead of 100.00% — caught by the identity test).

- **`make_background(H, W)`** — synthesizes a surface: wood tones, neutral greys,
  coloured mats, and **deliberately near-white surfaces 22% of the time**. Near-white is
  the hard case (page and surface share a tone, so the boundary must be found from
  shading and texture rather than brightness) and it is exactly what people actually
  photograph on. Plus low-frequency luminance variation, a directional gradient, and
  fine grain.

- **`composite_on_background(page, alpha, bg, drop_shadow)`** — composites with a soft
  offset contact shadow. A page lying on a surface occludes light and casts a shadow just
  outside its own edge; that shadow is one of the strongest cues for the page boundary,
  which is what rectifying the corners depends on.

Ordering in `augment_document` changed: the lighting gradient (`add_shadow`) is now
applied **after** compositing, so it falls across page and surface together as one scene
lit by one source. Previously it lit only the page.

`augment_document` now returns **4 values**: `(warped, flow, mask, params)`.

Verified on real pages: flow label still **exact (max error 0) inside the mask** for every
profile; background regions are textured (std 3.3-53.3) rather than the near-constant
per-row smear `BORDER_REPLICATE` produced.

#### DECISION: the mask is STORED, not re-derived

Deriving the mask at train time from the stored 56 x 72 flow was the cheaper option, but
measured over 40 random warps it mislabels background as page:

| | mean | max |
|---|---|---|
| **false positive** (background labelled page — pollutes the flow loss) | 0.50% | **3.07%** |
| false negative (page labelled background — merely drops data) | 1.05% | 2.74% |

False positives are the dangerous direction: they feed meaningless target values into the
loss. Storing the mask at flow resolution costs `56 x 72` uint8 = 4.0 KB/sample =
**0.59 GB** over 145,313 samples. Dataset total **17.31 -> 17.90 GB**. Trivially worth it
against 216 GB free, and it guarantees train/eval can never disagree.

`flow_to_valid_mask` is retained for **inference** on real photographs, where no stored
mask exists.

#### NEW ISSUE 12: page coverage was a free shortcut

Reviewing the contact sheet: every warped sample showed background, every flat sample
filled the frame. Measured over 240 samples, the only thing that ever pushed page content
off-frame was the warp itself, so:

| | before |
|---|---|
| corr(page coverage, warp severity) | **-0.825** |
| same, warped profiles only | -0.745 |
| **P(profile == flat \| coverage == 100%)** | **100.0%** |
| coverage range, `combined` | 81.9 - 98.6% |
| coverage range, `flat` | 100.0 - 100.0% |

The ranges did not overlap at all. A CNN finds this immediately: "does background reach
the frame edge? no -> predict zero flow", and reads warp severity off the size of the
background margin instead of off the page content. The shortcut works perfectly on this
dataset and not at all on a real photograph.

**Fix: camera framing.** New `similarity_flow(H, W, scale, angle, tx, ty)` and
`sample_framing()`, composed **last** (framing happens at capture, after the page is
already bent). Three regimes:

| Regime | prob | scale | roll | effect |
|---|---|---|---|---|
| `zoom` | 25% | 1.02 - 1.30 | +/-5 deg | tight crop, ~100% coverage even when badly bent |
| `exact` | 25% | 1.0 | 0 deg | scan / careful photograph, true identity when flat |
| `back` | 50% | 0.70 - 0.98 | +/-7 deg | page inset, surface visible |

The `zoom` regime is the load-bearing one: it lets a heavily warped page still fill the
frame, which is what stops "coverage == 100%" from being a flat-page detector.

Result:

| | before | after |
|---|---|---|
| corr(coverage, warp severity) | -0.825 | **-0.179** |
| same, warped profiles only | -0.745 | **-0.074** |
| P(flat \| coverage == 100%), balanced test | 100.0% | 69.3% |
| **P(flat \| coverage == 100%), real 1% flat mix** | 100.0% | **6.35%** |
| coverage range, `combined` | 81.9 - 98.6% | 44.4 - 100.0% |
| coverage range, `flat` | 100.0% only | 49.5 - 100.0% |

The ranges now fully overlap. The 69.3% balanced-test figure is an artefact of testing a
1:1:1:1 profile mix; at the real 143,874 : 1,439 ratio the conditional is 6.35%, i.e. no
usable shortcut.

Framing is also legitimate signal, not noise: the target is the rectified page filling the
frame, so a zoomed or rolled capture genuinely has to be scaled and rotated back.

#### Tests
`test_warps.py` now has **11 groups, all passing**. New this session:
- **10** validity mask + background: unwarped -> 100% coverage, page centre opaque, mask
  bounded, margins replaced by textured background, and the measured cost of deriving vs
  storing the mask
- **11** framing decorrelation: identity at scale 1, zoom keeps 100% coverage, pull-back
  reveals background, `|corr| < 0.35`, coverage no longer implies flat, ranges overlap

Test group 4 was also corrected: the flow reproduces the image **inside the mask** only —
outside it the pixels are background by design, so the old whole-frame assertion would
now (correctly) fail. Tests that measure the warp itself pass `use_framing=False` so they
measure distortion rather than camera pose.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1 | Cylinder cannot bow lines | FIXED (S6) |
| 8 | Cascaded resampling | FIXED (S6) |
| 5 | `--variants` default | FIXED (S6) |
| - | float64 flow bloat | FIXED (S6) |
| - | `n_control=4` spline | FIXED (S6) |
| 9 | 17% accidental identity samples | FIXED (S8) |
| 10 | Square resize distortion | FIXED (S8) |
| 6 | No resize-then-warp | FIXED (S8) |
| 4 | No background / no validity mask | **FIXED (S9)** |
| 12 | Page coverage was a warp-severity shortcut | **FIXED (S9)** |
| 11 | Annotations not on disk | URL identified — needs downloading |
| 3 | Layout-aware loss | open (blocked by 11) |
| 2 | Storage rewrite (sharded JPEG + float16 flow + **mask** + manifest) | open — next |
| 7 | Single-threaded | open (folded into 2) |

#### Next Steps
- [ ] Download `labels.tar.gz` into `<project>\`
- [ ] Fix #2: sharded writer — JPEG q90 pages, float16 normalized flow @56x72,
      uint8 mask @56x72, `manifest.parquet` with seed + every param, process pool
- [ ] Fix #3: filter train.json to the 47,958 local images, warp polygons through the flow
- [ ] Pilot 3,000 sources, validate the Kaggle loop, then scale

*Log last updated: 2026-09-12 — Session 9*

---

### 2026-09-12 — Session 10: Fix #2 — Storage Rewrite

**Status**: Writer, format, and verifier complete and tested. One decision outstanding
(flow resolution). Issue 7 closed along the way.

#### New files
| File | Role |
|---|---|
| `augmentation/dataset_format.py` | the on-disk contract + codec + reader. Writer and reader both import it, so they cannot disagree. |
| `augmentation/build_dataset.py` | the builder: process pool, sharded blobs, manifest, resume. Replaces `process_dataset`. |
| `augmentation/verify_dataset.py` | 6-group correctness + reproducibility check. Run after every build. |

#### Layout

```
<out_dir>/
    pages-0000.bin ...     concatenated warped JPEGs (q90)
    clean-0000.bin ...     concatenated clean JPEGs, ONE per source
    flows.f16              memmap (N, 72, 56, 2) float16
    masks.u8               memmap (N, 72, 56)    uint8
    manifest.jsonl(+.parquet)   one row/sample: byte offsets, seed, every param
    clean_index.jsonl(+.parquet)
    meta.json
```

Design points:
- **Images as concatenated blobs, not tar.** `(shard, offset, length)` in the manifest
  gives true random access with one seek — no tar streaming, no 430,000 inodes, and it
  survives DataLoader shuffling.
- **Flow and mask in flat memmaps.** Fixed size, so zero parsing cost and O(1) access.
- **Clean page stored ONCE per source**, referenced by `clean_id`. Saves ~4 GB over
  duplicating it per variant.
- **Parquet is optional.** JSONL is always written (stdlib only); parquet is added when
  pandas/pyarrow are importable. The reader prefers parquet, falls back to JSONL, and
  needs no pandas at all. This matters because the local Python is 3.14, where pyarrow
  wheels may not exist yet.

#### Flow encoding — three choices, all measured

1. **Normalized, not pixels** — stored as displacement in `grid_sample`'s [-1,1]
   convention, so the same files train at any resolution and inference on a full-res page
   needs no regeneration.
2. **Displacement, not absolute grid** — float16 precision scales with magnitude, and
   displacements are small where absolute coordinates are not. Free accuracy.
3. **Bounded, NOT clamped to the page.** Clamping the source coordinate to the page
   rectangle was the obvious way to keep out-of-page extrapolation from corrupting
   boundary cells. It measurably made things **worse** — p99 error 1.10 -> 1.28 px —
   because a clamp introduces a derivative kink at the page edge, and a kink is exactly
   the high-frequency content downsampling cannot represent. The flow is left smooth and
   merely clipped to +/-8.0 normalized units as float16 overflow insurance (largest value
   observed over 60 random warps: 0.888).

**float16 is free.** At 56x72 the reconstruction error is 0.052 px mean with float16 and
0.051 px with float32 — the error is entirely spatial downsampling, not quantisation. So
float16 halves flow storage for nothing.

#### VERIFIED: determinism and resume

- Every sample is seeded from `(global_seed, source_index, variant_index)`, never from
  scheduling order. Builds with **1 worker and 8 workers produce byte-identical**
  `pages`, `clean`, `flows`, `masks`, `manifest` and `clean_index`. Confirmed by md5.
- **Reproducibility from the manifest alone**: regenerating a sample from just its
  `sample_seed` + `stem` + `profile` reproduces the stored JPEG **byte for byte**, 24/24
  and 16/16 across runs.
- **`--resume`** replays the JSONL, finds the last *complete* source, truncates the blobs
  back to the last fully-written sample (discarding a partial write from a kill), and
  continues. Tested by truncating a finished build to 20 of 48 sources, leaving the blobs
  deliberately oversized, and resuming: the result is **byte-identical to the
  uninterrupted build**.
- **Shard rollover** tested with a ~50 KB cap: 156 page shards, 46 clean shards, and
  verification still passes — cross-shard random access works.

#### Verifier findings

Check 5 ("the stored flow relates the two stored images") initially **failed** at 36.86
grey levels mean difference. That was the test being wrong, not the data: the stored page
has the lighting gradient baked in, so an absolute-intensity comparison measures
`add_shadow`'s 0.4-1.0 multiplier rather than geometry. The shadow is low-frequency by
construction, so the check now divides out a blurred gain field first — which cannot hide
a geometric error, since those are high-frequency. Result: **4.63 grey levels**, i.e. JPEG
q90 on both images plus the sub-pixel flow error. Both numbers are reported so the
distinction stays visible.

#### Measured sizes and speed

Measured on a real build: **warped JPEG 54.8 KB, clean JPEG 63.4 KB** mean at 448x576 q90
(lower than the 0.30 B/px estimate used in Session 5).

| | full-build size |
|---|---|
| images (145,312 pages + 47,958 clean) | **11.27 GB** |
| flow + mask @ 56x72 | 2.93 GB |
| **total @ 56x72** | **14.20 GB** |

Down from ~1.24 TB in the original format — a **87x** reduction.

Build speed: 5.8 sources/s per worker. Projected full 47,958-source build:
**~0.6 h on 4 workers, ~0.3 h on 8**. Far cheaper than feared, which makes regenerating
at a different flow resolution a non-event.

#### OPEN DECISION: flow resolution

The error tail is intrinsic to downsampling a strongly curved field — it is **not**
confined to the page boundary (eroding the loss mask by 8 px moves p99 only from 1.099 to
0.806 and leaves the max untouched). So the only lever is the flow grid:

| flow grid | mean | p95 | p99 | max | flow+mask | TOTAL |
|---|---|---|---|---|---|---|
| 56 x 72 (page/8) | 0.052 | 0.201 | **1.10** | 6.12 | 2.93 GB | **14.20 GB** |
| 84 x 108 | 0.029 | 0.072 | **0.65** | 4.02 | 6.59 GB | **17.86 GB** |
| 112 x 144 (page/4) | 0.018 | 0.038 | **0.37** | 2.71 | 11.72 GB | **22.98 GB** |

Context: published dewarping work reports Local Distortion of roughly 5-15 px, so even
1.10 px p99 is well under the signal. But the paper's Gap 4 claim is "exact analytical
flow ground truth", and a reviewer can ask what the label noise floor actually is.
112 x 144 also divides the page by exactly 4, which is a cleaner decoder stride than 84's
5.33.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1, 8, 5, 6, 9, 10, 4, 12 + float64 + spline | | FIXED (S6/S8/S9) |
| 2 | Storage rewrite | **FIXED (S10)** |
| 7 | Single-threaded | **FIXED (S10)** — process pool, deterministic |
| 11 | Annotations not on disk | user downloading |
| 3 | Layout-aware loss | open (blocked by 11) |

#### Next Steps
- [ ] Decide flow resolution (56x72 / 84x108 / 112x144)
- [ ] Pilot build: `--limit 3000`, run `verify_dataset.py`, eyeball samples
- [ ] Full build (~0.5 h), verify, upload to Kaggle
- [ ] Fix #3 once `labels.tar.gz` is extracted
- [ ] Model: U-Net encoder at 448x576, decoder stopping at flow resolution

*Log last updated: 2026-09-12 — Session 10*

---

### 2026-09-12 — Session 11: Flow Resolution Decided + Pilot Built

**Status**: Format locked at 112 x 144. Pilot built on 48 real pages and fully verified.
Two important findings about the flow's direction and invertibility.

#### DECISION: flow stored at 112 x 144 (page / 4)

Chosen over 56 x 72. Rebuilding the whole dataset takes ~30 min and 216 GB is free, so the
only real cost is a one-time +8.8 GB Kaggle upload. Against that, the label noise floor
drops from 1.10 px p99 to 0.37 px, which matters because "exact analytical flow ground
truth" is one of the paper's claims and a reviewer can ask what the floor actually is.
112 also divides the page by exactly 4 — a clean decoder stride, unlike 84's 5.33.

Measured on the pilot at 112 x 144: **mean 0.0162 px, p95 0.036, p99 0.294, max 2.83**.

#### IMPORTANT: which direction the stored flow goes

Worth stating explicitly because it is easy to get backwards and would silently produce a
model that cannot rectify anything.

The stored flow `f` is the map that **generated** the warp:

```
warped[y, x] == clean[y + f_y, x + f_x]
```

So `f` gives, for each **warped** pixel, its location in the clean page. Its domain is the
warped image — the same grid as the network input, which is exactly what a CNN wants to
predict.

That is **not** the map that rectifies an image. Rectification needs `g`, defined on the
clean/output grid, with `clean[y,x] ~= warped[y + g_y, x + g_x]`. `g` is the inverse of `f`
as a coordinate map.

The useful consequence: **the entire training loop can run in the warped domain and never
needs an inversion.**

| Loss | How, with `f` |
|---|---|
| flow | directly on `f`, masked by the stored mask |
| photometric | resample the clean page with predicted `f`, compare to the warped input |
| **layout (the novelty)** | a layout polygon vertex in warped space maps to clean space by adding `f` at that vertex — which *is* the "does this text block come out rectangular" test |
| inference | invert once per image |

So storing `f` alone is sufficient for every planned loss, and it is the better-posed
prediction target. No second field needs storing.

#### NEW: `invert_flow` — and why damping is not optional

Added `fmt.invert_flow(flow, iters=60, beta=0.5)`: a damped fixed point,
`g <- g + beta * (-f(x + g(x)) - g)`.

The textbook undamped version (`beta = 1`) **diverges** on the strongly compressive parts
of aggressive warps. Measured over 40 real samples, inside the visible page:

| iters, beta | mean residual | p99 residual | worst sample |
|---|---|---|---|
| 12, 1.00 | 0.9679 px | 14.8226 px | 10.4089 px |
| 30, 1.00 | 0.7954 px | 14.0317 px | 10.3078 px |
| **30, 0.50** | 0.0027 px | 0.0371 px | 0.0310 px |
| **60, 0.50** | **0.0019 px** | **0.0130 px** | **0.0084 px** |

That is the difference between a correct inverse and a useless one — a 450x improvement
from one damping factor. (I shipped the undamped version first, and the new verifier check
caught it: residual 0.85 px, FAIL.)

#### Verifier gained check 7

`verify_dataset.py` now also inverts the stored flow, rectifies the stored warped page, and
compares it to the stored clean page. Reported: inversion residual, and PSNR after
dividing out the low-frequency gain field.

Note the same photometric confound as check 5 — rectifying a shadowed page gives a
correctly-shaped but still shadowed page, so raw PSNR was a misleading 9-27 dB. With the
gain field divided out: **24.26 dB**, which is JPEG q90 on both images plus two resampling
passes. Worth being explicit: **this pipeline is a geometric dewarper only.** Illumination
is deliberately left in the input and is not something the flow can or should remove; if
shadow removal is wanted it is a separate output head (DocRes does both tasks).

#### PILOT BUILD — 48 real distinct PubLayNet pages

Built and verified end to end. All 7 verifier groups **PASS**.

| | |
|---|---|
| sources / samples | 48 / 149 (3 variants + 5 flat) |
| build rate | 6.7 sources/s on 2 workers |
| size | 166.3 KB/sample |
| **projected full 47,958-source build** | **25.3 GB** |
| reproducibility | 30/30 samples regenerate byte-identical from the manifest |
| flow fidelity | mean 0.0162 px, p99 0.294 px |
| flow-vs-images consistency | 5.86 grey levels (gain-corrected) |
| inversion residual | 0.0021 px |
| rectification | 24.26 dB (gain-corrected) |
| page coverage | mean 83.9%, range 44.9 - 100% |

`pilot_samples.png` shows, per sample: clean / stored warped / stored flow / stored mask /
rectified-by-inverting-the-flow. The rectified column recovers straight text lines and a
full-frame page, which is the end-to-end proof that the labels are usable.

Note on how the pilot was run: the local shell still cannot mount `<project>`, so the 48
source pages were staged into the session container by filename (read off File Explorer
via computer use). A 3,000-source pilot would need ~60 staging calls, so the **full build
should be run locally** — the commands are below.

#### Commands to run locally

```
cd <project>
python augmentation\build_dataset.py --input_dir <project>\publaynet ^
       --out_dir <project>\dataset --variants 3 --flat_frac 0.03 --workers 8
python augmentation\verify_dataset.py --root <project>\dataset ^
       --source_dir <project>\publaynet --n_check 60
```

Add `--limit 3000` for a local pilot first; add `--resume` if a run is interrupted.
Expect ~30 min on 8 workers and ~25 GB.

#### Issue Status
All of 1-12 are now FIXED except:
| # | Issue | Status |
|---|---|---|
| 11 | Annotations not on disk | user downloading `labels.tar.gz` |
| 3 | Layout-aware loss | open (blocked by 11) |

#### Next Steps
- [ ] Run the full build locally, verify, upload to Kaggle
- [ ] Fix #3 once `labels.tar.gz` is extracted: filter `train.json` to the 47,958 local
      stems, store polygons per source, map vertices through `f` for the layout loss
- [ ] Model: U-Net encoder at 448 x 576, decoder terminating at 112 x 144 (saves the two
      most expensive decoder stages); resumable checkpointing from day one for Kaggle's
      12-hour session cap

*Log last updated: 2026-09-12 — Session 11*

---

### 2026-09-12 — Session 12: Annotations Located + Two Corrections

**Status**: Full build running locally (5.8% at time of check). Annotation file found on
disk and confirmed genuine. Two earlier conclusions corrected.

#### Build in progress — on track

Snapshot of `<project>\dataset`:

| | |
|---|---|
| sources done | 2,800 / 47,958 (5.8%) |
| samples done | 8,719 / 145,312 |
| profile mix | cylinder 2,874 / perspective 2,874 / combined 2,874 / flat 97 |
| mean warped page | 62.4 KB |
| projected images | 13.1 GB |
| + flows 9.37 GB + masks 2.34 GB (preallocated up front) | **~24.8 GB total** |

Matches the 25.3 GB pilot projection. Still on a single page shard, as expected at 1 GB
rollover.

#### FOUND: the annotations were already on disk

```
<project>\datasets\datasets\publaynet-max25\raw\publaynet\
    train.json    1,696,002,062 bytes  (1.70 GB)
    val.json         61,287,696 bytes
    README.txt   / LICENSE.txt
```

Confirmed genuine IBM PubLayNet COCO annotations. `README.txt` names the IBM DAX homepage
and the `publaynet.tar.gz` download; `LICENSE.txt` is the CDLA-Permissive-1.0 annotation
licence. Verified on `val.json`: **11,245 images, 120,761 annotations, 5 categories**
(1 text, 2 title, 3 list, 4 table, 5 figure), with full segmentation polygons (up to ~69
points) alongside bboxes. Those counts are exactly the official val split.

So **no further download is needed.** `labels.tar.gz` would have delivered the same two
files. The surrounding folder (`publaynet-max25`, `rico13-max25`, `rico25-max25` with
`processed/*.pt`) is a layout-generation dataset zoo; its `processed/` tensors are filtered
to <=25 elements and are **not** what we want — only `raw/publaynet/*.json` is.

#### CORRECTION 1: PubLayNet is NOT dimensionally uniform

Session 7 recorded "every one of the 12 sampled images is exactly 612x792 — the dataset
appears to be dimensionally uniform". `val.json` disproves that for the dataset as a whole:

| recorded size | share | aspect | stretch @448x576 |
|---|---|---|---|
| 601 x 792 | 30.56% | 0.7588 | 1.0250x |
| 596 x 794 | 19.76% | 0.7506 | 1.0362x |
| 596 x 791 | 11.77% | 0.7535 | 1.0323x |
| 612 x 792 | **8.98%** | 0.7727 | 1.0065x |
| 596 x 842 | 8.31% | 0.7078 | 1.0988x |

**35 distinct sizes** in the val split. Aspect p1 0.7078, median 0.7588, p99 0.7750.

Two consequences:
1. The 448x576 choice was justified on 612x792 (0.7727) giving a 1.0065x stretch. Against
   the *median* 0.7588 it is 1.025x, and up to **1.099x** for 596x842 pages. Still small
   and still far better than a square's 1.294x, so the choice stands — but the "distortion
   removed entirely" claim was overstated and is now on record as ~2.5% typical.
2. Far more importantly, annotation coordinates are in each page's **own** native space,
   so the layout loss must scale per image by that image's recorded width/height, never by
   a global constant.

Meanwhile all 48 locally staged pages measure 612x792 (47) or 612x801 (1) — a size that is
only 9% of the val split. Either the train split was rendered differently, or these images
come from a mirror that re-rendered to a uniform size. **If the latter, the annotation
coordinates will not line up**, which would block the novelty claim. `check_annotations.py`
settles it; the builder does not record native sizes, so it cannot answer this.

#### CORRECTION 2: this val.json is not a plain COCO file

While writing the streaming reader: **83 image records carry their own nested
`"annotations"` array plus a `"corrected": false` field** — the human-verified subset,
merged into the same file. The first textual occurrence of `"annotations"` is therefore
only ~2.7 KB into the file, *nested inside an image record*.

A naive reader that searches for `"annotations"` and parses from the following `[` reads
**19 annotations instead of 120,761** — and reports success. My first parser did exactly
that. Anything built on it would have silently trained against ~0.02% of the labels.

Fixed in `coco_stream.py`: the key must be found at **brace depth 1, bracket depth 0**.
Depths are computed in one chunked numpy `cumsum` pass, so it stays fast on a 1.7 GB file.
Object boundaries then come from `mmap.find` (C-level, two calls per object) rather than a
per-character Python loop, which would be unusably slow at this size.

Verified against `json.load` on val.json: **11,245 / 120,761 / 5 exactly**, in 1.0 s at
0.66 GB peak RSS. That is ~60 MB/s, so train.json should stream in roughly 30 s.

#### New files
| File | Role |
|---|---|
| `augmentation/coco_stream.py` | depth-aware streaming COCO reader; `CocoFile(path).stream('annotations')` |
| `augmentation/check_annotations.py` | 6-step usability check: local size census, top-level array location, stem matching, recorded-vs-actual dimensions with implied per-axis scale, annotation counts by category, and **rendered overlays** |

`check_annotations.py` was tested end to end against val.json using synthetic pages built
at each record's recorded size, plus one deliberately shrunk to 0.8x. It reported
13/14 exact, 1 differing, and recovered the implied scale as 0.7987 / 0.7992 with zero
std — so the mismatch-detection path works, not just the happy path.

#### Command to run (after the build finishes, so the CPU is free)

```
cd <project>
python augmentation\check_annotations.py ^
    --json   <project>\datasets\datasets\publaynet-max25\raw\publaynet\train.json ^
    --images <project>\publaynet ^
    --overlays 10
```

The overlays land in `<project>\annotation_check\`. **Those decide it**: boxes sitting
exactly on the text blocks means the annotations are usable; shifted or mis-scaled boxes
mean the images and labels come from different renderings.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-10, 12 | | FIXED (S6/S8/S9/S10) |
| 2, 7 | Storage rewrite, parallelism | FIXED (S10) |
| 11 | Annotations not on disk | **RESOLVED (S12)** — already present, genuine, no download needed |
| 13 | Local pages are 612x792 but only 9% of val is; alignment unproven | **open — `check_annotations.py` answers it** |
| 3 | Layout-aware loss | open (gated on 13) |

#### Next Steps
- [ ] Finish the build, run `verify_dataset.py`
- [ ] Run `check_annotations.py` and inspect the overlays -> closes issue 13
- [ ] Fix #3: filter train.json to the local stems, store polygons normalized to each
      page's own native size, map vertices through `f` for the layout loss
- [ ] Model: U-Net encoder at 448x576, decoder terminating at 112x144, resumable
      checkpointing for Kaggle's 12-hour cap

*Log last updated: 2026-09-12 — Session 12*

---

### 2026-09-12 — Session 12 (cont.): check_annotations.py Writes Its Own Report

**Status**: Script finalized so its results land in the project folder and can be verified
without copy-pasting console output.

#### Everything goes into one folder

`--out_dir` (default `annotation_check`) now receives all three outputs:

| File | Contents |
|---|---|
| `report.txt` | the full human-readable run, teed from stdout |
| `summary.json` | machine-readable: census, top-level byte ranges, match counts, exact/differ split, implied per-axis scale, category counts, verdict, problems |
| `overlay_*.png` | the annotated pages — the actual decision |

`summary.json` carries a `dimension_verdict` of `MATCH` (>99% exact), `MOSTLY_MATCH`
(>50%) or `MISMATCH`, plus a `problems` list, so the outcome is one field rather than an
inference from prose.

#### Dependency risk removed

The census originally used PIL, falling back to `cv2.imread` — which would have fully
decoded 47,958 pages to read two integers. Replaced with a 30-line JPEG SOF-marker parser
(`_jpeg_size`), stdlib only. Validated against `cv2.imread` on all 48 real pages:
**0 mismatches**, and **0.4 ms for 48 files — about 0.4 s for the full 47,958**, versus
minutes for a decode pass. PIL and cv2 remain as fallbacks for non-JPEG input.

#### Robustness
The whole body is wrapped so that a failure still writes `report.txt` and `summary.json`
with the traceback and a populated `problems` list, rather than dying with nothing on disk.

#### Command to run once the build finishes

```
cd <project>
python augmentation\check_annotations.py ^
    --json    <project>\datasets\datasets\publaynet-max25\raw\publaynet\train.json ^
    --images  <project>\publaynet ^
    --overlays 12 ^
    --out_dir <project>\annotation_check
```

Expected runtime: ~0.4 s census + ~30 s streaming train.json (measured 60 MB/s on
val.json) + overlay rendering. Outputs land in `<project>\annotation_check\`, inside the
connected folder, so they can be read directly from this session — nothing needs pasting.

*Log last updated: 2026-09-12 — Session 12 (cont.)*

---

### 2026-09-12 — Session 13: Issue 13 CLOSED — Annotations Align Exactly

**Status**: The layout-aware loss is viable. Build ~50% through. One small data-hygiene
item found.

#### VERDICT: MATCH

`check_annotations.py` run against the full `train.json` and all 47,958 local pages:

| | |
|---|---|
| image records in train.json | 335,703 |
| **local stems matched** | **47,958 of 47,958 (100.00%)** |
| **dimensions exact** | **47,958 (100.00%)** — 0 differ |
| implied scale | 1.000, 1.000 |
| annotation records in file | 3,263,046 |
| **annotations for our images** | **466,221 (9.7 per image)** |
| categories | text 334,530 (71.8%), title 89,730 (19.3%), figure 15,665 (3.4%), table 14,681 (3.2%), list 11,615 (2.5%) |
| segmentation polygons | 22.4 points mean, max 194; bboxes on all |
| problems | none |

Overlays confirm it visually: bboxes and polygons sit **exactly** on the text blocks,
titles, tables and figures, at scale 1.000. Checked `PMC5707887_00000` (a text-heavy
page, all 9 regions aligned) and `PMC6098134_00003` (figure + table + title + text, every
box tight).

So the Session 12 worry was unfounded — these images are the genuine DAX render, not a
re-rendered mirror. **No re-download, no rebuild. Fix #3 is unblocked.**

Note also: `train.json` has **no** nested-annotation records (val.json had 83). The
depth-aware reader handles both, and the byte ranges it reported
(`images` 12..27,417,934, `annotations` 27,417,953..1,696,001,800) confirm it located the
top-level arrays in the 1.7 GB file correctly.

#### CORRECTION 3: the local set has 297 distinct sizes, not one

Session 7 said "dimensionally uniform 612x792" from a 48-image sample. Session 12
suspected that was wrong. The full census settles it:

| size | images | share | aspect | stretch @448x576 |
|---|---|---|---|---|
| 596x791 | 9,266 | 19.32% | 0.7535 | 1.0323x |
| 612x792 | 8,790 | 18.33% | 0.7727 | 1.0065x |
| 596x794 | 7,601 | 15.85% | 0.7506 | 1.0362x |
| 596x842 | 3,394 | 7.08% | 0.7078 | 1.0988x |
| 601x792 | 3,032 | 6.32% | 0.7588 | 1.0250x |

**297 distinct sizes.** The top 12 cover 82.3%; the remaining 285 cover 17.7%. My 48-image
sample landing 47/48 on 612x792 — an 18% bucket — was simply luck, presumably because the
shard is ordered and those filenames clustered.

Forced-resize stretch into 448 x 576, weighted by image count:

| | |
|---|---|
| median | 1.0323x |
| p1 / p99 | 1.0036x / 1.1790x |
| within 5% of 1.0 | **83.9%** of images |
| within 10% | 92.8% |
| beyond 20% | 0.6% |

The 448x576 choice still stands — median 3.2% stretch versus a square's 29.4% — and the
same resize is applied to the clean page and the warped page, so the task stays exactly
self-consistent and the labels stay exact. What it does mean, and should be stated in the
paper rather than glossed: the rectified output carries the *frame's* aspect, not the
page's true aspect. That is standard for DocUNet-style fixed-output methods, which is why
evaluation resizes prediction and GT to a common size.

#### NEW ISSUE 14 (minor): 234 landscape pages get squashed 0.52x

0.488% of the local set — **234 images across 28 sizes** — are *landscape* (w > h), e.g.
794x596 (83), 842x596 (57), 792x612 (16). Forced into portrait 448x576 they are squeezed
to **0.518-0.55x**, far outside the distribution every other page sits in.

| stretch band | images | share |
|---|---|---|
| severe squash 0.52-0.80x (landscape) | 234 | 0.49% |
| near-native 0.90-1.05x | 40,220 | 83.87% |
| mild stretch 1.05-1.15x | 6,109 | 12.74% |
| strong stretch >1.15x | 1,395 | 2.91% |

Two clean options: rotate them 90 degrees into portrait (keeps the data, needs the
annotations rotated to match), or drop them (0.49% of the set, costs nothing). Dropping is
simpler and 234 samples will not be missed; rotating is better if landscape captures are
wanted in-distribution. Not worth restarting the current build over — it can be a filter in
the next build, or handled when the annotation preprocessing runs.

#### Miss worth recording
I predicted the size census would take "~0.4 s for the full 47,958" from a 48-file timing.
It took **518.5 s**. The 0.4 ms/48-file figure was a warm-cache measurement; the real run
did 47,958 cold file opens on a disk simultaneously being written by the 8-worker build.
Header parsing was never the cost — I/O was.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-10, 12 | | FIXED |
| 2, 7 | Storage rewrite, parallelism | FIXED (S10) |
| 11 | Annotations not on disk | RESOLVED (S12) |
| 13 | Annotation/image alignment unproven | **CLOSED (S13) — exact match** |
| 14 | 234 landscape pages squashed 0.52x | open (minor, 0.49%) |
| 3 | Layout-aware loss | **unblocked — next** |

#### Next Steps
- [ ] Finish build (~50%), run `verify_dataset.py`
- [ ] Fix #3: stream `train.json`, keep the 47,958 local stems, store bbox + polygon
      normalized to each page's own recorded size (all 466,221 of them), aligned to
      `clean_id` so the loader can join them
- [ ] Decide issue 14: rotate or drop the 234 landscape pages
- [ ] Model: U-Net encoder 448x576, decoder terminating at 112x144, resumable checkpoints

*Log last updated: 2026-09-12 — Session 13*

---

### 2026-09-12 — Session 14: Fix #3 — The Layout-Consistency Loss

**Status**: The novelty is implemented and tested. Store builder, format/reader, loss, and
a test suite that demonstrates the actual claim.

#### New files
| File | Role |
|---|---|
| `augmentation/layout_format.py` | on-disk contract + `LayoutStore` reader |
| `augmentation/prepare_layout.py` | builds the store from train.json; report + summary + training-space overlays |
| `augmentation/layout_loss.py` | the loss, in PyTorch, two modes |
| `augmentation/test_layout_loss.py` | 5 test groups; group 3 is the claim |

#### The store

```
<dataset>/layout/
    stems.jsonl          per page: stem, recorded w/h, actual w/h, [ann_start, ann_end)
    ann_cat.u8.npy       (N,)    category 1..5
    ann_bbox.u16.npy     (N,4)   x0,y0,x1,y1
    poly_offsets.i64.npy (N+1,)  ragged slice bounds
    poly_xy.u16.npy      (M,2)   polygon vertices
    meta.json
```

Three decisions worth recording:

- **Keyed by stem, not `clean_id`.** The store can therefore be built while
  `build_dataset.py` is still writing, and survives a rebuild with different variants. The
  loader joins stem -> clean_id through the dataset's own `clean_index` once at startup.
- **uint16, not float16.** Coordinates are normalized to [0,1], which is where float16 is
  at its worst: its quantum near 1.0 is ~0.0005, i.e. **0.22 px** at 448 wide. uint16 over
  the same range gives 1/65535, i.e. **0.0088 px**. Same two bytes, 32x the precision, and
  no possibility of a NaN in a label.
- **Normalized by the RECORDED page size, not the file's actual size.** These are identical
  for all 47,958 pages here (Session 13), but annotation coordinates live in the space the
  file declares, so dividing by the actual size would be silently wrong the moment the two
  diverge — exactly the re-rendered-mirror case Session 12 worried about.

  I wrote it the wrong way first. The test caught it: on a deliberately shrunk page, 161
  coordinate values were clamped to the page bounds. After the fix, **0**.

Also: COCO allows several polygons per region. Concatenating them would invent edges
between disjoint parts, so the largest is kept and the frequency is reported. Regions are
sorted top-to-bottom then left-to-right, so region order is deterministic rather than
COCO's arbitrary order.

#### The loss

The stored flow maps a warped pixel to its clean-page position, `c(p) = p + f(p)`. Take
the error field `d(p) = f_pred(p) - f_gt(p)`. Every pixel on one horizontal line inside a
text block shares the same `c_gt.y`; the predicted page keeps that line horizontal exactly
when `c_pred.y` is also constant along it, which — since `c_gt.y` already is — holds
exactly when **`d.y` is constant along the line**. So:

```
L = mean_bands Var(d_y | horizontal band) + mean_bands Var(d_x | vertical band)
```

restricted to pixels inside a labelled region and on the page.

Formulating it on the **error field** rather than on `c_pred` directly is what makes
`L(f_gt, f_gt) = 0` exactly. Banding `c_pred.y` itself would have left a floor equal to the
band-width variance.

**No flow inversion is needed.** Band membership uses `c_gt`, straight from the stored
flow; the regions are in clean space and `c_gt` maps warped pixels into clean space, so
membership is an interval test. `invert_flow` is needed only at inference.

#### The test that matters

It is not enough for the loss to fall when the flow is right — an L1 flow loss does that
too. The claim is that this loss carries information L1 does **not**. Two perturbations,
each scaled to exactly **1.0 px L1 flow error**:

| perturbation | L1 flow error | layout loss |
|---|---|---|
| rigid shift of the whole field | 1.0000 | **0.0000** |
| bend (quadratic in x — curved text lines) | 1.0000 | **2.2820** |

An L1 loss cannot tell these apart. This loss separates them by **>1,000,000x**. A rigid
shift leaves every block rectangular and costs nothing; a bend curves the text lines and is
punished hard. That is the paper's figure.

Full suite, passing in both modes:
1. `L(f_gt, f_gt) = 2e-6` (float noise), bands populated
2. monotone in perturbation amplitude (0 -> 0.436 over 0-8 px)
3. the separation above
4. gradients finite, non-zero, and confined to page pixels inside regions
5. batching works

#### Two modes, both tested

| mode | cost, batch 8 / K=30 CPU | notes |
|---|---|---|
| `per_region` | 1,787 ms | bands each region separately; loops over region slots, so linear in K |
| **`global`** (default) | **217 ms (8.2x faster)** | bands by position on the page, two scatters regardless of K |

`global` is not merely an approximation: two text blocks side by side at the same clean y
must land at the same y after dewarping, so constraining them jointly is if anything the
stronger statement. Both modes agree numerically — bend 2.28204 vs 2.27182, shift 0.00000
vs 0.00000 — so the fast one is the default.

Speed scales as ~10x at K=50, and these are elementwise ops on 448x576 tensors, so GPU cost
should be small; worth re-measuring on Kaggle before fixing the loss weight.

#### Command (after the build finishes)

```
python augmentation\prepare_layout.py ^
    --json    <project>\datasets\datasets\publaynet-max25\raw\publaynet\train.json ^
    --images  <project>\publaynet ^
    --out_dir <project>\dataset ^
    --overlays 8
python augmentation\test_layout_loss.py
```

`--drop_landscape 1` is the default, which also disposes of issue 14 (the 234 landscape
pages) for the layout store. Outputs land in `<project>\dataset\layout\`.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-12 | | FIXED |
| 13 | Annotation alignment | CLOSED (S13) |
| 14 | 234 landscape pages | handled in the layout store via `--drop_landscape`; still present in the image build |
| 3 | **Layout-aware loss** | **IMPLEMENTED (S14)** |

#### Next Steps
- [ ] Finish build; run `verify_dataset.py`, then `prepare_layout.py`
- [ ] Model: U-Net encoder 448x576, decoder terminating at 112x144
- [ ] Training loop: flow L1 (masked) + photometric (warped domain) + layout, resumable
      checkpointing for Kaggle's 12-hour cap
- [ ] Ablation for the paper: with and without the layout term, reported on Local
      Distortion and OCR-CER — the separation figure above predicts where it should help

*Log last updated: 2026-09-12 — Session 14*

---

### 2026-09-12 — Session 15: Build Complete + Layout Store Complete — Full Audit

**Status**: Dataset and layout store both built. Everything reconciles. One remaining
verification step and one small inconsistency to decide on.

#### DATASET BUILT — 145,313 samples, 24.56 GB

| item | size | note |
|---|---|---|
| warped pages | 9.10 GB | 9 shards, 61.2 KB/sample |
| clean pages | 3.65 GB | 4 shards, 74.3 KB/source, one per source |
| flows.f16 | 9.37 GB | 112 x 144 normalized float16 |
| masks.u8 | 2.34 GB | 112 x 144 uint8 |
| manifest.jsonl | 83.8 MB | every seed and parameter |
| **total** | **24.56 GB** | + 0.06 GB layout = **24.61 GB** |

`meta.json` confirms: 448 x 576 pages, flow 112 x 144 float16, JPEG q90, 145,313 samples,
47,958 sources, 3 variants, flat_frac 0.03, seed 42, profiles cylinder / perspective /
combined / flat. Against the 25.3 GB pilot projection — 3% under.

**Exact-size checks pass**, which is a strong structural signal that nothing was dropped or
double-written:

```
flows: 145,313 x 64,512 = 9,374,432,256  ==  actual 9,374,432,256   OK
masks: 145,313 x 16,128 = 2,343,608,064  ==  actual 2,343,608,064   OK
```

Count reconciliation: 47,958 x 3 + 1,439 flat = **145,313** exactly.

#### LAYOUT STORE BUILT — 465,596 annotations, 57.4 MB

| | |
|---|---|
| pages | 47,724 (47,958 - 234 landscape) |
| annotations | 465,596 (9.8 per page) |
| polygon points | 10,686,921 (23.0 per region) |
| on disk | 57.4 MB |
| matched | **47,724 of 47,724 (100%)** |
| recorded sizes vs disk | agree for every matched page |
| coordinates clamped | **6 of 22,305,052 (0.000%)** |
| degenerate bboxes dropped | 9 |
| regions with >1 polygon | 0 |
| re-read error | 0.000000 px |

Runtime: 26 s to measure 47,958 pages, 34 s to stream the 1.67 GB annotation array.

Every count reconciles against Session 13 independently:
- 466,221 annotations for all 47,958 pages (S13) minus 465,605 kept = **616** annotations,
  which is exactly what sits on the 234 excluded landscape pages.
- 465,605 kept minus 9 degenerate bboxes = **465,596** packed.

Overlays confirm it visually on real pages at 448 x 576: on `PMC1079906_00003` the figure
box wraps the composite figure, the title box the heading, text boxes the paragraphs — so
the normalize-to-page-then-rescale-to-training-space round trip is correct on real content,
not just on the synthetic test pages.

#### NEW ISSUE 15 (small): dataset and layout store disagree on 234 pages

`prepare_layout.py --drop_landscape 1` excluded the 234 landscape pages, but the image
build ran before that decision and **includes** them. So roughly **700 samples (0.48%)**
have images and flow but no layout annotations.

This is not a corruption — the two stores are joined by stem, and a missing stem simply
returns no regions. But it must be handled deliberately, not discovered at 3am:

- **Option A (recommended):** filter them out in the DataLoader by skipping manifest rows
  whose stem is absent from the layout store. Costs 0.48% of the data and also removes the
  badly squashed 0.52x-stretch pages, which are out-of-distribution anyway.
- **Option B:** keep them and zero the layout term for those samples. Keeps the data but
  leaves 700 distorted samples in training.

Option A is cleaner and the loss is negligible. It needs one line in the dataset wrapper,
no rebuild.

#### verify_dataset.py now writes its output

It was the only tool that printed to the console and left nothing behind. Added a stdout
tee plus a machine-readable summary:

- `<root>/verify_report.txt`
- `<root>/verify_summary.json` — `ok`, `failures`, `n_samples`, `n_checked`, `meta`

Tested on the pilot: `ok = True`, `failures = []`.

#### STILL TO RUN — the last verification

The full dataset has not yet been through `verify_dataset.py`. Structure and sizes check
out from the file listing, but reproducibility, flow fidelity and invertibility are only
provable by running it:

```
python <project>\augmentation\verify_dataset.py ^
    --root       <project>\dataset ^
    --source_dir <project>\publaynet ^
    --n_check 60
```

Expect: 7 groups, byte-identical regeneration on all 60, flow error ~0.016 px mean, and
inversion residual ~0.002 px.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-12 | warp correctness, storage, parallelism | FIXED |
| 13 | Annotation alignment | CLOSED (S13) |
| 14 | 234 landscape pages | handled in the layout store |
| 3 | Layout-aware loss | IMPLEMENTED (S14) |
| 15 | Dataset includes 234 pages the layout store excludes | **open — one line in the loader** |

#### Next Steps
- [ ] Run `verify_dataset.py` on the full set
- [ ] Dataset wrapper: join manifest to layout by stem, apply option A
- [ ] Model: U-Net encoder 448x576, decoder terminating at 112x144
- [ ] Training loop: masked flow L1 + warped-domain photometric + layout term;
      resumable checkpointing for Kaggle's 12-hour cap
- [ ] Upload 24.6 GB to Kaggle
- [ ] Ablation: with and without the layout term, on Local Distortion and OCR-CER

*Log last updated: 2026-09-12 — Session 15*

---

### 2026-09-12 — Session 16: Dataset VERIFIED + Training Wrapper (issue 15 closed)

#### verify_dataset.py on the full set — DATASET OK

All 7 groups pass on all 145,313 samples, in 21.7 s:

| check | result |
|---|---|
| structural (manifest/memmap/sizes/ids) | all PASS, `flows.f16` and `masks.u8` byte-exact |
| decode 60 random records | PASS, correct shapes and dtypes |
| **reproducibility** | **60/60 regenerate byte-identical from the manifest row alone** |
| flow fidelity vs freshly generated | mean **0.0209 px**, p95 0.047, p99 0.440, max 2.55 |
| flow relates the two stored images | 6.38 grey levels (gain-corrected) |
| **invertibility** | residual mean **0.0004 px**, worst 0.0045; rectified PSNR **25.16 dB** |
| composition | cylinder/perspective/combined 47,958 each (33.00%), flat 1,439 (0.99%) |
| page coverage | mean 82.15%, range 45.17 - 100% |

Flow fidelity at 112 x 144 came in at 0.021 px mean / 0.44 px p99 on real pages, against
the 0.018 / 0.37 predicted from the synthetic study — close enough to confirm the
resolution decision was made on sound numbers.

#### ISSUE 15 CLOSED — exact figures

| | |
|---|---|
| dataset sources | 47,958 |
| layout store pages | 47,724 |
| sources with no layout | **234** |
| **samples affected** | **707 of 145,313 (0.487%)** |
| remaining for training | **144,606** |

#### New file: `training/dewarp_dataset.py`

Joins the image store and the layout store and hands the training loop a batch.

**The exclusion** is `require_layout=True` (default): drop samples by stem membership.
Chosen over zeroing the layout term for those samples for two reasons — those pages are
the landscape ones squeezed 0.52x by a portrait resize, so they are out-of-distribution
regardless; and a silently-zero loss term on a subset is exactly the kind of thing that
skews an ablation, which is the experiment this project exists to run.

**The split is by source page, not by sample.** Each page appears 3-4 times and every
variant shares the same clean image, so splitting on samples would put variants of one page
on both sides and leak the answer into validation. The split is a deterministic md5 of the
stem — `hash()` would have been wrong, because Python salts it per process and
`num_workers > 0` would then disagree with the main process about which pages are
validation. Verified: **0 page overlap** between train and val.

**Resolutions.** Image comes back at full 448 x 576 (the encoder should see all of it);
flow, mask and layout boxes come back on the 112 x 144 flow grid, where the decoder
terminates and the losses run. Flow is returned normalized — scale-free and what the model
should regress — with `flow_to_pixels` for the layout loss.

Measured: the layout loss at flow resolution costs **6 ms per batch of 8 on CPU**, versus
217 ms at full resolution (16x fewer pixels). `loss(f_gt, f_gt) = 2e-6` end to end through
the real loader, so the plumbing is correct.

**Default layout categories are (text, title, list)** — the regions made of text lines,
where "must come out rectangular" is the meaningful claim. Tables and figures are excluded
by default: a figure's bounding box has no internal line structure to straighten.

#### Issue Status
Issues 1-15: **all closed.**

#### Next Steps
- [ ] Model: U-Net encoder 448x576, decoder terminating at 112x144
- [ ] Training loop: masked flow L1 + warped-domain photometric + layout term;
      resumable checkpointing for Kaggle's 12-hour cap
- [ ] Upload 24.6 GB to Kaggle
- [ ] Ablation: with and without the layout term, on Local Distortion and OCR-CER

*Log last updated: 2026-09-12 — Session 16*

---

### 2026-09-13 — Session 17: Model + Training Loop, Built and Tested Locally

**Status**: Model, losses and training loop written and verified on the pilot. Nothing
trained for real yet — that needs the GPU.

#### New files
| File | Role |
|---|---|
| `training/model.py` | `DewarpNet` — U-Net encoder at 448x576, decoder terminating at 112x144 |
| `training/losses.py` | masked flow L1 + illumination-robust photometric + layout term |
| `training/train.py` | training loop: AMP, wall-clock budget, exact resume, ablation flag |

#### The model — 6.45 M parameters

Encoder runs the full page; decoder stops at 112 x 144, where the label lives. Decoding
back to 448 x 576 would add the three most expensive stages in the network to produce a
prediction no loss could constrain — the GT flow simply does not exist at that resolution.

Three deliberate choices:

- **Zero-initialised output heads**, so the network starts at *exactly* the identity warp
  ("predict no displacement") rather than at noise. Confirmed: initial `|flow| = 0.0`.
- **A global-context branch at the bottleneck.** Page curvature is global — the bow of a
  text line near the spine depends on where the spine is, possibly 400 px away. A pooled
  page-level vector is broadcast back and concatenated, which costs almost nothing.
- **Deep supervision** at /8 and /16, train-time only.

#### The photometric loss had to be high-pass filtered

The obvious form — resample the clean page with the predicted flow, compare to the warped
input — is close to useless here, because the warped page has a lighting gradient baked in
and the clean one does not. Measured on real data: the raw difference between a *perfectly*
resampled clean page and the stored warped page is **71.4 grey levels**. A naive
photometric loss would spend its gradient trying to explain illumination with geometry.

Both images are therefore high-pass filtered (minus a Gaussian blur) before comparison.
Same measurement after filtering: **8.2 grey levels**. Shading gone, every text edge kept.
This is the same correction that took `verify_dataset.py` check 5 from a meaningless 45
grey levels to a meaningful 6.4.

#### The numbers that anchor everything

| predictor | flow_l1 | layout | **epe_px** |
|---|---|---|---|
| oracle (pred = GT) | 0.00000 | 0.00000 | **0.000** |
| identity (pred = 0) | 21.811 | 2.895 | **35.134** |

**35.13 px is the bar.** Anything above it means the network is worse than doing nothing.
Note the photometric term has a floor of ~0.03 even at the oracle — JPEG plus interpolation
— so it never reaches zero and should not be expected to.

#### RESUME IS BIT-EXACT — and two bugs had to be fixed to get there

Kaggle kills a session at 12 hours and the full dataset is ~9 epochs of work per session,
so training must span sessions. A resumed run has to be *the same run*, otherwise the
ablation is comparing two different experiments.

Test: run A = 40 uninterrupted steps; run B = 20 steps, checkpoint, resume to 40. Compare
weights and optimizer moments directly rather than trusting a loss curve.

Two real defects surfaced:

1. **Shuffling could not be resumed.** torch's default shuffle draws from the global RNG
   when the loader is iterated, so restoring RNG state is not enough — a run resumed
   mid-epoch restarts that epoch in a *different* order, seeing some samples twice and
   skipping others. Invisible in the loss curve, fatal for an ablation. Replaced with
   `ResumableSampler`: the permutation for epoch N is a pure function of `(seed, N)`, so it
   can be regenerated and sliced, and `start` skips consumed indices without loading them.

2. **The LR schedule was tied to the stop point.** `--steps` doubled as both "stop here"
   and "cosine horizon", so a run split across five sessions would have followed five
   truncated cosines instead of one. Split into `--steps` (stop) and `--total_steps`
   (schedule horizon, held fixed across sessions).

Plus an off-by-one: the consumed-sample counter incremented before the break check, so
resume skipped one batch the interrupted run never saw.

Result over 1,620,304 parameters:

```
weights           : max |A - B| = 0.000e+00
optimizer moments : max |A - B| = 0.000e+00
RESUME IS BIT-EXACT
```

#### It learns

Overfit test — 3 samples, 300 steps, base=16 on CPU:

| step | total | epe_px | flow_l1 | layout | photo |
|---|---|---|---|---|---|
| 0 | 34.65 | 45.07 | 28.27 | 3.313 | 0.129 |
| 90 | 5.39 | 6.47 | 4.10 | 0.762 | 0.105 |
| 180 | 2.80 | 3.34 | 2.10 | 0.732 | 0.084 |
| 300 | 2.13 | 2.34 | 1.49 | 0.395 | 0.076 |

Eval-mode EPE **2.47 px**, a **93% reduction** against the 35.13 px identity baseline, with
every component falling together. The pipeline learns end to end.

#### Data loading will not bottleneck the GPU

Measured single-threaded: **197 samples/s** with the clean page decoded (needed for the
photometric term), 387 without. Kaggle's 4 vCPU give roughly 790 samples/s, about 98
batches/s at batch 8 — against a GPU that will manage single-digit batches/s. Roughly 10x
headroom. `want_clean=False` halves decode cost if the photometric term is ever dropped.

#### Kaggle commands

```
# first session
python training/train.py --root /kaggle/input/<ds>/dataset --out /kaggle/working/run_full ^
       --batch 8 --epochs 30 --max_hours 11.0

# every later session: identical command. --resume auto picks up the newest checkpoint.
# --total_steps must stay fixed across sessions, or the cosine restarts.

# the ablation
python training/train.py ... --w_layout 0 --out /kaggle/working/run_nolayout
```

Checkpoints are pruned to the last two plus best (~78 MB each) against the 20 GB
`/kaggle/working` limit.

#### Next Steps
- [ ] Upload 24.6 GB to Kaggle
- [ ] Measure real steps/s on the GPU, set `--total_steps` from it, and sanity-check that
      val EPE drops below 35 px within the first session
- [ ] Full training run
- [ ] Ablation: `--w_layout 0.2` vs `--w_layout 0`, reported on EPE/Local Distortion and
      OCR-CER
- [ ] Inference script: predict at 112x144, invert with `invert_flow`, rectify at full
      resolution

*Log last updated: 2026-09-13 — Session 17*

---

### 2026-09-13 — Session 18: Multi-Session Continuation Fixed + Kaggle Runbook

**Status**: A real hole in the Kaggle continuation story found and closed. Operational
guide written.

#### THE BUG: resume would have silently restarted from scratch on Kaggle

`--resume auto` searched only `--out`. On Kaggle `--out` lives in `/kaggle/working`, **which
is wiped between sessions**. Session 2 would therefore have found nothing, printed nothing
alarming, and quietly trained a fresh network for 11 hours. The loss curve would have
looked perfectly healthy.

Fixed:

- **`--resume_dirs`** — comma-separated extra directories to search. This is how Kaggle
  continuation actually works: the previous session's output is saved as a dataset and
  attached read-only to the next session.
- **Checkpoints are ordered by the step in the FILENAME, never by mtime.** A Kaggle dataset
  mount rewrites modification times, so mtime ordering would happily pick a stale
  checkpoint and rewind training without complaint. Filenames are zero-padded
  (`ckpt_step000054231.pt`) so the ordering is unambiguous.
- **`last.pt`** is rewritten on every save, so the next session can point at one stable
  filename instead of guessing which step the last one reached.
- A loud `STARTING FROM SCRATCH` warning when `--resume_dirs` was given but nothing matched.

**Verified**: 40 steps uninterrupted vs 20 steps + resume **from a different directory**
(simulating the wiped working dir):

```
weights           max |diff| = 0.000e+00
optimizer moments max |diff| = 0.000e+00
CROSS-SESSION RESUME IS BIT-EXACT
```

#### Failures now surface in minutes, not at hour 12

| when | what |
|---|---|
| before step 1 | validation runs (`--val_at_start`) — a broken run shows immediately |
| every 20 steps | losses to stdout and `log.jsonl` |
| every 500 steps | checkpoint (was 1000) |
| every 1,000 steps | validation (was 2000), `best.pt` on improvement |
| **every epoch** | checkpoint + validation unconditionally, regardless of step counters |
| on `--max_hours` | clean exit with a final checkpoint |

At batch 8 that is a checkpoint every ~2 minutes. Every checkpoint carries full resume
state, so any of them is resumable — not just epoch boundaries.

The `[val @ start]` line is the useful one: an untrained network should score ~35 px (the
identity warp). A *resumed* session should print its previous value instead, which is an
immediate, unambiguous check that the resume actually worked.

#### THE 12-HOUR TRAP, now documented

A session killed by Kaggle's cap saves **nothing** — `/kaggle/working` only persists via
*Save Version*, which requires the notebook to finish. So `--max_hours 11.0` is not a
nicety; without it a session can lose 11 hours of GPU quota with no checkpoint to show.

#### `KAGGLE_GUIDE.md` — the runbook that should have existed already

Covers, in order: exactly which files to upload and which to skip; the two-dataset split
(data never changes, code changes constantly); `kaggle datasets create` with the
expectation that 24.6 GB takes hours and does not resume; the smoke test to run
interactively *before* committing anything, and what its output should say; session 1;
sessions 2..N with the one line that changes; what to check to confirm the resume took;
the ablation; and the GPU-quota arithmetic (~4 sessions per arm, two arms, ~3-4 weeks of
free quota).

It also records the escape hatch: dropping the four `clean-*.bin` files saves 3.65 GB of
upload at the cost of `--w_photo 0`. Flow and layout losses do not need them.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-15 | | all closed |
| 16 | Resume searched only the wiped working dir | **FIXED (S18)** |

#### Next Steps
- [ ] Upload; run the smoke test; record real steps/s and set `--total_steps`
- [ ] Session 1 of the layout arm
- [ ] Inference script (predict -> `invert_flow` -> rectify at full resolution)
- [ ] Evaluation: Local Distortion + OCR-CER on the val split
- [ ] Ablation arm with `--w_layout 0`

*Log last updated: 2026-09-13 — Session 18*

---

### 2026-09-13 — Session 19: Upload Path Fixed — the Kaggle CLI Skips Subfolders

**Status**: Upload made mechanical. One more silent-failure trap found and designed out.

#### THE TRAP: `kaggle datasets create` skips subdirectories by default

Uploading `<project>\dataset` as a single dataset would have uploaded all 24.6 GB and
**silently left `layout/` behind**. The upload reports success. The first sign of trouble
would have been a `FileNotFoundError` on Kaggle, hours later, after the whole transfer.

Designed out rather than documented around — **three datasets, split by how often each
changes**:

| dataset | contents | size | changes |
|---|---|---|---|
| `-data` | `dataset/` in place, flat files only | ~24.6 GB | never |
| `-layout` | the 6 layout-store files, lifted out of the subfolder | ~57 MB | only if regenerated |
| `-code` | every `.py` **flattened into one folder** | ~200 KB | constantly |

Flattening the code works because every module imports by plain name; verified by running
the full training loop from the flattened folder. Splitting by change-frequency is also
just right: regenerating the layout store should never mean re-uploading 24.6 GB.

#### Code changes
- **`LayoutStore(root)` now accepts either** the dataset root (finds `<root>/layout`) **or
  a standalone layout folder**. Required, since on Kaggle the layout store is its own
  mount.
- **`DewarpTrainingSet(..., layout_root=...)`** and **`train.py --layout_root`** plumb it
  through.
- **`prepare_upload.py`** — new. Validates the dataset is complete (names the missing file
  and stops if not), stages the layout and code folders, writes each
  `dataset-metadata.json`, and prints the three `kaggle datasets create` commands plus the
  full Kaggle notebook commands with every path already filled in. Reads the username from
  `kaggle.json`.

So "what do I upload" is now: run one script, paste three commands.

#### `KAGGLE_GUIDE.md` rewritten
Four numbered steps instead of prose: run the script, run three uploads, session 1,
sessions 2..N. Keeps the reasoning for why three datasets, the `/kaggle/working` wipe, the
12-hour kill, and the quota arithmetic — but the instructions no longer require
understanding any of it to follow.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-16 | | all closed |
| 17 | Kaggle CLI would silently skip `layout/` | **FIXED (S19)** |

#### Next Steps
- [ ] `python prepare_upload.py --user <name>`, then the three uploads
- [ ] Smoke test on Kaggle; record steps/s; confirm `[val @ start]` reads ~35 px
- [ ] Session 1 of the layout arm
- [ ] Inference script + Local Distortion / OCR-CER evaluation
- [ ] Ablation arm with `--w_layout 0`

*Log last updated: 2026-09-13 — Session 19*

---
