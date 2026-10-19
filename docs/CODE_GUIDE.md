# Code guide

A map of every file: what it does, when you run it, and what it reads and writes.
Every module starts with a long docstring that explains *why* it is built the
way it is, often with the measurement that decided it. Read those before
changing anything.

**Import convention:** modules import each other by plain name. Scripts in
`training/` add `../augmentation` to `sys.path` themselves. On Kaggle all `.py`
files are flattened into one folder (`tools/prepare_upload.py`), which works
for the same reason.

---

## `augmentation/` — synthetic data and the loss

| file | role | run it? |
|---|---|---|
| `warp_augment.py` | All warp physics. `cylindrical_flow`, `perspective_flow`, `wrinkle_flow` (creases), `similarity_flow` + `sample_framing` (camera framing), `make_background` + `composite_on_background`, `add_shadow`, `compose_flows`. `augment_document()` builds one sample (warped image, full-resolution pixel flow, mask, parameters). `DISTORTION_PROFILES` / `FLAT_PROFILE` define cylinder / perspective / combined / flat. Its `process_dataset` / CLI is the **legacy loose-file writer**; use `build_dataset.py` instead. | library |
| `dataset_format.py` | The on-disk contract shared by writer and reader: geometry constants, `encode_flow`/`decode_flow` (normalized float16 at 112×144), masks, JPEG codec, **`invert_flow`** (damped fixed point), `DewarpDataset` (raw reader). | library |
| `build_dataset.py` | Builds the training store from a folder of PubLayNet JPEGs: process pool, deterministic per-sample seeds, append-only blobs, **resumable** (`--resume`). Use `--limit 3000` for a pilot. | **yes** (step 4 of REPRODUCE) |
| `verify_dataset.py` | Checks a built store: sizes, decoding, byte-exact regeneration from the manifest, flow fidelity, flow ↔ image consistency, profile mix, invertibility. Writes `verify_summary.json`. | yes, after every build |
| `coco_stream.py` | Streaming reader for PubLayNet's 1.7 GB `train.json`. It finds **top-level** keys only, because `val.json` has nested `"annotations"` arrays that fool naive parsers. | library |
| `check_annotations.py` | Before building the layout store, proves that the annotations line up with *your* image files: size census plus drawn overlays. | yes, once |
| `prepare_layout.py` | Builds the layout store (`dataset/layout/`) from `train.json`, keeping only the pages you have on disk, and drops landscape pages. Writes overlays in training coordinates. | **yes** |
| `layout_format.py` | Layout-store format and the `LayoutStore` reader (`regions_for`, `bboxes_array`, scaled to any target size). | library |
| `layout_loss.py` | **The layout-consistency loss** (`layout_consistency_loss`): `global` mode (used) and `per_region` mode. Also `boxes_to_tensor` and `flow_from_normalized`. | library |
| `test_warps.py` | Regression tests for the warp pipeline. Every test guards a bug that actually shipped once. | yes, after any warp change |
| `test_layout_loss.py` | Proves the loss's claims: zero at the truth, zero for a rigid shift, large for a bend of equal L1, finite gradients only on the page. | yes, after any loss change |
| `preview_augmentation.py` | Quick visual check of the warps on one image. | optional |

## `training/` — model, training, evaluation

| file | role | run it? |
|---|---|---|
| `model.py` | `DewarpNet` (U-Net, global context, deep supervision, zero-init heads) and `apply_flow_norm` (grid_sample with the dataset's flow convention). | library |
| `losses.py` | `DewarpLoss`: masked flow L1, high-pass photometric, aux heads, layout loss. `endpoint_error`. The layout branch is skipped when `w_layout == 0`. | library |
| `dewarp_dataset.py` | `DewarpTrainingSet`: joins the image store and the layout store. It splits by **page stem** (`stem_split`, md5 < 0.02 → val), drops pages without layout, and returns image (448×576) plus flow, mask and boxes on the flow grid. Also `collate`. | library |
| `train.py` | The training loop: AMP, cosine LR, `--max_hours` clean exit, **exact resume** (`--resume_dirs`, `auto` on Kaggle), checkpoint pruning, periodic validation, `log.jsonl`. | **yes** (via the notebook) |
| `inference.py` | Load any checkpoint (`load_checkpoint` accepts a file or a folder), `predict_flow_norm`, `rectify`, `dewarp`. CLI: `--image` dewarps one photo; `--root` writes a preview sheet; `--full_val` writes `<out>.epe.json` with EPE per profile and the layout metric (`layout_metric`). | **yes** |
| `compare_arms.py` | Prints the two-arm comparison from two `.epe.json` files, both per-page and batch-pooled 𝒞. | yes |
| `identity_epe.py` | EPE of the do-nothing (zero-flow) prediction on a split; needs only the dataset. | yes |
| `extract_ocr_subset.py` | Freezes the 334-page stratified OCR subset into `ocr_subset.npz`, so OCR can run without the 25 GB store. Selection is by a seeded hash, never by results. | yes |
| `ocr_eval.py` | The synthetic OCR test: 3× re-synthesis, Tesseract, paired statistics; resumable. Writes `OCR_RESULT.txt`, `per_page.csv`, raw texts and example images. | yes (about 1 h on 2 CPU cores) |

## `notebooks/`

| file | role |
|---|---|
| `dewarp_kaggle.ipynb` | **The notebook that trained both arms.** Environment check, mount discovery, upload-integrity check, code-version check, one `MODE` switch (`smoke`, `session1`, `resume`, `ablation`, `ablation_resume`), live log, output check. See [KAGGLE_GUIDE.md](KAGGLE_GUIDE.md). |
| `dewarp_eval.ipynb` | Optional Kaggle notebook to look at a finished session: loss curve, preview sheet, EPE by profile. Running `training/inference.py` locally does the same. |

## `evaluation/docunet/` — real photos

| file | role |
|---|---|
| `setup_octave_eval.sh` | Unpacks the official `eval.zip`, applies `octave/siftflow_octave.patch` and compiles the SIFT-flow MEX files for GNU Octave. |
| `octave/eval_dir.m` | MS-SSIM + LD for a folder of rectified images (follows DocTr's `ssim_ld_eval.m`). Resumable CSV. |
| `octave/sanity.m` | Identical / shifted / different-document checks of the Octave port. |
| `octave/compat/ssim.m`, `imfilter_nd.m` | MATLAB-compatible SSIM, and channel-wise filtering for the 128-channel SIFT array. |
| `install_tesseract_501.sh` | Builds Tesseract 5.0.1, the version behind the published OCR numbers. |
| `make_input_set.py` | Writes the unrectified photos (document 64 rotated) with benchmark naming, for the "Distorted" row. |
| `rectify.py` | Rectifies all 130 photos with a checkpoint (full resolution, white fill, document 64 rotated). |
| `ocr.py` | ED / CER exactly as DocTr's `OCR_eval.py`, Settings 1 and 2; cached, resumable. |
| `analyse.py` | Paired comparison of two methods: document-level bootstrap CIs and Wilcoxon tests. |
| `run_all.sh` | The whole real-photo evaluation for both arms in one script. |

## `paper/`

| file | role |
|---|---|
| `main.tex`, `main.pdf`, `refs.bib` | The draft paper (IEEEtran). Build with `latexmk -pdf main.tex`. |
| `refs_notes.md` | How every reference was verified, what each paper does relative to ours, and the published DocUNet table with sources. |
| `figs/` | All figures (PDF for the paper, PNG for viewing). |
| `scripts/paths.py` | Where the scripts find their inputs (`DEWARP_OUTPUTS`, `DOCUNET_DATA`). |
| `scripts/style.py` | Shared matplotlib style (IEEE column widths, fonts, colours). |
| `scripts/fig_*.py` | One script per figure. They reproduce the committed figures pixel-for-pixel. |
| `scripts/subset_paired.py` | Per-page paired EPE / 𝒞 statistics on the 334-page subset (Table I, bottom). |
| `scripts/export_subset_boxes.py` | Exports the subset's layout boxes needed by `subset_paired.py`. |

## `tools/`

| file | role |
|---|---|
| `prepare_upload.py` | Stages the three Kaggle datasets (data in place, layout copied out of its subfolder, code flattened), writes their `dataset-metadata.json` and prints the upload commands. It reads only your Kaggle *username*. |
| `inspect_dataset.py` | Reconnaissance of a raw PubLayNet image folder: sizes, a contact sheet of real warps, round-trip checks, storage projection. |

---

## Data stores on disk (not in git)

```
dataset/                         built by build_dataset.py  (~25 GB)
  pages-0000.bin ...             warped JPEGs (q90), concatenated
  clean-0000.bin ...             clean JPEGs, one per source page
  flows.f16                      (N, 144, 112, 2) float16, normalized backward flow
  masks.u8                       (N, 144, 112) uint8 page mask
  manifest.jsonl                 one row per sample: offsets, profile, seed, every warp parameter
  clean_index.jsonl              one row per source page
  meta.json                      format version and geometry   (copy: results/dataset_build/)
  layout/                        built by prepare_layout.py  (~57 MB)
    stems.jsonl  ann_cat.u8.npy  ann_bbox.u16.npy  poly_offsets.i64.npy  poly_xy.u16.npy  meta.json
```

The format docstring in `dataset_format.py` still says `manifest.parquet`; the
store that was actually built (format version 2) uses `.jsonl` for both tables.

## Code provenance

The code in `augmentation/`, `training/` and `tools/` is the code that produced
the results. The training runs used exactly these modules, verified by MD5
against the code snapshot each Kaggle session saved. For publication, only
non-functional edits were made:

- Example paths in docstrings and default CLI paths were changed from the
  original Windows project folder to repo-relative ones (`dataset`,
  `data/publaynet/train`, `.`).
- Two docstring sentences claiming that no prior dewarping work uses layout
  were corrected (see the paper's related work).
- `training/compare_arms.py` was rewritten as a general command-line tool.

No logic was changed; `test_warps.py` and `test_layout_loss.py` pass.
