# Reproducing everything

From raw PubLayNet to the paper's figures. Each step lists what it needs, what it
produces, roughly how long it takes, and what a correct result looks like. All
commands run from the repository root.

You can start part-way:

- **Have no dataset but want to evaluate?** With the released weights you can run
  the real-photo evaluation (step 9) and three figures (`shift_bend`, `profiles`,
  `curves`) right away. Steps 7–8 need the built dataset (steps 2–4), but not training.
- **Want only the figures?** Most of the result files are committed; see step 10.

```
1 environment           5 upload to Kaggle         9 real photos (DocUNet)
2 PubLayNet pages+json  6 train both arms          10 figures + paper
3 layout store          7 synthetic evaluation
4 dataset build+verify  8 synthetic OCR test
```

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python augmentation/test_warps.py          # -> ALL TESTS PASSED
python augmentation/test_layout_loss.py    # -> ALL TESTS PASSED
```

## 2. PubLayNet source pages and annotations

PubLayNet (CDLA-Permissive-1.0) is linked from
<https://github.com/ibm-aur-nlp/PubLayNet>. You need two things:

- **The training images.** We used 47,958 pages, about the size of one of the
  seven training archives (about 13 GB each). We did not record *which* archive
  they came from, so to collect exactly this set you may need to download several
  or all training archives and let `select_pages.py` pick the pages.
- **The annotation file `train.json`** (COCO format, about 1.7 GB), from the
  labels archive.

Put the exact page set in one folder. **This matters**: samples are seeded by
their position in the sorted file list, so a different set of pages gives
different warps.

```bash
python tools/select_pages.py --src /path/to/publaynet/train_images --dst data/publaynet/train
# list: results/dataset_build/source_pages.txt.gz (47,958 stems; 234 marked no_layout_landscape)
```

If you only want *a* dataset rather than *the same* one, any folder of PubLayNet
pages works.

## 3. Layout store (about 10–15 minutes; the image-size census alone took 9 minutes)

```bash
mkdir -p outputs
python augmentation/check_annotations.py --json data/publaynet/train.json \
       --images data/publaynet/train --overlays 10 --out_dir outputs/annotation_check
#   check: the drawn boxes in outputs/annotation_check/*.png sit on the text

python augmentation/prepare_layout.py --json data/publaynet/train.json \
       --images data/publaynet/train --out_dir dataset --overlays 8
#   expect: n_pages 47,724, n_annotations 465,596, n_landscape 234
#           (compare results/dataset_build/layout_summary.json)
```

This writes `dataset/layout/` (57 MB).

## 4. Build and verify the dataset

Build time is a few hours on 2 CPU cores, under an hour on 8; the dataset is
about 25 GB.

```bash
python augmentation/build_dataset.py --input_dir data/publaynet/train --out_dir dataset_pilot --limit 3000   # optional pilot, separate folder
python augmentation/build_dataset.py --input_dir data/publaynet/train --out_dir dataset              # full; add --resume after an interruption
python augmentation/verify_dataset.py --root dataset --source_dir data/publaynet/train --n_check 60
#   expect: 145,313 samples (143,874 warped + 1,439 flat), "ok": true
#           (compare results/dataset_build/dataset_meta.json and verify_summary.json)
```

Optional checks:

- `python tools/inspect_dataset.py --root data/publaynet/train --out_dir outputs`
  writes a contact sheet of real warps.
- `python augmentation/preview_augmentation.py --image <page.jpg>` shows one
  page's warps.

## 5. Upload to Kaggle

Training ran on Kaggle's free T4 GPUs. Staging splits the work into three
datasets, because `kaggle datasets create` silently skips subfolders:

```bash
pip install kaggle      # and put your API token where the CLI expects it
python tools/prepare_upload.py --project . --user <your-kaggle-username>
# then run the three printed `kaggle datasets create -p ...` commands: code, layout, data (25 GB, hours)
```

## 6. Train both arms

About 17 GPU-hours per arm, in two Kaggle sessions each. Import
`notebooks/dewarp_kaggle.ipynb` into Kaggle and follow
[KAGGLE_GUIDE.md](KAGGLE_GUIDE.md):

| session | `MODE` | attach |
|---|---|---|
| arm 1, session 1 | `session1` | the 3 datasets |
| arm 1, session 2 | `resume` | + the saved output of arm 1, session 1 |
| arm 2, session 1 | `ablation` | the 3 datasets (separate notebook) |
| arm 2, session 2 | `ablation_resume` | + the saved output of arm 2, session 1 |

Do not change `EPOCHS = 30`, `TOTAL_STEPS = 540_000` or `BATCH = 8` between
sessions or between arms. (We used `HOURS = 11` for our arm and `10` for the
ablation; with exact resume the session length does not change the run.)

- **Check:** a resumed session prints `[resume] last.pt: epoch …` and a
  starting validation EPE of about 4 px. If it prints `STARTING FROM SCRATCH`,
  stop it.
- **Expect:** the final checkpoint is `ckpt_step000531990.pt`; the training log
  ends at a subset validation EPE of about 3.76 (ours) or 3.90 (ablation).

Without Kaggle, run `training/train.py` directly with the same flags, e.g.:

```bash
python training/train.py --root dataset --out runs/run_layout --batch 8 --epochs 30 --total_steps 540000
python training/train.py --root dataset --out runs/run_nolayout --batch 8 --epochs 30 --total_steps 540000 --w_layout 0
```

## 7. Synthetic evaluation (CPU: about 8 to 25 minutes per arm)

Use your checkpoints or the released weights.

```bash
mkdir -p outputs
python training/identity_epe.py --root dataset      # do-nothing baseline: 39.88 px
python training/inference.py --ckpt weights/dewarpnet_layout_lambda0.2.pt --root dataset \
       --n 6 --out outputs/final_layout.png --full_val --mask_invalid --device cpu
python training/inference.py --ckpt weights/dewarpnet_ablation_lambda0.pt --root dataset \
       --n 6 --out outputs/final_ablation.png --full_val --mask_invalid --device cpu
python training/compare_arms.py --ours outputs/final_layout.png.epe.json \
       --ablation outputs/final_ablation.png.epe.json
#   expect: EPE 3.662 vs 3.689, per-page C 0.4398 vs 0.4714 (results/synthetic_val/COMPARISON.txt)
```

**Paired per-page statistics on the 334-page subset** (Table I, bottom). These
need the subset from step 8:

```bash
python paper/scripts/export_subset_boxes.py --layout_root dataset
python paper/scripts/subset_paired.py > outputs/subset_paired.txt   # also rewrites results/synthetic_val/subset_paired.json
```

## 8. Synthetic OCR test (about 1 hour on 2 cores)

```bash
sudo apt install tesseract-ocr            # we used 5.3.4 for this test
python training/extract_ocr_subset.py --root dataset --out outputs/ocr_subset.npz       # 334 pages, ~70 MB
OMP_THREAD_LIMIT=1 python training/ocr_eval.py --subset outputs/ocr_subset.npz \
       --ckpt layout=weights/dewarpnet_layout_lambda0.2.pt \
       --ckpt ablation=weights/dewarpnet_ablation_lambda0.pt \
       --out outputs/ocr_results --workers 2
#   expect: mean CER input 63.05, layout 28.61, ablation 30.37, true flow 13.70
#           (results/ocr_synthetic/OCR_RESULT.txt)
```

## 9. Real photos: DocUNet benchmark (about 1 hour)

Follow [../evaluation/docunet/README.md](../evaluation/docunet/README.md): get
the benchmark, run `setup_octave_eval.sh` and `install_tesseract_501.sh` once,
then:

```bash
bash evaluation/docunet/run_all.sh data/docunet /opt/tess501/tess.sh
#   expect (outputs/docunet/paired_analysis.json):
#   distorted 0.2493 / 20.52 / CER S1 0.5092; ours 0.3243 / 16.98 / 0.3630; ablation 0.3220 / 16.85 / 0.3898
```

## 10. Figures and paper

The scripts read committed results where possible. Three of them also need the
generated files from steps 8–9, under `outputs/` (or `$DEWARP_OUTPUTS`).

| figure | script | needs |
|---|---|---|
| Fig. 1 `pipeline` | `fig_pipeline.py` | `outputs/ocr_subset.npz`, weights |
| Fig. 2 `shift_bend` | `fig_shift_bend.py` | nothing |
| Fig. 3 `profiles` | `fig_profiles.py` | committed results only |
| Fig. 4 `curves` | `fig_curves.py` | committed training logs |
| Fig. 5 `synth_qual` | `fig_synth.py` | `outputs/ocr_subset.npz`, weights |
| Fig. 6 `docunet_qual` | `fig_docunet.py` | `outputs/docunet/{distorted,ours,ablation}`, `data/docunet/scan` (or `$DOCUNET_DATA`) |

```bash
for f in fig_pipeline fig_shift_bend fig_profiles fig_curves fig_synth fig_docunet; do
  python paper/scripts/$f.py
done
cd paper && latexmk -pdf main.tex      # needs TeX Live with IEEEtran (texlive-publishers)
```

The scripts were checked to regenerate the committed PNGs pixel-for-pixel.

---

## Exactness: what reproduces bit-for-bit and what does not

| step | exact? |
|---|---|
| dataset build | **yes**, given the same 47,958 pages (`verify_dataset.py` regenerates samples byte for byte; rebuilding the first 40 pages with this repo reproduced all 120 of their warped samples byte for byte. Only the choice of *flat* samples depends on the total page count) |
| training on Kaggle | resume restores the exact state (bit-exact in a CPU test); GPU runs are close but not bit-identical, because cuDNN/AMP kernels are nondeterministic |
| synthetic EPE / 𝒞 with the released weights | yes, up to float summation order (CPU threads) |
| DocUNet rectification | yes, with `--threads 1` |
| MS-SSIM / LD in Octave | yes, on the same Octave 8.4 + image 2.14 |
| OCR | yes, with the same Tesseract build and language model |
