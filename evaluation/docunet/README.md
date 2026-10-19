# Real-photo evaluation: DocUNet benchmark

The DocUNet benchmark (Ma et al., CVPR 2018) has 130 phone photos of 65 paper
documents, each with a flatbed scan as ground truth. This folder scores a
checkpoint on it with **the same protocol as the published numbers**. Our score
for the raw, unrectified photos reproduces the published one:

| raw photos ("Distorted") | MS-SSIM | LD | ED (S1) | CER (S1) | ED (S2) | CER (S2) |
|---|---|---|---|---|---|---|
| published (DocGeoNet / DocScanner tables) | 0.2459 | 20.51 | 1552.22 | 0.5089 | 2111.56 | 0.5352 |
| this pipeline | 0.2493 | 20.52 | 1546.52 | 0.5092 | 2092.78 | 0.5310 |

## 1. Get the benchmark

From <http://vision.cs.stonybrook.edu/~kema/docwarp/> download `crop.zip`
(the photos), `scan.zip` (the ground truth) and `eval.zip` (the official MATLAB
evaluation code). Unzip the first two into one folder, for example
`data/docunet/`, so that you have `data/docunet/crop/*.png` and
`data/docunet/scan/*.png`. None of these files are redistributed in this repo.

## 2. One-time setup

```bash
# GNU Octave + image package + build tools (Ubuntu)
sudo apt install octave octave-image octave-dev patch unzip g++
bash evaluation/docunet/setup_octave_eval.sh /path/to/eval.zip
cd evaluation/docunet/octave && octave --no-gui --quiet sanity.m ../../../data/docunet/scan/ && cd -
# expected: identical 1.0001/0.000, shift 5 px 0.8431/2.493, other document 0.3100/50.504

# Tesseract 5.0.1 (the version behind the published OCR numbers)
bash evaluation/docunet/install_tesseract_501.sh       # -> /opt/tess501/tess.sh
pip install pytesseract rapidfuzz
```

**Why Octave:** the official code is MATLAB. `setup_octave_eval.sh` applies a
small patch (`octave/siftflow_octave.patch`) to the SIFT-flow sources so that
they compile as Octave MEX files (64-bit `mwSize`, Linux build flag, header
case), and swaps one `imfilter` call for `compat/imfilter_nd.m`, because
Octave's `imfilter` refuses the 128-channel dense-SIFT array. `compat/ssim.m`
re-implements MATLAB R2019a's default `ssim`, which Octave lacks.

**Why Tesseract 5.0.1:** ED and CER move noticeably with the engine version. On
the raw photos, 5.0.1 gives CER 0.509 (published 0.509), while Ubuntu 24.04's
5.3.4 gives 0.484. Compare only numbers produced by the same build.

## 3. Run

Everything at once, for both released models:

```bash
bash evaluation/docunet/run_all.sh data/docunet /opt/tess501/tess.sh
```

Or step by step:

```bash
python evaluation/docunet/rectify.py --ckpt weights/dewarpnet_layout_lambda0.2.pt \
    --crop_dir data/docunet/crop --out_dir outputs/docunet/ours
cd evaluation/docunet/octave
octave --no-gui --quiet eval_dir.m ../../../outputs/docunet/ours/ _rec.png ../../../outputs/docunet/ours_msld.csv ../../../data/docunet/scan/
cd -
python evaluation/docunet/ocr.py --rec_dir outputs/docunet/ours --scan_dir data/docunet/scan \
    --cache outputs/docunet/ocr_ours_t501.jsonl --gt_cache outputs/docunet/ocr_scan_t501.jsonl \
    --tesseract /opt/tess501/tess.sh
```

To evaluate **another method**, put its outputs in a folder as `<i>_<k>_rec.png`
(i = 1..65, k = 1, 2, with document 64 already rotated) and run `eval_dir.m` and
`ocr.py` on that folder.

## Protocol details (as in DocTr's `ssim_ld_eval.m` and `OCR_eval.py`)

- **Rectification:** the network sees a 448×576 resize of each photo. The flow
  is upsampled to full resolution, inverted there and applied once, with white
  fill outside the photo. `rectify.py` uses 1 torch thread by default: with more
  threads, floating-point summation order changes a few border pixels, so
  outputs are no longer bit-identical to ours.
- **Corrected benchmark:** both photos of document 64 are rotated by 180°
  before anything else, as in DocGeoNet, DocScanner and DocTr++.
- **MS-SSIM / LD:** grey images. The scan is resized to an area of 598,400
  pixels, the result is resized to the scan's size, then the official
  `evalUnwarp`. Scores are averaged over all 130 images.
- **OCR:** default `pytesseract.image_to_string`, applied to the full-resolution
  scan and to the rectified image as is. ED is the Levenshtein distance of the
  raw strings, and CER = ED / len(reference). The reported value is the average
  of the means over the `_1` and `_2` photos.
  - Setting 1 (DocTr) uses 30 documents (60 photos).
  - Setting 2 (DewarpNet) uses 25 documents (50 photos).
  - The lists are in `ocr.py`.
- **Paired statistics** (`analyse.py`): confidence intervals resample whole
  documents, because the two photos of a document are not independent;
  Wilcoxon tests treat images as independent.

## Our numbers

See `results/docunet/`, which holds per-image MS-SSIM/LD, per-image OCR ED/CER
(no text) and the paired analyses with Tesseract 5.0.1 and 5.3.4. A summary is
in [../../docs/RESULTS.md](../../docs/RESULTS.md).
