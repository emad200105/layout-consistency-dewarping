#!/usr/bin/env bash
# Full DocUNet real-photo evaluation of both arms, as reported in the paper.
#
#   bash evaluation/docunet/run_all.sh <docunet_dir> [tesseract]
#
# <docunet_dir> must contain crop/ and scan/ (from crop.zip and scan.zip);
# setup_octave_eval.sh must have been run once. [tesseract] defaults to
# /opt/tess501/tess.sh (install_tesseract_501.sh). Results go to
# outputs/docunet/. Runtime on 2 CPU cores: ~10 min rectification per arm,
# ~10-20 min MS-SSIM/LD per arm, ~5 min OCR per arm.
set -euo pipefail
DATA=$(cd "${1:?usage: run_all.sh <docunet_dir> [tesseract]}" && pwd)
TESS=${2:-/opt/tess501/tess.sh}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
OUT="$REPO/outputs/docunet"
mkdir -p "$OUT"
cd "$REPO"

python evaluation/docunet/make_input_set.py --crop_dir "$DATA/crop" --out_dir "$OUT/distorted"
python evaluation/docunet/rectify.py --ckpt weights/dewarpnet_layout_lambda0.2.pt --crop_dir "$DATA/crop" --out_dir "$OUT/ours"
python evaluation/docunet/rectify.py --ckpt weights/dewarpnet_ablation_lambda0.pt  --crop_dir "$DATA/crop" --out_dir "$OUT/ablation"

for m in distorted ours ablation; do
  (cd evaluation/docunet/octave && octave --no-gui --quiet eval_dir.m "$OUT/$m/" _rec.png "$OUT/${m}_msld.csv" "$DATA/scan/")
  python evaluation/docunet/ocr.py --rec_dir "$OUT/$m" --scan_dir "$DATA/scan" \
      --cache "$OUT/ocr_${m}_t501.jsonl" --gt_cache "$OUT/ocr_scan_t501.jsonl" --tesseract "$TESS"
done

python evaluation/docunet/analyse.py \
  --msld_a "$OUT/ours_msld.csv" --msld_b "$OUT/ablation_msld.csv" --msld_in "$OUT/distorted_msld.csv" \
  --ocr_a "$OUT/ocr_ours_t501.jsonl" --ocr_b "$OUT/ocr_ablation_t501.jsonl" \
  --ocr_in "$OUT/ocr_distorted_t501.jsonl" --ocr_gt "$OUT/ocr_scan_t501.jsonl" \
  --out "$OUT/paired_analysis.json" --per_image_csv "$OUT/ocr_per_image.csv"
