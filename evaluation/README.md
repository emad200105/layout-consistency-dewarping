# Evaluation

- **Synthetic validation** (EPE, layout distortion 𝒞): `training/inference.py --full_val`
  and `training/compare_arms.py`.
- **Synthetic OCR:** `training/extract_ocr_subset.py` + `training/ocr_eval.py`.
- **Real photos (DocUNet benchmark):** [docunet/](docunet/README.md).

Commands and expected outputs for all three: [../docs/REPRODUCE.md](../docs/REPRODUCE.md).
