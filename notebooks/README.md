# Notebooks

| notebook | where | what |
|---|---|---|
| `dewarp_kaggle.ipynb` | Kaggle (GPU) | **The notebook that trained both arms.** One `MODE` switch: `smoke`, `session1`, `resume`, `ablation`, `ablation_resume`. Use one copy per arm. Full instructions: [../docs/KAGGLE_GUIDE.md](../docs/KAGGLE_GUIDE.md). |
| `dewarp_eval.ipynb` | Kaggle (optional) | Looks at a finished session: loss curve, preview sheet, EPE by profile. Running `training/inference.py` on your own machine does the same and is simpler. |

The notebooks expect three Kaggle datasets made by `tools/prepare_upload.py`:
the code (flattened), the layout store, and the 25 GB data store. They find them
by marker files, so the dataset names and mount paths do not matter.
