# Training on Kaggle — runbook

Both arms of the paper were trained with `notebooks/dewarp_kaggle.ipynb` on
Kaggle's free GPUs, two sessions per arm (about 11 h + 6.5 h). This guide explains how to run
it and what to check. Most of it is about Kaggle's habits, which cost far more
time than the model ever did.

---

## Step 1 — stage the upload (on your machine)

```bash
pip install kaggle          # API token: Kaggle -> Settings -> Create New Token
python tools/prepare_upload.py --project . --user <your-kaggle-username>
```

It refuses to continue if the dataset is incomplete, and names the missing file.
It then stages `upload/layout/` and `upload/code/`, writes each
`dataset-metadata.json`, and prints the commands for step 2. The script reads
only your *username* from the Kaggle config, never the key.

## Step 2 — upload three datasets

Upload the small ones first, so you can build the notebook while the big one
uploads.

| # | what | size | how often |
|---|---|---|---|
| 1 | **code** — every `.py`, flattened into one folder | ~0.2 MB | whenever the code changes |
| 2 | **layout** — the six layout-store files | 57 MB | only if regenerated |
| 3 | **data** — `dataset/` in place | 25 GB | **once** |

```bash
kaggle datasets create -p upload/code
kaggle datasets create -p upload/layout
kaggle datasets create -p dataset          # hours; does not resume
```

To re-upload an existing dataset, use
`kaggle datasets version -p . -m "what changed"`.

**Why three datasets:** `kaggle datasets create` **skips subdirectories**.
Uploading `dataset/` alone would silently leave `layout/` behind, and you would
first notice hours later. The code is flattened into one folder; every module
imports by plain name, so that works.

**Licence label:** `prepare_upload.py` writes `CC0-1.0` into each
`dataset-metadata.json`. Our datasets were private. If you make yours public,
change the data and layout datasets to PubLayNet's licence (CDLA-Permissive-1.0)
first.

## Step 3 — create the notebook

Kaggle → Code → New Notebook → File → Import Notebook →
`notebooks/dewarp_kaggle.ipynb`. **Use one notebook per arm**, e.g. "dewarp" and
"dewarp-ablation". Each arm then has its own chain of saved outputs, and a
resume can never pick up the other arm's checkpoint.

Session options:

| setting | value | why |
|---|---|---|
| Accelerator | **GPU T4 x2** | cell 1 asserts a GPU; training uses one |
| Persistence | Files only | convenience only; **not** a backup (see below) |
| Environment | **Pin to original environment** | a mid-run library update could change numerics under a resumed checkpoint |
| Internet | off | nothing is downloaded |

Then **Add Data** and attach the three datasets.

## Step 4 — which files survive

| path | writable | survives |
|---|---|---|
| `/kaggle/input/...` | no | always |
| `/kaggle/working/` | yes | **only through a saved version** |
| anywhere else | yes | never |

*Save Version* has two modes:

- **Save & Run All (Commit)** starts a **fresh batch session** that re-runs the
  notebook from the top. Its `/kaggle/working` becomes the version's output.
- **Quick Save** keeps the notebook and the current interactive files.

**The mistake that costs a session:** training interactively for 11 hours and
*then* pressing Save & Run All. That throws the work away and starts the same
11 hours again in the background.

**Recommended:**

1. Run notebook sections 1–4 interactively (about 3 minutes), then section 5–6
   with `MODE = 'smoke'`.
2. Set the real `MODE`.
3. **Save & Run All (Commit)** and close the tab.

## Step 5 — one switch decides what a session does

Notebook section 5 ("Pick what this session does"):

```python
MODE = 'smoke'   # 'smoke' | 'session1' | 'resume' | 'ablation' | 'ablation_resume'
EPOCHS      = 30        # both arms MUST use the same number
TOTAL_STEPS = 540_000   # LR schedule horizon; identical in every session of both arms
BATCH       = 8
HOURS       = 11.0      # clean exit before Kaggle's 12-hour kill
```

| `MODE` | what | needs attached |
|---|---|---|
| `smoke` | 50 steps, about 2 minutes | the 3 datasets |
| `session1` | our arm (λ = 0.2) from scratch | the 3 datasets |
| `resume` | our arm, continuing | + the previous session's **saved output** |
| `ablation` | λ = 0 from scratch | the 3 datasets |
| `ablation_resume` | λ = 0, continuing | + the previous ablation session's saved output |

Only one training mode per run: *Run All* must perform exactly one training run.

### What a healthy start looks like

```
train 141,869 / val 2,737            <- both stores loaded; 707 landscape samples dropped
[val @ start] epe ~37 px             <- fresh start: an untrained net scores about the identity warp
```

For `resume` and `ablation_resume`:

```
[paths] resume candidate /kaggle/input/.../run_layout
[resume] last.pt: epoch 19, step 341,833, ...
[val @ start] epe ~4 px              <- MUST be about 4, not 37
```

If it says **`STARTING FROM SCRATCH`** or `epe 37`, stop the run immediately:
the previous output is not attached.

### Known failure messages

- **`EMPTY SPLIT`**: the layout store and the image dataset do not belong
  together. It prints the stem overlap.
- **`pinned to an OLD version of the code dataset`**: Kaggle pins the dataset
  version a notebook was created with. Remove the code input, add it again, and
  rerun sections 2 and 4.
- A mount-path `FileNotFoundError` even though the data is attached: never
  hard-code `/kaggle/input/...`. `train.py --root auto --layout_root auto
  --resume_dirs auto` find the stores by marker files (`flows.f16`,
  `stems.jsonl`, `last.pt`).

## Step 6 — sessions after the first

1. Check that the previous version **finished** and has an output. A session
   killed at 12 hours saves nothing; `--max_hours` prevents that.
2. In the arm's own notebook, *Add Data* → *Your Work* (or *Your Datasets*) →
   that version's output.
3. Change `MODE` to `resume` (or `ablation_resume`). **Nothing else changes.**
4. Save & Run All (Commit).

Resume restores the weights, optimizer moments, AMP scaler, schedule position,
epoch, step, position within the epoch, and all RNG states. In a CPU test a
resumed run was bit-identical to an uninterrupted one (compared tensor by
tensor); on GPU, cuDNN/AMP nondeterminism still applies.

## Step 7 — get the results

```bash
kaggle kernels output <your-kaggle-username>/<notebook-slug> -p runs/session2
python training/inference.py --ckpt runs/session2/run_layout/ckpt_step000531990.pt \
       --root dataset --n 6 --out preview.png --full_val
```

`--ckpt` also accepts a folder, but then it prefers `best.pt`, then `last.pt`,
then the newest `ckpt_step*.pt`. For the paper, **both arms were scored at the
same final checkpoint**, `ckpt_step000531990.pt`, given explicitly. `best.pt` is
selected on validation, and reporting it on the same validation set would be
biased.

## Budget

| | |
|---|---|
| steps per epoch | 17,733 (batch 8) |
| speed on one T4 | 8.6 steps/s with the layout loss, 9.2 without |
| 30 epochs | 16–18 GPU-hours per arm, in 2 sessions |
| free quota | 30 GPU-hours per week, so both arms need about 1.2 weeks of quota |
| output size | about 300 MB per session (checkpoints of 78 MB, pruned to the last two + `best.pt` + `last.pt`) |

**Training longer:** raise `EPOCHS` and `TOTAL_STEPS` **together**
(17,733 × epochs), and do it for **both** arms. Raising only `EPOCHS` trains the
extra epochs at a learning rate of almost zero.
