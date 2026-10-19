# 📄 Document Dewarping Project — Log

> **Note (added for the public repository).** These are the unedited working notes kept
> during development, with the help of an AI assistant (Claude). Only personal names, the
> Kaggle account name and local machine paths were replaced (`<kaggle-user>`, `<project>`),
> and a section about the assistant's own sandbox was removed. Some early statements were
> later corrected (notably the novelty claims of session 2; see the paper's related work).
> Where these notes and `docs/RESULTS.md` / the paper disagree, the latter are authoritative.


> **Goal**: Build a model that corrects wrinkles, curvature, and distortions in document images (e.g., photos of bent or curved pages) using a novel deep-learning approach trained on synthetic augmentations of the PubLayNet dataset.

---

## 🧭 CURRENT STATE — READ THIS FIRST

*A standing snapshot, rewritten whenever it changes. The session log below is
chronological and long; this section alone should be enough to pick the project
back up cold. Last rewritten: 2026-10-18, after session 36.*

### Paper status (session 36) — DRAFT DONE

- Draft paper: `<project>\paper\main.pdf` (IEEE conference, 8 pages), source `paper\main.tex`, bib `paper\refs.bib` (36 web-verified refs, notes in `paper\refs_notes.md`), figures `paper\figs\` (made by `paper\scripts\fig_*.py`), all numbers in `paper\results\`.
- Real-photo test: `<project>\docunet_eval\` (README.txt says how). DocUNet benchmark, 130 photos, Octave port of the official MATLAB eval + Tesseract 5.0.1. Our reproduction of the published "Distorted" row matches (0.249/20.52/1546.5/0.509 vs 0.246/20.51/1552.2/0.509).
- DocUNet results (corrected benchmark, Tess 5.0.1): ours MS-SSIM 0.324, LD 16.98, ED1 1061.4, CER1 0.363, ED2 1555.9, CER2 0.406; ablation 0.322 / 16.85 / 1081.1 / 0.390 / 1513.8 / 0.395. Arms NOT significantly different on real photos (all p > 0.19). Both far behind Doc3D-trained SOTA (e.g. DocTr 0.511/7.76).
- Paired 334-page synthetic stats (new): EPE diff not significant (p=0.59); layout metric C 6.1% lower (p=1.5e-9), C_h 8.3%, C_v 3.8%.
- Full-val C as per-page mean (correct statistic): ours 0.4398 vs ablation 0.4713 = 6.7% reduction. NOTE the 0.5274/0.5631 in FINAL_COMPARISON.txt are batch-pooled (batch 16), a different statistic — do not mix.
- Convention in the paper: relative reduction = (ablation - ours) / ablation.
- Authors/affiliation in main.tex are placeholders.
- Open: more seeds; real-photo gap; user to fill author names and pick a venue.

### Where everything is

| what | where |
|---|---|
| project root | `<project>` |
| code | `<project>\augmentation\*.py`, `<project>\training\*.py` |
| built dataset | `<project>\dataset\` — 24.6 GB, 145,313 samples |
| layout store | `<project>\dataset\layout\` — 57 MB |
| upload staging | `<project>\upload\code\`, `<project>\upload\layout\` |
| Kaggle notebook | `<project>\dewarp_kaggle.ipynb` (17 cells) |
| eval notebook | `<project>\dewarp_eval.ipynb` (12 cells) — optional; local is simpler |
| after-a-session steps | `<project>\AFTER_TRAINING.txt` — plain-language, for the user |
| session 2 steps | `<project>\SESSION_2.txt` |
| ablation steps | `<project>\ABLATION.txt` |
| **final result** | `<project>\FINAL_COMPARISON.txt` |
| **OCR result** | `<project>\ocr_results\` (OCR_RESULT.txt, per_page.csv, raw texts, examples) |
| OCR scripts | `training\extract_ocr_subset.py` -> `ocr_subset.npz` (334 pages, 70 MB); `training\ocr_eval.py` |
| run outputs | `runs\session1`, `runs\session2` (layout arm); `runs\abl1`, `runs\abl2` (ablation) |
| runbook | `<project>\KAGGLE_GUIDE.md` |
| this log | `<project>\PROJECT_LOG.md` — current state + sessions 20+ |
| older log | `<project>\PROJECT_LOG_ARCHIVE.md` — sessions 1-19, verbatim |
| Kaggle user | `<kaggle-user>` |
| Kaggle datasets | `publaynet-dewarp-code` / `-layout` / `-data`, all **private** |
| Kaggle mount shape | `/kaggle/input/datasets/<kaggle-user>/<slug>` — but **never hard-code it**, pass `auto` |

The local Linux workspace on the Windows machine stopped starting on 2026-09-13
(a Windows update from Sept 8). File staging and committing still work; anything
needing a shell has to be run by the user.

### The claim the paper rests on

A **layout-consistency loss**: PubLayNet ships semantic region boxes (text /
title / list / table / figure) for every page, and no dewarping work has used
them. A correctly dewarped text block must come out as an axis-aligned
rectangle. The loss is the variance of the *error field* `d = f_pred - f_gt`
within horizontal bands of each region. Demonstrated to do what it claims: at an
identical 1.0 px L1 error, a rigid shift scores **0.0000** and a bend scores
**2.2820**.

The experiment is one ablation: `--w_layout 0.2` against `--w_layout 0`,
everything else byte-identical.

### Constants that must not drift

| | |
|---|---|
| page | **448 x 576** |
| flow grid | **112 x 144** (quarter res; where the decoder ends and losses are computed) |
| flow | normalized displacement, `grid_sample` convention, float16 |
| layout coords | uint16, normalized by the **recorded** page size |
| model | `DewarpNet(base=32)`, 6.45 M params |
| batch | 8 |
| `TOTAL_STEPS` | **540,000** — identical in every session, or the cosine restarts |
| steps/epoch | 17,733 |
| split | by page stem (md5), 2% val — never by sample, variants share a clean image |

### Verified, with numbers

- dataset regenerates **byte-identical** from the manifest (60/60 samples)
- inference path reproduces the verifier: rectified-vs-clean **24.79 dB** against `verify_dataset`'s 25.16 dB, on a different subset
- flow error mean **0.0209 px**; inversion residual **0.0004 px**; rectification **25.16 dB**
- upload is **byte-exact** on Kaggle (`flows.f16` 9,374,432,256 B, `masks.u8` 2,343,608,064 B)
- resume is **bit-exact**: weights and optimizer moments `max |diff| = 0.000e+00`, including resume from a *different directory*
- model learns: overfit EPE 45.07 -> **2.47 px** against a 35.13 px identity baseline
- smoke test on Kaggle: `train 141,869 / val 2,737`, val epe **37.614 -> 36.683** in 50 steps
- throughput **8.63 steps/s** on one T4 (my 2.8 floor was 3x pessimistic)
- layout arm complete: **30 epochs / 531,990 steps in 17.6 h**
- **full val (2,737 samples): EPE 3.644 px** vs 35.13 identity — **89.6% better** (this is the reportable number)
- per profile: cylinder 2.25 / flat 2.38 / perspective 3.22 / combined 5.51
- layout consistency at 30 epochs: **0.5281** (h 0.2501, v 0.2779)
- converged: last 30 validations spread **0.055 px** — more epochs would not help

### Status

- [x] warping pipeline, dataset format, builder, verifier
- [x] layout store + layout loss + tests
- [x] model, losses, training loop, exact resume
- [x] uploaded to Kaggle; notebook built and smoke-tested
- [x] **session 1 of the layout arm — 20 epochs, EPE 37.6 -> 4.03 px**
- [x] session 2 finished the arm — **30 epochs complete**
- [x] **ablation arm complete — final 30-vs-30 result in `FINAL_COMPARISON.txt`**
  - EPE: **tie** (3.662 vs 3.689, 0.7%)
  - layout consistency: **layout arm 6.8% straighter** (0.5274 vs 0.5631); horizontal 9.4%
  - better on all four distortion types
- [x] **inference script** (`training/inference.py`) + **eval notebook** (`dewarp_eval.ipynb`)
- [x] **OCR-CER** on 332 val pages, both arms — result in `ocr_results\OCR_RESULT.txt`
  - layout arm **28.6% vs 30.4% CER** (+1.76 pts, 95% CI +0.78 to +2.79), better on 203 pages vs 126, Wilcoxon p = 4.6e-6
  - the gain is **concentrated on `combined`**: 29.8% vs 34.5% (15.8% fewer errors, 74 vs 25 pages, p = 5e-7);
    cylinder/perspective/flat not individually significant
- [ ] figures (`--mask_invalid`, OCR example strip, failure `PMC5786657_00000`); real-photo test; the paper

### To continue training

Open the notebook, attach all three datasets plus the previous session's saved
output, set `MODE = 'resume'` in cell 5, change nothing else, **Save & Run All**.
Confirm the log shows `[resume] last.pt: epoch N, step M`. If it says
`STARTING FROM SCRATCH`, stop — the previous output is not attached.

### Traps, each of which has already cost something

1. **Never hard-code a Kaggle mount path.** Three different shapes seen so far.
   Pass `auto`; `train.py` finds each store by a file inside it.
2. **`Save & Run All` re-runs the notebook from the top in a fresh session.** It
   does not photograph an interactive session. Never train interactively and
   then commit — that throws the work away and starts an identical run.
3. **A notebook pins the dataset version it was attached at.** Re-uploading code
   changes nothing until the input is removed and re-added. Cell 4 detects this.
4. **`/kaggle/working` is wiped** unless a version is saved, and a session killed
   at the 12-hour cap saves nothing. `--max_hours 11.0` exits inside the limit.
5. **`kaggle datasets create` skips subdirectories** — hence three datasets.
6. **Judge the smoke test on `final val epe` vs `[val @ start]`**, never on the
   per-step line: warmup is 500 steps, so at step 50 the LR is 3e-05 and the
   per-step numbers are batch noise.
7. `--resume_dirs auto` matters most of all: this failure is **silent**. Without
   it session 2 restarts from zero and the loss curve still looks plausible.


### Open decision

The instance has **two** T4s; the code uses one. `DataParallel` would give maybe
1.4-1.6x, cutting both arms from ~74 to ~50 GPU-h. Against it: new code on the
**resume path** — the most fragile part, and the one place a bug is silent — and
the bit-exact proof was single-GPU. It must be decided before session 1, since
changing effective batch mid-run makes the arms incomparable. **Decided: stay
single-GPU.**

---

## 📅 Session Log

Sessions 1-19 are in **`PROJECT_LOG_ARCHIVE.md`** — the full text, nothing
deleted. This file keeps the current state, the index below, and the sessions
still close enough to matter. Split on 2026-09-13 at 124 KB, because a log you
cannot read cold is not a log.

### Index of archived sessions

| # | what it established |
|---|---|
| 1 | Task, dataset, U-Net + flow-field architecture chosen over pixel-to-pixel |
| 2 | Novelty analysis; 8 SOTA papers reviewed; **the layout-consistency gap identified** |
| — | *(also archived: recommended paper angle, architecture discussion, decisions, folder structure, the Related Work paper table, tech stack)* |
| 3 | Augmentation pipeline built, then tested |
| 4 | Dataset scale decided |
| 5 | **Storage blocker** — naive format projected 1.24 TB. Format redesigned. Resolution locked |
| 6 | **Fix 1** — cylindrical warp had `flow_y == 0` and could not bow text at all. Replaced with 3-D cylinder + pinhole projection. Found a worse bug while verifying it |
| 7 | Pipeline run on real PubLayNet images |
| 8 | Resolution switch to 448x576; **fix 9** — deterministic profiles (17% of samples were accidental identities) |
| 9 | **Fix 4** — background + validity mask; **issue 12** — coverage leaked warp severity (corr -0.825), fixed with camera framing |
| 10 | **Fix 2** — storage rewrite; cascaded resampling replaced by compose-then-resample-once. Label error 4.9 grey levels -> exactly 0 |
| 11 | Flow resolution decided (112x144); pilot dataset built |
| 12 | Annotations located; naive COCO parser read 19 of 120,761 records — depth-aware streaming reader written |
| 13 | **Issue 13 closed** — annotations align exactly with images |
| 14 | **Fix 3 — the layout-consistency loss.** Rigid shift 0.0000 vs bend 2.2820 at identical L1 error |
| 15 | Full build + layout store complete; full audit |
| 16 | Dataset **VERIFIED** (byte-identical regeneration); training wrapper; issue 15 closed |
| 17 | Model + training loop built and tested locally; overfit EPE 45.07 -> 2.47 px |
| 18 | **Multi-session continuation fixed** — `ResumableSampler`, split `--steps`/`--total_steps`, bit-exact resume proven. Kaggle runbook written |
| 19 | Upload path fixed — `kaggle datasets create` skips subfolders, hence three datasets |

### Recent sessions

### 2026-09-13 — Session 20: Kaggle Mount Paths Are Not What the Docs Say

#### The symptom
Notebook with both small datasets **attached and listed in the Input panel**, and:

```
!ls /kaggle/input/
datasets

!cp /kaggle/input/publaynet-dewarp-code/*.py /kaggle/working/code/
cp: cannot stat '/kaggle/input/publaynet-dewarp-code/*.py': No such file or directory
```

My first read of this was wrong: I told the user nothing was attached. The Input panel
said otherwise. The real cause is that **Kaggle does not guarantee the mount path**. Older
notebooks use `/kaggle/input/<slug>`; this one nests it under a `datasets/` directory
(`/kaggle/input/datasets/<owner>/<slug>/versions/N`). Nothing outside the notebook can
tell you which you will get.

This is the worst failure shape there is: the thing that is broken (a path constant)
produces an error that points at something else entirely (a missing dataset), so the
natural response is to re-upload 22.9 GiB that was never the problem.

#### Fix — locate stores by content, never by path
`train.py` gained `find_under(base, marker)` and `resolve_input(spec, marker, label)`.
Passing the literal `auto` to `--root`, `--layout_root` or `--resume_dirs` walks
`/kaggle/input` (depth-capped at 7) for a file known to live inside that store:

| flag | marker |
|---|---|
| `--root auto` | `flows.f16` |
| `--layout_root auto` | `stems.jsonl` |
| `--resume_dirs auto` | `last.pt` |

Any other value passes through unchanged, so local runs are untouched. It prints what it
resolved, lists every candidate when ambiguous, and exits with an instruction — not a
stack trace — when a dataset is simply not attached yet.

`--resume_dirs auto` matters most: this exact path problem, hitting the resume flag in
session 2, is silent. The run starts from scratch, the loss curve looks plausible, and a
whole 11-hour session is lost.

#### Verified against the real failure
Built a fixture replicating the user's notebook exactly — `/kaggle/input` containing only
`datasets`, with the pilot dataset and its layout store at
`datasets/<kaggle-user>/<slug>/versions/1` — and ran `train.py --root auto --layout_root
auto` through to a checkpoint:

```
[paths] root = /kaggle/input/datasets/<kaggle-user>/publaynet-dewarp-data/versions/1
[paths] layout_root = /kaggle/input/datasets/<kaggle-user>/publaynet-dewarp-layout/versions/1
train 136  val 4  68 steps/epoch
[val @ start] epe 24.097px
finished at step 3
```

#### Issue 18 found while testing: an empty split crashed 200 lines downstream
The first fixture attached a layout store belonging to *different* pages. Every sample was
dropped for having no layout annotations, both splits came out empty, and the run
continued silently until `KeyError: 'epe_px'` inside the validation aggregator — a message
with no connection to the cause.

On Kaggle that is a realistic mistake (wrong dataset, or a stale version attached), so
`train.py` now refuses an empty split immediately and prints the diagnosis:

```
EMPTY SPLIT — train 0, val 0.
  manifest stems 48, layout stems 14, overlap 0
  The two stores share NO page stems: the layout dataset does not belong to this image dataset.
    manifest e.g. ['PMC509424_00001', ...]
    layout   e.g. ['PMC3335537_00001', ...]
```

#### Also updated
- `KAGGLE_GUIDE.md` step 3 rewritten around the discovery cell, with the `cp: cannot stat`
  error written out verbatim so it is searchable next time.
- `prepare_upload.py` no longer prints path constants it cannot know; it prints the
  four notebook cells, and `kaggle datasets version` for re-uploads.
- The `TO CONTINUE` banner at the end of a run now says `--resume_dirs auto`.

#### Note on the device
The local Linux workspace on the Windows machine stopped starting this session
("Workspace unavailable... a Windows update released September 8"). File staging and
committing still work, so edits go through those. Anything needing a shell has to be run
by the user.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-17 | | all closed |
| 18 | Kaggle mount path is not `/kaggle/input/<slug>` | **FIXED (S20)** — resolve by content |
| 19 | Empty split crashed downstream with an unrelated error | **FIXED (S20)** |

#### Next Steps
- [ ] Re-upload the code dataset (`kaggle datasets version -p . -m "auto paths"`) — 200 KB
- [ ] Finish the 22.9 GiB data upload, attach, smoke test
- [ ] Session 1 of the layout arm
- [ ] Inference script + Local Distortion / OCR-CER evaluation
- [ ] Ablation arm with `--w_layout 0`

*Log last updated: 2026-09-13 — Session 20*

---

### 2026-09-13 — Session 21: The Kaggle Notebook, Shipped and Tested

All three datasets are live on Kaggle (code 80 kB / 16 files, layout 57 MB,
data 25 GB), so the remaining gap was that the notebook existed only as
instructions in a chat. It is now a file: **`dewarp_kaggle.ipynb`**, 19 cells,
import-ready.

#### What the notebook does, and why each cell earns its place
| # | cell | the failure it prevents |
|---|---|---|
| 1 | GPU / torch / disk | discovering there is no accelerator after staging everything |
| 2 | locate mounts by content | the path problem from S20 |
| 3 | **blob-size arithmetic** | a truncated 25 GB upload killing an 11-hour run at hour 6 |
| 4 | `find_under` present in `train.py`? | Kaggle pinning the notebook to the OLD code version |
| 5 | smoke test, 50 steps | everything else, in two minutes |
| 6 | session 1 | — |
| 7 | read back `log.jsonl` | not knowing what the session actually achieved |
| 8 | resume cell | session 2 silently restarting from scratch |
| 9 | ablation | — |

Cell 3 is the new idea. The flow and mask blobs are fixed-stride, so their exact
byte length follows from `meta.json`: `flows.f16` must be
`n_samples x flow_h x flow_w x 2 x 2` and `masks.u8` a quarter of that. For the
real dataset that is 145,313 x 64,512 = **9,374,432,256 bytes** and
**2,343,608,064 bytes**. A partial upload cannot satisfy that arithmetic, and
checking costs two seconds against an 11-hour downside.

Cell 4 closes a trap I had not considered: **a Kaggle notebook pins the dataset
version it was attached at.** Pushing `kaggle datasets version` for the code and
rerunning changes nothing — the notebook keeps using the old copy, so the `auto`
path fix from S20 would appear not to work, with no message saying why. The cell
greps the mounted `train.py` for `find_under` and stops with the fix spelled out.

#### Tested against a replica, not just written
Built `/kaggle/input` containing only `datasets`, with all three stores at
`datasets/<kaggle-user>/<slug>/versions/N` — the user's actual mount shape — and
ran the notebook's cells against it:

```
CODE    /kaggle/input/datasets/<kaggle-user>/publaynet-dewarp-code/versions/2
LAYOUT  /kaggle/input/datasets/<kaggle-user>/publaynet-dewarp-layout/versions/1
DATA    /kaggle/input/datasets/<kaggle-user>/publaynet-dewarp-data/versions/1
  flows.f16     9,612,288 bytes  expected     9,612,288  OK
  masks.u8      2,403,072 bytes  expected     2,403,072  OK
DATASET OK
code version OK ... 15 files staged
```

Then the smoke-test cell verbatim, and the resume cell verbatim with a previous
run planted under `/kaggle/input`:

```
[paths] resume candidate /kaggle/input/datasets/<kaggle-user>/dewarp-run-1/versions/1
[resume] last.pt: epoch 0, step 4, 32 samples into the epoch
finished at step 6
```

So the continuation path is confirmed working *through a nested mount*, which is
the combination that would have failed silently in session 2.

`nbformat.validate` passes; all 19 cells carry ids.

#### `KAGGLE_GUIDE.md` rewritten start to finish
Five steps: stage, upload, notebook, session 1, sessions 2+. Adds `kaggle
datasets version` for re-uploads (the guide only ever showed `create`), the two
named error messages and their fixes, and the corrected 74 MB checkpoint size.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-19 | | all closed |
| 20 | Notebook pins the old dataset version after a re-upload | **FIXED (S21)** — cell 4 detects it |

#### Next Steps
- [ ] Import the notebook, attach all three datasets, run cells 1-5
- [ ] Record steps/s from the smoke test — it sets the session budget
- [ ] Session 1 of the layout arm
- [ ] Inference script + Local Distortion / OCR-CER evaluation
- [ ] Ablation arm with `--w_layout 0`

*Log last updated: 2026-09-13 — Session 21*

---

### 2026-09-13 — Session 22: Kaggle File Persistence, and a Workflow Error in My Own Guide

The user asked what the **Session options** panel means for files. Answering it
surfaced a mistake in the runbook I wrote in S18 and repeated in S21.

#### What I had wrong
The guide said: run the training cell, *then* **Save Version → Save & Run All
(Commit)**. That is wrong, and expensively so. **Save & Run All starts a fresh
batch session and re-runs the notebook from the top.** It does not photograph the
interactive session. Following my instruction literally means training for 11
hours interactively and then throwing it away to start an identical 11-hour run
in the background — two sessions of a 30 GPU-h weekly quota for one session of
progress.

Correct workflow: validate interactively with cells 1-5 (~3 min), then commit and
let the **batch** session do the training. (*Quick Save* with output included is
the other valid path — it does snapshot `/kaggle/working` — but an interactive
session that idles out or disconnects loses everything, so it is the weaker
option for an 11-hour run.)

#### The notebook had a matching structural bug
It contained four unguarded training cells (smoke, session 1, resume, ablation).
Under *Run All* — which is exactly what a commit does — that attempts all four in
one 12-hour session and finishes none. Since the commit path is now the
recommended one, this would have bitten on the first real session.

Restructured around a single switch:

```python
MODE = 'smoke'   # 'smoke' | 'session1' | 'resume' | 'ablation'
```

with one run cell that builds the argument list from `PRESETS[MODE]`. *Run All*
now performs exactly one training run, and moving between sessions is a
one-string edit.

#### Every `!` cell replaced with a streaming subprocess
`!` lines cannot be conditional, which the `MODE` switch requires. The run cell
uses `Popen(..., stdout=PIPE)` and echoes lines as they arrive. **`python -u` is
load-bearing**: a child writing to a pipe rather than a tty block-buffers, so
without it an 11-hour run shows a blank cell for an hour at a time, in the batch
log too.

#### Storage facts, written into both the notebook and the guide
| path | writable | survives |
|---|---|---|
| `/kaggle/input/...` | no | always |
| `/kaggle/working/` | yes | only via a saved version |
| `/kaggle/temp/`, `/tmp` | yes | never |

**Persistence: "Files only"** keeps `/kaggle/working` across *interactive*
restarts. It does **not** produce anything attachable as an input to the next
notebook — only a saved version's output does, and that is what the continuation
chain runs on. Convenience, not backup.

**Environment: "Pin to original environment"** — recommended to keep pinned. This
run spans weeks and 4+ sessions; an environment update mid-run could change the
torch version under a resumed checkpoint and alter numerics silently, which would
make session 4 incomparable to session 1 and quietly invalidate the ablation.

Output cap is ~20 GB; this run stays near **300 MB** (74 MB checkpoints, pruned
to last two + `best.pt` + `last.pt`), so size is not a risk.

I could not load Kaggle's own docs to cite (the pages are JS-rendered and return
only metadata to a fetcher), so the size cap is stated as approximate.

#### Also added
`assert torch.cuda.is_available()` in cell 1. Forgetting the accelerator is easy
and the failure is silent — training would run on CPU at roughly 1/50th speed and
the session would end a few hundred steps in.

#### Verified
All 17 cells executed end to end against a replica of the user's mount shape
(`/kaggle/input` containing only `datasets`): environment, discovery, integrity
arithmetic, code staging, config, the run cell through to a checkpoint, and the
summary reading `log.jsonl` back. `nbformat.validate` passes; every cell is pure
Python and compiles.

#### Issue Status
| # | Issue | Status |
|---|---|---|
| 1-20 | | all closed |
| 21 | Guide told the user to train interactively then commit — commit re-runs from scratch | **FIXED (S22)** |
| 22 | Four unguarded training cells would all fire under Run All | **FIXED (S22)** — `MODE` switch |

#### Next Steps
- [ ] Cells 1-5 interactively, `MODE = 'smoke'` — record steps/s
- [ ] `MODE = 'session1'`, Save & Run All (Commit)
- [ ] Session 2: attach the output, `MODE = 'resume'`, confirm the `[resume]` line
- [ ] Inference script + Local Distortion / OCR-CER evaluation
- [ ] Ablation arm

*Log last updated: 2026-09-13 — Session 22*

---

### 2026-09-13 — Session 23: Smoke Test on Kaggle — Everything Passes

First run on real hardware. 2x Tesla T4, `cuda True`, AMP on, `/kaggle/working`
19.5 GiB (so the output cap is 19.5 GB, not the ~20 GB I had been saying).

#### Mount paths
Resolved to `/kaggle/input/datasets/<kaggle-user>/<slug>` — **no `versions/N`
segment**, a third shape on top of the two I already knew about. The
resolve-by-content approach from S20 handled it without a change, which is the
whole argument for not hard-coding paths.

#### Dataset integrity — exact
```
145,313 samples | page 448x576 | flow 112x144
  flows.f16      9,374,432,256 bytes  expected   9,374,432,256  OK
  masks.u8       2,343,608,064 bytes  expected   2,343,608,064  OK
  9 page shards, 4 clean shards, 11.87 GiB of JPEG
  manifest.jsonl  145,313 rows | clean_index.jsonl  47,958 rows
DATASET OK
```
The 25 GB upload is byte-exact. Nothing was truncated.

#### Split
`train 141,869 / val 2,737`, 707 dropped (0.487%) — sums to 144,606 exactly as
predicted, val share 1.89%. **17,733 steps/epoch.**

#### The check criterion I wrote was wrong
The guide said "check `epe_px` is falling across the 50 steps". That is not a
usable test and I should have seen it when I wrote it: **warmup is 500 steps**, so
at step 50 the LR is `3e-4 x 51/500 = 3.06e-05` — a tenth of the way up. The
per-step numbers (38.8 / 38.9 / 47.3) are batch composition noise, not a trend;
a batch of 8 mixes cylinder, perspective and combined profiles whose difficulty
differs by more than that spread.

The signal that *is* comparable is the fixed validation set:
**37.614 px -> 36.683 px, -0.931 px in 50 steps at 10% warmup LR.** Right
direction, and about the magnitude to expect. Guide corrected.

#### Throughput
50 steps in 0.3 min including loader spin-up = **>=2.8 steps/s**, steady state
likely higher. Budget at that floor:

| steps/s | min/epoch | epochs per 11 h | sessions per arm | GPU-h both arms |
|---|---|---|---|---|
| 2.8 | 106 | 6.3 | 4.8 | 106 |
| 4.0 | 74 | 8.9 | 3.4 | 74 |
| 5.0 | 59 | 11.2 | 2.7 | 59 |

At 30 GPU-h/week that is 2.5-3.5 weeks for both arms. The real number comes out
of session 1's `log.jsonl`.

#### Open decision, deliberately deferred
The instance has **two** T4s and the code uses one. `DataParallel` would give
maybe 1.4-1.6x (not 2x — batch 8 splits to 4/GPU, poor utilisation, plus
gather/scatter), so ~74 GPU-h becomes ~50. Against that: it is new code on the
**resume path**, which is the most fragile part of the system and the one place a
silent bug costs whole sessions rather than minutes, and the bit-exact resume
proof from S18 was single-GPU. It also has to be decided before session 1, since
changing effective batch mid-run makes the arms incomparable. **Recommended: go
single-GPU.** Verified beats fast here.

#### Note
`TOTAL_STEPS = 540,000` against an actual 30 x 17,733 = 531,990. The cosine
therefore ends at 98.5% of its horizon, LR ~1.6e-07 instead of 0. Negligible, and
not worth the risk of a constant that has to stay identical across five sessions
being edited.

#### Next Steps
- [ ] `MODE = 'session1'`, Save & Run All (Commit)
- [ ] Read steps/s out of session 1's log; firm up the session budget
- [ ] Session 2: attach the output, `MODE = 'resume'`, confirm the `[resume]` line
- [ ] Inference script + Local Distortion / OCR-CER evaluation
- [ ] Ablation arm

*Log last updated: 2026-09-13 — Session 23*

---

### 2026-09-13 — Session 24: Inference + the Evaluation Notebook

Session 1 is training. The user's point was fair: I log metrics every epoch but
save no images, so there is no way to *look* at the model. They proposed a web
app for uploading a checkpoint and some images.

#### Why not a web app
A browser cannot run a 74 MB PyTorch checkpoint. It would need an ONNX export
plus `onnxruntime-web`, and inside an Artifact the CSP allows the library script
from a CDN but blocks the `.wasm` it then fetches — so it would fail, and only
at runtime. A server-backed app means gradio on Kaggle with internet on, burning
GPU quota to serve a UI.

The checkpoint is already sitting on the machine that has the GPU, the data and
torch. Rendering the comparison there and looking at a PNG is strictly less
machinery for strictly more information — it can show ground truth alongside,
which a drag-and-drop demo cannot.

#### `training/inference.py` (new, 412 lines) — this was owed anyway
It was already on the todo list. Provides `load_checkpoint`, `predict_flow_norm`,
`rectify`, `dewarp`, `preview_from_dataset`, `epe_by_profile`, and a CLI with a
`--image` mode for the user's own photos (CPU is fine, 6.45 M params).

Four things that had to be right:

**1. The inversion direction.** Stored `f` satisfies `warped[y,x] =
clean[y+f_y, x+f_x]` — domain is the warped image, which is what a CNN wants to
predict, but it is *not* the rectifying map. That needs `g` with `clean[y,x] =
warped[y+g_y, x+g_x]`. Getting it backwards does not crash; it returns a
plausible image warped about twice as much.

**2. Resolution.** Flow is normalized, so a 112x144 prediction upsamples to any
size and rectification runs at the input's own resolution. The coarse grid costs
smoothness, not sharpness.

**3. The PSNR metric — caught by a disagreement with our own verifier.** My first
version scored 13.26 dB where `verify_dataset.py` reports 25.16 dB on the same
operation. Two causes, both real:
- the mask must be **page visibility after rectification**
  (`apply_flow(mask, g) > 0.999`), not "source landed in bounds" — otherwise
  background is being scored against paper;
- the warp bakes in a **multiplicative shadow**, so PSNR must be gain-corrected
  by a low-pass ratio, or it measures lighting rather than geometry and punishes
  a perfect dewarp for a dark corner.

Matching the verifier exactly: **24.79 dB** vs its 25.16 dB on a different
subset. Now comparable across the project, and the honest ceiling for the model.

**4. `base` is read off the weights.** `head.weight` is `(2, base*2, 3, 3)`.
Found by loading an old `--base 16` checkpoint and getting 60 lines of
`size mismatch` that name every tensor and never the flag. A checkpoint carries
no record of the width it was trained with, so it is inferred.

#### The contact sheet
Four columns per sample: **input | PREDICTED | rectified with TRUE flow |
ground truth**, spread across profiles rather than taking the first n.

Ground-truth layout boxes are drawn on columns 2 and 3. In a flat page those
boxes *are* axis-aligned rectangles around text, so drawn on the model's output
they show directly whether dewarped text came out straight — the novelty claim,
made visible. Column 3 is the ceiling, so the gap between 2 and 3 is what is left
to learn; comparing 2 against 4 would be comparing against something
unreachable, since rectification cannot restore detail the warp destroyed.

#### `epe_by_profile`
The aggregate hides the answer. On the 40-step checkpoint: `combined` 49.9 px,
`perspective` 35.7, `cylinder` 31.6, `flat` 15.3, aggregate 37.0. A model can
sit at a decent average and be failing entirely on `combined`, which is the
hardest profile and the closest to a real photograph. `flat` doubles as a sanity
check — if it is not the lowest of the four, something is wrong beyond needing
more training.

#### `dewarp_eval.ipynb` (12 cells)
Read-only; trains nothing. Finds the code/data/layout mounts *and the training
run's saved output* by content, stages the code, plots the curve, renders the
sheet, prints the per-profile table, writes `eval_summary.json`.

One bug caught while testing it against a real `log.jsonl`: `train.py` tags
validations four ways — `val_at_start`, `val`, `val_epoch`, `final` — and the
photometric key is `photometric`, not `photo`. Filtering for `split == 'val'`
alone silently drops the **epoch-boundary** validations, which are the ones worth
looking at. Fixed; epoch ends are now marked separately on the plot.

#### Verified
Both notebooks executed end to end against a replica mount, including a planted
run output. The inference math was checked independently of model quality by
pushing the **ground-truth** flow through the same rectify path and matching the
verifier's number.

#### Note
`inference.py` must be in the Code dataset before the eval notebook can run:
`cd <project>\upload\code && kaggle datasets version -p . -m "inference"`.
Safe while session 1 runs — a notebook pins the dataset version it attached.

#### Next Steps
- [ ] Re-upload the code dataset (200 KB) so it carries `inference.py`
- [ ] When session 1 finishes: save the version, then run `dewarp_eval.ipynb`
- [ ] Session 2: `MODE = 'resume'`
- [ ] OCR-CER evaluation (the other half of what the paper reports)
- [ ] Ablation arm

*Log last updated: 2026-09-13 — Session 24*

---

### 2026-09-13 — Session 25: Inference Belongs on the Local PC, Not Kaggle

The user asked why inference needs Kaggle at all. It does not, and I had not
thought it through — I built the eval notebook because the *checkpoint* is on
Kaggle and stopped there, without noticing that everything else it needs is
already on their machine: the 24.6 GB dataset, the layout store, torch, OpenCV.
Only the 74 MB weights file has to come down.

I had also told them to re-upload the code dataset so the eval notebook could
import `inference.py`. Unnecessary. Withdrawn.

#### Measured on CPU, base=16, container
| | |
|---|---|
| single photo | **2.2 s** (including interpreter + torch start) |
| 4-sample contact sheet | **6.3 s** |
| `epe_by_profile` | 8.7 samples/s -> ~23 s for 200 |

Scaling to the real `base=32`, 200 samples is ~90 s. Nothing here wants a GPU:
the model is 6.45 M params and `invert_flow` is 60 `cv2.remap` passes.

The lesson is the ordinary one — "where does the artifact live" was the wrong
question; "where does the *work* need to happen" was the right one, and the
answer was the machine that already holds 24.6 GB of inputs.

#### `--profile_table N` added
While writing the instructions I found myself typing "ask Claude for the
per-type table, it is one small script", which is a bad sentence to put in a
runbook. `epe_by_profile` already existed; it just was not reachable from the
CLI. Now:

```
python training\inference.py --ckpt best.pt --root dataset --n 6 \
       --out preview.png --profile_table 200
```

prints the table, with the `flat`-should-be-lowest sanity note printed alongside
it rather than living only in my head.

#### `AFTER_TRAINING.txt`
Plain-language, no markdown, written for the user rather than for me. Four steps,
how to read the sheet, what each printed number means and what counts as bad, a
preview of session 2, and an index of what every file in the folder is for.
Deliberately in the folder rather than in chat, since chat is what gets lost.

The eval notebook stays as an option — it also draws the loss curve — but local
is the documented default.

#### Next Steps
- [ ] Session 1 finishes -> download `best.pt`, run the two local commands
- [ ] Read the sheet and the numbers; decide whether to keep training
- [ ] Session 2: `MODE = 'resume'`
- [ ] OCR-CER evaluation
- [ ] Ablation arm

*Log last updated: 2026-09-13 — Session 25*

---

### 2026-09-15 — Session 26: Session 1 Results — It Works

#### Headline
| | |
|---|---|
| steps | **341,833** in 11.00 h |
| throughput | **8.63 steps/s** — my 2.8 floor was 3x pessimistic |
| epochs | **20 of 30** |
| val EPE | **37.614 -> 4.030 px** |
| vs identity (35.13 px) | **88.5% better** |
| layout term | 3.867 -> 0.555 |

The model learns, the pipeline holds, and the continuation design was never
exercised in anger because one session got two thirds of the way through.

#### Epoch-end validation
```
 0  9.065     5  6.127    10  4.926    15  4.295
 1  7.408     6  5.615    11  4.689    16  4.195
 2  7.285     7  5.264    12  4.956    17  4.303
 3  6.530     8  5.143    13  4.608    18  4.119
 4  6.089     9  5.037    14  4.624
```

#### Is another session worth it? Yes, and the reason is the schedule
The last five epoch-ends read 4.624 / 4.295 / 4.195 / 4.303 / 4.119 — a 0.504 px
spread that looks like a plateau if you squint. It is not one. At step 341,833
of a 540,000-step cosine horizon the LR is **8.93e-05, still 30% of peak**. The
anneal to 1.63e-07 has not happened yet, and that final decay is where a cosine
schedule usually buys its last real drop.

Reading the flatness as convergence and stopping here would be mistaking "LR
still high enough to bounce" for "nothing left to learn". 190,157 steps remain =
10.7 epochs = **6.1 h**, comfortably inside one session.

#### Two path bugs in my own instructions
**1.** I told the user `runs\session1\best.pt`. `kaggle kernels output`
downloads the whole `/kaggle/working` tree, so the real path is
`runs\session1\run_layout\best.pt`. The error named a path that looks exactly
right, which is the worst kind.

Fixed properly rather than by correcting the docs: `--ckpt` now accepts **a file
or any folder above one** and searches for `best.pt`, then `last.pt`, then the
highest step checkpoint, printing what it picked. `--ckpt runs\session1` works
regardless of how Kaggle nests the download.

**2.** The banner printed the path the user typed rather than the file actually
loaded — harmless until the folder form made them different. Now prints the
resolved file.

#### Note on the downloaded output
It also contains `code/` (the staged modules) and `dewarp.log` (2.75 MB of
console output), plus `ckpt_step000341500.pt` and `ckpt_step000341833.pt`
alongside `best.pt` and `last.pt`. Pruning worked: four checkpoints, ~310 MB,
against the 19.5 GB cap.

#### Next Steps
- [x] Looked at `preview.png` — 5 of 6 clearly dewarped, 1 hard failure
- [ ] Session 2: `MODE = 'resume'`, ~6.1 h, finishes the arm at 30 epochs
- [ ] Then the ablation arm (`MODE = 'ablation'`), ~2 sessions at 8.63 steps/s
- [ ] OCR-CER evaluation

*Log last updated: 2026-09-15 — Session 26*

---

### 2026-09-15 — Session 27: Looking at the Pictures

Ran the sheet on `best.pt` (step 341,000, 20 epochs), 6 val samples spread
across profiles.

```
mean EPE            4.705 px
PSNR input         12.73 dB
PSNR predicted     16.00 dB   (+3.26)
PSNR true flow     25.90 dB   <- ceiling
```

| sample | profile | EPE px |
|---|---|---|
| PMC2744922_00003 | flat | **1.127** |
| PMC5633055_00002 | cylinder | 1.629 |
| PMC5295534_00001 | cylinder | 2.474 |
| PMC5954076_00001 | perspective | 3.945 |
| PMC5596512_00002 | combined | 4.845 |
| PMC5786657_00000 | combined | **14.208** |

#### What the pictures show
Five of six are genuinely dewarped: page upright and filling the frame, text
lines horizontal, and — the test that matters — the ground-truth layout boxes
sitting on the text blocks rather than cutting across them. Column 2 is close
enough to column 3 that they take a moment to tell apart on the two cylinder
rows.

**Row 5 is a real failure**, not noise: a severely warped page where the output
comes back compressed and the boxes land nowhere near the text. 14.2 px against a
4.7 px mean. Both failures are `combined`, which is the profile that stacks
cylinder + perspective + wrinkle and is the closest thing here to an actual
photograph.

The ordering is exactly as predicted — flat < cylinder < perspective < combined —
so the sanity check passes and the error is concentrated where the physics is
hardest, which is the good kind of concentration.

#### On the PSNR gap
16.0 dB against a 25.9 dB ceiling looks alarming next to a 4.7 px EPE. It is
mostly a property of the metric: text is the highest-frequency content on a page,
so a few pixels of residual misalignment destroys PSNR far faster than it
destroys legibility. Worth stating plainly in the paper rather than letting a
reviewer find it — and worth being the reason OCR-CER, not PSNR, is the
readability number we report.

#### Still worth running session 2
Nothing here changes the S26 conclusion. The LR is at 30% of peak with the whole
anneal ahead of it, and the hard tail (`combined`) is exactly what a low-LR
refinement phase tends to help most.

#### `SESSION_2.txt` written
Six steps, plain language, in the folder. The load-bearing one is step 6: check
the run prints `[resume] last.pt: epoch 19, step 341,833` and `[val @ start] epe
~4`. If it says 37, the previous output is not attached and the session is about
to redo 11 hours of finished work.

#### Next Steps
- [ ] Session 2: `MODE = 'resume'`, ~6.1 h, finishes the arm
- [ ] Optional while it runs: `--profile_table 200` for a per-profile baseline
- [ ] Ablation arm (`MODE = 'ablation'`), ~2 sessions
- [ ] OCR-CER evaluation — the readability number, since PSNR understates it

*Log last updated: 2026-09-15 — Session 27*

---

### 2026-09-15 — Session 28: Going Past 30 Epochs — the Mechanics and the Cost

User asked what it takes to train longer. Recorded because the answer has a trap
in it and a deadline-shaped constraint.

#### The trap: two numbers, not one
`--epochs` is the stop condition. `TOTAL_STEPS` is the cosine horizon. Raising
epochs alone buys nothing — LR has already annealed to **1.63e-07** by step
531,990, so the extra epochs run at effectively zero learning rate and the curve
flatlines in a way that looks like convergence but is just a dead schedule.

#### Extending after 30 is a warm restart, and that is fine
Because LR is a pure function of `(gstep, TOTAL_STEPS)`, raising the horizon on a
resumed run makes LR jump back up and anneal again — SGDR by accident, and a
published technique rather than a hack. From step 531,990:

| new epochs | TOTAL_STEPS | LR jumps to | % of peak |
|---|---|---|---|
| 35 | 620,655 | 1.49e-05 | 5% |
| 40 | 709,320 | 4.40e-05 | 15% |
| 45 | 797,985 | 7.51e-05 | 25% |
| 50 | 886,650 | 1.04e-04 | 35% |

Must be documented in the paper as a schedule extension at step 531,990, and
applied **identically** to the ablation.

#### Cost, at the measured 8.63 steps/s
1 epoch = 34.2 min. An 11-h session = 19.3 epochs.

| epochs | hours/arm | **both arms** |
|---|---|---|
| 30 | 17.1 | **34.2 h** |
| 40 | 22.8 | **45.7 h** |
| 50 | 28.5 | **57.1 h** |
| 60 | 34.2 | **68.5 h** |

At 30 GPU-h/week: 30 epochs is ~1.2 weeks for both arms, 50 epochs ~2 weeks. The
ablation is not optional — it is the claim — so every number in the first column
is really the number in the third.

#### Recommendation: finish 30 first, then decide
Deciding now would also mean changing `TOTAL_STEPS` mid-arm, re-stretching a
schedule session 1 already ran under. Deciding after 30 is clean, honest and
cheap. Extending later is easy; un-spending quota is not.

#### And epochs may be the wrong lever
If 30 epochs lands short, the candidates are a wider model (`--base 48`; it is
only 6.45 M params) or a flow grid finer than 112x144 — the 14.2 px `combined`
failure in S27 has the shape of a resolution limit on sharp wrinkle detail, not
an undertraining one. Both mean starting over, so they are decisions to make on
evidence, after seeing the annealed curve and a per-profile table.

#### Next Steps
- [ ] Session 2: `MODE = 'resume'`, ~6.1 h, finishes 30 epochs
- [ ] Read the annealed curve; decide extend / bigger model / stop
- [ ] Ablation arm, matching whatever was decided
- [ ] OCR-CER evaluation

*Log last updated: 2026-09-15 — Session 28*

---

### 2026-09-15 — Session 29: Layout Arm Done; Notebook Gets Two More Modes

30 epochs complete. Results not yet downloaded — the ablation is the critical
path, so it starts first and the inspection happens while it trains.

#### Why the ablation does not have to wait for the inspection
The open "should we train past 30 epochs" question does not block it. Extending
later is a warm restart **on both arms**, and both resume from their own
checkpoints, so nothing done now is wasted whichever way that decision goes. The
only thing that would be wasted is idle GPU time.

#### Two new modes
`'ablation'` existed; `'ablation_resume'` did not. The notebook told the user to
"edit `PRESETS['ablation']` to add `--resume_dirs auto`" for later sessions —
hand-editing a dict, mid-experiment, on the arm whose whole value is being
identical to the other one. Now five modes, verified to emit exactly the right
command:

```
session1        ... --epochs 30 --max_hours 11.0
resume          ... --epochs 30 --max_hours 11.0 --resume_dirs auto
ablation        ... --epochs 30 --max_hours 11.0 --w_layout 0
ablation_resume ... --epochs 30 --max_hours 11.0 --w_layout 0 --resume_dirs auto
```

The two arms now differ by exactly `--w_layout 0` and the output directory.

#### `EPOCHS` is a named constant now
It was the literal `'30'` in four separate preset strings. Four places to edit
for an epoch change is three too many when the thing that must hold is *both
arms used the same number*. `EPOCHS` and `TOTAL_STEPS` sit together in cell 5
with a note that they move together and apply to both arms.

#### A trap worth naming: one previous run at a time
Both resume modes search `/kaggle/input` for a `last.pt`. Attach both arms'
outputs to one notebook and the ablation could resume from the **layout** arm's
checkpoint — producing a control run contaminated by the treatment, with a
plausible-looking curve and no error anywhere. Hence the instruction to run the
ablation in its own notebook, written into `ABLATION.txt` with the reason.

#### Next Steps
- [ ] Ablation session 1: new notebook, `MODE = 'ablation'`
- [ ] Meanwhile: download session 2, `preview2.png` + `--profile_table 200`
- [ ] Watch quota — ~17 h used of 30/week, ablation needs ~17 more
- [ ] Ablation session 2: `MODE = 'ablation_resume'`
- [ ] Compare arms; then OCR-CER on both

*Log last updated: 2026-09-15 — Session 29*

---

### 2026-09-15 — Session 30: Layout Arm Final Numbers, and Two Measurement Bugs

#### The layout arm is done
| | |
|---|---|
| steps | 531,990 (30 epochs) |
| total wall time | 17.6 h across 2 sessions |
| val EPE | **3.756 px** final, 3.715 best (epoch 28) |
| vs identity 35.13 px | **89.3% better** |
| layout term | 3.867 -> 0.526 |

Session 2 epoch-ends: 4.113, 3.976, 3.911, 3.865, 3.793, 3.876, 3.771, 3.726,
3.757, 3.715, 3.756.

#### The extend-past-30 question is now settled: no
The last 30 validations span **0.055 px** (3.715 to 3.770). The LR anneal ran to
1.6e-07 and delivered 4.030 -> 3.756, a real 6.8% gain, and then the curve went
flat in a way the S26 plateau was not. Further epochs would buy noise. Both arms
stay at 30.

#### Bug 1 — `--profile_table` crashed on Windows
`OSError: [Errno 22]` from `multiprocessing`, with a 15-line "when serializing"
chain ending at `numpy.memmap`. Windows **spawns** DataLoader workers instead of
forking, so the dataset must pickle, and it holds memmaps that cannot. Invisible
on Kaggle because Linux forks. `num_workers=0` in `epe_by_profile`; evaluation is
not throughput-bound, the forward pass dominates.

#### Bug 2 — my "input" baseline moved with the model
The preview reported 4.705 px at 20 epochs and 4.991 px at 30, on what should be
the same 6 samples, which reads as a regression. Two causes, both mine:

1. Six samples is not a measurement. One of them is the 14.2 px outlier from S27,
   and it dominates the mean. The full-val figure went the right way over the
   same interval.
2. Worse, the PSNR baseline was scored on the **prediction's** visibility mask,
   so "PSNR input" — the same untouched input image — scored 12.73 dB at 20
   epochs and 12.71 dB at 30. A baseline that moves with the thing it is
   measuring is not a baseline. All three columns now score on the
   ground-truth mask, which depends only on the data.

#### A third thing, and it matters for the paper
Training-time validation uses `--val_batches 60` = **480 of 2,737 val samples**,
and `val_at_start` uses only 20 batches = 160. That is why session 2 opened at
`3.706` and read `4.104` one hundred steps later: two different subsets, not a
model that got worse. The 480-sample subset is consistent across the whole
project so the trend is sound, but it is a subsample, and `best.pt` is selected
on it.

Added `--full_val` to evaluate the entire split. That is the number to report and
the one the two arms must be compared on.

#### Next Steps
- [ ] Re-run locally with `--full_val` for the reportable layout-arm number
- [ ] Ablation session 1 (`MODE = 'ablation'`) in its own notebook
- [ ] Ablation session 2 (`MODE = 'ablation_resume'`)
- [ ] Compare arms on full val; then OCR-CER

*Log last updated: 2026-09-15 — Session 30*

---

### 2026-09-15 — Session 31: Full-Validation Numbers for the Layout Arm

First evaluation over the **entire** val split rather than the 480-sample
training-time subset.

```
FULL VAL, 2,737 samples, best.pt (step 514,257)

  cylinder      2.250    n=901
  flat          2.376    n= 34
  perspective   3.221    n=901
  combined      5.509    n=901
  ALL           3.644
```

**3.644 px against a 35.13 px identity baseline: 89.6% reduction.** This is the
number for the paper. The 480-sample subset read 3.756, so it was mildly
pessimistic and the whole project's trend line stands.

#### The "flat must be lowest" check was a bad check
It fired: cylinder 2.250 beats flat 2.376. That is not a defect, it is my rule
being wrong on two counts.

1. **n = 34.** `flat_frac` is 0.03, so the val split holds 34 flat samples
   against 901 of each other profile. The error bar swamps a 0.13 px gap.
2. **Flat is not the same as small flow.** `FLAT_PROFILE` still gets the camera
   framing similarity transform, so a page can sit at half scale and rotated —
   large, smooth displacements everywhere. "No bending" never implied "little to
   predict".

Replaced with the check that actually discriminates: **`combined` must be the
worst.** It stacks cylinder + perspective + wrinkle and is the closest thing to
a photograph; if it ever stops being the hardest, something is wrong.

Worth carrying into the paper: report per-profile numbers **with counts**, and
say plainly that the flat cell is thin.

#### The streaks are resampling, not the model
Rows 4 and 5 of the sheet show long vertical smears. They appear in the
**ground-truth-flow column too**, which settles it: where the page extended
past the input frame there is no source pixel, and `BORDER_REPLICATE` extends
the edge pixel outward forever.

Not a model error, but it reads as one in a figure, and a real system would emit
the page region only. Added `--mask_invalid`, which paints the off-page region
flat using the visibility mask. For figures, use it.

#### The 6-sample sheet, 20 vs 30 epochs (same samples)
```
flat        1.13 -> 0.79     cylinder    1.63 -> 1.82
cylinder    2.47 -> 2.03     perspective 3.95 -> 6.69
combined    4.84 -> 4.65     combined   14.21 -> 13.95
```
Four better, two worse, and the mean rose only because the perspective sample
moved 2.7 px. Full-val perspective is 3.221, so that sample is an outlier rather
than a trend — which is the argument against reading six samples as a
measurement, again.

`PMC5786657_00000` is failing consistently at ~14 px across both checkpoints. A
named, reproducible worst case is worth keeping: it is the honest failure figure
for the paper, and the thing to check first if the architecture ever changes.

#### Next Steps
- [ ] Ablation session 1 (`MODE = 'ablation'`), own notebook
- [ ] Ablation session 2 (`MODE = 'ablation_resume'`)
- [ ] Compare arms on **full val, per profile** — same command, both checkpoints
- [ ] Regenerate figures with `--mask_invalid`
- [ ] OCR-CER on both arms

*Log last updated: 2026-09-15 — Session 31*

---

### 2026-09-19 — Session 32: Ablation Half Done — EPE Says Nothing, and the Metric That Matters Was Never Logged

#### Matched-epoch EPE: a dead heat
Both arms at the same epoch, same 480-sample val subset:

```
epoch  layout  ablation   diff        epoch  layout  ablation   diff
    0   9.065     8.666  -0.400          9   5.037     5.055  +0.018
    1   7.408     7.668  +0.260         10   4.926     4.984  +0.058
    2   7.285     7.107  -0.177         11   4.689     4.992  +0.303
    3   6.530     6.848  +0.318         12   4.956     4.613  -0.343
    4   6.089     6.022  -0.067         13   4.608     4.665  +0.058
    5   6.127     5.931  -0.195         14   4.624     4.748  +0.124
    6   5.615     5.545  -0.070         15   4.295     4.393  +0.098
    7   5.264     5.389  +0.124         16   4.195     4.369  +0.174
    8   5.143     5.431  +0.288         17   4.303     4.420  +0.117
```

Mean difference **+0.038 px (0.8%)**. The layout arm leads 12 of 18, the sign
flips seven times, and the gap is an order of magnitude below the epoch-to-epoch
wobble within either arm. With one seed each this is a tie, and calling it
anything else would be reading noise.

The ablation's config is confirmed clean: `w_layout 0.0`, everything else
identical including `seed 1234` and `total_steps 540000`.

#### The measurement that should decide this does not exist
`DewarpLoss` wraps the entire layout branch in `if self.w_layout > 0`, so the
ablation never computes the layout term and never logs it. I had told the user
twice that the control arm "logs it as a diagnostic". It does not. That is the
one number designed to separate these two arms, and it was missing from the arm
it exists to measure.

The mistake is instructive rather than fatal: the quantity is a pure function of
a checkpoint and the val set, so it is fully recoverable offline.

#### `layout_metric()` added to `inference.py`
Loads any checkpoint, runs the val split, and computes
`layout_consistency_loss` as a metric — overall, horizontal/vertical split, and
per profile. Identical code path for both arms, so the numbers are comparable by
construction. Folded into `--full_val`, so one command now yields EPE-by-profile
*and* layout consistency.

Per-profile is computed one sample at a time because the loss reduces over the
batch. Wasteful, irrelevant at eval, and the per-profile split is exactly where
an effect concentrated on hard warps would show.

#### Why this is not yet bad news
The layout loss was **built** to be invisible to EPE. It penalises bending and
scores a rigid shift at 0.0000 — that was demonstrated in S14 as the argument
for it. An arm that matches on EPE while bending less is the outcome the design
predicts, not a failure of it. But it is now an unverified claim, and the next
measurement either supports it or ends it.

If the layout metric also ties, the honest reading is that the loss does not do
what it was meant to at `w_layout 0.2`, and the follow-up is a weight sweep at
reduced epochs rather than more of the same.

#### Also fixed
The `'flat' should be the LOWEST` line still printed from the code after S31
retired the rule in the docs. Replaced with the check that discriminates.

#### Next Steps
- [ ] Run `--full_val` on **both** arms at matched checkpoints; read the layout metric
- [ ] Finish ablation (`MODE = 'ablation_resume'`, ~7.7 h)
- [ ] Re-compare at 30 vs 30
- [ ] OCR-CER — the arbiter if the layout metric is also close
- [ ] If both tie: weight sweep (0.05 / 0.2 / 0.5) at 10 epochs

*Log last updated: 2026-09-19 — Session 32*

---

### 2026-09-19 — Session 33: The Ablation Comparison, and It Has the Right Shape

Both arms through the same offline evaluator, full val, 2,737 samples.
Layout arm at epoch 19, ablation at epoch 18.

```
                          LAYOUT  ABLATION   ablation worse by
EPE (all)                  3.909     4.067                4.0%
LAYOUT CONSISTENCY        0.5587    0.6065                8.6%
  horizontal              0.2661    0.2968               11.5%
  vertical                0.2926    0.3097                5.8%
```

**The layout arm wins on both, and the gap on the layout metric is 2.12x the gap
on EPE.**

That ratio is the result, not the win itself. A loss that simply made the model
better overall would move both numbers by the same proportion. This one moves
the quantity it optimises twice as far — which is the behaviour S14 predicted
when it showed a rigid shift scoring 0.0000 against a bend scoring 2.2820.

The horizontal/vertical split says more: **11.5% on horizontal against 5.8% on
vertical.** Horizontal consistency is text lines coming out straight across the
page, which is the component that decides whether a line is readable. The loss
helps most precisely there.

#### Per profile
```
                 --- EPE px ---            --- layout metric ---
              lay     abl   worse        lay      abl   worse
cylinder    2.508   2.662    6.1%     0.3039   0.3227    6.2%
flat        2.778   2.823    1.6%     0.2939   0.3168    7.8%
perspective 3.536   3.700    4.6%     0.3889   0.4263    9.6%
combined    5.725   5.886    2.8%     0.7292   0.7932    8.8%
```
The EPE column is erratic — 1.6% to 6.1% with no pattern. The layout column is
consistent, 6-10% everywhere, and largest on the two profiles with real
perspective in them. A stable effect on the targeted metric and a noisy one on
the untargeted metric is what a real mechanism looks like.

#### Why this reverses the S32 reading
S32 compared matched epochs on the 480-sample training subset and found +0.038 px
mean over epochs 0-17 — a tie. Two things were wrong with reading that as the
answer:

1. The gap **grows late**: +0.098, +0.174, +0.117 at epochs 15/16/17 against
   noise before that. Averaging over all 18 epochs buried the signal under the
   early ones, where neither arm had learned much for the layout term to shape.
2. It was EPE, on 480 samples. The full-val gap at ep19/18 is 4.0%, consistent
   with that late trend and measured on 5.7x more data.

#### What this does not yet establish
- **One seed per arm.** A single pair cannot carry a significance claim. State it
  as a limitation rather than rounding it away.
- **One epoch apart.** The layout arm is at 19, the ablation at 18. Worth ~0.15%
  on EPE at this stage (its own ep18 -> ep19 moved 4.119 -> 4.113), so it
  explains almost none of a 4% gap — but it is not zero and the final comparison
  must be 30 vs 30.
- **The ablation is not finished.** Extrapolating the layout arm's ep19 -> ep29
  trajectory (EPE -6.8%, layout -5.5%) onto the ablation predicts it lands near
  EPE 3.79 / layout 0.573, holding both gaps roughly where they are. Prediction,
  not result.

#### Next Steps
- [ ] Finish the ablation (~6.2 h remaining, running)
- [ ] Re-run both at 30 vs 30 — the comparison that goes in the paper
- [ ] OCR-CER on both arms: an 11.5% horizontal-consistency gap should show up as
      a character-error gap, and if it does not, that is worth knowing
- [ ] Figures with `--mask_invalid`

*Log last updated: 2026-09-19 — Session 33*

---

### 2026-10-18 — Session 34: The Final 30-vs-30 Result

After a month's gap. The ablation's second session had finished on 2026-09-19
(steps 329,280 -> 531,990 in 6.11 h, config identical, `w_layout 0.0`) but was
never downloaded. Downloaded to `runs\abl2`; final epoch-end 3.903 (480 subset)
against the layout arm's 3.756.

Both arms scored at the **same final checkpoint, step 531,990**, not `best.pt`:
`best.pt` is picked on validation, so reporting it on the same validation set
flatters whichever arm got luckier. Run by the user in Windows Python (~8 min each).

```
                        LAYOUT  ABLATION  ablation worse
EPE px                   3.662     3.689            0.7%
LAYOUT CONSISTENCY      0.5274    0.5631            6.8%
  horizontal            0.2492    0.2726            9.4%
  vertical              0.2782    0.2905            4.4%

per profile      n |  EPE lay    abl     gap | LAYOUT lay     abl     gap
cylinder       901 |    2.274  2.256   -0.8% |     0.2742  0.2923    6.6%
flat            34 |    2.389  2.511    5.1% |     0.2684  0.2791    4.0%
perspective    901 |    3.241  3.288    1.5% |     0.3563  0.3819    7.2%
combined       901 |    5.519  5.568    0.9% |     0.6953  0.7471    7.5%
```

#### The finding
**On EPE the arms tie. On straightness the layout arm is clearly ahead.**

0.7% EPE is nothing — on cylinder the ablation is marginally ahead. Layout
consistency is 6.8% better overall, better on **all four** profiles, and largest
on horizontal (9.4%): text lines coming out straight across the page.

This is a cleaner result than "wins on both". It says exactly what the method
claims: the layout loss straightens text **at no cost in average error**, and
average error cannot see the difference. The standard dewarping metric is blind
to the property this loss improves — which is the argument for reporting a layout
metric at all.

#### The gaps moved in opposite directions during the anneal
```
                  EPE gap  layout gap   ratio
ep19 vs ep18         4.0%        8.6%    2.1x
ep29 vs ep29         0.7%        6.8%    9.2x
```
The ablation caught up on EPE as the learning rate fell. It did **not** catch up
on straightness. Nothing in its objective pushes on that, so annealing refined
what it was already optimising and left the rest. Hard to explain as noise,
because noise does not prefer one metric.

#### Honest limits — these go in the paper
- **One seed per arm.** No significance claim across training runs. Consistency
  across four profiles and two checkpoints supports the result; it does not
  replace seeds.
- **Synthetic validation only**, from the same warp generator as training. Real
  photos untested.
- **A paired per-sample test is cheap** (both arms score the same 2,737 pages):
  how often is the layout arm straighter, page by page? Needs `inference.py` to
  save per-sample values (it saves aggregates only). Measures variation over
  pages, not seeds.
- **The layout metric is our own.** Reviewers will ask whether it tracks
  something they care about. That is what OCR-CER is for.

#### Practical notes
- The local Linux shell on the user's PC works again; torch CPU + OpenCV
  installed. **Background jobs die when the command returns** (each call is its
  own sandbox). I told the user a 45-minute evaluation was running when it had
  died after 20 seconds, and only found out when they asked for the process name.
  Long jobs go to the user's Windows Python, which is also ~3x faster.
- The PC's log briefly lacked Session 33; it is back in its original form.
- `training\compare_arms.py` added (reads `final_layout/ablation.png.epe.json`).

#### Next Steps
- [ ] OCR-CER on both arms — does 9.4% straighter horizontally mean fewer
      character errors?
- [ ] Per-sample paired comparison (small change to `inference.py`)
- [ ] Figures with `--mask_invalid`, including the failure `PMC5786657_00000`
- [ ] A few real phone photos
- [ ] Write the paper

*Log last updated: 2026-10-18 — Session 34*

---

### 2026-10-18 — Session 35: OCR Character Error Rate — Straighter Text Reads Better

The readability test of the S34 result: does a 6.8% straightness gain (9.4%
horizontal) translate into fewer OCR errors?

#### Building the test took two failed attempts
1. **OCR of the stored pages measured nothing.** At 448x576 (~53 dpi), with the
   page shrunk inside the frame and JPEG-compressed, letters are 3-4 px tall.
   Rectifying with the *ground-truth* flow still scored 66-71% CER. A test whose
   ceiling is 70% wrong cannot separate two models.
2. **Lighting was a red herring.** Shadow made Tesseract return empty strings on
   some pages, but flattening illumination did not fix the ceiling. Resolution
   was the real limit.

#### Protocol that works: high-resolution re-synthesis
The DocUNet pattern: predict the flow from a small image, apply it to a big one.
Clean page upscaled 3x (1344x1728); its own ground-truth warp applied at that
size; each model predicts from the **original stored 448x576 photo** (its
training distribution); the normalized flow is inverted and applied to the
high-res page. Geometry is identical, so the prediction is valid at 3x. Tests
geometry only — lighting is deliberately out of scope. Reference = Tesseract on
the clean page at 3x (DocUNet protocol; PubLayNet has no transcriptions).
Tesseract 5 LSTM, `--oem 1 --psm 3`, one OpenMP thread (default threading was
5x slower: 11.5 s vs 2.3 s per page). Full protocol in the `ocr_eval.py` header.

Subset: 334 val pages chosen **before** any OCR by seeded stem hash — 100
cylinder, 100 perspective, 100 combined, all 34 flat. 2 dropped (reference
under 200 characters). Both arms at step 531,990.

#### Result (332 pages)
```
              mean CER   corpus CER   median
input           63.05%       62.82%   71.17%     page left bent
layout          28.61%       22.82%   20.52%
ablation        30.37%       24.55%   21.69%
true_flow       13.70%        9.92%    7.03%     ceiling

PAIRED: ablation minus layout = +1.76 pts (95% bootstrap CI +0.78 to +2.79)
        ablation has 6.2% more character errors
        layout better on 203 pages, worse on 126, tied 3; Wilcoxon p = 4.6e-6

BY PROFILE      n   layout  ablation   diff [95% CI]            rel     wins    p
cylinder       98    32.5%    33.4%   +0.93 [-1.05, +3.08]    +2.9%   54-42  0.19
flat           34    24.6%    23.6%   -0.97 [-4.37, +1.95]    -3.9%   19-15  0.91
perspective   100    25.0%    25.5%   +0.57 [-0.89, +1.86]    +2.3%   56-44  0.11
combined      100    29.8%    34.5%   +4.69 [+2.95, +6.62]   +15.8%   74-25  5e-7
all but comb. 232    28.1%    28.6%   +0.50 [-0.65, +1.65]    +1.8%  129-101 0.056
```

#### Reading it
**Overall: yes, the straighter model reads better** — 6.2% fewer character
errors, consistent page by page (203 vs 126), and the interval excludes zero.

**But the gain is concentrated on `combined`**, the warp that stacks curve +
perspective + wrinkle and is closest to a real photograph: 15.8% fewer errors,
74 pages to 25. On the three milder profiles the effect is a small positive
trend that is not significant on its own (and flat, n=34, is a wash).

This is a sharper claim than a uniform gain, and a defensible one. The layout
*metric* gap was nearly uniform across profiles (6.6-7.5%). OCR only cashes it
in where residual bending is large enough to break Tesseract's line
segmentation. On mild warps both models are already inside OCR's tolerance, so
straighter-still buys nothing a reader would notice. **The paper should say
exactly this rather than lead with the 6.2% average**, which would hide where
the effect lives.

Context: both models recover most of what is recoverable — 69.8% (layout) vs
66.2% (ablation) of the input-to-ceiling improvement — and both remain well
above the 13.7% ceiling, so there is headroom for a better flow regressor that
is independent of this result.

#### Limits (for the paper)
- One training seed per arm; these statistics vary **pages**, not seeds.
- Reference text is OCR of the clean page, not ground-truth transcription, so
  CER partly measures agreement with Tesseract's own errors (standard protocol).
- Synthetic warps; 3x re-synthesis rather than real high-res photographs.
- Absolute CER is tied to this protocol and is not comparable to published
  DocUNet-benchmark numbers.

#### Practical
- First run died at page 43: **the cloud container also kills background jobs
  when Claude goes idle**. Resumed from the saved JSONL and finished by polling
  inside one turn. Third occurrence of "the job is running" being false in this
  project; the rule now: verify a process is alive in a later call before
  saying so.
- `device_commit_files` sometimes writes a stale earlier version when the same
  outputs filename is reused; committing from fresh filenames and re-reading the
  file on the PC afterwards avoids it.

#### Files
`ocr_results\OCR_RESULT.txt`, `per_page.csv` (per-page CER for every version,
for paired analysis or plots), `ocr_texts.jsonl` (every raw OCR string),
`examples\` (pages 0-5, all five versions at 3x).

#### Next Steps
- [ ] Figures: OCR example strip on a `combined` page; `--mask_invalid` sheets;
      the failure `PMC5786657_00000`
- [ ] A few real phone photos (qualitative)
- [ ] Write the paper — results are complete for the core claim

### Session 36 — 2026-10-18: real-photo benchmark, figures, paper draft

- DocUNet benchmark downloaded to the cloud workspace; official eval code compiled for GNU Octave 8.4 (SIFTflow mex patched, MATLAB-compatible ssim.m, imfilter_nd.m). Calibrated: distorted crops give MS-SSIM 0.2497 / LD 20.498 (uncorrected) and 0.2493 / 20.518 (doc 64 rotated), published 0.25 (0.2459) / 20.51.
- OCR protocol (DocTr OCR_eval.py, Settings 1 and 2): Tesseract 5.3.4 gave distorted CER 0.484 (published 0.509), so built Tesseract 5.0.1 from source: 0.5092 / ED 1546.5. Matches.
- Rectified all 130 photos with both final checkpoints (net sees 448x576 resize, flow upsampled and inverted at full res, white fill). Results in Paper status above. With Tess 5.3.4 the S1 CER gap was significant (0.344 vs 0.384, p=0.015) but not with 5.0.1 (p=0.19), so no real-photo OCR claim.
- Real-photo failure modes: crumpled paper, page boundary misjudged on busy backgrounds, non-article documents.
- Subagent built and verified the bibliography (36 entries). Closest prior work: ForCenNet (ICCV25, OCR-midline curvature loss), MataDoc (arXiv, equal-y text-line loss), AADD (AAAI26, row/col variance on whole grid), D2Dewarp (CVPR26, PubLayNet line masks as targets), LA-DocFlatten (TOG). We claim region-level, layout-box, free-label loss + controlled ablation, NOT "first layout-aware".
- Paired per-page EPE and C on the 334-page OCR subset (boxes exported to `ocr_results\subset_boxes_flowgrid.json`).
- Figures: pipeline, shift-vs-bend, curves, per-profile bars, synthetic qualitative, DocUNet qualitative (picked by rule).
- Paper written (IEEEtran) and checked by an independent review subagent (~250 numbers verified); its fixes applied (per-page C, sign conventions, softer real-photo claims, units of L_lay).
- Delivered: `paper\`, `docunet_eval\`, `paper_bundle_s36.zip`.

*Log last updated: 2026-10-18 — Session 36*
