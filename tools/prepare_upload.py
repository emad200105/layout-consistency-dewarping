"""
prepare_upload.py
=================
Stages everything for Kaggle and prints the exact commands to run.

    python prepare_upload.py --user <your-kaggle-username>

After it finishes you run three `kaggle datasets create` commands that it
prints. Nothing else to decide.

WHY THREE DATASETS AND NOT ONE
------------------------------
`kaggle datasets create -p <folder>` SKIPS subdirectories by default. Upload
`dataset/` as one dataset and the `layout/` folder is silently
left behind — the upload "succeeds", and the first sign of trouble is a
FileNotFoundError hours later on Kaggle. So anything in a subfolder has to be
its own dataset, or flattened.

  1. data   ~24.6 GB  uploaded ONCE, never changes
  2. layout ~57 MB    separate because of the subfolder rule above, and because
                      regenerating it should never mean re-uploading 24.6 GB
  3. code   ~200 KB   flattened into one folder (every module imports by plain
                      name, so a flat layout works) and re-uploaded whenever
                      the code changes, which is often

Splitting on "how often does this change" is also just correct: you will
re-upload the code many times and the data never.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

DATA_REQUIRED = ['meta.json', 'manifest.jsonl', 'clean_index.jsonl',
                 'flows.f16', 'masks.u8']
LAYOUT_REQUIRED = ['stems.jsonl', 'meta.json', 'ann_cat.u8.npy',
                   'ann_bbox.u16.npy', 'poly_offsets.i64.npy', 'poly_xy.u16.npy']


def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return f"{n:.1f} {u}" if u != 'B' else f"{n} B"
        n /= 1024


def dir_size(p, recursive=True):
    t = 0
    for root, dirs, files in os.walk(p):
        if not recursive:
            dirs[:] = []
        for f in files:
            try:
                t += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return t


def write_meta(folder, slug, title, user):
    meta = {"title": title, "id": f"{user}/{slug}",
            "licenses": [{"name": "CC0-1.0"}]}
    with open(Path(folder) / 'dataset-metadata.json', 'w') as fh:
        json.dump(meta, fh, indent=2)


def kaggle_user():
    """
    Find the Kaggle username without assuming where the CLI keeps its creds.

    kaggle 2.x does not necessarily leave a kaggle.json in ~/.kaggle — it may
    use the env vars, XDG config, or %APPDATA%. So try all of them, and fall
    back to asking the CLI itself, which is authoritative by definition.
    """
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']

    candidates = [
        Path.home() / '.kaggle' / 'kaggle.json',
        Path.home() / '.config' / 'kaggle' / 'kaggle.json',
    ]
    if os.environ.get('KAGGLE_CONFIG_DIR'):
        candidates.insert(0, Path(os.environ['KAGGLE_CONFIG_DIR']) / 'kaggle.json')
    if os.environ.get('APPDATA'):
        candidates.append(Path(os.environ['APPDATA']) / 'kaggle' / 'kaggle.json')
    for p in candidates:
        try:
            if p.exists():
                u = json.load(open(p)).get('username')
                if u:
                    return u
        except Exception:                                        # noqa: BLE001
            pass

    # Last resort: ask the CLI. If `kaggle datasets list -m` works, the CLI
    # knows who you are even when this script cannot find the file.
    import subprocess
    for cmd in (['kaggle', 'config', 'view'],
                [sys.executable, '-m', 'kaggle', 'config', 'view']):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            for line in (out.stdout or '').splitlines():
                if 'username' in line.lower():
                    parts = line.replace(':', ' ').replace('=', ' ').split()
                    if parts:
                        return parts[-1].strip()
        except Exception:                                        # noqa: BLE001
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".",
                    help="repo root, containing dataset/ (built) and the code")
    ap.add_argument("--user", default=None,
                    help="Kaggle username; read from kaggle.json if omitted")
    ap.add_argument("--prefix", default="publaynet-dewarp",
                    help="dataset slug prefix")
    a = ap.parse_args()

    proj = Path(a.project)
    data = proj / 'dataset'
    layout_src = data / 'layout'
    stage = proj / 'upload'

    user = a.user or kaggle_user()
    if not user:
        print("!! Could not determine your Kaggle username automatically.")
        print("   The CLI may still be authenticated — this only means the")
        print("   credentials are not where this script looked.")
        print("   Just pass it:   python prepare_upload.py --user <your-username>")
        print("   (it is the first part of the 'ref' column in "
              "`kaggle datasets list -m`)")
        sys.exit(1)

    print("=" * 74)
    print(f"project : {proj}")
    print(f"kaggle  : {user}")
    print("=" * 74)

    ok = True
    for f in DATA_REQUIRED:
        if not (data / f).exists():
            print(f"  !! MISSING {data / f}")
            ok = False
    if not list(data.glob('pages-*.bin')):
        print(f"  !! MISSING {data}\\pages-*.bin")
        ok = False
    for f in LAYOUT_REQUIRED:
        if not (layout_src / f).exists():
            print(f"  !! MISSING {layout_src / f}")
            ok = False
    if not ok:
        print("\nFix the above first — build_dataset.py and prepare_layout.py "
              "must have finished.")
        sys.exit(1)

    # ---- 1. data: upload in place, just drop the metadata file in ----
    write_meta(data, f"{a.prefix}-data", "PubLayNet Dewarping Dataset", user)
    flat = dir_size(data, recursive=False)
    n_pages = len(list(data.glob('pages-*.bin')))
    n_clean = len(list(data.glob('clean-*.bin')))
    print(f"\n1. DATA   {data}")
    print(f"   uploaded in place; {n_pages} page shards + {n_clean} clean shards "
          f"+ flows + masks + manifests")
    print(f"   {human(flat)}   (the layout/ subfolder is NOT included — that is dataset 2)")

    # ---- 2. layout: copy the six files out of the subfolder ----
    lay = stage / 'layout'
    lay.mkdir(parents=True, exist_ok=True)
    for f in LAYOUT_REQUIRED:
        shutil.copy2(layout_src / f, lay / f)
    write_meta(lay, f"{a.prefix}-layout", "PubLayNet Dewarping Layout Store", user)
    print(f"\n2. LAYOUT {lay}")
    print(f"   {len(LAYOUT_REQUIRED)} files, {human(dir_size(lay))}")

    # ---- 3. code: flatten every module into one folder ----
    code = stage / 'code'
    if code.exists():
        shutil.rmtree(code)
    code.mkdir(parents=True, exist_ok=True)
    copied = []
    for sub in ('augmentation', 'training'):
        src = proj / sub
        if not src.exists():
            continue
        for p in sorted(src.glob('*.py')):
            shutil.copy2(p, code / p.name)
            copied.append(p.name)
    write_meta(code, f"{a.prefix}-code", "PubLayNet Dewarping Code", user)
    print(f"\n3. CODE   {code}")
    print(f"   {len(copied)} modules flattened into one folder "
          f"({human(dir_size(code))})")
    print(f"   {', '.join(copied)}")

    print("\n" + "=" * 74)
    print("RUN THESE THREE COMMANDS (code and layout first — they are quick)")
    print("=" * 74)
    print(f"\n  cd {code}")
    print(f"  kaggle datasets create -p .")
    print(f"\n  cd {lay}")
    print(f"  kaggle datasets create -p .")
    print(f"\n  cd {data}")
    print(f"  kaggle datasets create -p .          <-- {human(flat)}, expect hours")
    print("\n  (re-uploading a dataset that already exists is `kaggle datasets "
          "version -p . -m \"note\"`)")

    print("\n" + "=" * 74)
    print("THEN, IN A KAGGLE NOTEBOOK (GPU on, internet off)")
    print("=" * 74)
    print(r"""
  # add all three datasets as inputs. DO NOT hard-code the mount paths:
  # Kaggle mounts a dataset at /kaggle/input/<slug> in some notebooks and at
  # /kaggle/input/datasets/<owner>/<slug>/versions/N in others, so a literal
  # path fails with "No such file or directory" while the Input panel happily
  # shows the dataset attached. Cell 1 locates everything by content.

  # --- cell 1: find the mounts -------------------------------------------
  import os
  from pathlib import Path

  def find_dir(marker, depth=7):
      base = Path('/kaggle/input'); hits = []
      for root, dirs, files in os.walk(base):
          if len(Path(root).relative_to(base).parts) > depth:
              dirs[:] = []; continue
          if marker in files:
              hits.append(root)
      return hits

  CODE   = (find_dir('train.py')    or [''])[0]
  LAYOUT = (find_dir('stems.jsonl') or [''])[0]
  DATA   = (find_dir('flows.f16')   or [''])[0]
  print('CODE  ', CODE); print('LAYOUT', LAYOUT); print('DATA  ', DATA)

  # --- cell 2: stage the code --------------------------------------------
  !mkdir -p /kaggle/working/code && cp {CODE}/*.py /kaggle/working/code/
  !ls /kaggle/working/code

  # --- cell 3: smoke test, 50 steps, ~2 minutes --------------------------
  # train.py resolves 'auto' the same way cell 1 does, so the flags never
  # have to change between notebooks.
  !cd /kaggle/working/code && python train.py \
      --root auto --layout_root auto \
      --out /kaggle/working/smoke \
      --batch 8 --steps 50 --total_steps 540000 --max_hours 0.2

  # --- cell 4: the real run ----------------------------------------------
  !cd /kaggle/working/code && python train.py \
      --root auto --layout_root auto \
      --out /kaggle/working/run_layout \
      --batch 8 --epochs 30 --total_steps 540000 \
      --max_hours 11.0 --workers 4
""")
    print("  Session 2 onward: add the PREVIOUS session's saved output as a")
    print("  fourth input and append one flag:")
    print("      --resume_dirs auto")
    print("\n  Everything else stays byte-identical, especially --total_steps.")
    print("=" * 74)


if __name__ == "__main__":
    main()
