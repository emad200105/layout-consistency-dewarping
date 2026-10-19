"""
select_pages.py
===============
Collect exactly the 47,958 PubLayNet training pages this project used into one
folder, so that build_dataset.py reproduces the same dataset bit for bit.

    python tools/select_pages.py --src /path/to/publaynet/train --dst data/publaynet/train

WHY THE EXACT SET MATTERS
-------------------------
build_dataset.py sorts the image files and seeds every sample from
(global seed, position in that sorted list, variant). A folder with even one
extra or missing page shifts every later position, and with it every warp. The
list in results/dataset_build/source_pages.txt.gz is the sorted set of page
stems that was built (it is about one of PubLayNet's seven train archives).
Pages marked `no_layout_landscape` (234 of them) were built but are dropped by
the training loader, because prepare_layout.py excludes landscape pages. Keep
them in the folder anyway: removing them changes the seeds of the rest.

`--src` may be any folder tree that contains the PubLayNet training JPEGs
(<stem>.jpg); subfolders are searched. Files are hard-linked when possible and
copied otherwise; `--mode symlink` is also available.
"""

import argparse
import gzip
import os
import shutil
from pathlib import Path

LIST = Path(__file__).resolve().parents[1] / 'results' / 'dataset_build' / 'source_pages.txt.gz'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='folder tree with PubLayNet train JPEGs')
    ap.add_argument('--dst', default='data/publaynet/train')
    ap.add_argument('--list', default=str(LIST))
    ap.add_argument('--mode', choices=['link', 'copy', 'symlink'], default='link')
    a = ap.parse_args()

    stems = [line.split('\t')[0].strip() for line in gzip.open(a.list, 'rt') if line.strip()]
    want = set(stems)
    print(f'{len(stems):,} pages in the list')

    found = {}
    for dirpath, _, names in os.walk(a.src):
        for n in names:
            st, ext = os.path.splitext(n)
            if ext.lower() in ('.jpg', '.jpeg', '.png') and st in want and st not in found:
                found[st] = Path(dirpath) / n
    missing = [s for s in stems if s not in found]
    print(f'{len(found):,} found under {a.src}, {len(missing):,} missing')
    if missing:
        print('  first missing:', ', '.join(missing[:5]))
        raise SystemExit('Not all pages are available; the build would not be reproducible.')

    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    extra = [p for p in dst.iterdir() if p.is_file() and p.stem not in want]
    if extra:
        raise SystemExit(f'{dst} already holds {len(extra)} files not in the list; use an empty folder.')
    for st in stems:
        src, out = found[st], dst / found[st].name
        if out.exists():
            continue
        if a.mode == 'symlink':
            out.symlink_to(src.resolve())
        elif a.mode == 'link':
            try:
                os.link(src, out)
            except OSError:
                shutil.copy2(src, out)
        else:
            shutil.copy2(src, out)
    print(f'done: {len(stems):,} pages in {dst}')


if __name__ == '__main__':
    main()
