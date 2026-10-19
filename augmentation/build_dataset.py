"""
build_dataset.py
================
Builds the training dataset in the format defined by `dataset_format.py`.

Replaces `process_dataset` in warp_augment.py, which wrote loose PNGs and
full-resolution .npy flows — about 1.24 TB and 430,000+ files for the full run.
This writes ~18 GB in a handful of files.

    python augmentation/build_dataset.py --input_dir data/publaynet/train \\
                                         --out_dir   dataset

    # pilot first — 3,000 sources, a few minutes, ~1.1 GB
    python augmentation/build_dataset.py --input_dir ... --out_dir ... --limit 3000

DETERMINISM
-----------
Every sample's RNG is seeded from (global_seed, source_index, variant_index),
never from the order work happens to be scheduled in. So the dataset is
bit-identical whether it is built with 1 worker or 16, and any single sample
can be regenerated later from its manifest row alone.

RESUME
------
The manifest is appended as JSONL while the run proceeds, and the blob files
are append-only. `--resume` replays the JSONL to find where it stopped,
truncates the blobs back to the last fully-written sample (discarding any
partial write from a kill), and carries on. A multi-hour run does not have to
start over.
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_format as fmt                                       # noqa: E402
from warp_augment import (augment_document, DISTORTION_PROFILES,    # noqa: E402
                          FLAT_PROFILE, TRAIN_W, TRAIN_H)

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

# OpenCV spawns its own threads; with a process pool that oversubscribes the
# CPU badly and can run slower than single-threaded.
cv2.setNumThreads(1)


# ─────────────────────────────────────────────
# WORKER
# ─────────────────────────────────────────────

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)
    cv2.setNumThreads(1)


def _build_one(task):
    """
    Build every variant of one source image.

    Runs in a worker process. Returns encoded bytes and arrays; all file I/O
    happens in the parent so the blobs stay append-only and ordered.
    """
    src_idx, path, is_flat = task
    W, H = _CFG['width'], _CFG['height']
    seed = _CFG['seed']

    img = cv2.imread(str(path))
    if img is None:
        return src_idx, None, f"unreadable: {path}"

    clean = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    clean_jpg = fmt.encode_jpeg(clean, _CFG['quality'])

    profiles = [DISTORTION_PROFILES[v % len(DISTORTION_PROFILES)]
                for v in range(_CFG['variants'])]
    if is_flat:
        profiles.append(FLAT_PROFILE)

    variants = []
    for v, prof in enumerate(profiles):
        p = dict(prof)
        name = p.pop('name')

        # Seeded per (seed, source, variant) — never from scheduling order.
        s = (seed * 1_000_003 + src_idx * 97 + v) & 0x7FFFFFFF
        random.seed(s)
        np.random.seed(s)

        warped, flow_px, mask, params = augment_document(clean, **p)

        variants.append({
            'profile': name,
            'variant': v,
            'sample_seed': int(s),
            'jpeg': fmt.encode_jpeg(warped, _CFG['quality']),
            'flow': fmt.encode_flow(flow_px),
            'mask': fmt.encode_mask(mask),
            'params': {k: (float(x) if isinstance(x, (int, float)) and
                           not isinstance(x, bool) else x)
                       for k, x in params.items() if k != 'warps_applied'},
        })

    return src_idx, {'stem': Path(path).stem,
                     'clean_jpeg': clean_jpg,
                     'variants': variants}, None


# ─────────────────────────────────────────────
# BLOB WRITER
# ─────────────────────────────────────────────

class BlobWriter:
    """Append-only sharded blob file. Rolls to a new shard past `max_bytes`."""

    def __init__(self, root: Path, prefix: str, max_bytes: int):
        self.root, self.prefix, self.max_bytes = Path(root), prefix, max_bytes
        self.shard = 0
        self.offset = 0
        self.fh = None

    def _path(self, shard=None):
        return self.root / f'{self.prefix}-{(self.shard if shard is None else shard):04d}.bin'

    def open(self, shard: int, offset: int):
        self.close()
        self.shard, self.offset = shard, offset
        # Truncate away any partial tail from a previous killed run.
        with open(self._path(), 'ab') as fh:
            pass
        with open(self._path(), 'r+b') as fh:
            fh.truncate(offset)
        self.fh = open(self._path(), 'ab')

    def write(self, data: bytes):
        if self.fh is None:
            self.open(0, 0)
        if self.offset and self.offset + len(data) > self.max_bytes:
            self.shard += 1
            self.offset = 0
            self.fh.close()
            self.fh = open(self._path(), 'wb')
        shard, offset = self.shard, self.offset
        self.fh.write(data)
        self.offset += len(data)
        return shard, offset, len(data)

    def close(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None


# ─────────────────────────────────────────────
# MAIN BUILD
# ─────────────────────────────────────────────

def collect_images(input_dir: Path):
    files = []
    for dirpath, _, names in os.walk(input_dir):
        for n in names:
            if n.lower().endswith(IMAGE_EXTS):
                files.append(Path(dirpath) / n)
    return sorted(files)


def build(input_dir, out_dir, variants=3, flat_frac=0.03,
          width=TRAIN_W, height=TRAIN_H, quality=fmt.JPEG_QUALITY,
          workers=None, seed=42, limit=None, shard_gb=1.0, resume=False):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_images(Path(input_dir))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"No images found under {input_dir}")

    n_src = len(files)
    flat_ids = set(random.Random(seed ^ 0x5EED).sample(
        range(n_src), int(round(n_src * flat_frac))))
    n_total = n_src * variants + len(flat_ids)

    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    max_bytes = int(shard_gb * (1 << 30))

    print(f"Sources          : {n_src:,}")
    print(f"Variants/source  : {variants}  (+{len(flat_ids):,} flat)")
    print(f"Total samples    : {n_total:,}")
    print(f"Page             : {width} x {height}  JPEG q{quality}")
    print(f"Flow             : {fmt.FLOW_W} x {fmt.FLOW_H} {np.dtype(fmt.FLOW_DTYPE).name} "
          f"normalized   ({fmt.FLOW_ITEM_BYTES/1024:.1f} KB/sample)")
    print(f"Mask             : {fmt.FLOW_W} x {fmt.FLOW_H} uint8            "
          f"({fmt.MASK_ITEM_BYTES/1024:.1f} KB/sample)")
    print(f"Workers          : {workers}")
    print(f"Out              : {out_dir}")
    print()

    # Fixed-size arrays -> preallocated memmaps, indexed by sample id.
    flows = np.memmap(out_dir / 'flows.f16', dtype=fmt.FLOW_DTYPE,
                      mode='r+' if resume and (out_dir / 'flows.f16').exists() else 'w+',
                      shape=(n_total, fmt.FLOW_H, fmt.FLOW_W, 2))
    masks = np.memmap(out_dir / 'masks.u8', dtype=fmt.MASK_DTYPE,
                      mode='r+' if resume and (out_dir / 'masks.u8').exists() else 'w+',
                      shape=(n_total, fmt.FLOW_H, fmt.FLOW_W))

    man_path = out_dir / 'manifest.jsonl'
    cln_path = out_dir / 'clean_index.jsonl'

    man_rows, cln_rows = [], []
    start_src = 0
    if resume and man_path.exists():
        with open(man_path) as fh:
            man_rows = [json.loads(l) for l in fh if l.strip()]
        if cln_path.exists():
            with open(cln_path) as fh:
                cln_rows = [json.loads(l) for l in fh if l.strip()]
        # Keep only whole source images: a source is complete when all of its
        # expected variants are present.
        done = {}
        for r in man_rows:
            done[r['source_index']] = done.get(r['source_index'], 0) + 1
        complete = {i for i, c in done.items()
                    if c == variants + (1 if i in flat_ids else 0)}
        start_src = (max(complete) + 1) if complete else 0
        man_rows = [r for r in man_rows if r['source_index'] < start_src]
        cln_rows = [r for r in cln_rows if r['source_index'] < start_src]
        print(f"RESUME: {len(man_rows):,} samples from {start_src:,} complete "
              f"sources; rebuilding from source {start_src:,}")

    pages = BlobWriter(out_dir, 'pages', max_bytes)
    cleans = BlobWriter(out_dir, 'clean', max_bytes)
    if man_rows:
        last = man_rows[-1]
        pages.open(last['page_shard'], last['page_offset'] + last['page_length'])
    else:
        pages.open(0, 0)
    if cln_rows:
        last = cln_rows[-1]
        cleans.open(last['shard'], last['offset'] + last['length'])
    else:
        cleans.open(0, 0)

    man_fh = open(man_path, 'w')
    cln_fh = open(cln_path, 'w')
    for r in man_rows:
        man_fh.write(json.dumps(r) + '\n')
    for r in cln_rows:
        cln_fh.write(json.dumps(r) + '\n')
    man_fh.flush()
    cln_fh.flush()

    sample_id = len(man_rows)
    clean_id = len(cln_rows)

    tasks = [(i, files[i], i in flat_ids) for i in range(start_src, n_src)]
    cfg = dict(width=width, height=height, quality=quality,
               variants=variants, seed=seed)

    t0 = time.time()
    n_done = 0
    n_bad = 0
    bytes_pages = 0

    def flush_row(row, fh):
        fh.write(json.dumps(row) + '\n')

    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for src_idx, res, err in pool.imap(_build_one, tasks, chunksize=8):
            n_done += 1
            if res is None:
                n_bad += 1
                print(f"  [SKIP] {err}")
                continue

            shard, off, ln = cleans.write(res['clean_jpeg'])
            flush_row({'clean_id': clean_id, 'source_index': src_idx,
                       'stem': res['stem'], 'shard': shard,
                       'offset': off, 'length': ln}, cln_fh)

            for var in res['variants']:
                flows[sample_id] = var['flow']
                masks[sample_id] = var['mask']
                pshard, poff, plen = pages.write(var['jpeg'])
                bytes_pages += plen
                row = {'sample_id': sample_id, 'clean_id': clean_id,
                       'source_index': src_idx, 'stem': res['stem'],
                       'variant': var['variant'], 'profile': var['profile'],
                       'page_shard': pshard, 'page_offset': poff,
                       'page_length': plen, 'sample_seed': var['sample_seed']}
                row.update(var['params'])
                flush_row(row, man_fh)
                sample_id += 1

            clean_id += 1

            if n_done % 200 == 0:
                man_fh.flush()
                cln_fh.flush()
                el = time.time() - t0
                rate = n_done / el
                eta = (len(tasks) - n_done) / max(rate, 1e-9)
                print(f"  [{n_done:,}/{len(tasks):,}] {rate:5.1f} src/s  "
                      f"elapsed {el/60:5.1f}m  eta {eta/60:6.1f}m  "
                      f"pages {bytes_pages/1e9:5.2f} GB")

    man_fh.close()
    cln_fh.close()
    pages.close()
    cleans.close()
    flows.flush()
    masks.flush()
    del flows, masks

    # Re-read the JSONL we just wrote and emit parquet alongside it.
    with open(man_path) as fh:
        all_man = [json.loads(l) for l in fh if l.strip()]
    with open(cln_path) as fh:
        all_cln = [json.loads(l) for l in fh if l.strip()]
    t1 = fmt.write_table(all_man, out_dir, 'manifest')
    t2 = fmt.write_table(all_cln, out_dir, 'clean_index')

    meta = {
        'format_version': fmt.FORMAT_VERSION,
        'page_w': width, 'page_h': height,
        'flow_w': fmt.FLOW_W, 'flow_h': fmt.FLOW_H,
        'flow_dtype': np.dtype(fmt.FLOW_DTYPE).name,
        'flow_units': 'normalized displacement, grid_sample convention',
        'jpeg_quality': quality,
        'n_samples': len(all_man), 'n_sources': len(all_cln),
        'variants_per_source': variants, 'flat_frac': flat_frac,
        'seed': seed,
        'profiles': [p['name'] for p in DISTORTION_PROFILES] + [FLAT_PROFILE['name']],
        'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    with open(out_dir / 'meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file())
    el = time.time() - t0
    print()
    print(f"Done in {el/60:.1f} min ({n_done/max(el,1e-9):.1f} sources/s)")
    print(f"  samples   : {len(all_man):,}   sources: {len(all_cln):,}"
          + (f"   skipped: {n_bad}" if n_bad else ""))
    print(f"  tables    : {t1} / {t2}")
    print(f"  on disk   : {total/1e9:.2f} GB  ({total/max(len(all_man),1)/1024:.1f} KB/sample)")
    if len(all_man):
        proj = total / len(all_man) * (47958 * variants + int(47958 * flat_frac))
        print(f"  projected full 47,958-source build: {proj/1e9:.1f} GB")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the dewarping dataset")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--flat_frac", type=float, default=0.03)
    ap.add_argument("--width", type=int, default=TRAIN_W)
    ap.add_argument("--height", type=int, default=TRAIN_H)
    ap.add_argument("--quality", type=int, default=fmt.JPEG_QUALITY)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N source images (use for a pilot)")
    ap.add_argument("--shard_gb", type=float, default=1.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    build(a.input_dir, a.out_dir, variants=a.variants, flat_frac=a.flat_frac,
          width=a.width, height=a.height, quality=a.quality,
          workers=a.workers, seed=a.seed, limit=a.limit,
          shard_gb=a.shard_gb, resume=a.resume)
