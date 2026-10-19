"""
layout_format.py
================
On-disk contract for the PubLayNet layout annotations, and the reader for them.

This is the data behind the paper's layout-consistency loss: PubLayNet ships
semantic layout regions (text / title / list / table / figure) for every page.
The layout-consistency loss needs those regions
in the training coordinate space, fast to fetch, for 466,221 annotations across
47,958 pages.

LAYOUT
------
    <out_dir>/layout/
        stems.jsonl         one row per source page: stem, native w/h,
                            annotation range [ann_start, ann_end)
        ann_cat.u8.npy      (N,)     category 1..5
        ann_bbox.u16.npy    (N, 4)   x0, y0, x1, y1
        poly_offsets.i64.npy (N+1,)  slice bounds into poly_xy
        poly_xy.u16.npy     (M, 2)   polygon vertices
        meta.json

KEYED BY STEM, NOT clean_id
---------------------------
Rows are keyed by page stem and sorted, so this store can be built while the
image dataset is still being written, and stays valid if the dataset is rebuilt
with different variants or a different flat fraction. The loader joins
stem -> clean_id through the dataset's own clean_index once at startup.

COORDINATES: uint16, NORMALIZED TO THE PAGE
-------------------------------------------
Every coordinate is stored as round(v / page_dimension * 65535).

Normalized, because annotations are in each page's OWN native pixel space and
the local set has 297 distinct page sizes — there is no global constant that
converts them. Normalizing at build time means the loader multiplies by
whatever training resolution it wants and nothing has to be regenerated.

uint16 rather than float16, because the values live in [0, 1] where float16 is
at its worst: its quantum near 1.0 is ~0.0005, i.e. 0.22 px at 448 wide.
uint16 over the same range gives 1/65535, i.e. 0.0068 px. Same two bytes,
32x the precision, and no chance of a NaN in a label.
"""

import json
from pathlib import Path

import numpy as np

LAYOUT_VERSION = 1
U16_MAX = 65535.0

CAT_NAMES = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
CAT_COLORS = {
    1: (60, 180, 75),
    2: (255, 140, 0),
    3: (200, 60, 220),
    4: (0, 200, 255),
    5: (0, 80, 255),
}

FILES = {
    'cat': 'ann_cat.u8.npy',
    'bbox': 'ann_bbox.u16.npy',
    'poly_off': 'poly_offsets.i64.npy',
    'poly_xy': 'poly_xy.u16.npy',
    'stems': 'stems.jsonl',
    'meta': 'meta.json',
}


def quantize(vals, size):
    """Normalize by `size` and quantize to uint16, clamped to the page."""
    v = np.asarray(vals, dtype=np.float64) / float(size)
    n_clamped = int(np.count_nonzero((v < 0.0) | (v > 1.0)))
    v = np.clip(v, 0.0, 1.0)
    return np.rint(v * U16_MAX).astype(np.uint16), n_clamped


def dequantize(u16, size):
    """uint16 -> pixel coordinates at the given target size."""
    return np.asarray(u16, dtype=np.float32) / U16_MAX * float(size)


class LayoutStore:
    """
    Random-access reader.

        store = LayoutStore('dataset')
        regions = store.regions_for('PMC5707887_00000', 448, 576)

    `regions_for` returns coordinates already scaled to the requested target
    size, i.e. directly in the training image's pixel space.
    """

    def __init__(self, root):
        """
        `root` may be the dataset root (the store is found at <root>/layout) or
        the layout folder itself.

        Both are supported because the Kaggle CLI SKIPS subdirectories by
        default — `kaggle datasets create -p dataset/` would upload every big
        file and silently leave `layout/` behind, and the first sign of trouble
        would be a confusing error hours into an upload. Shipping the layout
        store as its own small dataset avoids that entirely, and it is only
        57 MB, so regenerating it never means re-uploading 24.6 GB.
        """
        p = Path(root)
        if (p / FILES['stems']).exists():
            self.root = p
        elif (p / 'layout' / FILES['stems']).exists():
            self.root = p / 'layout'
        else:
            raise FileNotFoundError(
                f"no layout store at {p} or {p/'layout'} "
                f"(looking for {FILES['stems']})")
        with open(self.root / FILES['meta']) as fh:
            self.meta = json.load(fh)
        if self.meta['layout_version'] != LAYOUT_VERSION:
            raise ValueError(f"layout version {self.meta['layout_version']} "
                             f"!= reader {LAYOUT_VERSION}")

        self.cat = np.load(self.root / FILES['cat'], mmap_mode='r')
        self.bbox = np.load(self.root / FILES['bbox'], mmap_mode='r')
        self.poly_off = np.load(self.root / FILES['poly_off'], mmap_mode='r')
        self.poly_xy = np.load(self.root / FILES['poly_xy'], mmap_mode='r')

        self.stems = []
        with open(self.root / FILES['stems']) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.stems.append(json.loads(line))
        self.by_stem = {r['stem']: i for i, r in enumerate(self.stems)}

    def __len__(self):
        return len(self.stems)

    def n_annotations(self):
        return int(self.cat.shape[0])

    def regions_for(self, stem, out_w, out_h, want_poly=True):
        """
        Regions for one page, scaled into an (out_w, out_h) image.

        Returns a list of dicts: category id/name, bbox as (x0, y0, x1, y1),
        and the polygon as an (P, 2) float32 array when requested.
        """
        i = self.by_stem.get(stem)
        if i is None:
            return []
        row = self.stems[i]
        lo, hi = row['ann_start'], row['ann_end']
        out = []
        for j in range(lo, hi):
            x0, y0, x1, y1 = self.bbox[j]
            reg = {
                'category_id': int(self.cat[j]),
                'category': CAT_NAMES.get(int(self.cat[j]), '?'),
                'bbox': (float(dequantize(x0, out_w)), float(dequantize(y0, out_h)),
                         float(dequantize(x1, out_w)), float(dequantize(y1, out_h))),
            }
            if want_poly:
                a, b = int(self.poly_off[j]), int(self.poly_off[j + 1])
                if b > a:
                    pts = np.asarray(self.poly_xy[a:b], dtype=np.float32)
                    reg['polygon'] = np.stack([
                        pts[:, 0] / U16_MAX * out_w,
                        pts[:, 1] / U16_MAX * out_h], axis=-1)
                else:
                    reg['polygon'] = np.zeros((0, 2), np.float32)
            out.append(reg)
        return out

    def bboxes_array(self, stem, out_w, out_h):
        """
        Vectorised variant for the training loop: (K, 4) float32 boxes and
        (K,) int64 categories, with no per-region Python dicts.
        """
        i = self.by_stem.get(stem)
        if i is None:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.int64))
        row = self.stems[i]
        lo, hi = row['ann_start'], row['ann_end']
        b = np.asarray(self.bbox[lo:hi], dtype=np.float32) / U16_MAX
        b[:, 0] *= out_w
        b[:, 2] *= out_w
        b[:, 1] *= out_h
        b[:, 3] *= out_h
        return b, np.asarray(self.cat[lo:hi], dtype=np.int64)
