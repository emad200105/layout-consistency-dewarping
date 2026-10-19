"""
dataset_format.py
=================
The on-disk contract for the dewarping dataset. Writer and reader both import
from here so they can never disagree.

WHY NOT LOOSE FILES
-------------------
The original pipeline wrote one PNG and one full-resolution float64 .npy per
sample. At 47,958 sources x 3 variants that is ~1.24 TB, 430,000+ files, and
unusable on Kaggle. This format stores the same information in ~18 GB.

LAYOUT
------
    <out_dir>/
        pages-0000.bin ...      concatenated warped JPEGs
        clean-0000.bin ...      concatenated clean JPEGs, ONE per source image
        flows.f16               memmap (N, FLOW_H, FLOW_W, 2) float16
        masks.u8                memmap (N, FLOW_H, FLOW_W)    uint8
        manifest.parquet        one row per sample; byte offsets + every param
        clean_index.parquet     one row per source image
        meta.json               format version, geometry, counts

Images are raw JPEG bytes concatenated into blob files, with (shard, offset,
length) in the manifest. That gives true random access with a single seek —
no tar streaming, no per-file inode overhead, and it survives being shuffled
by a DataLoader. Flow and mask are fixed-size, so they go in flat memmaps and
cost zero parsing.

The clean page is stored ONCE per source and referenced by all of its variants
(`clean_id`), rather than duplicated 3-4x. That saves ~4 GB.

FLOW ENCODING — three deliberate choices
----------------------------------------
1. NORMALIZED, not pixels. Stored as a displacement in [-1,1] grid units, the
   convention torch.nn.functional.grid_sample uses. The same file therefore
   works at any inference resolution; nothing has to be regenerated to train
   at a different size.

2. DISPLACEMENT, not absolute grid. float16 has ~11 bits of mantissa, so
   precision scales with magnitude. Absolute grid coordinates span the whole
   [-1,1] range (~0.0005 quantum, ~0.11 px at 448 wide); displacements are
   usually small, where float16 is far finer. Same storage, better accuracy.

3. OUT-OF-PAGE FLOW IS BOUNDED, NOT CLAMPED TO THE PAGE. Outside the page the
   warp formulae extrapolate freely and a homography can blow up without
   bound, which risks corrupting boundary cells during downsampling.
   Clamping the source coordinate to the page rectangle was the obvious fix
   and measurably made things WORSE (p99 error 1.10 -> 1.28 px), because the
   clamp introduces a derivative kink at the page edge and a kink is exactly
   the high-frequency content that downsampling cannot represent. The flow is
   therefore left smooth and merely clipped to +/-FLOW_BOUND normalized
   units, far outside anything real (the largest value observed over 60
   random warps was 0.888), purely as float16 overflow insurance.

MEASURED ACCURACY (448x576 page, error inside the page mask, in pixels)
-----------------------------------------------------------------------
    flow grid    dtype |   mean     p95     p99      max
     28 x  36      f16 |  0.209   1.029   3.539   12.865
     56 x  72      f16 |  0.052   0.201   1.099    6.122
     84 x 108      f16 |  0.029   0.072   0.653    4.016
    112 x 144      f16 |  0.018   0.038   0.373    2.713

float16 vs float32 is indistinguishable (0.052 vs 0.051 mean at 56x72): the
error is entirely spatial downsampling, not quantisation. So float16 halves
the flow storage for free.

The error tail is NOT confined to the page boundary — eroding the loss mask by
8 px only moves p99 from 1.099 to 0.806 and leaves the max untouched — so it is
intrinsic to 8x downsampling of a strongly curved field. The only real lever is
FLOW_W/FLOW_H, which is why they are a single constant.
"""

import json
from pathlib import Path

import cv2
import numpy as np

FORMAT_VERSION = 2

# Geometry. Imported from warp_augment so there is exactly one source of truth.
try:
    from warp_augment import TRAIN_W, TRAIN_H, FLOW_W, FLOW_H
except ImportError:                                    # standalone reader use
    TRAIN_W, TRAIN_H = 448, 576
    FLOW_W, FLOW_H = 56, 72

JPEG_QUALITY = 90

# float16 overflow insurance only; real values stay well under 1.0.
FLOW_BOUND = 8.0

FLOW_DTYPE = np.float16
MASK_DTYPE = np.uint8

FLOW_SHAPE = (FLOW_H, FLOW_W, 2)
MASK_SHAPE = (FLOW_H, FLOW_W)
FLOW_ITEM_BYTES = int(np.prod(FLOW_SHAPE)) * np.dtype(FLOW_DTYPE).itemsize
MASK_ITEM_BYTES = int(np.prod(MASK_SHAPE)) * np.dtype(MASK_DTYPE).itemsize


# ─────────────────────────────────────────────
# FLOW / MASK CODEC
# ─────────────────────────────────────────────

def encode_flow(flow_px: np.ndarray) -> np.ndarray:
    """
    Full-resolution pixel flow -> stored float16 normalized flow at FLOW_H x FLOW_W.

    Args:
        flow_px: (H, W, 2) float32 backward flow in pixel units.

    Returns:
        (FLOW_H, FLOW_W, 2) float16, normalized displacement in [-1, 1] units.
    """
    H, W = flow_px.shape[:2]

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    # Left smooth across the page edge on purpose — clamping the source
    # coordinate to the page introduces a derivative kink there and measurably
    # worsened reconstruction (p99 1.10 -> 1.28 px). Only a loose bound is
    # applied, as float16 overflow insurance.
    nx = 2.0 * flow_px[..., 0] / (W - 1.0)
    ny = 2.0 * flow_px[..., 1] / (H - 1.0)
    norm = np.clip(np.stack([nx, ny], axis=-1), -FLOW_BOUND, FLOW_BOUND).astype(np.float32)

    small = cv2.resize(norm, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
    return small.astype(FLOW_DTYPE)


def decode_flow(stored: np.ndarray, out_w: int = TRAIN_W,
                out_h: int = TRAIN_H) -> np.ndarray:
    """
    Stored float16 normalized flow -> pixel flow at (out_h, out_w).

    Because the stored flow is normalized it is resolution-independent: pass
    any output size, including the native size of a full-resolution page at
    inference time.
    """
    norm = cv2.resize(stored.astype(np.float32), (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR)
    dx = norm[..., 0] * (out_w - 1.0) / 2.0
    dy = norm[..., 1] * (out_h - 1.0) / 2.0
    return np.stack([dx, dy], axis=-1).astype(np.float32)


def encode_mask(mask: np.ndarray) -> np.ndarray:
    """Full-resolution float mask in [0,1] -> stored uint8 at FLOW_H x FLOW_W."""
    small = cv2.resize(mask.astype(np.float32), (FLOW_W, FLOW_H),
                       interpolation=cv2.INTER_AREA)
    return np.clip(small * 255.0 + 0.5, 0, 255).astype(MASK_DTYPE)


def decode_mask(stored: np.ndarray, out_w: int = TRAIN_W,
                out_h: int = TRAIN_H) -> np.ndarray:
    """Stored uint8 mask -> float mask in [0,1] at (out_h, out_w)."""
    up = cv2.resize(stored.astype(np.float32) / 255.0, (out_w, out_h),
                    interpolation=cv2.INTER_LINEAR)
    return np.clip(up, 0.0, 1.0)


def encode_jpeg(img: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    ok, buf = cv2.imencode('.jpg', img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def decode_jpeg(raw: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("JPEG decode failed")
    return img


def invert_flow(flow_px: np.ndarray, iters: int = 60,
                beta: float = 0.5) -> np.ndarray:
    """
    Invert a backward flow by fixed-point iteration.

    WHICH DIRECTION THE STORED FLOW GOES — read this before writing a loss
    ---------------------------------------------------------------------
    The stored flow `f` is the map that GENERATED the warp:

        warped[y, x] == clean[y + f_y, x + f_x]

    so `f` gives, for each WARPED pixel, its corresponding location in the
    clean page. Its domain is the warped image — the same grid as the network
    input, which is exactly what a CNN wants to predict.

    That is NOT the map needed to rectify an image. Rectification needs `g`
    defined on the clean/output grid:

        clean[y, x] ~= warped[y + g_y, x + g_x]

    `g` is the inverse of `f` as a coordinate map, and this function computes it.

    The practical consequence is that the whole training loop can run in the
    WARPED domain and never needs an inversion:

      * flow loss        — directly on `f`, masked by the stored mask
      * photometric loss — resample the clean page with predicted `f` and
                           compare against the warped input
      * layout loss      — a layout polygon vertex in warped space maps to
                           clean space by simply adding `f` at that vertex,
                           which is precisely the "does this text block come
                           out rectangular" test
      * inference        — invert once per image, here

    Iteration: a DAMPED fixed point, g <- g + beta * (-f(x + g(x)) - g).

    The undamped version (beta = 1) diverges on the strongly compressive parts
    of aggressive warps. Measured over 40 real samples, inside the visible page:

        iters, beta     mean resid    p99 resid    worst sample
           12, 1.00       0.9679 px    14.8226 px     10.4089 px
           30, 1.00       0.7954       14.0317        10.3078
           30, 0.50       0.0027        0.0371         0.0310
           60, 0.50       0.0019        0.0130         0.0084

    So beta = 0.5 is not a tuning nicety, it is the difference between a
    correct inverse and a useless one. The defaults cost ~60 remaps, which is
    irrelevant at inference (once per image).
    """
    H, W = flow_px.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    g = np.zeros_like(flow_px)
    for _ in range(iters):
        sx = np.clip(gx + g[..., 0], 0, W - 1).astype(np.float32)
        sy = np.clip(gy + g[..., 1], 0, H - 1).astype(np.float32)
        fx = cv2.remap(flow_px[..., 0], sx, sy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        fy = cv2.remap(flow_px[..., 1], sx, sy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
        target = np.stack([-fx, -fy], axis=-1)
        g = g + beta * (target - g)          # damped; beta=1 diverges
    return g.astype(np.float32)


# ─────────────────────────────────────────────
# TABLE I/O  (parquet when available, JSONL always)
# ─────────────────────────────────────────────

def write_table(rows, root: Path, name: str):
    """Write a list of dicts as JSONL (always) and parquet (if available)."""
    root = Path(root)
    with open(root / f'{name}.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(root / f'{name}.parquet', index=False)
        return 'parquet+jsonl'
    except Exception:                                            # noqa: BLE001
        return 'jsonl'


def _load_table(root: Path, name: str):
    """Read a table as a dict of columns. Prefers parquet, falls back to JSONL."""
    pq = Path(root) / f'{name}.parquet'
    if pq.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(pq)
            return {c: df[c].tolist() for c in df.columns}
        except Exception:                                        # noqa: BLE001
            pass
    rows = []
    with open(Path(root) / f'{name}.jsonl') as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return {}
    return {k: [r.get(k) for r in rows] for k in rows[0]}


# ─────────────────────────────────────────────
# READER
# ─────────────────────────────────────────────

class DewarpDataset:
    """
    Random-access reader for the format above. No torch dependency — wrap it in
    a torch Dataset by returning tensors from __getitem__ if you want one.

    Blob file handles are opened lazily and per-instance, so this is safe to
    use from DataLoader worker processes (each worker gets its own handles;
    handles are never inherited across a fork).
    """

    def __init__(self, root, out_w: int = None, out_h: int = None,
                 want_clean: bool = True):
        self.root = Path(root)
        with open(self.root / 'meta.json') as fh:
            self.meta = json.load(fh)
        if self.meta['format_version'] != FORMAT_VERSION:
            raise ValueError(f"format version {self.meta['format_version']} "
                             f"!= reader {FORMAT_VERSION}")

        self.out_w = out_w or self.meta['page_w']
        self.out_h = out_h or self.meta['page_h']
        self.want_clean = want_clean

        self.manifest = _load_table(self.root, 'manifest')
        self.clean_index = _load_table(self.root, 'clean_index')

        n = len(self.manifest['page_offset'])
        fh, fw = self.meta['flow_h'], self.meta['flow_w']
        self.flows = np.memmap(self.root / 'flows.f16', dtype=FLOW_DTYPE,
                               mode='r', shape=(n, fh, fw, 2))
        self.masks = np.memmap(self.root / 'masks.u8', dtype=MASK_DTYPE,
                               mode='r', shape=(n, fh, fw))

        self._p_shard = np.asarray(self.manifest['page_shard'], np.int64)
        self._p_off = np.asarray(self.manifest['page_offset'], np.int64)
        self._p_len = np.asarray(self.manifest['page_length'], np.int64)
        self._clean_id = np.asarray(self.manifest['clean_id'], np.int64)
        self._profile = list(self.manifest['profile'])
        self._c_shard = np.asarray(self.clean_index['shard'], np.int64)
        self._c_off = np.asarray(self.clean_index['offset'], np.int64)
        self._c_len = np.asarray(self.clean_index['length'], np.int64)

        self._page_fh = {}
        self._clean_fh = {}

    def __len__(self):
        return len(self._p_off)

    def _read(self, cache, prefix, shard, offset, length):
        fh = cache.get(shard)
        if fh is None:
            fh = open(self.root / f'{prefix}-{shard:04d}.bin', 'rb')
            cache[shard] = fh
        fh.seek(offset)
        return fh.read(length)

    def __getitem__(self, i):
        warped = decode_jpeg(self._read(self._page_fh, 'pages',
                                        int(self._p_shard[i]),
                                        int(self._p_off[i]),
                                        int(self._p_len[i])))
        flow = decode_flow(self.flows[i], self.out_w, self.out_h)
        mask = decode_mask(self.masks[i], self.out_w, self.out_h)

        if (warped.shape[1], warped.shape[0]) != (self.out_w, self.out_h):
            warped = cv2.resize(warped, (self.out_w, self.out_h),
                                interpolation=cv2.INTER_AREA)

        out = {'warped': warped, 'flow': flow, 'mask': mask,
               'index': i, 'profile': self._profile[i]}

        if self.want_clean:
            c = int(self._clean_id[i])
            clean = decode_jpeg(self._read(self._clean_fh, 'clean',
                                           int(self._c_shard[c]),
                                           int(self._c_off[c]),
                                           int(self._c_len[c])))
            if (clean.shape[1], clean.shape[0]) != (self.out_w, self.out_h):
                clean = cv2.resize(clean, (self.out_w, self.out_h),
                                   interpolation=cv2.INTER_AREA)
            out['clean'] = clean
        return out

    def __del__(self):
        for cache in (getattr(self, '_page_fh', {}), getattr(self, '_clean_fh', {})):
            for fh in cache.values():
                try:
                    fh.close()
                except Exception:                                # noqa: BLE001
                    pass
