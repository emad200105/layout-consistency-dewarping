"""
dewarp_dataset.py
=================
The torch Dataset that joins the three stores and hands the training loop
everything a step needs.

    from dewarp_dataset import DewarpTrainingSet, collate

    train = DewarpTrainingSet('dataset', split='train')
    val   = DewarpTrainingSet('dataset', split='val')
    loader = DataLoader(train, batch_size=8, shuffle=True, num_workers=4,
                        collate_fn=collate, pin_memory=True)

THE 234-PAGE EXCLUSION (issue 15)
---------------------------------
The image build ran before the decision to drop landscape pages, so it contains
234 sources the layout store does not. That is **707 of 145,313 samples
(0.487%)** with images and flow but no layout regions.

`require_layout=True` (the default) drops them by stem membership. Two reasons
it is the right call rather than zeroing the layout term for those samples:

  * those same pages are the landscape ones, squeezed 0.52x by the portrait
    448x576 resize — far outside the distribution every other page sits in, so
    they are training noise regardless of the layout question;
  * a silently-zero loss term on a subset is the kind of thing that quietly
    skews an ablation, which is exactly the experiment this project exists to
    run.

It costs 0.487% of the data and needs no rebuild. Set `require_layout=False`
to keep them.

THE SPLIT IS BY SOURCE PAGE, NOT BY SAMPLE
------------------------------------------
Each page appears 3-4 times (cylinder / perspective / combined / sometimes
flat) and every variant shares the SAME clean image. Splitting on samples would
put variants of one page on both sides and leak the answer straight into
validation. The split is therefore a deterministic hash of the page stem, so it
is stable across runs, machines and rebuilds without storing a split file.

WHAT RESOLUTION THINGS COME BACK AT
-----------------------------------
The image is full 448 x 576, because the encoder should see all of it. Flow,
mask and layout boxes come back on the 112 x 144 flow grid, because that is
where the decoder terminates and where the losses are computed — 16x cheaper
than doing it at full resolution, and no less correct: region membership and
band structure are resolution-independent.

Flow is returned BOTH normalized (what the model should regress: scale-free and
well-conditioned) and in pixel units on the flow grid (what the layout loss
needs).
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))

import dataset_format as fmt                                      # noqa: E402
import layout_format as lf                                        # noqa: E402


def stem_split(stem, val_frac=0.02, test_frac=0.0):
    """
    Deterministic split from the page stem alone.

    md5 rather than hash(): Python's hash is salted per process, so num_workers
    > 0 would disagree with the main process about which pages are validation.
    """
    h = int(hashlib.md5(stem.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < val_frac:
        return 'val'
    if h < val_frac + test_frac:
        return 'test'
    return 'train'


class DewarpTrainingSet(Dataset):
    def __init__(self, root, split='train', val_frac=0.02, test_frac=0.0,
                 require_layout=True, want_clean=True, layout_root=None,
                 flow_w=None, flow_h=None, max_regions=32,
                 layout_categories=(1, 2, 3), verbose=True):
        """
        Args:
            root:         dataset root (the folder holding meta.json).
            layout_root:  where the layout store lives. Defaults to
                          <root>/layout. On Kaggle the layout store is its own
                          dataset, so pass its mount point.
            split:        'train', 'val', 'test' or 'all'.
            require_layout: drop samples whose page has no layout annotations.
            layout_categories: which region types feed the layout loss.
                          Default (text, title, list) — the regions made of
                          text lines, where "must come out rectangular" is the
                          meaningful claim. Tables and figures are excluded by
                          default: a figure's bounding box has no internal
                          line structure to straighten.
            max_regions:  cap on regions per sample, largest-area first.
        """
        self.root = Path(root)
        self.images = fmt.DewarpDataset(self.root, want_clean=want_clean)
        self.layout = lf.LayoutStore(layout_root or self.root)
        self.want_clean = want_clean
        self.flow_w = flow_w or self.images.meta['flow_w']
        self.flow_h = flow_h or self.images.meta['flow_h']
        self.page_w = self.images.meta['page_w']
        self.page_h = self.images.meta['page_h']
        self.max_regions = max_regions
        self.cats = set(layout_categories)

        stems = self.images.manifest['stem']
        n_all = len(stems)
        keep = []
        n_no_layout = 0
        for i, stem in enumerate(stems):
            has = stem in self.layout.by_stem
            if require_layout and not has:
                n_no_layout += 1
                continue
            if split != 'all' and stem_split(stem, val_frac, test_frac) != split:
                continue
            keep.append(i)
        self.index = np.asarray(keep, dtype=np.int64)
        self._stems = stems

        if verbose:
            print(f"[DewarpTrainingSet] split={split!r}  "
                  f"{len(self.index):,} of {n_all:,} samples")
            if require_layout:
                print(f"                    dropped {n_no_layout:,} "
                      f"({100*n_no_layout/max(n_all,1):.3f}%) with no layout "
                      f"annotations (issue 15)")

    def __len__(self):
        return len(self.index)

    def _regions(self, stem):
        """Region boxes on the FLOW grid, largest first, filtered by category."""
        b, c = self.layout.bboxes_array(stem, self.flow_w, self.flow_h)
        if len(b) == 0:
            return np.zeros((0, 4), np.float32)
        sel = np.array([int(x) in self.cats for x in c], dtype=bool)
        b = b[sel]
        if len(b) == 0:
            return np.zeros((0, 4), np.float32)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        order = np.argsort(-area)[:self.max_regions]
        return b[order].astype(np.float32)

    def __getitem__(self, k):
        i = int(self.index[k])
        stem = self._stems[i]

        warped = self.images._read(self.images._page_fh, 'pages',
                                   int(self.images._p_shard[i]),
                                   int(self.images._p_off[i]),
                                   int(self.images._p_len[i]))
        warped = fmt.decode_jpeg(warped)

        # Flow and mask straight off the stored grid — no resampling, so the
        # label the model sees is exactly the label that was written.
        flow_n = np.asarray(self.images.flows[i], dtype=np.float32)      # (h,w,2)
        mask = np.asarray(self.images.masks[i], dtype=np.float32) / 255.0

        out = {
            'warped': torch.from_numpy(
                np.ascontiguousarray(warped[:, :, ::-1].transpose(2, 0, 1))
            ).float().div_(255.0),
            'flow_norm': torch.from_numpy(flow_n.transpose(2, 0, 1)).float(),
            'mask': torch.from_numpy(mask).float().unsqueeze(0),
            'boxes': torch.from_numpy(self._regions(stem)),
            'stem': stem,
            'profile': self.images._profile[i],
            'sample_id': i,
        }
        if self.want_clean:
            c = int(self.images._clean_id[i])
            clean = fmt.decode_jpeg(
                self.images._read(self.images._clean_fh, 'clean',
                                  int(self.images._c_shard[c]),
                                  int(self.images._c_off[c]),
                                  int(self.images._c_len[c])))
            out['clean'] = torch.from_numpy(
                np.ascontiguousarray(clean[:, :, ::-1].transpose(2, 0, 1))
            ).float().div_(255.0)
        return out


def collate(batch):
    """Stack, padding the ragged region lists into (B, Kmax, 4) + validity."""
    out = {}
    for key in ('warped', 'clean', 'flow_norm', 'mask'):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    out['stem'] = [b['stem'] for b in batch]
    out['profile'] = [b['profile'] for b in batch]
    out['sample_id'] = torch.tensor([b['sample_id'] for b in batch])

    K = max(1, max(len(b['boxes']) for b in batch))
    B = len(batch)
    boxes = torch.zeros(B, K, 4)
    valid = torch.zeros(B, K)
    for i, b in enumerate(batch):
        n = len(b['boxes'])
        if n:
            boxes[i, :n] = b['boxes']
            valid[i, :n] = 1.0
    out['boxes'] = boxes
    out['box_valid'] = valid
    return out


def flow_to_pixels(flow_norm, w, h):
    """Normalized displacement -> pixel units on a (w, h) grid."""
    out = flow_norm.clone()
    out[:, 0] = out[:, 0] * (w - 1) / 2.0
    out[:, 1] = out[:, 1] * (h - 1) / 2.0
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    from torch.utils.data import DataLoader

    tr = DewarpTrainingSet(a.root, split='train')
    va = DewarpTrainingSet(a.root, split='val')
    print(f"\ntrain {len(tr):,}   val {len(va):,}   "
          f"val share {100*len(va)/max(len(tr)+len(va),1):.2f}%")

    # No page may appear in both splits.
    tr_stems = {tr._stems[int(i)] for i in tr.index}
    va_stems = {va._stems[int(i)] for i in va.index}
    overlap = tr_stems & va_stems
    print(f"pages: train {len(tr_stems):,}  val {len(va_stems):,}  "
          f"OVERLAP {len(overlap)}  -> {'OK' if not overlap else 'LEAK!'}")

    dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=0,
                    collate_fn=collate)
    b = next(iter(dl))
    print("\nbatch shapes:")
    for k, v in b.items():
        print(f"  {k:12s} {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}"
              f"{'' if torch.is_tensor(v) else ' len=' + str(len(v))}")
    print(f"  regions per sample: {b['box_valid'].sum(1).tolist()}")
    print(f"  profiles: {b['profile']}")
