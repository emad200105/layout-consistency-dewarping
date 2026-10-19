"""
identity_epe.py
===============
EPE of the do-nothing prediction (zero flow, i.e. the identity warp) on a split,
computed exactly like `inference.epe_by_profile`: per page, over page pixels
(mask > 0.5) on the 112x144 grid, in 448x576 page pixels, then averaged.

    python training/identity_epe.py --root dataset --out results/synthetic_val/identity_epe.json

Needs no model and no GPU; it reads only the stored flows and masks.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))
from dewarp_dataset import DewarpTrainingSet   # noqa: E402

PAGE_W, PAGE_H = 448, 576


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset')
    ap.add_argument('--layout_root', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    ds = DewarpTrainingSet(a.root, split=a.split, layout_root=a.layout_root,
                           want_clean=False, verbose=False)
    per, by = [], {}
    for k in range(len(ds)):
        i = int(ds.index[k])
        f = np.asarray(ds.images.flows[i], dtype=np.float32)
        m = np.asarray(ds.images.masks[i], dtype=np.float32) / 255.0 > 0.5
        dx, dy = f[..., 0] * (PAGE_W - 1) / 2.0, f[..., 1] * (PAGE_H - 1) / 2.0
        e = float(np.sqrt(dx * dx + dy * dy)[m].mean()) if m.any() else 0.0
        per.append(e)
        by.setdefault(ds.images._profile[i], []).append(e)
    res = {'split': a.split, 'n': len(per), 'identity_epe_px': float(np.mean(per)),
           'by_profile': {p: float(np.mean(v)) for p, v in sorted(by.items())},
           'counts': {p: len(v) for p, v in sorted(by.items())}}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
