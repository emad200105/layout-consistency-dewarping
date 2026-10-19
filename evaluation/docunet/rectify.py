"""
Rectify the DocUNet benchmark photos with a trained checkpoint.

    python evaluation/docunet/rectify.py --ckpt weights/dewarpnet_layout_lambda0.2.pt \
        --crop_dir data/docunet/crop --out_dir outputs/docunet/ours

For each photo `<i>_<k> copy.png` (the benchmark's file names):
  * the network sees a 448x576 resize of the photo (its training size);
  * the predicted backward flow is bilinearly upsampled to the photo's full
    resolution, inverted there (damped fixed point, 60 iterations, beta 0.5 —
    the same inversion used everywhere in this project) and the photo is
    resampled once;
  * pixels whose source lies outside the photo are filled white;
  * the result is written losslessly as `<out_dir>/<i>_<k>_rec.png`.

`--fix_doc64` (default on) rotates the two photos of document 64 by 180 degrees
before rectification. They are upside down relative to their scan in the
released benchmark; DocGeoNet, DocScanner and DocTr++ report numbers on this
"corrected" benchmark, and so do we.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
sys.path.insert(0, str(ROOT / 'augmentation'))
from inference import load_checkpoint, predict_flow_norm, norm_to_px   # noqa: E402
import dataset_format as fmt                                              # noqa: E402


def rectify_white(img, flow_norm):
    h, w = img.shape[:2]
    g = fmt.invert_flow(norm_to_px(flow_norm, w, h), iters=60, beta=0.5)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + g[..., 0], gy + g[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--crop_dir', required=True, help="folder with '<i>_<k> copy.png'")
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fix_doc64', type=int, default=1)
    ap.add_argument('--n_docs', type=int, default=65, help='documents 1..n (for quick tests)')
    ap.add_argument('--threads', type=int, default=1,
                    help='torch threads; 1 reproduces our outputs bit-exactly (more threads change '
                         'a few border pixels through floating-point summation order)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    net, info = load_checkpoint(a.ckpt)
    print(info, flush=True)
    for i in range(1, a.n_docs + 1):
        for k in (1, 2):
            dst = out / f'{i}_{k}_rec.png'
            if dst.exists():
                continue
            t = time.time()
            img = cv2.imread(os.path.join(a.crop_dir, f'{i}_{k} copy.png'), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f'missing {i}_{k} copy.png in {a.crop_dir}')
            if a.fix_doc64 and i == 64:
                img = cv2.rotate(img, cv2.ROTATE_180)
            rec = rectify_white(img, predict_flow_norm(net, img))
            tmp = str(dst) + '.tmp.png'
            cv2.imwrite(tmp, rec)
            os.replace(tmp, dst)
            print(f'{i}_{k} {img.shape[1]}x{img.shape[0]} {time.time() - t:.1f}s', flush=True)
    print('RECT_DONE')


if __name__ == '__main__':
    main()
