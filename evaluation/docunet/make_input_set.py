"""
Write the unrectified benchmark photos as `<i>_<k>_rec.png`, with document 64
rotated by 180 degrees, so the "Distorted" baseline row is scored by exactly the
same pipeline (and naming) as the models.

    python evaluation/docunet/make_input_set.py --crop_dir data/docunet/crop \
                                                --out_dir outputs/docunet/distorted
"""
import argparse
import os
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fix_doc64', type=int, default=1)
    ap.add_argument('--n_docs', type=int, default=65)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(1, a.n_docs + 1):
        for k in (1, 2):
            img = cv2.imread(os.path.join(a.crop_dir, f'{i}_{k} copy.png'), cv2.IMREAD_UNCHANGED)
            if a.fix_doc64 and i == 64:
                img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(out / f'{i}_{k}_rec.png'), img)
    print('done')


if __name__ == '__main__':
    main()
