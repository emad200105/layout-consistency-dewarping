"""
extract_ocr_subset.py
=====================
Pull a fixed, stratified sample of VALIDATION pages out of the 25 GB dataset
into one small .npz, so OCR evaluation can run anywhere without the dataset.

    python training/extract_ocr_subset.py --root dataset --out ocr_subset.npz

What is stored per page: the warped JPEG and the clean JPEG exactly as they sit
in the blob files (raw bytes, no re-encoding, so no extra compression loss),
the ground-truth flow and the page mask. Plus stem and profile.

Selection is a pure function of the stem hash and a seed, NOT of how either
model does on the page. Choosing pages after looking at results would bias the
comparison, so the subset is fixed before any OCR is run.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))

from dewarp_dataset import DewarpTrainingSet                     # noqa: E402

PER_PROFILE = {'cylinder': 100, 'perspective': 100, 'combined': 100,
               'flat': 10 ** 9}          # flat: take all (only ~34 in val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset')
    ap.add_argument('--layout_root', default=None)
    ap.add_argument('--out', default='ocr_subset.npz')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    ds = DewarpTrainingSet(a.root, split='val', layout_root=a.layout_root,
                           want_clean=True, verbose=False)
    im = ds.images

    # order every val sample by a seeded hash of its stem+id: deterministic,
    # independent of model behaviour, stable across machines
    def key(k):
        i = int(ds.index[k])
        return hashlib.md5(f"{a.seed}:{ds._stems[i]}:{i}".encode()).hexdigest()

    order = sorted(range(len(ds)), key=key)
    picked, count = [], {}
    for k in order:
        prof = im._profile[int(ds.index[k])]
        if count.get(prof, 0) < PER_PROFILE.get(prof, 0):
            picked.append(k)
            count[prof] = count.get(prof, 0) + 1

    warped, clean, flows, masks, stems, profs = [], [], [], [], [], []
    for k in picked:
        i = int(ds.index[k])
        warped.append(bytes(im._read(im._page_fh, 'pages', int(im._p_shard[i]),
                                     int(im._p_off[i]), int(im._p_len[i]))))
        c = int(im._clean_id[i])
        clean.append(bytes(im._read(im._clean_fh, 'clean', int(im._c_shard[c]),
                                    int(im._c_off[c]), int(im._c_len[c]))))
        flows.append(np.asarray(im.flows[i], dtype=np.float16))
        masks.append(np.asarray(im.masks[i], dtype=np.uint8))
        stems.append(ds._stems[i])
        profs.append(im._profile[i])

    np.savez(a.out,
             warped=np.array(warped, dtype=object),
             clean=np.array(clean, dtype=object),
             flow=np.stack(flows), mask=np.stack(masks),
             stem=np.array(stems), profile=np.array(profs),
             meta=np.array([f"split=val seed={a.seed} n={len(picked)} "
                            f"per_profile={count}"]))
    mb = Path(a.out).stat().st_size / 2 ** 20
    print(f"wrote {a.out}  {len(picked)} pages  {mb:.1f} MB  {count}")


if __name__ == '__main__':
    main()
