"""
Export the PubLayNet region boxes (text, title, list; at most 32 per page,
largest first) for every page of the OCR subset, on the 112x144 flow grid —
exactly what DewarpTrainingSet feeds the layout loss. subset_paired.py needs it.

    python paper/scripts/export_subset_boxes.py --layout_root dataset
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import OCR_SUBSET, SUBSET_BOXES, need   # noqa: E402
from layout_format import LayoutStore               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout_root', required=True, help='dataset root or its layout/ folder')
    a = ap.parse_args()
    store = LayoutStore(a.layout_root)
    z = np.load(need(OCR_SUBSET, 'python training/extract_ocr_subset.py ...'), allow_pickle=True)
    out = {}
    for st in z['stem']:
        b, c = store.bboxes_array(str(st), 112, 144)
        if len(b):
            b = b[np.array([int(x) in (1, 2, 3) for x in c], bool)]
        if len(b):
            area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
            b = b[np.argsort(-area)[:32]]
        out[str(st)] = np.round(np.asarray(b, dtype=float).reshape(-1, 4), 4).tolist()
    SUBSET_BOXES.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(SUBSET_BOXES, 'w'))
    print(f'{len(out)} pages, {sum(len(v) for v in out.values())} boxes -> {SUBSET_BOXES}')


if __name__ == '__main__':
    main()
