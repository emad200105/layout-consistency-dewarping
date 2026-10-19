"""
OCR edit distance (ED) and character error rate (CER) on the DocUNet benchmark,
following DocTr's OCR_eval.py exactly:

  * reference text = pytesseract.image_to_string(scan/<i>.png)  (full-res scan, defaults)
  * test text      = pytesseract.image_to_string(<rec_dir>/<i>_<k>_rec.png)  (as is, defaults)
  * ED  = Levenshtein distance between the raw strings; CER = ED / len(reference)
  * reported value = (mean over the k=1 photos + mean over the k=2 photos) / 2
  * Setting 1 (DocTr, 30 documents = 60 photos) and Setting 2 (DewarpNet,
    25 documents = 50 photos)

    python evaluation/docunet/ocr.py --rec_dir outputs/docunet/ours \
        --scan_dir data/docunet/scan --cache outputs/docunet/ocr_ours_t501.jsonl \
        --gt_cache outputs/docunet/ocr_scan_t501.jsonl --tesseract /opt/tess501/tess.sh

Use the SAME Tesseract build for every row you compare. Published numbers used
Tesseract 5.0.1 (see install_tesseract_501.sh). OCR text is cached per image
(JSONL), so runs are resumable.
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault('OMP_THREAD_LIMIT', '1')   # Tesseract's OpenMP is slower here
import numpy as np                                 # noqa: E402
import pytesseract                                 # noqa: E402
from PIL import Image                              # noqa: E402
from rapidfuzz.distance import Levenshtein         # noqa: E402

S1 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 24, 27, 30, 31, 32, 36, 38, 40, 41, 44, 45,
      46, 47, 48, 50, 51, 52, 53]
S2 = [1, 9, 10, 12, 19, 20, 21, 22, 23, 24, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 44, 45,
      46, 47, 49]


def load(path):
    d = {}
    if path and os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line)
            d[r['key']] = r['text']
    return d


def _ocr(job):
    key, path, cmd = job
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return key, pytesseract.image_to_string(Image.open(path))


def fill(cache, jobs, workers):
    have = load(cache)
    todo = [j for j in jobs if j[0] not in have]
    os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
    with open(cache, 'a', encoding='utf-8') as f, ProcessPoolExecutor(workers) as ex:
        for n, (k, t) in enumerate(ex.map(_ocr, todo), 1):
            f.write(json.dumps({'key': k, 'text': t}) + '\n')
            f.flush()
            print(f'  ocr {n}/{len(todo)} {k}', flush=True)
    return load(cache)


def score(gt, rec, lis):
    out = {}
    for k in (1, 2):
        eds = [Levenshtein.distance(gt[str(i)], rec[f'{i}_{k}']) for i in lis]
        cers = [e / len(gt[str(i)]) for e, i in zip(eds, lis)]
        out[k] = (np.mean(eds), np.mean(cers))
    return (out[1][0] + out[2][0]) / 2, (out[1][1] + out[2][1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rec_dir', required=True, help="folder with '<i>_<k>_rec.png'")
    ap.add_argument('--scan_dir', required=True)
    ap.add_argument('--cache', required=True, help='JSONL cache of this folder\'s OCR text')
    ap.add_argument('--gt_cache', required=True, help='JSONL cache of the scans\' OCR text '
                    '(one per Tesseract version)')
    ap.add_argument('--tesseract', default=None, help='tesseract executable (default: on PATH)')
    ap.add_argument('--tail', default='_rec.png')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    ids = sorted(set(S1) | set(S2))
    gt = fill(a.gt_cache, [(str(i), os.path.join(a.scan_dir, f'{i}.png'), a.tesseract) for i in ids],
              a.workers)
    rec = fill(a.cache, [(f'{i}_{k}', os.path.join(a.rec_dir, f'{i}_{k}{a.tail}'), a.tesseract)
                         for i in ids for k in (1, 2)], a.workers)
    for name, lis in (('Setting 1 (DocTr, 60 photos)', S1), ('Setting 2 (DewarpNet, 50 photos)', S2)):
        ed, cer = score(gt, rec, lis)
        print(f'{name}: ED {ed:.2f}  CER {cer:.4f}')


if __name__ == '__main__':
    main()
