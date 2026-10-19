"""
Paired comparison of two rectification methods on the DocUNet benchmark.

    python evaluation/docunet/analyse.py \
        --msld_a outputs/docunet/ours_msld.csv  --msld_b outputs/docunet/ablation_msld.csv \
        --msld_in outputs/docunet/distorted_msld.csv \
        --ocr_a outputs/docunet/ocr_ours_t501.jsonl --ocr_b outputs/docunet/ocr_ablation_t501.jsonl \
        --ocr_in outputs/docunet/ocr_distorted_t501.jsonl --ocr_gt outputs/docunet/ocr_scan_t501.jsonl \
        --out outputs/docunet/paired.json

All differences are A minus B. With --per_image_csv, A is written as 'ours' and B as
'ablation' (the naming of results/docunet/ocr_per_image_*.csv). Confidence intervals (95%, 10,000 resamples)
resample DOCUMENTS, because the two photos of one document are not independent;
the Wilcoxon signed-rank test treats images as independent.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr import S1, S2, load   # noqa: E402


def boot_doc(diff, docs, rng, n=10000):
    u = np.unique(docs)
    idx = {d: np.where(docs == d)[0] for d in u}
    m = [np.mean(np.concatenate([diff[idx[d]] for d in rng.choice(u, len(u))])) for _ in range(n)]
    return [float(x) for x in np.percentile(m, [2.5, 97.5])]


def main():
    ap = argparse.ArgumentParser()
    for k in ('msld_a', 'msld_b', 'msld_in', 'ocr_a', 'ocr_b', 'ocr_in', 'ocr_gt'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--per_image_csv', default=None,
                    help='also write per-image OCR ED/CER (no text) for input, A and B')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {}
    M = {k: pd.read_csv(getattr(a, 'msld_' + k)).set_index('name') for k in ('a', 'b', 'in')}
    names = M['a'].index.tolist()
    docs = np.array([int(n.split('_')[0]) for n in names])
    for met in ('ms_ssim', 'ld'):
        va, vb, vi = (M[k].loc[names, met].values for k in ('a', 'b', 'in'))
        d = va - vb
        out[met] = dict(input=float(vi.mean()), a=float(va.mean()), b=float(vb.mean()),
                        diff_a_minus_b=float(d.mean()), ci=boot_doc(d, docs, rng),
                        p=float(wilcoxon(d).pvalue), n=len(names))
    gt = load(a.ocr_gt)
    rec = {k: load(getattr(a, 'ocr_' + k)) for k in ('a', 'b', 'in')}
    for sname, lis in (('S1', S1), ('S2', S2)):
        keys = [f'{i}_{k}' for i in lis for k in (1, 2)]
        dd = np.array([i for i in lis for _ in (1, 2)])
        ed = {m: np.array([Levenshtein.distance(gt[k.split('_')[0]], rec[m][k]) for k in keys]) for m in rec}
        cer = {m: ed[m] / np.array([len(gt[k.split('_')[0]]) for k in keys]) for m in rec}
        d = cer['a'] - cer['b']
        out['ocr_' + sname] = dict(n=len(keys), **{f'cer_{m}': float(cer[m].mean()) for m in rec},
                                   **{f'ed_{m}': float(ed[m].mean()) for m in rec},
                                   cer_diff_a_minus_b=float(d.mean()), ci=boot_doc(d, dd, rng),
                                   p=float(wilcoxon(d).pvalue))
    if a.per_image_csv:
        import csv
        ids = sorted(set(S1) | set(S2))
        with open(a.per_image_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['image', 'in_setting1', 'in_setting2', 'ref_chars', 'ed_distorted', 'cer_distorted',
                        'ed_ours', 'cer_ours', 'ed_ablation', 'cer_ablation'])
            for i in ids:
                for k in (1, 2):
                    key, g = f'{i}_{k}', gt[str(i)]
                    row = [key, int(i in S1), int(i in S2), len(g)]
                    for m in ('in', 'a', 'b'):
                        e = Levenshtein.distance(g, rec[m][key])
                        row += [e, round(e / len(g), 6)]
                    w.writerow(row)
    txt = json.dumps(out, indent=1)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt)


if __name__ == '__main__':
    main()
