"""
compare_arms.py
===============
Side-by-side comparison of the two training arms on the full validation set.

    python training/compare_arms.py --ours results/synthetic_val/final_layout.epe.json \
                                    --ablation results/synthetic_val/final_ablation.epe.json

Both inputs are the `<out>.epe.json` files that `inference.py --full_val` writes
next to its preview image. They hold the EPE per warp profile and the layout
distortion metric C (`layout_metric` in inference.py).

TWO VERSIONS OF C — READ THIS
-----------------------------
`layout_metric` reports C in two ways:

  * `loss` / `horizontal` / `vertical`: computed per mini-batch of 16 pages
    (band statistics pooled over the batch, then the square root, then averaged
    over batches). This is what training logs.
  * `by_profile`: computed per page and averaged. This is the statistic the
    paper uses. The all-pages per-page mean is recovered exactly by weighting
    the per-profile means by their sample counts.

The two are different numbers (e.g. 0.5274 vs 0.4398 for the layout arm), so
never mix them in one table. Relative reductions here follow the paper's
convention: (ablation - ours) / ablation.
"""

import argparse
import json


def red(ours, abl):
    return 100.0 * (abl - ours) / abl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ours', required=True, help='epe.json of the arm with the layout loss')
    ap.add_argument('--ablation', required=True, help='epe.json of the arm without it')
    ap.add_argument('--identity_epe', type=float, default=39.88,
                    help='EPE of the zero-flow (identity) prediction on the same set '
                         '(training/identity_epe.py; 39.88 px on our validation set)')
    ap.add_argument('--out', default=None, help='optional text file to write')
    a = ap.parse_args()

    L = json.load(open(a.ours))
    A = json.load(open(a.ablation))
    le, ae = L['epe_by_profile'], A['epe_by_profile']
    ll, al = L['layout_metric'], A['layout_metric']
    cnt = ll['counts']
    n = sum(cnt.values())

    def per_page(m):
        return sum(m['by_profile'][k] * cnt[k] for k in cnt) / n

    lines = []
    p = lines.append
    p(f"Full validation set, n = {n}   (reduction = (ablation - ours) / ablation)")
    p(f"{'':30s}{'ours':>10}{'ablation':>10}{'reduction':>11}")
    p('-' * 61)
    p(f"{'EPE (page px)':30s}{le['ALL']:10.3f}{ae['ALL']:10.3f}{red(le['ALL'], ae['ALL']):10.1f}%")
    pl, pa = per_page(ll), per_page(al)
    p(f"{'C, per-page mean (paper)':30s}{pl:10.4f}{pa:10.4f}{red(pl, pa):10.1f}%")
    p(f"{'C, batch-pooled (train log)':30s}{ll['loss']:10.4f}{al['loss']:10.4f}{red(ll['loss'], al['loss']):10.1f}%")
    p(f"{'  C_h, batch-pooled':30s}{ll['horizontal']:10.4f}{al['horizontal']:10.4f}"
      f"{red(ll['horizontal'], al['horizontal']):10.1f}%")
    p(f"{'  C_v, batch-pooled':30s}{ll['vertical']:10.4f}{al['vertical']:10.4f}"
      f"{red(ll['vertical'], al['vertical']):10.1f}%")
    p('')
    p(f"{'per profile':14s}{'n':>5}{'EPE ours':>10}{'abl':>8}{'red':>8}   {'C ours':>8}{'abl':>8}{'red':>8}")
    for k in ('flat', 'cylinder', 'perspective', 'combined'):
        lc, ac = ll['by_profile'][k], al['by_profile'][k]
        p(f"{k:14s}{cnt[k]:5d}{le[k]:10.3f}{ae[k]:8.3f}{red(le[k], ae[k]):7.1f}%   "
          f"{lc:8.4f}{ac:8.4f}{red(lc, ac):7.1f}%")
    p('')
    p(f"vs identity ({a.identity_epe} px): ours {100 * (1 - le['ALL'] / a.identity_epe):.1f}% lower EPE, "
      f"ablation {100 * (1 - ae['ALL'] / a.identity_epe):.1f}%")
    txt = '\n'.join(lines)
    print(txt)
    if a.out:
        open(a.out, 'w').write(txt + '\n')


if __name__ == '__main__':
    main()
