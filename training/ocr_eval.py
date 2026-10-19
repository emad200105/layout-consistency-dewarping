"""
ocr_eval.py
===========
OCR character error rate (CER) for the layout arm vs the ablation, on a fixed
subset of validation pages. This is the readability test: does a model that
produces straighter text lines (the layout metric) also produce text that OCR
reads more accurately?

    python training/ocr_eval.py --subset ocr/ocr_subset.npz \
        --ckpt layout=runs/layout_final.pt --ckpt ablation=runs/ablation_final.pt \
        --out ocr/results --workers 2

Resumable: OCR results are appended to <out>/ocr.jsonl as they finish, and a
re-run skips anything already there.

PROTOCOL (state all of this in the paper)
-----------------------------------------
* WHY NOT OCR THE STORED PAGES: they are 448x576 (~53 dpi). In the warped
  photo the page is also shrunk inside the frame and JPEG-compressed, so letters
  are 3-4 px tall. Measured: even rectification with the GROUND-TRUTH flow then
  scores 66-71% CER, so the test could not tell a good model from a bad one.

* HIGH-RESOLUTION RE-SYNTHESIS (the DocUNet-benchmark pattern: predict the flow
  from a small image, apply it to a full-resolution photo):
    1. clean page upscaled 3x (bicubic)            -> C   (1344x1728)
    2. the page's own ground-truth warp applied to C -> W   (same geometry as the
       stored training photo, but without the shrink/JPEG/shadow loss)
    3. each model predicts the flow from the ORIGINAL stored 448x576 warped
       photo — exactly the input distribution it was trained on — and that flow
       (normalized units, so resolution-free) is inverted and applied to W.
  Geometry is identical between the stored photo and W, so the predicted flow is
  valid for W. This isolates GEOMETRY, which is what the layout loss targets.
  Lighting/shadow is deliberately not part of this test.

* Reference text = Tesseract on C (the clean page at the same 3x), not human
  transcription — the standard DocUNet protocol. PubLayNet has no transcriptions.
* Engine: Tesseract 5 LSTM (`--oem 1 --psm 3`), English, one OpenMP thread.
* Off-page pixels are white in every version (BORDER_CONSTANT 255 on every
  remap), so frame-edge smear is never read as characters.
* Versions scored per page:
      input      W, unrectified                       (the "do nothing" bar)
      layout     W rectified with the layout-arm model
      ablation   W rectified with the ablation model
      true_flow  W rectified with the ground-truth flow (the ceiling)
* Pages whose reference has < MIN_REF_CHARS characters are dropped (mostly
  figures/tables: CER on 40 characters is noise). The count is reported.
* CER = Levenshtein(hyp, ref) / len(ref) after collapsing whitespace.
  Reported as mean of per-page CER, and corpus CER (sum ED / sum len).
* Layout vs ablation is PAIRED (same pages): mean difference with a bootstrap
  95% CI, Wilcoxon signed-rank p, and win/tie/loss counts. This measures
  variation over PAGES. It says nothing about variation over training seeds —
  there is one seed per arm.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault('OMP_THREAD_LIMIT', '1')   # 4x faster: see note in main()

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))

PAGE_W, PAGE_H = 448, 576
MIN_REF_CHARS = 200
TESS_CONFIG = '--oem 1 --psm 3'
VERSIONS = ('input', 'layout', 'ablation', 'true_flow')


# ─────────────────────────────────────────────
# Stages 1+2 — per page: build every version at high res, OCR it
# (one task per page; workers load both models once)
# ─────────────────────────────────────────────

SCALE = 3
_W = {}                                    # per-worker state


def _init_worker(ckpts, subset):
    import torch
    import inference as inf
    torch.set_num_threads(1)
    _W['inf'] = inf
    _W['nets'] = {k: inf.load_checkpoint(v, 'cpu')[0] for k, v in ckpts.items()}
    _W['d'] = np.load(subset, allow_pickle=True)
    w, h = PAGE_W * SCALE, PAGE_H * SCALE
    _W['wh'] = (w, h)
    _W['grid'] = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))


def _remap(img, f_px):
    gx, gy = _W['grid']
    return cv2.remap(img, (gx + f_px[..., 0]).astype(np.float32),
                     (gy + f_px[..., 1]).astype(np.float32), cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def build_versions(i):
    """All five images for page i, as BGR uint8 at SCALE x training size."""
    import dataset_format as fmt
    inf, d = _W['inf'], _W['d']
    w, h = _W['wh']
    clean = cv2.imdecode(np.frombuffer(d['clean'][i], np.uint8), cv2.IMREAD_COLOR)
    warped = cv2.imdecode(np.frombuffer(d['warped'][i], np.uint8), cv2.IMREAD_COLOR)
    C = cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC)
    gt = cv2.resize(d['flow'][i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    Whr = _remap(C, inf.norm_to_px(gt, w, h))

    def unwarp(fn):
        return _remap(Whr, fmt.invert_flow(inf.norm_to_px(fn, w, h), iters=60, beta=0.5))

    out = {'clean': C, 'input': Whr, 'true_flow': unwarp(gt)}
    for k, net in _W['nets'].items():
        # predicted from the ORIGINAL stored photo — the training distribution
        out[k] = unwarp(inf.predict_flow_norm(net, warped, 'cpu', out_w=w, out_h=h))
    return out


def ocr_img(img):
    import pytesseract
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ' '.join(pytesseract.image_to_string(g, config=TESS_CONFIG).split())


def _page_task(args):
    i, save_dir = args
    t = time.time()
    imgs = build_versions(i)
    if save_dir is not None:
        for k, im in imgs.items():
            cv2.imwrite(str(Path(save_dir) / f'{i}_{k}.jpg'), im,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    texts = {k: ocr_img(im) for k, im in imgs.items()}
    return {'i': i, 'texts': texts, 'sec': round(time.time() - t, 1)}


def run_pages(subset, ckpts, out, workers, n_save=6):
    jsonl = out / 'ocr.jsonl'
    out.mkdir(parents=True, exist_ok=True)
    save_dir = out / 'examples'
    save_dir.mkdir(exist_ok=True)
    n = len(np.load(subset, allow_pickle=True)['stem'])
    done = set()
    if jsonl.exists():
        done = {json.loads(l)['i'] for l in open(jsonl, encoding='utf-8')}
    todo = [i for i in range(n) if i not in done]
    print(f'  pages: {len(done)} done, {len(todo)} to go, {workers} workers', flush=True)
    if not todo:
        return
    t0 = time.time()
    with open(jsonl, 'a', encoding='utf-8') as fh, \
            ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                initargs=(ckpts, subset)) as ex:
        futs = [ex.submit(_page_task, (i, str(save_dir) if i < n_save else None))
                for i in todo]
        for k, f in enumerate(as_completed(futs), 1):
            fh.write(json.dumps(f.result(), ensure_ascii=False) + '\n')
            fh.flush()
            if k % 10 == 0 or k == len(todo):
                el = time.time() - t0
                print(f'  page {k}/{len(todo)}  {el/60:.1f} min elapsed, '
                      f'~{el/k*(len(todo)-k)/60:.0f} min left', flush=True)


# ─────────────────────────────────────────────
# Stage 3 — score
# ─────────────────────────────────────────────

def score(subset, jsonl, out_dir):
    from rapidfuzz.distance import Levenshtein
    from scipy.stats import wilcoxon

    d = np.load(subset, allow_pickle=True)
    stems, profs = list(d['stem']), list(d['profile'])
    texts = {}
    for line in open(jsonl, encoding='utf-8'):
        r = json.loads(line)
        for k, t in r['texts'].items():
            texts[(r['i'], k)] = t

    rows, dropped = [], 0
    for i in range(len(stems)):
        ref = texts.get((i, 'clean'))
        if ref is None or any((i, v) not in texts for v in VERSIONS):
            continue
        if len(ref) < MIN_REF_CHARS:
            dropped += 1
            continue
        row = {'i': i, 'stem': stems[i], 'profile': profs[i], 'ref_chars': len(ref)}
        for v in VERSIONS:
            ed = Levenshtein.distance(texts[(i, v)], ref)
            row[f'ed_{v}'] = ed
            row[f'cer_{v}'] = ed / len(ref)
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'per_page.csv', 'w', encoding='utf-8') as fh:
        keys = list(rows[0].keys())
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(str(r[k]) for k in keys) + '\n')

    def block(sel, label):
        lines = [f'{label}  (n={len(sel)})']
        for v in VERSIONS:
            mean_cer = np.mean([r[f'cer_{v}'] for r in sel])
            corpus = sum(r[f'ed_{v}'] for r in sel) / sum(r['ref_chars'] for r in sel)
            med = np.median([r[f'cer_{v}'] for r in sel])
            ed = np.mean([r[f'ed_{v}'] for r in sel])
            lines.append(f'  {v:10s} mean CER {100*mean_cer:6.2f}%   corpus CER {100*corpus:6.2f}%'
                         f'   median {100*med:6.2f}%   mean ED {ed:7.1f}')
        return lines

    rep = []
    rep.append(f'OCR EVALUATION — Tesseract 5 LSTM, {SCALE}x high-res re-synthesis, reference = OCR(clean page at {SCALE}x)')
    rep.append(f'pages scored {len(rows)}   dropped (reference < {MIN_REF_CHARS} chars) {dropped}')
    rep.append('')
    rep += block(rows, 'ALL PAGES')
    for p in ('cylinder', 'flat', 'perspective', 'combined'):
        sel = [r for r in rows if r['profile'] == p]
        if sel:
            rep.append('')
            rep += block(sel, p)

    # paired layout vs ablation — overall, then per profile
    def paired(sel, label):
        aa = np.array([r['cer_layout'] for r in sel])
        bb = np.array([r['cer_ablation'] for r in sel])
        dd = bb - aa
        rg = np.random.default_rng(0)
        bt = [dd[rg.integers(0, len(dd), len(dd))].mean() for _ in range(10000)]
        l_, h_ = np.percentile(bt, [2.5, 97.5])
        nz_ = dd[dd != 0]
        p_ = wilcoxon(nz_).pvalue if len(nz_) > 10 else float('nan')
        return (f'  {label:20s} n={len(sel):3d}  layout {100*aa.mean():5.1f}%  ablation {100*bb.mean():5.1f}%'
                f'  diff {100*dd.mean():+5.2f} pts [{100*l_:+.2f}, {100*h_:+.2f}]'
                f'  rel {100*dd.mean()/aa.mean():+5.1f}%  wins {(dd>0).sum()}-{(dd<0).sum()}  p={p_:.3g}')

    a = np.array([r['cer_layout'] for r in rows])
    b = np.array([r['cer_ablation'] for r in rows])
    diff = b - a                                   # >0 means layout arm better
    rng = np.random.default_rng(0)
    boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    nz = diff[diff != 0]
    p = wilcoxon(nz).pvalue if len(nz) > 10 else float('nan')
    win, loss = int((diff > 0).sum()), int((diff < 0).sum())
    tie = len(diff) - win - loss
    rep.append('')
    rep.append('PAIRED: layout arm vs ablation, same pages')
    rep.append(f'  mean CER  layout {100*a.mean():.2f}%   ablation {100*b.mean():.2f}%')
    rep.append(f'  ablation minus layout: {100*diff.mean():+.2f} pts  '
               f'(95% bootstrap CI {100*lo:+.2f} to {100*hi:+.2f})')
    rep.append(f'  relative: ablation has {100*diff.mean()/a.mean():+.1f}% more character errors')
    rep.append(f'  layout better on {win} pages, worse on {loss}, tied on {tie}')
    rep.append(f'  Wilcoxon signed-rank p = {p:.4g}')
    rep.append('  (variation over PAGES only: one training seed per arm)')
    rep.append('')
    rep.append('PAIRED BY PROFILE  (diff = ablation minus layout; >0 means layout arm reads better)')
    for prof in ('cylinder', 'flat', 'perspective', 'combined'):
        sel = [r for r in rows if r['profile'] == prof]
        if sel:
            rep.append(paired(sel, prof))
    rest = [r for r in rows if r['profile'] != 'combined']
    rep.append(paired(rest, 'all except combined'))
    rep.append('')
    i_ = np.array([r['cer_input'] for r in rows]); t_ = np.array([r['cer_true_flow'] for r in rows])
    rep.append(f'Share of the possible improvement (input -> true-flow ceiling) achieved: '
               f'layout {100*(i_-a).mean()/(i_-t_).mean():.1f}%, ablation {100*(i_-b).mean()/(i_-t_).mean():.1f}%')

    txt = '\n'.join(rep)
    (out_dir / 'OCR_RESULT.txt').write_text(txt + '\n', encoding='utf-8')
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', default='ocr/ocr_subset.npz')
    ap.add_argument('--ckpt', action='append', required=True,
                    help='name=path, give twice: layout=... ablation=...')
    ap.add_argument('--out', default='ocr/results')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--stage', default='all', choices=['all', 'ocr', 'score'])
    a = ap.parse_args()

    ckpts = dict(c.split('=', 1) for c in a.ckpt)
    assert set(ckpts) == {'layout', 'ablation'}, 'need layout=... and ablation=...'
    out = Path(a.out)
    # OMP_THREAD_LIMIT=1 is set at import. Tesseract's default OpenMP threading
    # spent more time in the kernel than reading: 11.5 s/page vs 2.3 s/page
    # with one thread, and one-thread jobs then run side by side.
    if a.stage in ('all', 'ocr'):
        run_pages(a.subset, ckpts, out, a.workers)
    if a.stage in ('all', 'score'):
        score(a.subset, out / 'ocr.jsonl', out)

if __name__ == '__main__':
    main()
