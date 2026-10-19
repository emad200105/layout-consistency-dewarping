"""Qualitative synthetic-validation figure. Rows = the FIRST page of each warp profile
in the hash-ordered OCR subset (no cherry-picking). Columns: stored input photo,
rectified by the layout arm, by the ablation, by the true flow, and the clean page.
Prints the per-row CERs from the OCR test (per_page.csv) for reference."""
import sys, numpy as np, cv2, torch, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGS, RESULTS, WEIGHTS, OCR_SUBSET, need
from style import plt, COL2
import inference as inf, dataset_format as fmt
torch.set_num_threads(1)
z = np.load(need(OCR_SUBSET, 'python training/extract_ocr_subset.py --root dataset --out outputs/ocr_subset.npz'), allow_pickle=True)
prof = list(z['profile']); stems = list(z['stem'])
order = ['flat', 'cylinder', 'perspective', 'combined']
rows = [prof.index(p) for p in order]
nets = {k: inf.load_checkpoint(WEIGHTS[k])[0] for k in ('layout', 'ablation')}
S = 2                                   # display at 2x (896x1152) so text is legible
def unwarp(img, fn):
    h, w = img.shape[:2]
    g = fmt.invert_flow(inf.norm_to_px(fn, w, h), iters=60, beta=0.5)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + g[..., 0], gy + g[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
pp = pd.read_csv(RESULTS / 'ocr_synthetic' / 'per_page.csv').set_index('i')
cols = ['input', 'layout', 'ablation', 'true_flow']
heads = ['input', 'ours', 'ablation', 'true flow']
fig, axs = plt.subplots(2, 8, figsize=(COL2, COL2 / 8 * 576 / 448 * 2 + 0.25))
for r, i in enumerate(rows):
    warped = cv2.imdecode(np.frombuffer(z['warped'][i], np.uint8), cv2.IMREAD_COLOR)
    clean = cv2.imdecode(np.frombuffer(z['clean'][i], np.uint8), cv2.IMREAD_COLOR)
    h, w = warped.shape[0] * S, warped.shape[1] * S
    W2 = cv2.resize(warped, (w, h), interpolation=cv2.INTER_CUBIC)
    im = {'input': W2, 'clean': cv2.resize(clean, (w, h), interpolation=cv2.INTER_CUBIC),
          'true_flow': unwarp(W2, cv2.resize(z['flow'][i].astype(np.float32), (w, h)))}
    for k, net in nets.items():
        im[k] = unwarp(W2, inf.predict_flow_norm(net, warped, 'cpu', out_w=w, out_h=h))
    for c, k in enumerate(cols):
        ax = axs[r // 2, (r % 2) * 4 + c]; ax.imshow(im[k][:, :, ::-1]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
        ax.set_title(f'{order[r]}: {heads[c]}' if c == 0 else heads[c], fontsize=6.5, pad=1.5)
    print(order[r], i, stems[i], {k: round(pp.loc[i, f'cer_{k}'], 3) for k in ('input', 'layout', 'ablation', 'true_flow')})
fig.subplots_adjust(wspace=0.04, hspace=0.12)
fig.savefig(FIGS / 'synth_qual.pdf', dpi=300); fig.savefig(FIGS / 'synth_qual.png', dpi=160)
