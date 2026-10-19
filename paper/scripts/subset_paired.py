"""Per-page EPE and layout consistency C for both arms on the fixed 334-page subset
(the OCR subset), so the synthetic metrics get paired per-page statistics too.

Needs outputs/ocr_subset.npz (training/extract_ocr_subset.py) and
outputs/subset_boxes_flowgrid.json (paper/scripts/export_subset_boxes.py).
Writes results/synthetic_val/subset_paired.json and prints the table."""
import sys, json, numpy as np, torch, cv2
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, WEIGHTS, OCR_SUBSET, SUBSET_BOXES, need
import inference as inf
from layout_loss import layout_consistency_loss
from scipy.stats import wilcoxon
torch.set_num_threads(1)
z = np.load(need(OCR_SUBSET, 'python training/extract_ocr_subset.py --root dataset --out outputs/ocr_subset.npz'), allow_pickle=True)
boxes = json.load(open(need(SUBSET_BOXES, 'python paper/scripts/export_subset_boxes.py --layout_root dataset')))
nets = {k: inf.load_checkpoint(WEIGHTS[k])[0] for k in ('layout', 'ablation')}
res = {k: {'epe': [], 'C': [], 'Ch': [], 'Cv': []} for k in nets}
for i in range(len(z['stem'])):
    w = cv2.imdecode(np.frombuffer(z['warped'][i], np.uint8), cv2.IMREAD_COLOR)
    x = torch.from_numpy(np.ascontiguousarray(w[:, :, ::-1].transpose(2, 0, 1))).float().div(255)[None]
    gt = torch.from_numpy(z['flow'][i].astype(np.float32).transpose(2, 0, 1))[None]
    m = torch.from_numpy(z['mask'][i].astype(np.float32) / 255.0)[None, None]
    b = torch.tensor(boxes[str(z['stem'][i])], dtype=torch.float32).reshape(1, -1, 4)
    v = torch.ones(1, b.shape[1])
    for k, net in nets.items():
        with torch.no_grad():
            p = net(x)['flow']
        d = p - gt
        dpx = torch.stack([d[:, 0] * 447 / 2, d[:, 1] * 575 / 2], 1)
        mm = (m > 0.5).float()
        epe = float((torch.sqrt((dpx ** 2).sum(1, keepdim=True) + 1e-12) * mm).sum() / mm.sum().clamp(min=1))
        fw, fh = 112, 144
        pp = torch.stack([p[:, 0] * (fw - 1) / 2, p[:, 1] * (fh - 1) / 2], 1)
        gg = torch.stack([gt[:, 0] * (fw - 1) / 2, gt[:, 1] * (fh - 1) / 2], 1)
        r = layout_consistency_loss(pp, gg, m, b, v, n_bands=24, mode='global')
        res[k]['epe'].append(epe); res[k]['C'].append(float(r['loss'])); res[k]['Ch'].append(float(r['horizontal'])); res[k]['Cv'].append(float(r['vertical']))
prof = np.array([str(p) for p in z['profile']])
out = {'profile': prof.tolist(), **{k: {m: list(map(float, v)) for m, v in d.items()} for k, d in res.items()}}
json.dump(out, open(RESULTS / 'synthetic_val' / 'subset_paired.json', 'w'))
rng = np.random.default_rng(0)
def report(name, L, A, sel):
    L, A = np.asarray(L)[sel], np.asarray(A)[sel]; d = A - L
    bs = [d[j].mean() for j in rng.integers(0, len(d), (10000, len(d)))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'{name:10s} n={sel.sum():3d} lay {L.mean():.4f} abl {A.mean():.4f} red {100*d.mean()/A.mean():+.2f}% '
          f'CI[{100*lo/A.mean():+.2f},{100*hi/A.mean():+.2f}] wins {int((d>0).sum())}-{int((d<0).sum())} p={wilcoxon(d).pvalue:.2e}')
for met in ('epe', 'C', 'Ch', 'Cv'):
    print('==', met)
    report('ALL', res['layout'][met], res['ablation'][met], np.ones(len(prof), bool))
    for p in ('flat', 'cylinder', 'perspective', 'combined'):
        report(p, res['layout'][met], res['ablation'][met], prof == p)
