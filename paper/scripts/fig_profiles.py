import sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import RESULTS, FIGS
from style import plt, LAY, ABL, COL1
import json
# full-validation numbers, both arms at step 531,990 (n = 2,737), per-page means
L = json.load(open(RESULTS / 'synthetic_val' / 'final_layout.epe.json'))
A = json.load(open(RESULTS / 'synthetic_val' / 'final_ablation.epe.json'))
prof = ['flat', 'cylinder', 'perspective', 'combined']
n = L['counts']
epe = {p: (L['epe_by_profile'][p], A['epe_by_profile'][p]) for p in prof}
lay = {p: (L['layout_metric']['by_profile'][p], A['layout_metric']['by_profile'][p]) for p in prof}
rel = lambda t: 100 * (t[1] - t[0]) / t[1]            # reduction vs. the ablation, %
df = pd.read_csv(RESULTS / 'ocr_synthetic' / 'per_page.csv')
rng = np.random.default_rng(0)
ocr, ocr_ci, nocr = {}, {}, {}
for p in prof:
    s = df[df.profile == p]; L, A = s.cer_layout.values, s.cer_ablation.values
    ocr[p] = 100 * (A.mean() - L.mean()) / A.mean(); nocr[p] = len(s)
    b = []
    for _ in range(10000):
        j = rng.integers(0, len(s), len(s)); b.append(100 * (A[j].mean() - L[j].mean()) / A[j].mean())
    ocr_ci[p] = np.percentile(b, [2.5, 97.5])
    print(p, round(ocr[p], 2), np.round(ocr_ci[p], 2))
x = np.arange(len(prof)); w = 0.26
fig, ax = plt.subplots(figsize=(COL1, 1.9))
ax.axhline(0, color='k', lw=0.5)
ax.bar(x - w, [rel(epe[p]) for p in prof], w, color='0.65', label='flow EPE')
ax.bar(x, [rel(lay[p]) for p in prof], w, color=LAY, label='layout consistency $\\mathcal{C}$')
o = np.array([ocr[p] for p in prof]); ci = np.array([ocr_ci[p] for p in prof]).T
ax.bar(x + w, o, w, color='#e0a33a', label='OCR CER', yerr=[o - ci[0], ci[1] - o],
       error_kw=dict(lw=0.6, capsize=1.5, capthick=0.6))
ax.set_xticks(x); ax.set_xticklabels([f'{p}\n(n={n[p]} / {nocr[p]})' for p in prof], fontsize=6.5)
ax.set_ylabel('reduction vs. ablation (%)')
ax.legend(frameon=False, loc='upper left', ncol=1, fontsize=6.5)
ax.set_ylim(-20, 25); ax.grid(axis='x', visible=False)
fig.savefig(FIGS / 'profiles.pdf'); fig.savefig(FIGS / 'profiles.png', dpi=200)
