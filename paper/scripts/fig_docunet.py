"""DocUNet qualitative figure. Images picked by RULE, not by eye: rank the 130 photos by
the layout arm's MS-SSIM gain over the input and take the 90th, 50th and 10th percentile
images plus the single worst one (a failure case)."""
import sys, numpy as np, cv2, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGS, RESULTS, DOCUNET_OUT, DOCUNET_DATA
from style import plt, COL2
R = RESULTS / 'docunet'
L = pd.read_csv(R / 'ours_msssim_ld.csv').set_index('name'); A = pd.read_csv(R / 'ablation_msssim_ld.csv').set_index('name')
D = pd.read_csv(R / 'distorted_msssim_ld.csv').set_index('name')
g = (L.ms_ssim - D.ms_ssim).sort_values()
n = len(g); pick = [g.index[int(round(q * (n - 1)))] for q in (0.9, 0.5, 0.1)] + [g.index[0]]
tags = ['90th pct.', 'median', '10th pct.', 'worst']
print(list(zip(pick, tags)))
cols = [('input', lambda m: DOCUNET_OUT / 'distorted' / f'{m}_rec.png'),
        ('ours', lambda m: DOCUNET_OUT / 'ours' / f'{m}_rec.png'),
        ('ablation', lambda m: DOCUNET_OUT / 'ablation' / f'{m}_rec.png'),
        ('scan (GT)', lambda m: DOCUNET_DATA / 'scan' / f'{m.split("_")[0]}.png')]
fig, axs = plt.subplots(2, 8, figsize=(COL2, 2 * COL2 / 8 * 1.3 + 0.35))
for r, (m, t) in enumerate(zip(pick, tags)):
    for c, (h, f) in enumerate(cols):
        ax = axs[r // 2, (r % 2) * 4 + c]
        im = cv2.imread(str(f(m)))
        if im is None: raise SystemExit(f'missing {f(m)} (run evaluation/docunet first)')
        im = cv2.resize(im, (300, 390), interpolation=cv2.INTER_AREA)
        ax.imshow(im[:, :, ::-1]); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4)
        ax.set_title(f'{m} ({t})' if c == 0 else h, fontsize=6.3, pad=1.5)
        sc = {'input': D, 'ours': L, 'ablation': A}.get(h)
        if sc is not None:
            ax.text(0.5, -0.02, f'{sc.loc[m, "ms_ssim"]:.2f} / {sc.loc[m, "ld"]:.1f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=5.8)
fig.subplots_adjust(wspace=0.04, hspace=0.2)
fig.savefig(FIGS / 'docunet_qual.pdf', dpi=300); fig.savefig(FIGS / 'docunet_qual.png', dpi=160)
