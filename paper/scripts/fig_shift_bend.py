"""Illustrates the layout-consistency loss: a rigid shift and a bend with the SAME
flow L1 get very different layout losses. Numbers come from the training code."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGS
from style import plt, LAY, ABL, COL1
from layout_loss import layout_consistency_loss
H, W = 144, 112                       # flow grid used by the loss
ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
boxes = np.array([[12, 14, 100, 62], [12, 74, 100, 130]], np.float32)   # two text blocks (clean px)
def field(kind, amp):
    d = np.zeros((2, H, W), np.float32)
    if kind == 'shift': d[1] = amp
    if kind == 'bend':  d[1] = amp * ((xs - W / 2) / (W / 2)) ** 2
    return d
def losses(d):
    gt = torch.zeros(1, 2, H, W); pr = torch.from_numpy(d)[None]
    m = torch.ones(1, 1, H, W)
    b = torch.from_numpy(boxes)[None]; v = torch.ones(1, len(boxes))
    r = layout_consistency_loss(pr, gt, m, b, v, n_bands=24)
    inside = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes: inside |= (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
    return float(np.abs(d).sum(0)[inside].mean()), float(r['loss'])
# choose the bend amplitude so that its L1 inside the regions equals the shift's
s_amp = 3.0
l1_s, lay_s = losses(field('shift', s_amp))
grid = np.linspace(1, 20, 2000); l1b = np.array([losses(field('bend', a))[0] for a in grid[::50]])
b_amp = float(np.interp(l1_s, l1b, grid[::50]))
l1_b, lay_b = losses(field('bend', b_amp))
print(f'shift: L1 {l1_s:.3f} lay {lay_s:.4f} | bend amp {b_amp:.3f}: L1 {l1_b:.3f} lay {lay_b:.4f}')
fig, axs = plt.subplots(1, 3, figsize=(COL1, 1.75))
titles = [('ground truth', None), ('rigid shift', (l1_s, lay_s)), ('bend', (l1_b, lay_b))]
for ax, kind, (t, v) in zip(axs, ['none', 'shift', 'bend'], titles):
    d = field(kind, s_amp if kind == 'shift' else b_amp)[1, 0]   # d_y along x (constant in y)
    ax.add_patch(plt.Rectangle((0, 0), W, H, fc='white', ec='0.3', lw=0.6))
    for x0, y0, x1, y1 in boxes:
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec=LAY, lw=0.6, ls='--'))
        for y in np.arange(y0 + 4, y1 - 2, 5.5):
            xx = np.linspace(x0 + 3, x1 - 3, 80)
            dd = np.interp(xx, np.arange(W), d)
            ax.plot(xx, y + dd, color='0.15', lw=0.9, solid_capstyle='butt')
    ax.set_xlim(-2, W + 2); ax.set_ylim(H + 2, -2); ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(t, fontsize=7.5, pad=2)
    if v: ax.text(W / 2, H + 12, f'$L_1$ in boxes={v[0]:.2f}\n$\\mathcal{{L}}_{{lay}}$={v[1]:.2f}', ha='center', va='top', fontsize=6.5)
    else: ax.text(W / 2, H + 12, '$L_1$ in boxes=0\n$\\mathcal{L}_{lay}$=0', ha='center', va='top', fontsize=6.5)
fig.subplots_adjust(wspace=0.08, bottom=0.22)
fig.savefig(FIGS / 'shift_bend.pdf'); fig.savefig(FIGS / 'shift_bend.png', dpi=200)
