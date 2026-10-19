"""Overview figure (Fig. 1). Uses page 7 of the OCR subset (PMC6091653_00001, the first
cylinder page) and its PubLayNet text/title/list boxes in 448x576 page pixels
(read once from the layout store with layout_format.LayoutStore.bboxes_array)."""
import sys, numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import FIGS, WEIGHTS, OCR_SUBSET, need
from style import plt, LAY, COL2
import inference as inf, dataset_format as fmt
from matplotlib.patches import FancyBboxPatch
torch.set_num_threads(1)
z = np.load(need(OCR_SUBSET, 'python training/extract_ocr_subset.py --root dataset --out outputs/ocr_subset.npz'), allow_pickle=True)
i = 7   # PMC6091653_00001, cylinder (first cylinder page of the subset)
boxes = [[234.4, 53.3, 417.2, 184.0], [32.0, 53.4, 214.8, 216.8], [234.4, 191.0, 344.4, 203.5], [234.4, 206.6, 417.2, 255.1], [32.0, 225.8, 214.8, 405.6], [234.4, 262.1, 355.8, 274.6], [234.4, 277.9, 417.2, 443.9], [32.0, 414.6, 214.8, 479.5], [234.4, 450.9, 330.5, 463.5], [234.4, 466.6, 417.2, 482.2], [234.4, 485.7, 417.2, 534.2], [32.0, 486.5, 127.9, 499.0], [32.0, 502.1, 214.8, 534.4]]
warped = cv2.imdecode(np.frombuffer(z['warped'][i], np.uint8), cv2.IMREAD_COLOR)
clean = cv2.imdecode(np.frombuffer(z['clean'][i], np.uint8), cv2.IMREAD_COLOR)
net, _ = inf.load_checkpoint(WEIGHTS['layout'])
fn = inf.predict_flow_norm(net, warped, 'cpu', out_w=112, out_h=144)
fpx = inf.norm_to_px(cv2.resize(fn, (448, 576)), 448, 576)
mag, ang = cv2.cartToPolar(fpx[..., 0], fpx[..., 1])
hsv = np.zeros((576, 448, 3), np.uint8); hsv[..., 0] = ang * 180 / np.pi / 2; hsv[..., 1] = np.clip(mag / np.percentile(mag, 99) * 255, 0, 255); hsv[..., 2] = 255
flowviz = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
g = fmt.invert_flow(fpx, iters=60, beta=0.5)
gx, gy = np.meshgrid(np.arange(448, dtype=np.float32), np.arange(576, dtype=np.float32))
rect = cv2.remap(warped, gx + g[..., 0], gy + g[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
fig = plt.figure(figsize=(COL2, 1.65))
FH = 1.65
def img_ax(x, y, w, im, title, boxes_=None):
    h = w * 576 / 448 * COL2 / FH
    ax = fig.add_axes([x, y, w, h]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4)
    ax.set_title(title, fontsize=6.8, pad=2)
    if boxes_:
        for x0, y0, x1, y1 in boxes_:
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=LAY, alpha=0.18, ec=LAY, lw=0.5))
    return ax, h
W = 0.1; Y = 0.03
_, h = img_ax(0.0, Y, W, warped[:, :, ::-1], 'warped photo')
img_ax(0.33, Y, W, flowviz, 'backward flow $f$')
img_ax(0.62, Y, W, rect[:, :, ::-1], 'rectified page')
img_ax(0.88, Y, W, clean[:, :, ::-1], 'flat page + layout boxes', boxes)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
yc = Y + h * 0.45
def box(x, y, w, hh, txt, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle='round,pad=0.006', fc=fc, ec='0.25', lw=0.6))
    ax.text(x + w / 2, y + hh / 2, txt, ha='center', va='center', fontsize=6.3, linespacing=1.25)
def arrow(x0, y0, x1, y1, ls='-'):
    ax.annotate('', (x1, y1), (x0, y0), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='0.2', ls=ls, mutation_scale=7))
box(0.12, yc - 0.22, 0.19, 0.44, 'U-Net encoder-decoder\n(global-context\nbottleneck, 6.45 M)\n\n$448{\\times}576 \\rightarrow 112{\\times}144{\\times}2$', '#eef3fa')
arrow(0.1, yc, 0.12, yc); arrow(0.31, yc, 0.33, yc)
box(0.455, yc - 0.14, 0.145, 0.28, 'invert $f$ (damped\nfixed point) and\nresample at full res.', '#eef7ee')
arrow(0.43, yc, 0.455, yc); arrow(0.60, yc, 0.62, yc)
ax.text(0.455 + 0.0725, yc - 0.16, 'inference', fontsize=5.8, ha='center', va='top', color='0.35')
top = Y + h + 0.085
box(0.33, top + 0.025, 0.65, 0.18, 'training: $\\mathcal{L} = \\mathcal{L}_{flow}(f, f^*) + 0.5\\,\\mathcal{L}_{photo} + 0.2\\,\\mathcal{L}_{aux} + \\lambda\\,\\mathcal{L}_{lay}(f, f^*, \\mathrm{boxes})$\n'
    'ours: $\\lambda = 0.2$        ablation: $\\lambda = 0$        (everything else identical)', '#fff6e6')
arrow(0.38, Y + h + 0.075, 0.38, top + 0.015)
arrow(0.93, Y + h + 0.075, 0.93, top + 0.015)
fig.savefig(FIGS / 'pipeline.pdf'); fig.savefig(FIGS / 'pipeline.png', dpi=200)
