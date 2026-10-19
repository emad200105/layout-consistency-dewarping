"""
inference.py
============
Run a trained checkpoint: predict the flow, invert it, rectify the page.

    # look at what the model does on held-out val samples, with ground truth
    python inference.py --ckpt best.pt --root <dataset> --layout_root <layout> \
                        --n 8 --out preview.png

    # dewarp your own photo of a page
    python inference.py --ckpt best.pt --image my_photo.jpg --out fixed.png

WHY INVERSION IS NEEDED (the part that is easy to get backwards)
---------------------------------------------------------------
The stored flow `f` — the thing the network is trained to predict — is the map
that GENERATED the warp:

    warped[y, x] == clean[y + f_y, x + f_x]

Its domain is the warped image, which is exactly the grid a CNN wants to
predict on. But it is NOT the map that rectifies. Rectification needs `g`,
defined on the OUTPUT grid:

    clean[y, x] ~= warped[y + g_y, x + g_x]

`g` is the inverse of `f` as a coordinate map. `dataset_format.invert_flow`
computes it by damped fixed-point iteration, and the damping is not optional:
undamped, it diverges on the compressive parts of strong warps (0.97 px mean
residual vs 0.0019 px with beta=0.5).

Getting this backwards does not crash. It produces a plausible-looking image
that is warped roughly twice as much, which is why it is written out here.

RESOLUTION
----------
The network sees 448x576 and emits flow on a 112x144 grid, but the flow is
stored in NORMALIZED units ([-1,1], the grid_sample convention), which are
resolution-independent. So the same prediction upsamples to any output size and
the rectification happens at the ORIGINAL resolution of the input photo — the
network's coarse grid costs smoothness, not sharpness.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))

import dataset_format as fmt                                      # noqa: E402
from model import DewarpNet                                       # noqa: E402

PAGE_W, PAGE_H = 448, 576


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def infer_base(state):
    """
    Recover the `base` width from the weights themselves.

    `head.weight` is (2, base*2, 3, 3), so the width is readable off the
    checkpoint. Worth doing rather than asking: a checkpoint carries no record
    of the flag it was trained with, and the failure mode is a wall of
    `size mismatch` lines that says nothing about which number to pass.
    """
    w = state.get('head.weight')
    return int(w.shape[1]) // 2 if w is not None else 32


def resolve_ckpt(path):
    """
    Accept a checkpoint file OR any folder above one.

    `kaggle kernels output` downloads the whole /kaggle/working tree, so the
    weights land at <download>/run_layout/best.pt, not <download>/best.pt. That
    extra level is invisible until FileNotFoundError names a path that looks
    exactly right. Searching for it is a two-line fix for a ten-minute confusion.
    Prefers best.pt, then last.pt, then the highest-numbered step checkpoint.
    """
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists():
        raise SystemExit(f'no such file or folder: {p}')
    for name in ('best.pt', 'last.pt'):
        hits = sorted(p.rglob(name))
        if hits:
            if len(hits) > 1:
                print(f'[ckpt] {len(hits)} matches for {name}, using the first:')
                for h in hits:
                    print(f'       {h}')
            print(f'[ckpt] {hits[0]}')
            return hits[0]
    steps = sorted(p.rglob('ckpt_step*.pt'))
    if steps:
        print(f'[ckpt] {steps[-1]}')
        return steps[-1]
    raise SystemExit(f'no checkpoint (best.pt / last.pt / ckpt_step*.pt) under {p}')


def load_checkpoint(path, device='cpu', base=None):
    """Load any checkpoint this project writes — best/last/ckpt_step*."""
    path = resolve_ckpt(path)
    blob = torch.load(path, map_location='cpu', weights_only=False)
    state = blob.get('model', blob)
    base = base or infer_base(state)
    net = DewarpNet(base=base)
    net.load_state_dict(state)
    net.eval().to(device)
    info = {k: blob.get(k) for k in ('epoch', 'gstep', 'best') if k in blob}
    info['base'] = base
    return net, info


@torch.no_grad()
def predict_flow_norm(net, img_bgr, device='cpu', out_w=None, out_h=None):
    """
    Predict normalized flow for one BGR image.

    Returns (out_h, out_w, 2) float32. Defaults to the input's own size, so the
    caller rectifies at full resolution rather than at the network's grid.
    """
    h0, w0 = img_bgr.shape[:2]
    out_w = out_w or w0
    out_h = out_h or h0

    net_in = cv2.resize(img_bgr, (PAGE_W, PAGE_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(
        np.ascontiguousarray(net_in[:, :, ::-1].transpose(2, 0, 1))
    ).float().div_(255.0).unsqueeze(0).to(device)

    flow = net(x)['flow']                                  # (1, 2, 144, 112)
    flow = torch.nn.functional.interpolate(
        flow, size=(out_h, out_w), mode='bilinear', align_corners=False)
    return flow[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


# ─────────────────────────────────────────────
# RECTIFICATION
# ─────────────────────────────────────────────

def norm_to_px(flow_norm, w, h):
    """Normalized displacement -> pixels on a (w, h) grid."""
    out = np.empty_like(flow_norm)
    out[..., 0] = flow_norm[..., 0] * (w - 1) / 2.0
    out[..., 1] = flow_norm[..., 1] * (h - 1) / 2.0
    return out


def rectify(img_bgr, flow_norm, iters=60, beta=0.5):
    """
    Undo the warp. Returns (rectified, in_bounds_mask, g_px).

    `g_px` is handed back because scoring needs it: the honest mask is not
    "did the source land in bounds" but "does this output pixel show page",
    and that is `page_visibility(page_mask, g_px)`.
    """
    h, w = img_bgr.shape[:2]
    f_px = norm_to_px(flow_norm, w, h)
    g_px = fmt.invert_flow(f_px, iters=iters, beta=beta)

    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    sx = (gx + g_px[..., 0]).astype(np.float32)
    sy = (gy + g_px[..., 1]).astype(np.float32)
    out = cv2.remap(img_bgr, sx, sy, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    valid = ((sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1))
    return out, valid.astype(np.uint8) * 255, g_px


def dewarp(net, img_bgr, device='cpu', iters=60, beta=0.5):
    flow_norm = predict_flow_norm(net, img_bgr, device)
    rect, valid, _ = rectify(img_bgr, flow_norm, iters, beta)
    return rect, valid, flow_norm


# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────

def label(img, text, color=(255, 255, 255)):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
                cv2.LINE_AA)
    return out


def draw_boxes(img, boxes, color=(60, 220, 60)):
    """
    Ground-truth layout rectangles, in CLEAN page coordinates.

    This is the whole novelty claim made visible: in the clean page these boxes
    ARE axis-aligned rectangles, so drawing them on the RECTIFIED output shows
    directly whether dewarped text blocks came out straight. Text that drifts
    out of its box is residual curvature the flow loss alone would not have
    flagged.
    """
    out = img.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (int(round(x0)), int(round(y0))),
                      (int(round(x1)), int(round(y1))), color, 1)
    return out


def page_visibility(mask_page, g_px):
    """
    Which output pixels show page after rectification.

    `mask_page` marks page pixels in the WARPED frame; pushing it through the
    same inverse map says where page ends up in the rectified frame. Scoring
    outside this is scoring background against paper, which is why a naive
    in-bounds mask reports ~12 dB worse than the real figure.
    """
    h, w = g_px.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    m = cv2.remap(mask_page.astype(np.float32),
                  (gx + g_px[..., 0]).astype(np.float32),
                  (gy + g_px[..., 1]).astype(np.float32),
                  cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                  borderValue=0.0)
    return m > 0.999


def psnr_page(clean_bgr, rect_bgr, vis, gain_correct=True):
    """
    PSNR on the visible page — the SAME metric verify_dataset.py reports, so
    numbers here are directly comparable to the 25.16 dB pipeline ceiling.

    Grayscale and gain-corrected, because the warp bakes in a multiplicative
    shadow. Rectifying does not remove the shadow, so without dividing out a
    low-pass ratio the score measures lighting, not geometry — which would
    punish a perfect dewarp for a dark page corner.
    """
    a = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(rect_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gain_correct:
        gain = np.clip(cv2.GaussianBlur(a, (0, 0), 25)
                       / np.maximum(cv2.GaussianBlur(b, (0, 0), 25), 1.0), 0.2, 3.0)
        b = b * gain
    d = (a - b)[vis]
    if d.size == 0:
        return float('nan')
    mse = max(float((d ** 2).mean()), 1e-9)
    return 10.0 * np.log10(255.0 ** 2 / mse)


# ─────────────────────────────────────────────
# VAL PREVIEW
# ─────────────────────────────────────────────

def fill_invalid(img, vis, value=235):
    """
    Paint the non-page region a flat colour instead of leaving it smeared.

    `cv2.remap` with BORDER_REPLICATE extends the edge pixel outward wherever
    the source coordinate falls outside the input, which produces long vertical
    streaks on any page that extended past the frame. Those streaks appear in
    the ground-truth-flow column too, so they are a property of resampling, not
    a model error — but in a figure they read as one, and a real system would
    emit the page region only.
    """
    out = img.copy()
    out[~vis] = value
    return out


def preview_from_dataset(net, root, layout_root, n, device, out_path,
                         split='val', seed=0, max_regions=24,
                         mask_invalid=False):
    """
    Contact sheet over held-out samples: input, prediction, truth.

    Deliberately the VAL split and deliberately spread across distortion
    profiles — a sheet of easy cylinders would look great and mean nothing.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dewarp_dataset import DewarpTrainingSet

    ds = DewarpTrainingSet(root, split=split, layout_root=layout_root,
                           want_clean=True, verbose=False)
    if len(ds) == 0:
        raise SystemExit('no samples in that split')

    # Spread across profiles rather than taking the first n.
    rng = np.random.RandomState(seed)
    by_profile = {}
    for k in range(len(ds)):
        by_profile.setdefault(ds.images._profile[int(ds.index[k])], []).append(k)
    picks = []
    profs = sorted(by_profile)
    while len(picks) < n and any(by_profile.values()):
        for p in profs:
            if by_profile[p] and len(picks) < n:
                picks.append(by_profile[p].pop(rng.randint(len(by_profile[p]))))

    rows, stats = [], []
    for k in picks:
        s_ = ds[k]
        warped = (s_['warped'].numpy().transpose(1, 2, 0)[:, :, ::-1] * 255
                  ).astype(np.uint8)
        clean = (s_['clean'].numpy().transpose(1, 2, 0)[:, :, ::-1] * 255
                 ).astype(np.uint8)

        # Page mask at full res: which warped pixels are page, not background.
        mask_px = cv2.resize(s_['mask'].numpy()[0], (PAGE_W, PAGE_H),
                             interpolation=cv2.INTER_LINEAR)

        pred_norm = predict_flow_norm(net, warped, device)
        rect, _, g_pred = rectify(warped, pred_norm)
        vis_pred = page_visibility(mask_px, g_pred)

        gt_norm = cv2.resize(s_['flow_norm'].numpy().transpose(1, 2, 0),
                             (PAGE_W, PAGE_H), interpolation=cv2.INTER_LINEAR)
        gt_rect, _, g_gt = rectify(warped, gt_norm)
        vis_gt = page_visibility(mask_px, g_gt)

        # EPE on the flow grid inside the stored mask — the training metric.
        m = s_['mask'].numpy()[0] > 0.5
        fh_, fw_ = s_['flow_norm'].shape[1], s_['flow_norm'].shape[2]
        fp = cv2.resize(pred_norm, (fw_, fh_), interpolation=cv2.INTER_LINEAR)
        fg = s_['flow_norm'].numpy().transpose(1, 2, 0)
        d = norm_to_px(fp - fg, PAGE_W, PAGE_H)
        epe = float(np.sqrt((d ** 2).sum(-1))[m].mean()) if m.any() else float('nan')

        boxes = s_['boxes'].numpy()
        sx_ = PAGE_W / max(ds.flow_w, 1)
        sy_ = PAGE_H / max(ds.flow_h, 1)
        boxes = boxes[:max_regions] * np.array([sx_, sy_, sx_, sy_], np.float32)

        # All three scored on the GROUND-TRUTH visible region. Using the
        # prediction's own mask made the "input" baseline move with the model,
        # so the same untouched input scored differently at 20 and 30 epochs —
        # a baseline that is not a baseline. vis_gt depends only on the data.
        p_in = psnr_page(clean, warped, vis_gt)
        p_pred = psnr_page(clean, rect, vis_gt)
        p_gt = psnr_page(clean, gt_rect, vis_gt)
        stats.append(dict(stem=s_['stem'], profile=s_['profile'], epe_px=epe,
                          psnr_input=p_in, psnr_predicted=p_pred,
                          psnr_true_flow=p_gt))

        show_pred, show_gt = rect, gt_rect
        if mask_invalid:
            show_pred = fill_invalid(rect, vis_pred)
            show_gt = fill_invalid(gt_rect, vis_gt)

        rows.append(np.hstack([
            label(warped, f"input  {s_['profile']}  {p_in:.1f} dB"),
            label(draw_boxes(show_pred, boxes),
                  f"PREDICTED  epe {epe:.2f} px  {p_pred:.1f} dB", (120, 255, 120)),
            label(draw_boxes(show_gt, boxes),
                  f'rectified with TRUE flow  {p_gt:.1f} dB', (120, 200, 255)),
            label(clean, 'ground truth'),
        ]))

    sheet = np.vstack([np.pad(r, ((0, 6), (0, 0), (0, 0)), constant_values=32)
                       for r in rows])
    cv2.imwrite(str(out_path), sheet)

    mean_epe = float(np.nanmean([r['epe_px'] for r in stats]))
    mean_in = float(np.nanmean([r['psnr_input'] for r in stats]))
    mean_pr = float(np.nanmean([r['psnr_predicted'] for r in stats]))
    mean_gt = float(np.nanmean([r['psnr_true_flow'] for r in stats]))
    summary = dict(n=len(stats), mean_epe_px=mean_epe,
                   mean_psnr_input=mean_in, mean_psnr_predicted=mean_pr,
                   mean_psnr_true_flow=mean_gt, samples=stats)
    Path(str(out_path) + '.json').write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {out_path}   ({sheet.shape[1]}x{sheet.shape[0]})")
    print(f"  mean EPE            {mean_epe:8.3f} px")
    print(f"  PSNR input vs clean {mean_in:8.2f} dB   <- the bar to beat")
    print(f"  PSNR dewarped       {mean_pr:8.2f} dB   "
          f"({mean_pr - mean_in:+.2f} dB)")
    print(f"  PSNR with TRUE flow {mean_gt:8.2f} dB   <- ceiling for this pipeline")
    return summary


@torch.no_grad()
def epe_by_profile(net, ds, n=200, device='cpu', seed=0, batch=16):
    """
    Mean EPE per distortion profile over `n` samples.

    Broken out by profile because the aggregate hides the thing worth knowing.
    A model can sit at a good average while failing entirely on `combined` —
    the hardest profile and the one closest to a real photograph — and the
    single number would never say so.
    """
    from torch.utils.data import DataLoader, Subset
    from dewarp_dataset import collate

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ds))[:min(n, len(ds))]
    # num_workers=0 deliberately. Windows spawns workers instead of forking, so
    # the dataset has to pickle — and it holds numpy memmaps, which cannot be.
    # The failure is an OSError deep inside multiprocessing that names no cause.
    # Evaluation is not throughput-bound anyway; the forward pass dominates.
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=batch, shuffle=False,
                    num_workers=0, collate_fn=collate)

    tot, cnt = {}, {}
    for b in dl:
        pred = net(b['warped'].to(device))['flow']
        gt = b['flow_norm'].to(device)
        m = (b['mask'].to(device) > 0.5).float()
        d = pred - gt
        d = torch.stack([d[:, 0] * (PAGE_W - 1) / 2.0,
                         d[:, 1] * (PAGE_H - 1) / 2.0], 1)
        e = torch.sqrt((d ** 2).sum(1, keepdim=True) + 1e-12) * m
        per = (e.flatten(1).sum(1) / m.flatten(1).sum(1).clamp(min=1)).cpu().numpy()
        for prof, v in zip(b['profile'], per):
            tot[prof] = tot.get(prof, 0.0) + float(v)
            cnt[prof] = cnt.get(prof, 0) + 1

    out = {k: tot[k] / cnt[k] for k in sorted(tot)}
    out['ALL'] = sum(tot.values()) / max(sum(cnt.values()), 1)
    return out, cnt


@torch.no_grad()
def layout_metric(net, ds, n=400, device='cpu', seed=0, batch=16,
                  n_bands=24, mode='global'):
    """
    Layout consistency as a pure METRIC, computed offline from a checkpoint.

    WHY THIS HAS TO LIVE HERE
    -------------------------
    `DewarpLoss` guards the whole layout branch behind `if self.w_layout > 0`,
    so the ablation arm — which exists precisely to be compared on this
    quantity — never computes it and never logs it. The number is recoverable
    after the fact from any checkpoint, which is what this does, identically for
    both arms.

    It is the measurement that decides the paper. EPE averages every kind of
    error together; this one responds to BENDING and ignores rigid shift, which
    is the whole claim. Lower is straighter.
    """
    from torch.utils.data import DataLoader, Subset
    from dewarp_dataset import collate
    from layout_loss import layout_consistency_loss

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(ds))[:min(n, len(ds))]
    dl = DataLoader(Subset(ds, idx.tolist()), batch_size=batch, shuffle=False,
                    num_workers=0, collate_fn=collate)

    agg = {'loss': 0.0, 'horizontal': 0.0, 'vertical': 0.0}
    by_prof, cnt_prof, nb = {}, {}, 0
    for b in dl:
        gt = b['flow_norm'].to(device)
        pred = net(b['warped'].to(device))['flow']
        mask = b['mask'].to(device)
        fw, fh = gt.shape[-1], gt.shape[-2]
        p_px = torch.stack([pred[:, 0] * (fw - 1) / 2.0,
                            pred[:, 1] * (fh - 1) / 2.0], dim=1)
        g_px = torch.stack([gt[:, 0] * (fw - 1) / 2.0,
                            gt[:, 1] * (fh - 1) / 2.0], dim=1)
        r = layout_consistency_loss(p_px, g_px, mask,
                                    b['boxes'].to(device),
                                    b['box_valid'].to(device),
                                    n_bands=n_bands, mode=mode)
        for k in agg:
            agg[k] += float(r[k].detach())
        nb += 1

        # per-profile needs one sample at a time, since the loss reduces over
        # the batch. Cheap enough at eval, and the per-profile split is where
        # an effect concentrated on hard warps would show up.
        for i, prof in enumerate(b['profile']):
            ri = layout_consistency_loss(p_px[i:i+1], g_px[i:i+1], mask[i:i+1],
                                         b['boxes'][i:i+1].to(device),
                                         b['box_valid'][i:i+1].to(device),
                                         n_bands=n_bands, mode=mode)
            by_prof[prof] = by_prof.get(prof, 0.0) + float(ri['loss'].detach())
            cnt_prof[prof] = cnt_prof.get(prof, 0) + 1

    out = {k: v / max(nb, 1) for k, v in agg.items()}
    out['by_profile'] = {k: by_prof[k] / cnt_prof[k] for k in sorted(by_prof)}
    out['counts'] = cnt_prof
    out['n'] = int(sum(cnt_prof.values()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='preview.png')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--base', type=int, default=None,
                    help='override; read from the checkpoint otherwise')
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--beta', type=float, default=0.5)
    # dataset mode
    ap.add_argument('--root', default=None)
    ap.add_argument('--layout_root', default=None)
    ap.add_argument('--split', default='val')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--profile_table', type=int, default=0,
                    help='also print mean EPE by distortion type over N samples')
    ap.add_argument('--mask_invalid', action='store_true',
                    help='paint the off-page region flat instead of letting '
                         'BORDER_REPLICATE smear it into streaks. Use for '
                         'figures.')
    ap.add_argument('--full_val', action='store_true',
                    help='evaluate the ENTIRE split, not a subsample. This is '
                         'the number to report: training-time validation uses '
                         'only 60 batches, so its epe_px is a subset estimate.')
    # single-image mode
    ap.add_argument('--image', default=None)
    a = ap.parse_args()

    net, info = load_checkpoint(a.ckpt, a.device, a.base)
    print(f"base={info['base']}"
          + (f"  epoch {info.get('epoch')}, step {info.get('gstep'):,}"
             if info.get('gstep') is not None else ''))

    if a.image:
        img = cv2.imread(a.image, cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f'cannot read {a.image}')
        rect, valid, _ = dewarp(net, img, a.device, a.iters, a.beta)
        cv2.imwrite(a.out, rect)
        side = Path(a.out).with_name(Path(a.out).stem + '_sidebyside.png')
        cv2.imwrite(str(side), np.hstack([label(img, 'input'),
                                          label(rect, 'dewarped')]))
        print(f"wrote {a.out}  and  {side}   ({img.shape[1]}x{img.shape[0]})")
        return

    if not a.root:
        raise SystemExit('pass --image <file>, or --root <dataset> for a val preview')
    preview_from_dataset(net, a.root, a.layout_root, a.n, a.device, a.out,
                         split=a.split, seed=a.seed,
                         mask_invalid=a.mask_invalid)

    if a.profile_table or a.full_val:
        from dewarp_dataset import DewarpTrainingSet
        ds = DewarpTrainingSet(a.root, split=a.split, layout_root=a.layout_root,
                               want_clean=False, verbose=False)
        n = len(ds) if a.full_val else a.profile_table
        res, cnt = epe_by_profile(net, ds, n=n, device=a.device, seed=a.seed)
        print(f"\n  EPE by distortion type ({sum(cnt.values()):,} samples"
              f"{', FULL ' + a.split + ' split' if a.full_val else ''})")
        print(f"  {'profile':12s} {'EPE px':>9s} {'n':>6s}")
        for k, v in res.items():
            print(f"  {k:12s} {v:9.3f} {cnt.get(k, sum(cnt.values())):6d}")
        print("  'combined' must be the worst — it stacks cylinder + perspective")
        print("  + wrinkle. flat and cylinder often tie; flat has only ~34 val")
        print("  samples, so do not read a winner between those two.")
        lay = layout_metric(net, ds, n=n, device=a.device, seed=a.seed)
        print(f"\n  LAYOUT CONSISTENCY ({lay['n']:,} samples) — lower is straighter")
        print(f"    overall    {lay['loss']:9.4f}"
              f"   (horizontal {lay['horizontal']:.4f}, vertical {lay['vertical']:.4f})")
        for k, v in lay['by_profile'].items():
            print(f"    {k:12s} {v:9.4f}")
        print("    This is the metric the layout loss optimises. EPE averages")
        print("    all error together; this one responds to BENDING only.")

        Path(str(a.out) + '.epe.json').write_text(json.dumps(
            {'ckpt': str(a.ckpt), 'split': a.split, 'full_val': bool(a.full_val),
             'n': sum(cnt.values()), 'epe_by_profile': res, 'counts': cnt,
             'layout_metric': lay},
            indent=2))
        print(f"  wrote {a.out}.epe.json")


if __name__ == '__main__':
    main()
