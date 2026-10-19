"""
layout_loss.py
==============
The layout-consistency loss — the paper's novelty, in code.

THE IDEA
--------
PubLayNet labels every page with semantic regions (text / title / list / table
/ figure). A correctly dewarped page has a property no pixel loss states
directly: each of those regions must come out as an AXIS-ALIGNED RECTANGLE.
Text lines inside a text block must be horizontal; the block's left and right
margins must be vertical.

That is a constraint on the predicted geometry that holds regardless of how
accurate the flow is in absolute terms, and it is exactly the property
downstream OCR depends on. (Related work: text-line and layout cues have been
used for dewarping before, e.g. MataDoc and ForCenNet put OCR text-line losses on
the flow and D2Dewarp uses PubLayNet-derived line masks as targets; this loss
differs in using ground-truth 2D region boxes and band variances of the flow
error. See docs/METHOD.md and the paper's related-work section.)

THE FORMULATION — AND WHY IT IS NOT JUST THE FLOW LOSS AGAIN
------------------------------------------------------------
The stored flow `f` maps a warped pixel to its position in the clean page:

    c(p) = p + f(p)

Take the error field between prediction and truth:

    d(p) = c_pred(p) - c_gt(p) = f_pred(p) - f_gt(p)

Now consider one horizontal line inside a text block. Every pixel on it shares
the same `c_gt.y` by definition. The predicted page keeps that line horizontal
if and only if `c_pred.y` is also constant along it — which, since `c_gt.y` is
already constant, holds exactly when **`d.y` is constant along the line**.

So the loss is the VARIANCE of `d.y` within each horizontal band of a region,
and of `d.x` within each vertical band:

    L = mean_bands Var(d_y | band)  +  mean_bands Var(d_x | band)

This is a genuinely different signal from an L1 flow loss:

  * a constant offset of a whole block costs NOTHING — the block is still a
    rectangle, merely translated, and the flow loss already penalises that;
  * what it penalises is the *spatial variation* of the error inside a block,
    which is precisely the residual bending that leaves text lines curved.

With `f_pred == f_gt` the loss is exactly 0, by construction.

NO FLOW INVERSION IS NEEDED
---------------------------
Grouping pixels into bands uses `c_gt`, which comes straight from the stored
flow. The regions are given in clean space, and `c_gt` maps warped pixels into
clean space, so membership is a plain interval test. The expensive inverse
(dataset_format.invert_flow, ~60 remaps) is needed only at inference.

OVERLAPPING REGIONS
-------------------
PubLayNet regions can nest and overlap, so the implementation loops over the
region slot rather than assigning each pixel a single owner. A pixel inside two
regions contributes to both, which is correct — both must come out rectangular.
"""

import torch
import torch.nn.functional as F


def _banded_variance(values, band_id, n_bins, weights, min_count):
    """
    Weighted variance of `values` within each band, summed over bands that
    carry at least `min_count` weight.

    Uses scatter-add of w, w*v and w*v^2, so the whole thing is three
    reductions regardless of how many pixels or bands are involved — no
    per-band gather, and it runs unchanged on GPU.
    """
    w = weights
    wv = w * values
    wv2 = wv * values

    sw = torch.zeros(n_bins, device=values.device, dtype=values.dtype)
    swv = torch.zeros_like(sw)
    swv2 = torch.zeros_like(sw)
    sw.scatter_add_(0, band_id, w)
    swv.scatter_add_(0, band_id, wv)
    swv2.scatter_add_(0, band_id, wv2)

    keep = sw >= min_count
    if not torch.any(keep):
        return values.sum() * 0.0, 0

    sw_k = sw[keep]
    mean = swv[keep] / sw_k
    var = swv2[keep] / sw_k - mean * mean
    var = torch.clamp(var, min=0.0)          # guard float error near zero
    return var.mean(), int(keep.sum())


def layout_consistency_loss(flow_pred, flow_gt, mask, boxes, box_valid,
                            n_bands=24, min_count=16.0, reduce_sqrt=True,
                            mode='global'):
    """
    Args:
        flow_pred: (B, 2, H, W) predicted backward flow, PIXEL units,
                   channel 0 = dx, channel 1 = dy.
        flow_gt:   (B, 2, H, W) ground-truth flow, same convention.
        mask:      (B, 1, H, W) page-validity mask in [0, 1]. Background
                   pixels carry no meaningful flow and must not contribute.
        boxes:     (B, K, 4) region boxes in CLEAN-page pixel coordinates,
                   (x0, y0, x1, y1), zero-padded.
        box_valid: (B, K) bool/float, which box slots are real.
        n_bands:   bands per region per axis. 24 at 448x576 is ~24 px per band
                   vertically, a few text lines' worth.
        min_count: minimum accumulated weight for a band to count, so a band
                   clipped to a couple of pixels cannot dominate.
        reduce_sqrt: return the RMS deviation (pixels) rather than the
                   variance (pixels squared). Keeps the loss on the same scale
                   as an L1 flow loss so the weighting is interpretable.
        mode:      'global' (default) or 'per_region'.

                   'per_region' bands each region separately and loops over
                   region slots, so cost is linear in K — 1.56 s for batch 8,
                   K=30 on CPU.

                   'global' keeps the region mask (only pixels inside a labelled
                   region count) but bands by position on the PAGE, so the whole
                   loss is two scatters regardless of K. This is not a mere
                   approximation: two text blocks side by side at the same clean
                   y must land at the same y after dewarping, so constraining
                   them jointly is if anything the stronger statement. It is
                   also ~12x faster.

    Returns:
        dict with 'loss', 'horizontal', 'vertical', 'n_bands_used'.
    """
    B, _, H, W = flow_pred.shape
    dev = flow_pred.device
    K = boxes.shape[1]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=dev, dtype=flow_pred.dtype),
        torch.arange(W, device=dev, dtype=flow_pred.dtype),
        indexing='ij')

    # Clean-space position of every warped pixel, under the TRUE flow.
    cgt_x = xs.unsqueeze(0) + flow_gt[:, 0]
    cgt_y = ys.unsqueeze(0) + flow_gt[:, 1]

    # Prediction error, in clean space.
    d_x = flow_pred[:, 0] - flow_gt[:, 0]
    d_y = flow_pred[:, 1] - flow_gt[:, 1]

    m = mask[:, 0]
    batch_ix = torch.arange(B, device=dev).view(B, 1, 1).expand(B, H, W)

    # Pixels that sit inside at least one labelled region. Building this is
    # K cheap comparisons with no reduction, which is why 'global' mode can
    # drop the expensive per-region scatter loop.
    any_region = torch.zeros_like(m, dtype=torch.bool)
    for k in range(K):
        valid = box_valid[:, k]
        if not torch.any(valid.bool()):
            continue
        x0 = boxes[:, k, 0].view(B, 1, 1)
        y0 = boxes[:, k, 1].view(B, 1, 1)
        x1 = boxes[:, k, 2].view(B, 1, 1)
        y1 = boxes[:, k, 3].view(B, 1, 1)
        inside = ((cgt_x >= x0) & (cgt_x < x1) &
                  (cgt_y >= y0) & (cgt_y < y1) &
                  valid.bool().view(B, 1, 1))
        any_region |= inside

    if mode == 'global':
        w = m * any_region.to(m.dtype)
        if float(w.sum()) <= 0:
            z = flow_pred.sum() * 0.0
            return {'loss': z, 'horizontal': z, 'vertical': z, 'n_bands_used': 0}

        # Band by position on the page, not within a region.
        row = torch.clamp((cgt_y / H * n_bands).long(), 0, n_bands - 1)
        col = torch.clamp((cgt_x / W * n_bands).long(), 0, n_bands - 1)
        hid = (batch_ix * n_bands + row).reshape(-1)
        vid = (batch_ix * n_bands + col).reshape(-1)
        lh, used_h = _banded_variance(d_y.reshape(-1), hid, B * n_bands,
                                      w.reshape(-1), min_count)
        lv, used_v = _banded_variance(d_x.reshape(-1), vid, B * n_bands,
                                      w.reshape(-1), min_count)
        if reduce_sqrt:
            lh = torch.sqrt(lh + 1e-12)
            lv = torch.sqrt(lv + 1e-12)
        return {'loss': lh + lv, 'horizontal': lh, 'vertical': lv,
                'n_bands_used': used_h + used_v}

    if mode != 'per_region':
        raise ValueError(f"mode must be 'global' or 'per_region', got {mode!r}")

    tot_h = flow_pred.sum() * 0.0
    tot_v = flow_pred.sum() * 0.0
    used_h = used_v = 0
    n_slots = 0

    for k in range(K):
        valid = box_valid[:, k]
        if not torch.any(valid.bool()):
            continue
        n_slots += 1

        x0 = boxes[:, k, 0].view(B, 1, 1)
        y0 = boxes[:, k, 1].view(B, 1, 1)
        x1 = boxes[:, k, 2].view(B, 1, 1)
        y1 = boxes[:, k, 3].view(B, 1, 1)
        bw = torch.clamp(x1 - x0, min=1e-3)
        bh = torch.clamp(y1 - y0, min=1e-3)

        inside = ((cgt_x >= x0) & (cgt_x < x1) &
                  (cgt_y >= y0) & (cgt_y < y1))
        w = m * inside.to(m.dtype) * valid.to(m.dtype).view(B, 1, 1)
        if float(w.sum()) <= 0:
            continue

        row = torch.clamp(((cgt_y - y0) / bh * n_bands).long(), 0, n_bands - 1)
        hid = (batch_ix * n_bands + row).reshape(-1)
        lh, nh = _banded_variance(d_y.reshape(-1), hid, B * n_bands,
                                  w.reshape(-1), min_count)
        tot_h = tot_h + lh
        used_h += nh

        col = torch.clamp(((cgt_x - x0) / bw * n_bands).long(), 0, n_bands - 1)
        vid = (batch_ix * n_bands + col).reshape(-1)
        lv, nv = _banded_variance(d_x.reshape(-1), vid, B * n_bands,
                                  w.reshape(-1), min_count)
        tot_v = tot_v + lv
        used_v += nv

    denom = max(n_slots, 1)
    lh = tot_h / denom
    lv = tot_v / denom
    if reduce_sqrt:
        lh = torch.sqrt(lh + 1e-12)
        lv = torch.sqrt(lv + 1e-12)
    return {'loss': lh + lv, 'horizontal': lh, 'vertical': lv,
            'n_bands_used': used_h + used_v}


def boxes_to_tensor(per_sample_boxes, device='cpu', dtype=torch.float32):
    """
    Pad a list of (K_i, 4) arrays into (B, Kmax, 4) plus a validity mask.
    Region counts vary per page (9.7 on average, up to a few dozen).
    """
    B = len(per_sample_boxes)
    K = max((len(b) for b in per_sample_boxes), default=0)
    K = max(K, 1)
    out = torch.zeros(B, K, 4, device=device, dtype=dtype)
    valid = torch.zeros(B, K, device=device, dtype=dtype)
    for i, b in enumerate(per_sample_boxes):
        if len(b):
            t = torch.as_tensor(b, device=device, dtype=dtype)
            out[i, :len(b)] = t
            valid[i, :len(b)] = 1.0
    return out, valid


def flow_from_normalized(norm_flow, w, h):
    """
    (B, 2, H, W) normalized displacement, as stored, -> pixel units.
    Channel 0 scales with width, channel 1 with height.
    """
    out = norm_flow.clone()
    out[:, 0] = out[:, 0] * (w - 1) / 2.0
    out[:, 1] = out[:, 1] * (h - 1) / 2.0
    return out
