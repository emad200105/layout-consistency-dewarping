"""
losses.py
=========
The training objective: masked flow regression, an illumination-robust
photometric term, and the layout-consistency term.

    total = w_flow * flow_l1
          + w_photo * photometric
          + w_layout * layout
          + w_aux * (deep-supervision flow terms)

WHY THE PHOTOMETRIC TERM IS HIGH-PASS FILTERED
----------------------------------------------
The obvious photometric loss is: resample the clean page with the predicted
flow and compare it to the warped input. On this dataset that would be mostly
wrong, because the stored warped page has a lighting gradient baked in
(`add_shadow` multiplies by 0.4-1.0) and the clean page does not. Measured on
the real dataset, the raw difference between a perfectly resampled clean page
and the stored warped page is **45.29 grey levels** — almost all of it shadow,
not geometry. A naive photometric loss would spend its gradient trying to
explain illumination with geometry.

The shadow is low-frequency by construction, so both images are high-pass
filtered (subtract a Gaussian blur) before comparison. That removes the
shading while leaving every text edge — the structure the geometry actually
has to line up. The same trick is what took `verify_dataset.py` check 5 from a
meaningless 45 grey levels to a meaningful 6.4.

WHY EPE IS REPORTED IN FULL-RESOLUTION PIXELS
---------------------------------------------
The flow is predicted at 112 x 144 but the metric the dewarping literature
reports (Local Distortion) is in pixels of the actual page. Scaling the
end-point error to the 448 x 576 page keeps our numbers comparable to
published ones instead of quietly 4x better.
"""

import torch
import torch.nn.functional as F

from model import apply_flow_norm
from layout_loss import layout_consistency_loss


def _gauss_kernel(sigma, device, dtype):
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).view(1, 1, -1)


def high_pass(img, sigma=8.0):
    """Grayscale minus its Gaussian blur — kills smooth shading, keeps edges."""
    g = img.mean(dim=1, keepdim=True)
    k = _gauss_kernel(sigma, img.device, img.dtype)
    r = k.shape[-1] // 2
    blur = F.conv2d(F.pad(g, (r, r, 0, 0), mode='reflect'), k.view(1, 1, 1, -1))
    blur = F.conv2d(F.pad(blur, (0, 0, r, r), mode='reflect'), k.view(1, 1, -1, 1))
    return g - blur


def masked_mean(x, m, eps=1e-6):
    return (x * m).sum() / (m.sum() + eps)


def flow_l1(pred_norm, gt_norm, mask, page_w, page_h):
    """
    Masked L1 on the flow, expressed in full-resolution pixels.

    Masked because the flow outside the page is meaningless — it is whatever
    the warp formula extrapolated to, and asking the network to reproduce that
    would waste capacity on a label with no content.
    """
    if pred_norm.shape[-2:] != gt_norm.shape[-2:]:
        pred_norm = F.interpolate(pred_norm, size=gt_norm.shape[-2:],
                                  mode='bilinear', align_corners=True)
    d = (pred_norm - gt_norm).abs()
    d = torch.stack([d[:, 0] * (page_w - 1) / 2.0,
                     d[:, 1] * (page_h - 1) / 2.0], dim=1)
    return masked_mean(d, mask.expand_as(d))


def endpoint_error(pred_norm, gt_norm, mask, page_w, page_h):
    """Mean end-point error in full-resolution pixels — the Local Distortion proxy."""
    if pred_norm.shape[-2:] != gt_norm.shape[-2:]:
        pred_norm = F.interpolate(pred_norm, size=gt_norm.shape[-2:],
                                  mode='bilinear', align_corners=True)
    dx = (pred_norm[:, 0] - gt_norm[:, 0]) * (page_w - 1) / 2.0
    dy = (pred_norm[:, 1] - gt_norm[:, 1]) * (page_h - 1) / 2.0
    epe = torch.sqrt(dx * dx + dy * dy + 1e-12)
    return masked_mean(epe, mask[:, 0])


def photometric(pred_norm, clean, warped, mask, sigma=8.0):
    """
    Resample the clean page with the predicted flow; compare to the warped
    input. High-pass filtered so illumination cannot masquerade as geometry.
    """
    recon = apply_flow_norm(clean, pred_norm)
    a = high_pass(recon, sigma)
    b = high_pass(warped, sigma)
    m = F.interpolate(mask, size=a.shape[-2:], mode='bilinear',
                      align_corners=False)
    return masked_mean((a - b).abs(), m)


class DewarpLoss(torch.nn.Module):
    def __init__(self, page_w=448, page_h=576,
                 w_flow=1.0, w_photo=0.5, w_layout=0.2, w_aux=0.2,
                 layout_bands=24, layout_mode='global', photo_sigma=8.0):
        super().__init__()
        self.pw, self.ph = page_w, page_h
        self.w_flow = w_flow
        self.w_photo = w_photo
        self.w_layout = w_layout
        self.w_aux = w_aux
        self.layout_bands = layout_bands
        self.layout_mode = layout_mode
        self.photo_sigma = photo_sigma

    def forward(self, out, batch, want_photo=True):
        gt = batch['flow_norm']
        mask = batch['mask']
        pred = out['flow']
        logs = {}

        l_flow = flow_l1(pred, gt, mask, self.pw, self.ph)
        total = self.w_flow * l_flow
        logs['flow_l1'] = float(l_flow.detach())

        if self.w_aux > 0 and 'aux' in out:
            l_aux = sum(flow_l1(a, gt, mask, self.pw, self.ph) for a in out['aux'])
            l_aux = l_aux / max(len(out['aux']), 1)
            total = total + self.w_aux * l_aux
            logs['flow_aux'] = float(l_aux.detach())

        if self.w_photo > 0 and want_photo and 'clean' in batch:
            l_ph = photometric(pred, batch['clean'], batch['warped'], mask,
                               self.photo_sigma)
            total = total + self.w_photo * l_ph
            logs['photometric'] = float(l_ph.detach())

        if self.w_layout > 0:
            fw, fh = gt.shape[-1], gt.shape[-2]
            p_px = torch.stack([pred[:, 0] * (fw - 1) / 2.0,
                                pred[:, 1] * (fh - 1) / 2.0], dim=1)
            g_px = torch.stack([gt[:, 0] * (fw - 1) / 2.0,
                                gt[:, 1] * (fh - 1) / 2.0], dim=1)
            r = layout_consistency_loss(p_px, g_px, mask,
                                        batch['boxes'], batch['box_valid'],
                                        n_bands=self.layout_bands,
                                        mode=self.layout_mode)
            total = total + self.w_layout * r['loss']
            logs['layout'] = float(r['loss'].detach())
            logs['layout_h'] = float(r['horizontal'].detach())
            logs['layout_v'] = float(r['vertical'].detach())

        with torch.no_grad():
            logs['epe_px'] = float(endpoint_error(pred, gt, mask, self.pw, self.ph))
        logs['total'] = float(total.detach())
        return total, logs
