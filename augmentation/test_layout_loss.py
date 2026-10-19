"""
test_layout_loss.py
===================
Proves the layout-consistency loss does what the paper will claim it does.

    python augmentation/test_layout_loss.py

The load-bearing test is number 3. It is not enough for the loss to go down
when the flow is right — an L1 flow loss already does that. The claim is that
this loss carries information L1 does not, and the way to show it is a
perturbation that L1 punishes heavily while this loss ignores (a rigid shift of
a block, which leaves it rectangular) alongside one that L1 barely notices
while this loss punishes hard (a bend, which curves the text lines).
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from warp_augment import augment_document, DISTORTION_PROFILES, TRAIN_W, TRAIN_H  # noqa: E402
from layout_loss import layout_consistency_loss, boxes_to_tensor       # noqa: E402

W, H = TRAIN_W, TRAIN_H


def synthetic_page_with_layout():
    """A two-column page plus the clean-space boxes of its blocks."""
    import cv2
    img = np.full((H, W, 3), 252, np.uint8)
    boxes = []
    cv2.rectangle(img, (40, 30), (W - 40, 70), (20, 20, 20), -1)
    boxes.append((40, 30, W - 40, 70))                       # title
    for cx0 in (36, W // 2 + 8):
        cx1 = cx0 + (W // 2 - 44)
        y = 100
        while y < H - 150:
            bh = np.random.randint(90, 160)
            for yy in range(y + 6, y + bh - 6, 13):
                cv2.line(img, (cx0 + 4, yy), (cx1 - 6, yy), (35, 35, 35), 2)
            boxes.append((cx0, y, cx1, y + bh))
            y += bh + 18
    return img, np.array(boxes, np.float32)


def make_sample(seed, profile_idx=2):
    import random
    random.seed(seed)
    np.random.seed(seed)
    img, boxes = synthetic_page_with_layout()
    p = dict(DISTORTION_PROFILES[profile_idx])
    p.pop('name')
    p.update(use_shadow=False, use_background=False)
    _, flow, mask, _ = augment_document(img, **p)
    return flow, mask, boxes


def to_t(flow, mask):
    f = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
    return f, m


def main():
    fails = []
    MODE = os.environ.get('LAYOUT_MODE', 'global')
    print(f'MODE = {MODE}')

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
        if not cond:
            fails.append(name)

    flow, mask, boxes = make_sample(11)
    f_gt, m = to_t(flow, mask)
    bx, bv = boxes_to_tensor([boxes])
    print(f"sample: {W}x{H}, {len(boxes)} layout regions, "
          f"page coverage {float(m.mean())*100:.1f}%")

    print("\n" + "=" * 72)
    print("1. the true flow scores exactly zero")
    print("=" * 72)
    r = layout_consistency_loss(f_gt, f_gt, m, bx, bv, mode=MODE)
    print(f"      loss {float(r['loss']):.3e}  (h {float(r['horizontal']):.3e}, "
          f"v {float(r['vertical']):.3e}), {r['n_bands_used']} bands used")
    check("loss(f_gt, f_gt) == 0", float(r['loss']) < 1e-5)
    # 'global' bands the page (<= 2 * n_bands per sample); 'per_region' bands
    # every region separately, so it has many more.
    floor = 20 if MODE == 'global' else 100
    check("bands were actually populated", r['n_bands_used'] > floor,
          f"{r['n_bands_used']} (floor {floor} for mode '{MODE}')")

    print("\n" + "=" * 72)
    print("2. loss rises monotonically with random flow error")
    print("=" * 72)
    prev = -1.0
    mono = True
    for amp in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        g = torch.Generator().manual_seed(3)
        noise = torch.randn(f_gt.shape, generator=g) * amp
        # low-frequency noise: a plausible model error, not per-pixel static
        noise = torch.nn.functional.avg_pool2d(noise, 33, 1, 16)
        r = layout_consistency_loss(f_gt + noise, f_gt, m, bx, bv, mode=MODE)
        print(f"      amplitude {amp:4.1f} px -> loss {float(r['loss']):8.4f}")
        if float(r['loss']) < prev - 1e-6:
            mono = False
        prev = float(r['loss'])
    check("monotonically increasing", mono)

    print("\n" + "=" * 72)
    print("3. THE CLAIM: this loss sees what an L1 flow loss cannot")
    print("=" * 72)
    print("   Two perturbations, each scaled to the SAME L1 flow error.")
    print("   A rigid shift keeps every block rectangular; a bend curves the")
    print("   text lines. An L1 flow loss cannot tell them apart.\n")

    ys = torch.arange(H).view(1, 1, H, 1).float()
    xs = torch.arange(W).view(1, 1, 1, W).float()

    # (a) rigid translation of the whole field
    shift = torch.zeros_like(f_gt)
    shift[:, 1] = 1.0
    shift[:, 0] = 1.0

    # (b) a bend: vertical displacement varying quadratically across x,
    #     i.e. exactly the residual curvature that leaves text lines bowed
    bend = torch.zeros_like(f_gt)
    bend[:, 1] = ((xs - W / 2) / (W / 2)) ** 2 - 0.33

    print(f"      {'perturbation':<22s}{'L1 flow err':>13s}{'layout loss':>13s}"
          f"{'ratio':>9s}")
    results = {}
    for name, pert in (("rigid shift", shift), ("bend (curved lines)", bend)):
        scale = 1.0 / float(pert.abs().mean())
        p = pert * scale                       # normalise to L1 = 1.0 px
        l1 = float((p.abs() * m).sum() / (m.sum() * 2))
        r = layout_consistency_loss(f_gt + p, f_gt, m, bx, bv, mode=MODE)
        lay = float(r['loss'])
        results[name] = (l1, lay)
        print(f"      {name:<22s}{l1:13.4f}{lay:13.4f}{lay/max(l1,1e-9):9.2f}")

    l1_shift, lay_shift = results["rigid shift"]
    l1_bend, lay_bend = results["bend (curved lines)"]
    check("both perturbations have comparable L1 error",
          abs(l1_shift - l1_bend) / max(l1_shift, 1e-9) < 0.35,
          f"{l1_shift:.4f} vs {l1_bend:.4f}")
    check("rigid shift is nearly free (block stays rectangular)",
          lay_shift < 0.02, f"{lay_shift:.5f}")
    check("bend is punished hard (lines curve)", lay_bend > 0.1,
          f"{lay_bend:.5f}")
    check("the loss separates them by >10x",
          lay_bend > 10 * max(lay_shift, 1e-6),
          f"{lay_bend/max(lay_shift,1e-9):.0f}x")

    print("\n" + "=" * 72)
    print("4. gradients flow")
    print("=" * 72)
    fp = (f_gt + 0.4 * torch.randn(f_gt.shape, generator=torch.Generator().manual_seed(1))
          ).clone().requires_grad_(True)
    r = layout_consistency_loss(fp, f_gt, m, bx, bv, mode=MODE)
    r['loss'].backward()
    g = fp.grad
    check("gradient is finite", bool(torch.isfinite(g).all()))
    check("gradient is non-zero", float(g.abs().max()) > 0,
          f"max |grad| {float(g.abs().max()):.3e}")
    inside_frac = float((g.abs().sum(dim=1) > 0).float().mean())
    print(f"      gradient touches {inside_frac*100:.1f}% of pixels "
          f"(only those inside a region and on the page)")
    check("gradient is confined to the page", inside_frac < float(m.mean()) + 0.02)

    print("\n" + "=" * 72)
    print("5. batching and cost")
    print("=" * 72)
    import time
    fs, ms, bs = [], [], []
    for s in range(4):
        fl, mk, bb = make_sample(100 + s)
        a, b = to_t(fl, mk)
        fs.append(a)
        ms.append(b)
        bs.append(bb)
    fb = torch.cat(fs)
    mb = torch.cat(ms)
    bxb, bvb = boxes_to_tensor(bs)
    pred = fb + 0.3 * torch.randn(fb.shape, generator=torch.Generator().manual_seed(5))
    t0 = time.time()
    r = layout_consistency_loss(pred, fb, mb, bxb, bvb, mode=MODE)
    el = time.time() - t0
    print(f"      batch 4, K={bxb.shape[1]} slots: loss {float(r['loss']):.4f} "
          f"in {el*1000:.0f} ms on CPU")
    check("batched result is finite", bool(torch.isfinite(r['loss'])))

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
