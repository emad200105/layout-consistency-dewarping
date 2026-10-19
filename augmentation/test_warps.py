"""
test_warps.py
=============
Regression tests for the warp pipeline. Run after any change to
warp_augment.py:

    python augmentation/test_warps.py

Each test guards a bug that was actually shipped once:

  1. curl -> 0 is the identity              (coordinate-convention off-by-half)
  2. flow_y is non-zero                     (the old cylinder could not bow lines)
  3. off-centre lines bow, centre stays straight   (correct cylinder physics)
  4. the saved flow reproduces the saved image EXACTLY
                                            (cascaded resampling desynced them)
  5. every flow is float32                  (float64 doubled the dataset size)
  6. detail retention                       (cascaded resampling destroyed 85%)
  7. n_control=4 does not crash the spline
"""

import os
import sys

import cv2
import numpy as np
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warp_augment import (                                    # noqa: E402
    cylindrical_flow, perspective_flow, wrinkle_flow,
    cylindrical_warp, apply_flow, compose_flows, identity_flow,
    augment_document, DISTORTION_PROFILES, FLAT_PROFILE,
    flow_to_valid_mask, make_background, composite_on_background,
    similarity_flow, sample_framing,
    TRAIN_W, TRAIN_H, FLOW_W, FLOW_H,
)

# Test at the real training geometry.
W, H = TRAIN_W, TRAIN_H


def ruled_page(H=H, W=W, spacing=30):
    """A page of horizontal rules — line bow is unmistakable on these."""
    img = np.full((H, W, 3), 255, np.uint8)
    for y in range(spacing, H - spacing // 2, spacing):
        cv2.line(img, (30, y), (W - 30, y), (20, 20, 20), 2)
    for x in range(60, W - 30, 120):
        cv2.line(img, (x, 30), (x, H - 30), (200, 60, 60), 1)
    cv2.rectangle(img, (8, 8), (W - 9, H - 9), (0, 120, 0), 3)
    return img


def sharpness(im):
    return cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def bow_px(flow, row):
    """How much the source row varies across x — i.e. how bowed the line is."""
    ys = row + flow[row, :, 1]
    return float(ys.max() - ys.min())


def main():
    img = ruled_page()
    fails = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
        if not cond:
            fails.append(name)

    print("=" * 72)
    print("1. curl -> 0 is exactly the identity")
    print("=" * 72)
    f0 = cylindrical_flow(H, W, curl=1e-9)
    w0 = apply_flow(img, f0)
    check("zero flow", float(np.abs(f0).max()) < 1e-3,
          f"max|flow| = {np.abs(f0).max():.2e} px")
    check("identical pixels", int(np.abs(w0.astype(int) - img.astype(int)).max()) == 0)

    print()
    print("=" * 72)
    print("2. flow_y is non-zero — the old cylinder set it to exactly 0")
    print("=" * 72)
    for curl in (0.2, 0.45, 0.7):
        fl = cylindrical_flow(H, W, curl=curl, cam_dist=2.5)
        dy = float(np.abs(fl[..., 1]).max())
        check(f"curl={curl}: vertical displacement exists", dy > 1.0,
              f"|dx|max={np.abs(fl[...,0]).max():6.2f}  |dy|max={dy:6.2f} px")

    print()
    print("=" * 72)
    print("3. off-centre lines bow; the centre line stays straight")
    print("=" * 72)
    fl = cylindrical_flow(H, W, curl=0.55, cam_dist=2.2)
    b_top, b_mid, b_bot = bow_px(fl, 40), bow_px(fl, H // 2), bow_px(fl, H - 40)
    print(f"      top bow={b_top:6.2f}  centre bow={b_mid:6.3f}  bottom bow={b_bot:6.2f} px")
    check("centre line straight", b_mid < 0.5)
    check("top line bows", b_top > 5.0)
    check("bow is symmetric about centre", abs(b_top - b_bot) < 1.0)

    print()
    print("=" * 72)
    print("4. saved flow reproduces the saved image EXACTLY inside the mask")
    print("   (outside the mask the pixels are background, by design)")
    print("=" * 72)
    for v, prof in enumerate(DISTORTION_PROFILES):
        p = dict(prof)
        name = p.pop('name')
        p['use_shadow'] = False          # shadow is photometric, not in the flow
        random.seed(500 + v)
        np.random.seed(500 + v)
        _, flow, mask, _ = augment_document(img, **dict(p, use_framing=False))
        random.seed(500 + v)
        np.random.seed(500 + v)
        no_bg, _, _, _ = augment_document(img, **dict(p, use_background=False,
                                                         use_framing=False))
        err = np.abs(apply_flow(img, flow).astype(np.int16)
                     - no_bg.astype(np.int16))[mask > 0.999]
        check(f"v{v} {name}: flow is an exact label on page", int(err.max()) == 0,
              f"mean={err.mean():.6f} max={err.max()}")

    print()
    print("=" * 72)
    print("5. every flow is float32 (float64 doubled dataset size)")
    print("=" * 72)
    for nm, fn, kw in [("cylindrical_flow", cylindrical_flow, dict(curl=0.5)),
                       ("perspective_flow", perspective_flow, dict(max_shift_frac=0.1)),
                       ("wrinkle_flow", wrinkle_flow, dict(n_control=6, max_disp_frac=0.03))]:
        f = fn(H, W, **kw)
        check(f"{nm} float32", f.dtype == np.float32, str(f.dtype))
    comp = compose_flows(cylindrical_flow(H, W, curl=0.4),
                         perspective_flow(H, W, max_shift_frac=0.1))
    check("compose_flows float32", comp.dtype == np.float32, str(comp.dtype))

    print()
    print("=" * 72)
    print("6. detail retention — cascaded resampling used to destroy ~85%")
    print("=" * 72)
    s_clean = sharpness(img)
    for v, prof in enumerate(DISTORTION_PROFILES):
        p = dict(prof)
        name = p.pop('name')
        p['use_shadow'] = False
        random.seed(700 + v)
        np.random.seed(700 + v)
        warped, _, mask, _ = augment_document(img, **dict(p, use_background=False,
                                                            use_framing=False))
        ret = 100 * sharpness(warped) / s_clean
        check(f"v{v} {name}: retains >30% of detail", ret > 30.0, f"{ret:5.1f}%")

    print()
    print("=" * 72)
    print("8. warp presence is DETERMINISTIC — no accidental identity samples")
    print("   (gating once made v0 identity 19.5%, v1 29.6%, and v2 applied")
    print("    all three warps only 37% of the time)")
    print("=" * 72)
    for v, prof in enumerate(DISTORTION_PROFILES):
        p = dict(prof)
        name = p.pop('name')
        p['use_shadow'] = False
        n_ident = 0
        trials = 60
        for t in range(trials):
            random.seed(t)
            np.random.seed(t)
            _, flow, _, _ = augment_document(img, **dict(p, use_framing=False))
            if float(np.abs(flow).max()) < 0.01:
                n_ident += 1
        check(f"v{v} {name}: never identity", n_ident == 0,
              f"{n_ident}/{trials} identity")

    p = dict(FLAT_PROFILE)
    p.pop('name')
    p['use_shadow'] = False
    p['use_framing'] = False          # framing is camera pose, not distortion
    random.seed(1)
    np.random.seed(1)
    w_flat, f_flat, m_flat, _ = augment_document(img, **p)
    check("FLAT_PROFILE has exactly zero flow", float(np.abs(f_flat).max()) == 0.0)
    check("FLAT_PROFILE leaves pixels untouched",
          int(np.abs(w_flat.astype(int) - img.astype(int)).max()) == 0)
    check("FLAT_PROFILE mask is fully page", float(m_flat.min()) == 1.0,
          f"min alpha = {m_flat.min():.3f}")

    print()
    print("=" * 72)
    print("9. training geometry matches PubLayNet's native aspect")
    print("=" * 72)
    native = 612 / 792
    stretch = (TRAIN_W / TRAIN_H) / native
    check(f"{TRAIN_W}x{TRAIN_H} within 2% of native aspect", abs(stretch - 1) < 0.02,
          f"stretch = {stretch:.4f}x  (a 512x512 square would be {1/native:.4f}x)")
    check("both dims divisible by 32", TRAIN_W % 32 == 0 and TRAIN_H % 32 == 0,
          f"{TRAIN_W}/32={TRAIN_W/32}, {TRAIN_H}/32={TRAIN_H/32}")

    print()
    print("=" * 72)
    print("10. validity mask + background compositing")
    print("    (margins used to be BORDER_REPLICATE smear — a texture that")
    print("     exists in no real photograph)")
    print("=" * 72)
    m_id = flow_to_valid_mask(identity_flow(H, W), H, W)
    check("unwarped page -> 100% coverage", float(m_id.mean()) == 1.0,
          f"{m_id.mean()*100:.2f}%")

    fl = cylindrical_flow(H, W, curl=0.6, cam_dist=2.2, fit='contain')
    m_soft = flow_to_valid_mask(fl, H, W, soft=True)
    check("centre of page is fully opaque", float(m_soft[H // 2, W // 2]) == 1.0)
    check("mask is bounded in [0,1]",
          float(m_soft.min()) >= 0.0 and float(m_soft.max()) <= 1.0)

    random.seed(31)
    np.random.seed(31)
    p = dict(DISTORTION_PROFILES[2])
    p.pop('name')
    p['use_shadow'] = False
    comp, flow, mask, params = augment_document(img, **dict(p, use_framing=False))
    bgpx = comp[mask < 0.01]
    check("background replaced the margins", params.get('background') is True,
          f"page coverage = {params['page_coverage']*100:.1f}%")
    check("background is textured, not a constant smear", float(bgpx.std()) > 1.0,
          f"std = {bgpx.std():.2f} over {len(bgpx):,} px")

    # The mask MUST be stored, not re-derived from the downsampled flow:
    # derivation mislabels background as page, which pollutes the flow loss.
    up = cv2.resize(cv2.resize(flow, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA),
                    (W, H), interpolation=cv2.INTER_LINEAR)
    true_m = flow_to_valid_mask(flow, H, W, soft=False) > 0.5
    der_m = flow_to_valid_mask(up, H, W, soft=False) > 0.5
    fp = 100 * float(np.mean(der_m & ~true_m))
    print(f"      deriving the mask from {FLOW_W}x{FLOW_H} flow mislabels "
          f"{fp:.3f}% of background as page")
    print(f"      -> the dataset stores the mask ({FLOW_W*FLOW_H/1e3:.1f} KB/sample, 0.59 GB total)")

    print()
    print("=" * 72)
    print("11. camera framing decorrelates page coverage from warp severity")
    print("    (before framing, coverage identified 'flat' with 100% precision")
    print("     and correlated -0.83 with warp severity — a free shortcut)")
    print("=" * 72)
    f_id = similarity_flow(H, W, 1.0, 0.0, 0.0, 0.0)
    check("scale=1, angle=0 is exactly the identity", float(np.abs(f_id).max()) == 0.0)

    fz = similarity_flow(H, W, 1.25, 0.0, 0.0, 0.0)
    mz = flow_to_valid_mask(fz, H, W)
    check("zoom-in keeps 100% coverage", float(mz.mean()) == 1.0,
          f"{mz.mean()*100:.2f}%")
    fb = similarity_flow(H, W, 0.75, 0.0, 0.0, 0.0)
    mb = flow_to_valid_mask(fb, H, W)
    check("pull-back reveals background", float(mb.mean()) < 0.60,
          f"coverage {mb.mean()*100:.2f}%")

    profs = list(DISTORTION_PROFILES) + [FLAT_PROFILE]
    covs, sevs, names = [], [], []
    for t in range(160):
        q = dict(profs[t % 4])
        nm = q.pop('name')
        q.update(use_shadow=False, use_background=False)
        random.seed(t)
        np.random.seed(t)
        _, fw, _, _ = augment_document(img, **dict(q, use_framing=False))
        random.seed(t)
        np.random.seed(t)
        _, _, mk, _ = augment_document(img, **q)
        covs.append(float(mk.mean()))
        sevs.append(float(np.abs(fw).mean()))
        names.append(nm)
    r = float(np.corrcoef(np.array(covs), np.array(sevs))[0, 1])
    check("|corr(coverage, warp severity)| < 0.35", abs(r) < 0.35,
          f"r = {r:+.3f}  (was -0.825)")

    full = [n for n, c in zip(names, covs) if c > 0.9999]
    p_flat = np.mean([n == "flat" for n in full]) if full else 0.0
    check("coverage==100% no longer means 'flat'", p_flat < 0.90,
          f"P(flat | cov=100%) = {100*p_flat:.1f}%  (was 100.0%)")
    c_comb = [c for n, c in zip(names, covs) if n == "combined"]
    c_flat = [c for n, c in zip(names, covs) if n == "flat"]
    check("coverage ranges overlap across profiles",
          max(c_comb) > min(c_flat),
          f"combined {min(c_comb)*100:.1f}-{max(c_comb)*100:.1f}%, "
          f"flat {min(c_flat)*100:.1f}-{max(c_flat)*100:.1f}%")

    print()
    print("=" * 72)
    print("7. small control grids do not crash the spline")
    print("=" * 72)
    for n in (4, 5, 8):
        try:
            f = wrinkle_flow(H, W, n_control=n, max_disp_frac=0.03)
            check(f"n_control={n}", f.shape == (H, W, 2), f"|max|={np.abs(f).max():.2f}")
        except Exception as e:                                  # noqa: BLE001
            check(f"n_control={n}", False, repr(e))

    print()
    print("=" * 72)
    if fails:
        print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
