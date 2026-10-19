"""
warp_augment.py
===============
Synthetic Augmentation Pipeline for Document Dewarping.

For each clean PubLayNet image, this script applies one or more of:
  1. Cylindrical / book-spine warp (3-D cylinder + pinhole camera;
     bows text lines, which the old horizontal-only version could not do)
  2. Perspective distortion (simulates a tilted camera)
  3. Wrinkle / crease simulation (local random thin-plate-spline deformation)
  4. Shadow gradient overlay (simulates lighting variation on a curved page)

Outputs (per image):
  - images_clean/<name>.png       -> original (ground truth)
  - images_warped/<name>.png      -> synthetically distorted
  - flow_fields/<name>.npy        -> GT backward flow field (H, W, 2) in pixel units
                                     flow[y,x] = (dx, dy) such that
                                     clean_pixel(x+dx, y+dy) == warped_pixel(x, y)

Author: Document Dewarping Project
Date:   2026-09-12
"""

import cv2
import numpy as np
import os
import random
import argparse
from pathlib import Path
from scipy.interpolate import RectBivariateSpline

# ─────────────────────────────────────────────
# TARGET GEOMETRY
#
# PubLayNet pages are uniformly 612 x 792 (US Letter @ 72 dpi, verified over a
# random sample). 448 x 576 preserves that 0.7727 aspect ratio to within 0.7%,
# is 1.6% FEWER pixels than a 512 x 512 square, and both dimensions are
# divisible by 32 so a 5-stage U-Net downsamples cleanly. A square resize would
# have stretched every page horizontally by 1.294x for no benefit.
#
# Flow is stored at 1/4 resolution: 112 x 144.
#
# 1/8 (56 x 72) was the original plan on the argument that these warps have no
# high-frequency content — the wrinkle control grid is at most 8 x 8 and the
# cylinder and homography are analytic. Measured, that argument holds on
# average but not in the tail:
#
#     flow grid     mean     p95     p99      max   (px, inside the page mask)
#      56 x  72    0.052   0.201   1.099    6.122
#      84 x 108    0.029   0.072   0.653    4.016
#     112 x 144    0.018   0.038   0.373    2.713
#
# The tail is NOT at the page boundary — eroding the loss mask by 8 px moves
# p99 only 1.099 -> 0.806 — so it is intrinsic to downsampling a strongly
# curved field, and the grid is the only lever. 1/4 costs +8.8 GB on a single
# Kaggle upload and takes the label noise floor to 0.37 px p99, which matters
# because "exact analytical flow ground truth" is one of the paper's claims.
# /4 is also an exact decoder stride, unlike 84's 5.33.
# ─────────────────────────────────────────────

TRAIN_W, TRAIN_H = 448, 576
FLOW_W, FLOW_H = TRAIN_W // 4, TRAIN_H // 4      # 112 x 144

# ─────────────────────────────────────────────
# WARP FUNCTIONS
# ─────────────────────────────────────────────

def apply_flow(image: np.ndarray, flow: np.ndarray,
               border=cv2.BORDER_REPLICATE) -> np.ndarray:
    """
    Resample `image` with a backward flow field — the ONLY place resampling
    should happen.

    flow[y,x] = (dx, dy) means: warped[y,x] = image[y+dy, x+dx]
    """
    H, W = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    return cv2.remap(image,
                     (gx + flow[..., 0]).astype(np.float32),
                     (gy + flow[..., 1]).astype(np.float32),
                     interpolation=cv2.INTER_LINEAR, borderMode=border)


def identity_flow(H: int, W: int) -> np.ndarray:
    return np.zeros((H, W, 2), dtype=np.float32)


# ─────────────────────────────────────────────
# FLOW GENERATORS
#
# Each returns ONLY a flow field, computed from geometry alone — it never
# touches pixels. This is what lets augment_document() compose every warp
# first and resample exactly once (see the note on cascaded resampling in
# augment_document).
# ─────────────────────────────────────────────

def cylindrical_flow(H: int, W: int,
                     curl: float = 0.45,
                     cam_dist: float = 2.5,
                     spine_pos: float = 0.5,
                     horizontal: bool = False,
                     fit: str = "contain",
                     n_tab: int = 4096) -> np.ndarray:
    """
    Backward flow for a page bent around a cylinder and photographed.

    THE MODEL
    ---------
    A page point at arc-length u from the spine and height v sits in 3-D at

        X = R sin(u/R),    Y = v,    Z = R (1 - cos(u/R))

    (Z is depth: the page recedes from the camera away from the spine.)
    A pinhole camera at distance d gives

        x_img = f X / (d + Z)          y_img = f Y / (d + Z)

    y_img is divided by (d + Z), and Z varies with the HORIZONTAL position u.
    That coupling is the whole point: a horizontal text line at constant v
    projects to a y_img that changes across the page, so the line bows toward
    the image centre-line as the page recedes at the edges. Lines above centre
    bow down, lines below bow up, and the centre line stays straight — which
    is exactly what a photograph of an open book looks like.

    The previous implementation set y_src = y identically, making flow_y zero
    everywhere. It was a pure horizontal resample and could not bow a line at
    all, so it never produced the single most characteristic cue of spine
    curvature.

    INVERSION
    ---------
    x_img is a function of u alone and monotonic for |u/R| < pi/2, so it is
    inverted by tabulating x_img(u) densely and calling np.interp. Then
    Z = R(1 - cos(u/R)) per column, and v = y_img (d + Z) / f in closed form.

    Args:
        H, W:       Output size.
        curl:       Half-angle subtended by the page at the cylinder axis, in
                    radians. 0 = flat. Realistic book pages: 0.15 - 0.85.
        cam_dist:   Camera distance as a multiple of page width. Smaller =
                    stronger perspective, more pronounced bow. Realistic 1.6-4.
        spine_pos:  Bend-axis location as a fraction of width. 0.5 = open book
                    with a centred spine; 0 or 1 = single page curling from one
                    edge.
        horizontal: Spine runs horizontally instead (page curling at top/bottom).
        fit:        'contain' keeps the ENTIRE page inside the frame — no page
                    content is lost, which is what a dewarping GT needs.
                    'width' fills the frame width but pushes the page corners
                    out of frame, destroying that content permanently.
        n_tab:      Inversion lookup table size.

    Returns:
        flow: (H, W, 2) float32 backward flow in pixel units.
    """
    if horizontal:
        # Solve the vertical-spine problem transposed, then transpose back and
        # swap the flow channels.
        f_t = cylindrical_flow(W, H, curl=curl, cam_dist=cam_dist,
                               spine_pos=spine_pos, horizontal=False,
                               fit=fit, n_tab=n_tab)
        return np.ascontiguousarray(np.swapaxes(f_t, 0, 1)[..., ::-1])

    if abs(curl) < 1e-6:
        return identity_flow(H, W)

    # Arc-length extent, in source-pixel units so curl -> 0 is exactly the
    # identity on the pixel grid 0 .. W-1.
    u_lo = -spine_pos * (W - 1)
    u_hi = (1.0 - spine_pos) * (W - 1)

    # R from the wider side, keeping |u/R| < pi/2 on both.
    R = max(abs(u_lo), abs(u_hi)) / max(curl, 1e-6)

    d = cam_dist * W
    f = d                      # f == d makes curl -> 0 the identity

    u_tab = np.linspace(u_lo, u_hi, n_tab)
    ang = u_tab / R
    Z_tab = R * (1.0 - np.cos(ang))
    x_proj = f * (R * np.sin(ang)) / (d + Z_tab)

    span = x_proj[-1] - x_proj[0]
    if span <= 1e-9:
        return identity_flow(H, W)
    s_width = (W - 1) / span

    if fit == "contain":
        # The projection is vertically tallest at the spine, where Z = 0 and
        # |y_img| reaches (H-1)/2 (because f == d). s must not exceed 1 or the
        # page top/bottom would be pushed out of frame.
        s = min(s_width, 1.0)
    elif fit == "width":
        s = s_width
    else:
        raise ValueError(f"fit must be 'contain' or 'width', got {fit!r}")

    x_out = np.arange(W, dtype=np.float64)
    x_off = (W - 1) / 2.0 - s * (x_proj[0] + x_proj[-1]) / 2.0
    u_of_x = np.interp((x_out - x_off) / s, x_proj, u_tab)        # (W,)
    Z_of_x = R * (1.0 - np.cos(u_of_x / R))                       # (W,)

    cy = (H - 1) / 2.0
    y_img = (np.arange(H, dtype=np.float64) - cy) / s             # (H,)
    v = y_img[:, None] * (d + Z_of_x)[None, :] / f                # (H, W)

    flow_x = np.broadcast_to((u_of_x - u_lo) - x_out, (H, W))
    flow_y = v - (np.arange(H, dtype=np.float64) - cy)[:, None]

    return np.stack([flow_x, flow_y], axis=-1).astype(np.float32)


def perspective_flow(H: int, W: int,
                     max_shift_frac: float = 0.12) -> np.ndarray:
    """
    Backward flow for a random perspective distortion (camera held at an angle).

    Each corner is displaced by a random fraction of the image dimension, and
    the backward homography (warped -> source) is evaluated on the pixel grid.
    """
    margin_w = int(W * max_shift_frac)
    margin_h = int(H * max_shift_frac)

    src_pts = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])

    def rs(m):
        return random.randint(-m, m) if m > 0 else 0

    dst_pts = np.float32([
        [rs(margin_w),             rs(margin_h)],
        [W - 1 + rs(margin_w),     rs(margin_h)],
        [W - 1 + rs(margin_w),     H - 1 + rs(margin_h)],
        [rs(margin_w),             H - 1 + rs(margin_h)],
    ])

    # dst -> src, i.e. the backward direction we need.
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    # float32 throughout — the old code built this with np.ones(H*W), which is
    # float64 and silently promoted every composed flow to float64, doubling
    # the stored dataset size.
    coords = np.stack([gx.ravel(), gy.ravel(),
                       np.ones(H * W, dtype=np.float32)], axis=0)
    sc = M.astype(np.float32) @ coords
    sc = sc / sc[2:3, :]

    flow_x = sc[0].reshape(H, W) - gx
    flow_y = sc[1].reshape(H, W) - gy
    return np.stack([flow_x, flow_y], axis=-1).astype(np.float32)


def wrinkle_flow(H: int, W: int,
                 n_control: int = 6,
                 max_disp_frac: float = 0.03) -> np.ndarray:
    """
    Backward flow for local wrinkles / creases.

    A coarse n_control x n_control grid of control points is displaced randomly
    and interpolated to full resolution with a bicubic spline. Because the
    control grid is at most 8 x 8, the resulting field has no high-frequency
    content — which is what makes a 64 x 64 stored flow lossless enough.
    """
    max_dx = W * max_disp_frac
    max_dy = H * max_disp_frac

    cp_xs = np.linspace(0, W - 1, n_control)
    cp_ys = np.linspace(0, H - 1, n_control)

    disp_x = np.random.uniform(-max_dx, max_dx, (n_control, n_control))
    disp_y = np.random.uniform(-max_dy, max_dy, (n_control, n_control))

    k = min(3, n_control - 1)
    sx = RectBivariateSpline(cp_ys, cp_xs, disp_x, kx=k, ky=k)
    sy = RectBivariateSpline(cp_ys, cp_xs, disp_y, kx=k, ky=k)

    ys = np.arange(H, dtype=np.float64)
    xs = np.arange(W, dtype=np.float64)
    flow_x = sx(ys, xs).astype(np.float32)
    flow_y = sy(ys, xs).astype(np.float32)
    return np.stack([flow_x, flow_y], axis=-1)


def similarity_flow(H: int, W: int,
                    scale: float = 1.0,
                    angle_deg: float = 0.0,
                    tx: float = 0.0,
                    ty: float = 0.0) -> np.ndarray:
    """
    Backward flow for camera framing: zoom, roll, and off-centre aim.

    WHY THIS EXISTS — it breaks a shortcut, it is not cosmetic
    ----------------------------------------------------------
    Without it, how much background is visible is a direct readout of how
    badly the page is warped, because the only thing that ever pushed page
    content off-frame was the warp itself. Measured over 240 samples:

        corr(page coverage, mean |flow|) = -0.791
        P(profile == flat | coverage == 100%) = 100.0%

    A convolutional network finds that instantly. It would learn "does
    background reach the frame edge? no -> predict zero flow", and read warp
    severity off the size of the background margin rather than off the page
    content — the shortcut works perfectly on this dataset and not at all on
    a real photograph.

    Real captures frame a page at an arbitrary distance and angle, quite
    independently of whether it happens to be bent. Composing a random
    similarity transform restores that independence: a flat page can be shot
    from far back with plenty of desk visible, and a badly bent page can be
    cropped tight so it fills the frame.

    It is also a legitimate part of the task. The target is the rectified page
    filling the frame, so a zoomed or rolled capture genuinely has to be
    scaled and rotated back — this is signal, not noise.

    Args:
        H, W:      Output size.
        scale:     >1 zooms in (page fills the frame, corners clipped);
                   <1 pulls back (page inset, background visible).
        angle_deg: In-plane roll. Real photos are rarely perfectly square on.
        tx, ty:    Off-centre aim, in pixels.

    Returns:
        flow: (H, W, 2) float32 backward flow in pixel units.
    """
    if scale == 1.0 and angle_deg == 0.0 and tx == 0.0 and ty == 0.0:
        return identity_flow(H, W)

    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    th = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(th), np.sin(th)

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))

    # Invert  out = s * R * (src - c) + c + t
    u = (gx - cx - tx) / scale
    v = (gy - cy - ty) / scale
    x_src = (u * cos_t + v * sin_t) + cx
    y_src = (-u * sin_t + v * cos_t) + cy

    return np.stack([x_src - gx, y_src - gy], axis=-1).astype(np.float32)


def sample_framing(H: int, W: int):
    """
    Draw a plausible camera framing, in three regimes.

    The regimes are chosen so that page coverage is NOT a proxy for warp
    severity (see similarity_flow). Crucially, `zoom` lets a heavily warped
    page still fill the frame, which is what stops "coverage == 100%" from
    being a perfect flat-page detector.

    Returns:
        (scale, angle_deg, tx, ty, regime_name)
    """
    r = random.random()

    if r < 0.25:
        # Tight crop: page fills the frame, corners clipped. Coverage ~100%
        # even for a badly bent page.
        scale = random.uniform(1.02, 1.30)
        angle = random.uniform(-5.0, 5.0)
        regime = 'zoom'
    elif r < 0.50:
        # Exact framing, square on — a scan or a careful photograph.
        scale, angle, regime = 1.0, 0.0, 'exact'
    else:
        # Pulled back: page inset, surface visible around it.
        scale = random.uniform(0.70, 0.98)
        angle = random.uniform(-7.0, 7.0)
        regime = 'back'

    # Keep the page roughly in frame: translate only within the slack that the
    # scale leaves, so the subject does not wander off the edge.
    slack = max(0.0, 1.0 - min(scale, 1.0)) * 0.45
    tx = random.uniform(-slack, slack) * W
    ty = random.uniform(-slack, slack) * H
    return scale, angle, tx, ty, regime


# ─────────────────────────────────────────────
# VALIDITY MASK + BACKGROUND COMPOSITING
# ─────────────────────────────────────────────

def flow_to_valid_mask(flow: np.ndarray,
                       src_h: int = None,
                       src_w: int = None,
                       soft: bool = True) -> np.ndarray:
    """
    Which output pixels actually show page, and which show background.

    A backward flow says where each output pixel samples FROM in the source
    page. If that source coordinate falls outside the page rectangle, the
    output pixel is not page at all — it is background, and the flow value
    there is meaningless.

    This matters twice:
      * at generation time, to composite the page onto a background instead of
        smearing edge pixels outward with BORDER_REPLICATE;
      * at training time, because the flow loss MUST be masked here. Without
        the mask the model is optimised to predict specific garbage values in
        the margins, which is both unlearnable and a waste of capacity.

    The mask is derived from the flow rather than stored, so train and eval
    can never disagree, and it costs nothing on disk. The page boundary is a
    smooth curve, so deriving it from a downsampled flow is accurate to within
    a pixel or two — fine for masking a loss.

    Args:
        flow:   (H, W, 2) backward flow in pixel units.
        src_h:  Source page height. Defaults to the flow's own height, which
                is correct whenever the warp was generated at output size.
        src_w:  Source page width. Same default.
        soft:   Anti-alias the boundary over one pixel. Use True for
                compositing; either works for a loss mask.

    Returns:
        mask: (H, W) float32 in [0, 1]. 1 = page, 0 = background.
    """
    H, W = flow.shape[:2]
    if src_h is None:
        src_h = H
    if src_w is None:
        src_w = W

    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    x_src = gx + flow[..., 0]
    y_src = gy + flow[..., 1]

    # Signed distance (in pixels) from the source coordinate to the nearest
    # page edge. Positive inside.
    d = np.minimum.reduce([x_src,
                           (src_w - 1) - x_src,
                           y_src,
                           (src_h - 1) - y_src]).astype(np.float32)

    if soft:
        # A source coordinate exactly ON the page edge (d == 0) is fully valid:
        # it samples page pixel 0 exactly. The ramp therefore lives entirely
        # OUTSIDE the page — d = -0.5 is half off-page, d = -1 is fully off.
        # Offsetting by 0.5 instead would have made the whole 1-pixel frame
        # border half-transparent even for an unwarped page.
        return np.clip(d + 1.0, 0.0, 1.0)
    return (d >= 0.0).astype(np.float32)


def make_background(H: int, W: int) -> np.ndarray:
    """
    Synthesize a plausible surface for the page to sit on.

    Real document photos are taken on something — a desk, a table, a scanner
    bed, a sheet of paper. With BORDER_REPLICATE the margins were filled with
    smeared page pixels instead, a texture that exists nowhere in reality, so
    the model had no honest signal for where the page ends. Page-boundary
    localisation is most of what rectifying the corners depends on.

    Deliberately includes near-white backgrounds. Those are the hard case —
    page and surface have nearly the same tone, so the boundary must be found
    from shading and texture rather than brightness. Omitting them would leave
    the model helpless on exactly the scans people actually make.
    """
    kind = random.random()

    if kind < 0.22:
        # Near-white: scanner bed, white desk, paper on paper. Hard case.
        base = np.array([random.uniform(198, 246)] * 3, np.float32)
        base += np.random.uniform(-6, 6, 3).astype(np.float32)
    elif kind < 0.45:
        # Wood / warm desk tone (BGR).
        base = np.array([random.uniform(25, 95),
                         random.uniform(45, 125),
                         random.uniform(70, 165)], np.float32)
    elif kind < 0.70:
        # Neutral grey / dark surface.
        g = random.uniform(20, 130)
        base = np.array([g, g, g], np.float32) + np.random.uniform(-8, 8, 3).astype(np.float32)
    else:
        # Arbitrary cool / coloured surface (cloth, mat, painted table).
        base = np.random.uniform(20, 180, 3).astype(np.float32)

    bg = np.empty((H, W, 3), np.float32)
    bg[:] = base

    # Low-frequency luminance variation — a surface is never perfectly flat-lit.
    coarse = np.random.uniform(-1.0, 1.0, (max(2, H // 64), max(2, W // 64))).astype(np.float32)
    coarse = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_CUBIC)
    bg *= (1.0 + random.uniform(0.04, 0.18) * coarse)[..., None]

    # A directional gradient, as from a window or a single lamp.
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xs = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    ang = random.uniform(0, 2 * np.pi)
    ramp = np.cos(ang) * xs + np.sin(ang) * ys
    ramp = (ramp - ramp.min()) / max(ramp.max() - ramp.min(), 1e-6)
    bg *= (1.0 - random.uniform(0.05, 0.30) * ramp)[..., None]

    # Fine grain.
    bg += np.random.normal(0, random.uniform(1.5, 6.0), (H, W, 3)).astype(np.float32)

    return np.clip(bg, 0, 255).astype(np.uint8)


def composite_on_background(page: np.ndarray,
                            alpha: np.ndarray,
                            background: np.ndarray = None,
                            drop_shadow: bool = True) -> np.ndarray:
    """
    Lay the warped page over a background, with a soft contact shadow.

    A page lying on a surface occludes light and casts a shadow just outside
    its own edge. That shadow is one of the strongest cues for where the page
    boundary is, which is precisely what the network needs in order to
    rectify the corners.

    Args:
        page:        (H, W, 3) warped page, already resampled.
        alpha:       (H, W) float32 in [0, 1] from flow_to_valid_mask.
        background:  (H, W, 3) surface; synthesized if None.
        drop_shadow: Darken the surface just outside the page edge.

    Returns:
        (H, W, 3) uint8 composite.
    """
    H, W = page.shape[:2]
    if background is None:
        background = make_background(H, W)

    bg = background.astype(np.float32)

    if drop_shadow:
        sigma = random.uniform(2.0, 9.0)
        strength = random.uniform(0.18, 0.55)
        # Offset the page silhouette slightly, as a real light source would.
        dx = int(round(random.uniform(-0.5, 0.5) * sigma * 2))
        dy = int(round(random.uniform(-0.2, 1.0) * sigma * 2))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        sil = cv2.warpAffine(alpha, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sil = cv2.GaussianBlur(sil, (0, 0), sigma)
        # Only darken where the page is NOT, so the page itself is untouched.
        shade = np.clip(sil - alpha, 0.0, 1.0)
        bg *= (1.0 - strength * shade)[..., None]

    a = alpha[..., None]
    out = page.astype(np.float32) * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# IMAGE-LEVEL WRAPPERS
#
# Convenience only (used by preview_augmentation.py). The batch pipeline does
# NOT call these — it composes flows and resamples once.
# ─────────────────────────────────────────────

def cylindrical_warp(image, **kw):
    H, W = image.shape[:2]
    flow = cylindrical_flow(H, W, **kw)
    return apply_flow(image, flow), flow


def perspective_warp(image, max_shift_frac: float = 0.12):
    H, W = image.shape[:2]
    flow = perspective_flow(H, W, max_shift_frac=max_shift_frac)
    return apply_flow(image, flow), flow


def wrinkle_warp(image, n_control: int = 6, max_disp_frac: float = 0.03):
    H, W = image.shape[:2]
    flow = wrinkle_flow(H, W, n_control=n_control, max_disp_frac=max_disp_frac)
    return apply_flow(image, flow), flow


def add_shadow(image: np.ndarray,
               min_alpha: float = 0.35,
               max_alpha: float = 0.7):
    """
    Add a realistic gradient shadow to simulate page curvature lighting.
    
    Creates a smooth gradient that darkens one side / a strip of the image.
    Multiple shadow types are randomly chosen.
    
    Args:
        image:      Input BGR image
        min_alpha:  Minimum shadow darkness (0=black, 1=no shadow)
        max_alpha:  Maximum shadow darkness
    
    Returns:
        shadowed:   Image with shadow overlay (same shape as input)
    """
    H, W = image.shape[:2]
    shadow_type = random.choice(['left', 'right', 'center_dark', 'diagonal'])

    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    grid_x, grid_y = np.meshgrid(xs, ys)

    if shadow_type == 'left':
        # Darker on left, fades right
        alpha_map = np.clip(grid_x, 0, 1)
    elif shadow_type == 'right':
        alpha_map = np.clip(1 - grid_x, 0, 1)
    elif shadow_type == 'center_dark':
        # Page bent in middle — dark at center
        alpha_map = 1.0 - np.exp(-8 * (grid_x - 0.5) ** 2)
    else:  # diagonal
        alpha_map = np.clip((grid_x + grid_y) / 2.0, 0, 1)

    # Rescale to [min_alpha, 1.0]
    alpha_map = min_alpha + (1.0 - min_alpha) * alpha_map

    # Apply: darken the image by multiplying with alpha map
    alpha_map = alpha_map[:, :, np.newaxis]  # (H, W, 1)
    shadowed = (image.astype(np.float32) * alpha_map).astype(np.uint8)
    return shadowed


# ─────────────────────────────────────────────
# FLOW COMBINATION UTILITY
# ─────────────────────────────────────────────

def compose_flows(flow1: np.ndarray, flow2: np.ndarray) -> np.ndarray:
    """
    Compose two backward flow fields: first apply flow1, then flow2.
    
    Given:
      flow1: (H, W, 2) — first warp
      flow2: (H, W, 2) — second warp applied on top of first
    
    Returns:
      composed: (H, W, 2) float32 — combined backward flow, equal to
                f2(x) + f1(x + f2(x))

    Neither input depends on image content, so composing flows generated
    independently for the same (H, W) is exactly equivalent to generating the
    second warp from the first warp's output — which is what lets the pipeline
    resample only once.
    """
    H, W = flow1.shape[:2]
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # The combined source position: where does the output pixel come from in the original
    src_x1 = grid_x + flow2[..., 0]
    src_y1 = grid_y + flow2[..., 1]

    # Look up flow1 at those positions (bilinear interpolation)
    src_x1_c = np.clip(src_x1, 0, W - 1).astype(np.float32)
    src_y1_c = np.clip(src_y1, 0, H - 1).astype(np.float32)
    add_dx = cv2.remap(flow1[..., 0], src_x1_c, src_y1_c,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    add_dy = cv2.remap(flow1[..., 1], src_x1_c, src_y1_c,
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    composed = np.stack([flow2[..., 0] + add_dx,
                         flow2[..., 1] + add_dy], axis=-1)
    return composed.astype(np.float32)


# ─────────────────────────────────────────────
# MAIN AUGMENTATION FUNCTION
# ─────────────────────────────────────────────

def augment_document(image: np.ndarray,
                     use_cylinder: bool = True,
                     use_perspective: bool = True,
                     use_wrinkle: bool = True,
                     use_shadow: bool = True,
                     shadow_prob: float = 0.75,
                     use_background: bool = True,
                     use_framing: bool = True,
                     page_scale: float = None,
                     page_angle: float = None,
                     cylinder_curl: float = None,
                     cylinder_cam_dist: float = None,
                     cylinder_spine_pos: float = None,
                     cylinder_horizontal: bool = None,
                     perspective_shift: float = None,
                     wrinkle_n_control: int = None,
                     wrinkle_max_disp: float = None):
    """
    Apply a randomized combination of warps and shadow to a clean page.

    COMPOSE FIRST, RESAMPLE ONCE
    ----------------------------
    Every warp contributes only a flow field. All flows are composed, and the
    clean image is resampled exactly once at the end.

    The previous version resampled the image after each warp and composed the
    flows alongside. That was wrong in two ways, both of which hit profile v2
    (cylinder + perspective + wrinkle) hardest — the profile meant to be the
    most realistic:

      1. Cascaded bilinear interpolation destroys detail. Measured on a ruled
         test page, three sequential remaps retained 15.2% of the clean page's
         Laplacian variance; one composed remap retained 43.5%. The model was
         being trained to dewarp images blurrier than any real photograph.

      2. The stored flow no longer matched the stored image. Resampling the
         clean image with the saved flow reproduced the sequentially-warped
         image only to a mean absolute error of ~4 grey levels, with interior
         (not merely border) disagreement up to 235. The label was inconsistent
         with the input it labelled.

    Composing first fixes both exactly, and is faster — one remap, not three.

    WARP PRESENCE IS DETERMINISTIC
    ------------------------------
    Each enabled warp is ALWAYS applied; only its magnitude is random. An
    earlier version gated every warp behind `random.random() > threshold`,
    which made the named profiles lie about themselves. Measured over 20,000
    draws per profile: v0 'cylinder' produced no warp at all 19.5% of the time,
    v1 'perspective' 29.6%, and v2 'combined' applied all three warps only
    37.0% of the time. Dataset-wide that was a 17.1% geometric identity rate —
    about 24,600 of 143,874 pairs with zero warp, none of it intentional.

    Flat pages are still needed (a model that cannot leave an undistorted page
    alone is useless), but they are now produced deliberately via FLAT_PROFILE
    and a counted `flat_frac`, not as a side effect of three coin flips.

    `shadow_prob` stays stochastic on purpose — real captures are sometimes
    evenly lit, and a flat page with a strong lighting gradient teaches the
    model not to mistake shading for geometry.

    Returns:
        warped: Final image (H, W, 3) — page composited on a background
        flow:   Composed backward flow (H, W, 2) float32, pixel units
        mask:   (H, W) float32 in [0, 1]; 1 = page, 0 = background. The flow
                loss MUST be masked with this — flow values outside the page
                are meaningless.
        params: Every parameter used, for reproducibility
    """
    H, W = image.shape[:2]
    params = {}
    flow = identity_flow(H, W)

    # 1. Cylindrical / book-spine curvature
    if use_cylinder:
        curl = cylinder_curl if cylinder_curl is not None \
            else random.uniform(0.15, 0.85)
        cam_dist = cylinder_cam_dist if cylinder_cam_dist is not None \
            else random.uniform(1.6, 4.0)
        if cylinder_spine_pos is not None:
            spine_pos = cylinder_spine_pos
        elif random.random() < 0.45:
            # Single page curling away from one edge — the common case.
            spine_pos = random.uniform(0.0, 0.15) if random.random() < 0.5 \
                else random.uniform(0.85, 1.0)
        else:
            # Open book, spine near the middle.
            spine_pos = random.uniform(0.30, 0.70)
        horiz = cylinder_horizontal if cylinder_horizontal is not None \
            else (random.random() < 0.25)

        params.update(cylinder_curl=curl, cylinder_cam_dist=cam_dist,
                      cylinder_spine_pos=spine_pos, cylinder_horizontal=horiz)
        flow = compose_flows(flow, cylindrical_flow(
            H, W, curl=curl, cam_dist=cam_dist,
            spine_pos=spine_pos, horizontal=horiz, fit='contain'))

    # 2. Perspective / camera angle
    if use_perspective:
        shift = perspective_shift if perspective_shift is not None \
            else random.uniform(0.04, 0.13)
        params['perspective_shift'] = shift
        flow = compose_flows(flow, perspective_flow(H, W, max_shift_frac=shift))

    # 3. Wrinkles / creases
    if use_wrinkle:
        n_ctrl = wrinkle_n_control if wrinkle_n_control is not None \
            else random.randint(4, 8)
        max_disp = wrinkle_max_disp if wrinkle_max_disp is not None \
            else random.uniform(0.01, 0.04)
        params.update(wrinkle_n_control=n_ctrl, wrinkle_max_disp=max_disp)
        flow = compose_flows(flow, wrinkle_flow(
            H, W, n_control=n_ctrl, max_disp_frac=max_disp))

    # 4. Camera framing — zoom / roll / off-centre aim. Composed LAST because
    #    it happens at capture, after the page is already bent. This is what
    #    decouples how much background is visible from how badly the page is
    #    warped; without it, coverage was a -0.79-correlated readout of warp
    #    severity and a perfect flat-page detector. See similarity_flow.
    if use_framing:
        if page_scale is not None or page_angle is not None:
            scale = 1.0 if page_scale is None else page_scale
            angle = 0.0 if page_angle is None else page_angle
            tx = ty = 0.0
            regime = 'explicit'
        else:
            scale, angle, tx, ty, regime = sample_framing(H, W)
        params.update(page_scale=scale, page_angle=angle,
                      page_tx=tx, page_ty=ty, framing=regime)
        flow = compose_flows(flow, similarity_flow(H, W, scale, angle, tx, ty))

    # ---- the single resample --------------------------------------------
    # BORDER_REPLICATE is kept here only so that bilinear taps at the very
    # page edge use page colour rather than blending toward black. Everything
    # outside the page is replaced by background immediately below, so no
    # replicated pixel survives into the output.
    warped = apply_flow(image, flow)

    # ---- validity mask ---------------------------------------------------
    mask = flow_to_valid_mask(flow, src_h=H, src_w=W, soft=True)
    params['page_coverage'] = float(mask.mean())

    # 5. Background. The page is composited onto a synthesized surface with a
    #    contact shadow. Before this, margins were filled with smeared page
    #    pixels — a texture that exists in no real photograph — so the model
    #    had no honest cue for where the page ends.
    if use_background and params['page_coverage'] < 0.999:
        warped = composite_on_background(warped, mask, drop_shadow=True)
        params['background'] = True

    # 6. Shadow: photometric only, and applied AFTER compositing so the
    #    lighting gradient falls across the page and the surface together, as
    #    one scene lit by one source. Never touches the flow.
    if use_shadow and random.random() < shadow_prob:
        min_alpha = random.uniform(0.4, 0.65)
        params['shadow_min_alpha'] = min_alpha
        warped = add_shadow(warped, min_alpha=min_alpha)

    _meta = ('warps_applied', 'page_coverage', 'background', 'page_scale',
             'page_angle', 'page_tx', 'page_ty', 'framing', 'shadow_min_alpha')
    params['warps_applied'] = [k for k in params if k not in _meta]
    return warped, flow.astype(np.float32), mask, params


# ─────────────────────────────────────────────
# DISTORTION PROFILES
# ─────────────────────────────────────────────

# Each profile defines a named, distinct distortion scenario.
# Having multiple profiles per image means the model sees every
# clean image in several different distortion contexts,
# improving generalization.
#
# Profile dict keys map directly to augment_document() kwargs.
# Set a key to False to disable that warp type for this profile.

# 3 profiles chosen to cover the most distinct real-world distortion types.
# With 48k source images x 3 variants = 144,000 training pairs total.
#
#  v0  cylinder   — pure spine/book curvature + shadow
#  v1  perspective — pure camera angle/tilt + shadow
#  v2  combined   — cylinder + perspective + wrinkle + shadow (hardest, most realistic)

DISTORTION_PROFILES = [
    {
        # v0 — pure book-spine / bent-page curvature.
        # The most common real-world document distortion.
        'name': 'cylinder',
        'use_cylinder':    True,
        'use_perspective': False,
        'use_wrinkle':     False,
        'use_shadow':      True,
        'cylinder_curl':       None,   # random in [0.15, 0.85] rad
        'cylinder_cam_dist':   None,   # random in [1.6, 4.0] x page width
        'cylinder_spine_pos':  None,   # random, edge-biased
        'cylinder_horizontal': None,   # 25% chance of a horizontal spine
    },
    {
        # v1 — pure camera angle. A photo taken from the side or at a slant.
        'name': 'perspective',
        'use_cylinder':    False,
        'use_perspective': True,
        'use_wrinkle':     False,
        'use_shadow':      True,
        'perspective_shift': None,     # random in [0.04, 0.13]
    },
    {
        # v2 — everything at once. What a real hand-held capture of a bent,
        # slightly crumpled page actually looks like. All three warps are now
        # GUARANTEED to apply (previously all three landed only 37% of the
        # time, so this profile was mostly a two-warp profile in disguise).
        'name': 'combined',
        'use_cylinder':    True,
        'use_perspective': True,
        'use_wrinkle':     True,
        'use_shadow':      True,
        'cylinder_curl':       None,
        'cylinder_cam_dist':   None,
        'cylinder_spine_pos':  None,
        'cylinder_horizontal': None,
        'perspective_shift':   None,
        'wrinkle_n_control':   None,   # random in [4, 8]
        'wrinkle_max_disp':    None,   # random in [0.01, 0.04]
    },
]


# A deliberately flat page: zero flow, but shadow still applied most of the
# time. Emitted for a small counted fraction of source images (`flat_frac`) so
# the model learns to leave an undistorted page alone, and specifically learns
# that a lighting gradient is not geometry.
FLAT_PROFILE = {
    'name': 'flat',
    'use_cylinder':    False,
    'use_perspective': False,
    'use_wrinkle':     False,
    'use_shadow':      True,
}
# Note: flat samples still get random camera framing, so roughly a quarter of
# them are an exact zero-flow identity (the 'exact' regime) and the rest are a
# pure zoom/roll with background visible. That is deliberate — if 'flat' were
# the only profile that ever reached 100% page coverage, coverage alone would
# identify it (it did: 100% precision before framing was added).


# ─────────────────────────────────────────────
# BATCH PROCESSING
# ─────────────────────────────────────────────

def process_dataset(input_dir: str,
                    clean_dir: str,
                    warped_dir: str,
                    flow_dir: str,
                    variants_per_image: int = 3,
                    width: int = TRAIN_W,
                    height: int = TRAIN_H,
                    flat_frac: float = 0.03,
                    extensions: tuple = ('.png', '.jpg', '.jpeg'),
                    seed: int = 42):
    """
    Process every image in input_dir and save augmented pairs.

    RESIZE THEN WARP
    ----------------
    Each source page is resized to (width, height) BEFORE warping. Warping at
    native 612 x 792 and downsampling afterwards costs ~5x more CPU and throws
    the extra detail away anyway, since the stored sample is at training
    resolution regardless.

    FLAT SAMPLES
    ------------
    `flat_frac` of the source images additionally emit a `_v3_flat` variant:
    zero flow, shadow usually applied. This replaces the ~17% of accidentally
    unwarped samples the old random gating produced, with a small, counted,
    intentional number instead.

    NOTE: this still writes loose PNG + float32 .npy files, which is the
    Session 5 format decision NOT yet implemented (Fix #2). The sharded
    JPEG + float16-normalized-flow writer replaces this function.
    """
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_dir)
    clean_path, warped_path, flow_path = map(Path, (clean_dir, warped_dir, flow_dir))
    for d in (clean_path, warped_path, flow_path):
        d.mkdir(parents=True, exist_ok=True)

    image_files = []
    for ext in extensions:
        image_files.extend(input_path.rglob(f'*{ext}'))
    image_files = sorted(image_files)

    n_profiles = len(DISTORTION_PROFILES)
    n_flat = int(round(len(image_files) * flat_frac))
    flat_ids = set(random.Random(seed ^ 0x5EED).sample(
        range(len(image_files)), min(n_flat, len(image_files))))

    print(f"Found {len(image_files):,} images in {input_dir}")
    print(f"Resizing to {width} x {height}, then warping "
          f"({variants_per_image} variant(s) per image from {n_profiles} profiles)")
    print(f"Flat variants: {len(flat_ids):,} images ({100*flat_frac:.1f}%) "
          f"also get a zero-flow sample")
    print(f"Total pairs to generate: "
          f"{len(image_files)*variants_per_image + len(flat_ids):,}")
    print()

    total = 0
    n_written_flat = 0
    for i, img_path in enumerate(image_files):
        stem = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [SKIP] unreadable: {img_path}")
            continue

        clean = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(clean_path / f"{stem}.png"), clean)

        todo = [DISTORTION_PROFILES[v % n_profiles] for v in range(variants_per_image)]
        if i in flat_ids:
            todo.append(FLAT_PROFILE)

        for v, prof in enumerate(todo):
            profile = dict(prof)
            profile_name = profile.pop('name')
            warped, flow, mask, params = augment_document(clean, **profile)

            variant_name = f"{stem}_v{v}_{profile_name}"
            cv2.imwrite(str(warped_path / f"{variant_name}.png"), warped)
            np.save(str(flow_path / f"{variant_name}.npy"), flow.astype(np.float32))
            total += 1
            if profile_name == 'flat':
                n_written_flat += 1

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1:,}/{len(image_files):,}] {stem}  ({len(todo)} variants)")

    print(f"\nDone.")
    print(f"  Source images   : {len(image_files):,}")
    print(f"  Resolution      : {width} x {height}")
    print(f"  Warped variants : {total - n_written_flat:,}")
    print(f"  Flat variants   : {n_written_flat:,}")
    print(f"  Total pairs     : {total:,}")
    print(f"  Clean  -> {clean_dir}")
    print(f"  Warped -> {warped_dir}")
    print(f"  Flow   -> {flow_dir}")


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Document Dewarping Synthetic Augmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 3 variants per image (default) — 48k images -> 144k training pairs
  python warp_augment.py --input_dir D:/publaynet/train

  # 1 variant per image (quick smoke test)
  python warp_augment.py --input_dir D:/publaynet/train --variants 1

  # Custom output directories
  python warp_augment.py --input_dir D:/publaynet/train --variants 3 --clean_dir D:/data/clean --warped_dir D:/data/warped --flow_dir D:/data/flow
"""
    )
    parser.add_argument("--input_dir",  required=True,
                        help="Directory containing clean PubLayNet images")
    parser.add_argument("--clean_dir",  default="images_clean",
                        help="Output directory for clean image copies")
    parser.add_argument("--warped_dir", default="images_warped",
                        help="Output directory for warped images")
    parser.add_argument("--flow_dir",   default="flow_fields",
                        help="Output directory for .npy flow fields")
    parser.add_argument("--variants",   type=int, default=3,
                        help="Number of warped variants per image (default: 3, matching the 3 distortion profiles)")
    parser.add_argument("--width",      type=int, default=TRAIN_W,
                        help=f"Training width (default {TRAIN_W}, preserves PubLayNet aspect)")
    parser.add_argument("--height",     type=int, default=TRAIN_H,
                        help=f"Training height (default {TRAIN_H})")
    parser.add_argument("--flat_frac",  type=float, default=0.03,
                        help="Fraction of sources that also emit a zero-flow flat variant")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    process_dataset(
        input_dir=args.input_dir,
        clean_dir=args.clean_dir,
        warped_dir=args.warped_dir,
        flow_dir=args.flow_dir,
        variants_per_image=args.variants,
        width=args.width,
        height=args.height,
        flat_frac=args.flat_frac,
        seed=args.seed
    )
