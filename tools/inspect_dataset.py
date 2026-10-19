"""
inspect_dataset.py
==================
Dataset reconnaissance + live pipeline check on REAL PubLayNet images.

What it does:
  1. Reports the folder layout (and names any annotation JSON it finds).
  2. Counts images, total on-disk size, and the real width/height distribution.
  3. Picks N images AT RANDOM and runs the actual augmentation pipeline on
     them at the target training resolution, writing a contact sheet so the
     warps can be eyeballed on real text rather than a synthetic test page.
  4. Round-trip checks every generated sample: resampling the clean image with
     the saved flow must reproduce the saved warped image exactly.
  5. Projects the dataset storage cost.

Everything printed is also written to a report file, so the output can be
read back without scrolling a terminal.

Usage:
    python inspect_dataset.py
    python inspect_dataset.py --n_warp 8 --res 512 --seed 7
"""

import argparse
import os
import random
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


# ─────────────────────────────────────────────
# tee: print to stdout AND to the report file
# ─────────────────────────────────────────────

class Tee:
    def __init__(self, path):
        self.fh = open(path, 'w', encoding='utf-8')

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


# ─────────────────────────────────────────────
# 1. FOLDER LAYOUT
# ─────────────────────────────────────────────

def report_layout(say, root: Path, max_depth: int = 3):
    say("=" * 72)
    say(f"FOLDER LAYOUT under {root}")
    say("=" * 72)
    if not root.exists():
        say(f"  !! {root} does not exist.")
        return

    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = 0 if str(rel) == '.' else len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue

        n_img = sum(1 for f in filenames if f.lower().endswith(IMAGE_EXTS))
        others = [f for f in filenames if not f.lower().endswith(IMAGE_EXTS)]
        indent = "  " * depth
        label = str(rel) if str(rel) != '.' else '.'
        say(f"{indent}{label}/   images={n_img:,}  other={len(others):,}  subdirs={len(dirnames)}")

        # Annotation files matter — the layout-aware loss needs them.
        for f in others[:8]:
            try:
                mb = (Path(dirpath) / f).stat().st_size / 1e6
                say(f"{indent}    [file] {f}  ({mb:,.1f} MB)")
            except OSError:
                pass
        if len(others) > 8:
            say(f"{indent}    ... and {len(others) - 8:,} more non-image files")
    say("")


# ─────────────────────────────────────────────
# 2. IMAGE INVENTORY
# ─────────────────────────────────────────────

def collect_images(root: Path):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith(IMAGE_EXTS):
                files.append(Path(dirpath) / f)
    return files


def report_images(say, files, sample: int):
    say("=" * 72)
    say("IMAGE INVENTORY")
    say("=" * 72)
    if not files:
        say("  !! No image files found.")
        return None

    total = 0
    for p in files:
        try:
            total += p.stat().st_size
        except OSError:
            pass
    n = len(files)
    say(f"  Image files        : {n:,}")
    say(f"  Total on disk      : {total / 1e9:,.2f} GB")
    say(f"  Mean file size     : {total / n / 1e3:,.1f} KB")
    say(f"  Example filenames  :")
    for p in files[:5]:
        say(f"       {p.name}")
    say(f"  Extensions present : {dict(Counter(p.suffix.lower() for p in files))}")
    say("")

    if not HAVE_CV2:
        say("  (opencv unavailable — skipping dimension sampling)")
        return n

    files_sorted = sorted(files)
    step = max(1, n // sample)
    picks = files_sorted[::step][:sample]

    dims = []
    for p in picks:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            dims.append((w, h))
    if not dims:
        say("  !! Could not read any sampled image.")
        return n

    say("-" * 72)
    say(f"DIMENSION DISTRIBUTION  (sampled {len(dims):,} of {n:,})")
    say("-" * 72)
    counts = Counter(dims)
    say("  Most common (width x height):")
    for (w, h), c in counts.most_common(10):
        say(f"    {w:5d} x {h:5d}   {c:5d} imgs  ({100.0*c/len(dims):5.1f}%)")

    ws = sorted(w for w, _ in dims)
    hs = sorted(h for _, h in dims)
    q = lambda v, p: v[min(len(v) - 1, int(p * len(v)))]          # noqa: E731
    say("")
    say(f"  width  : min={ws[0]}  p25={q(ws,.25)}  median={q(ws,.5)}  p75={q(ws,.75)}  max={ws[-1]}")
    say(f"  height : min={hs[0]}  p25={q(hs,.25)}  median={q(hs,.5)}  p75={q(hs,.75)}  max={hs[-1]}")
    say(f"  distinct sizes in sample : {len(counts):,}")
    say(f"  median aspect ratio (w/h): {q(ws,.5)/q(hs,.5):.3f}")
    say(f"  -> 448x576 resize implies a horizontal stretch of "
        f"{(448/576)/(q(ws,.5)/q(hs,.5)):.4f}x   "
        f"(a 512x512 square would be {(1.0)/(q(ws,.5)/q(hs,.5)):.4f}x)")
    say("")
    return n


# ─────────────────────────────────────────────
# 3. LIVE PIPELINE CHECK ON RANDOM REAL IMAGES
# ─────────────────────────────────────────────

def run_pipeline_check(say, files, out_png: Path, n_warp: int,
                       width: int, height: int, seed: int):
    """
    Pick n_warp images at random, run all 3 distortion profiles on each at the
    target training resolution, verify the flow label is exact, and write a
    contact sheet.
    """
    here = Path(__file__).resolve().parent
    for cand in (here / 'augmentation', here):
        if (cand / 'warp_augment.py').exists():
            sys.path.insert(0, str(cand))
            break
    try:
        from warp_augment import (augment_document, apply_flow,
                                  DISTORTION_PROFILES, FLAT_PROFILE,
                                  TRAIN_W, TRAIN_H)
        if width is None:
            width, height = TRAIN_W, TRAIN_H
        profiles = list(DISTORTION_PROFILES) + [FLAT_PROFILE]
    except Exception:
        say("  !! Could not import warp_augment.py:")
        say(traceback.format_exc())
        return

    say("=" * 72)
    say(f"LIVE PIPELINE CHECK — {n_warp} random real images @ {width}x{height}")
    say(f"  profiles: {', '.join(p['name'] for p in profiles)}")
    say("=" * 72)

    rng = random.Random(seed)
    picks = rng.sample(files, min(n_warp, len(files)))

    rows = []
    all_ok = True
    for i, path in enumerate(picks):
        img = cv2.imread(str(path))
        if img is None:
            say(f"  [SKIP] unreadable: {path.name}")
            continue
        h0, w0 = img.shape[:2]

        # Resize-then-warp: this is the order the real pipeline will use, and
        # it is ~5x cheaper than warping at full page resolution.
        clean = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        say(f"  [{i+1}/{len(picks)}] {path.name}   native {w0}x{h0}  ->  {width}x{height}")

        panels = [("clean", clean)]
        for v, prof in enumerate(profiles):
            p = dict(prof)
            name = p.pop('name')
            random.seed(seed * 1000 + i * 10 + v)
            np.random.seed(seed * 1000 + i * 10 + v)

            # Geometry-only pass to verify the flow is an exact label
            # (shadow is photometric and deliberately not in the flow).
            p_geo = dict(p, use_shadow=False, use_background=False)
            w_geo, flow, mask, _ = augment_document(clean, **p_geo)
            err = np.abs(apply_flow(clean, flow).astype(np.int16)
                         - w_geo.astype(np.int16))[mask > 0.999]
            ok = int(err.max()) == 0 if err.size else True
            all_ok &= ok

            random.seed(seed * 1000 + i * 10 + v)
            np.random.seed(seed * 1000 + i * 10 + v)
            warped, flow2, mask2, params = augment_document(clean, **p)

            mag = np.sqrt(flow2[..., 0] ** 2 + flow2[..., 1] ** 2)
            say(f"        v{v} {name:11s} |dx|max={np.abs(flow2[...,0]).max():6.2f} "
                f"|dy|max={np.abs(flow2[...,1]).max():6.2f} "
                f"mean={mag.mean():5.2f}px  page={100*mask2.mean():5.1f}%  "
                f"{flow2.dtype}  label_exact={'YES' if ok else 'NO  <-- PROBLEM'}")
            panels.append((f"v{v} {name}", warped))
            if v == len(profiles) - 2:      # show the mask for v2 combined
                mvis = cv2.cvtColor((mask2 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                panels.append(("v2 mask", mvis))

        # build one row of the contact sheet
        th = 330
        tw = int(round(th * width / height))
        cells = []
        for label, im in panels:
            cell = cv2.resize(im, (tw, th))
            bar = np.full((22, tw, 3), 28, np.uint8)
            cv2.putText(bar, label, (3, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.40, (215, 215, 215), 1)
            cells.append(np.vstack([bar, cell]))
        row = np.hstack([np.hstack([c, np.full((c.shape[0], 4, 3), 70, np.uint8)])
                         for c in cells])
        tag = np.full((20, row.shape[1], 3), 12, np.uint8)
        cv2.putText(tag, f"{path.name}  ({w0}x{h0})", (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 210, 140), 1)
        rows.append(np.vstack([tag, row]))

    if rows:
        wmax = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], wmax - r.shape[1], 3), 12, np.uint8)])
                if r.shape[1] < wmax else r for r in rows]
        sheet = np.vstack([np.vstack([r, np.full((6, wmax, 3), 70, np.uint8)])
                           for r in rows])
        cv2.imwrite(str(out_png), sheet)
        say("")
        say(f"  Contact sheet written to: {out_png}")
    say(f"  Flow labels exact on every sample: {'YES' if all_ok else 'NO'}")
    say("")


# ─────────────────────────────────────────────
# 4. STORAGE PROJECTION
# ─────────────────────────────────────────────

def report_storage(say, n_source: int, variants: int = 3):
    say("=" * 72)
    say(f"STORAGE PROJECTION   ({n_source:,} sources x {variants} = "
        f"{n_source*variants:,} pairs)")
    say("=" * 72)
    BPP = 0.30                      # JPEG q=90 on a text page, empirical
    n_pairs = n_source * variants

    say("  CURRENT on-disk format (full-res PNG + full-res float64 flow):")
    for (w, h) in [(612, 792), (800, 1000)]:
        c = n_source * w * h * 3 * 0.45 / 1e9
        wa = n_pairs * w * h * 3 * 0.45 / 1e9
        f64 = n_pairs * w * h * 2 * 8 / 1e9
        say(f"    {w}x{h}: clean {c:6,.0f} GB | warped {wa:7,.0f} GB | "
            f"flow f64 {f64:8,.0f} GB | TOTAL {c+wa+f64:8,.0f} GB")
    say("")
    say("  PROPOSED (fixed-res JPEG q90 + low-res float16 normalized flow):")
    say(f"    {'train res':>11} | {'flow':>7} | {'clean':>8} | {'warped':>8} | {'flow':>8} | {'TOTAL':>9}")
    say("    " + "-" * 62)
    for (w, h) in ((256, 320), (320, 416), (384, 480), (448, 576), (512, 512), (512, 664)):
        fw, fh = w // 8, h // 8
        c = n_source * w * h * BPP / 1e9
        wa = n_pairs * w * h * BPP / 1e9
        fl = n_pairs * fw * fh * 2 * 2 / 1e9
        mark = "   <-- chosen" if (w, h) == (448, 576) else ""
        say(f"    {w:>4}x{h:<6} | {fw:>2}x{fh:<4} | {c:7,.2f}G | "
            f"{wa:7,.2f}G | {fl:7,.2f}G | {c+wa+fl:8,.2f}G{mark}")
    say("")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect PubLayNet + live pipeline check")
    ap.add_argument("--root", default="data/publaynet/train")
    ap.add_argument("--sample", type=int, default=400,
                    help="images opened for the dimension histogram")
    ap.add_argument("--n_warp", type=int, default=6,
                    help="random images to run the full pipeline on")
    ap.add_argument("--width", type=int, default=None,
                    help="training width (default: TRAIN_W from warp_augment)")
    ap.add_argument("--height", type=int, default=None,
                    help="training height (default: TRAIN_H from warp_augment)")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    say = Tee(out_dir / "inspect_report.txt")

    try:
        say(f"inspect_dataset.py   root={args.root}   seed={args.seed}")
        say(f"python={sys.version.split()[0]}  numpy={np.__version__}  "
            f"opencv={cv2.__version__ if HAVE_CV2 else 'MISSING'}")
        say("")

        root = Path(args.root)
        report_layout(say, root)
        files = collect_images(root)
        n = report_images(say, files, args.sample)
        if n and HAVE_CV2:
            run_pipeline_check(say, files, out_dir / "real_preview.png",
                               args.n_warp, args.width, args.height, args.seed)
        if n:
            report_storage(say, n, args.variants)
        say("DONE.")
    finally:
        say.close()
