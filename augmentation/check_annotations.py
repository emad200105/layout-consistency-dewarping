"""
check_annotations.py
====================
Decides whether a PubLayNet COCO annotation file can actually be used with the
page images on this disk — before anything is built on top of it.

    python augmentation/check_annotations.py \\
        --json   data/publaynet/train.json \\
        --images data/publaynet/train \\
        --overlays 10

WHY THIS IS NOT OPTIONAL
------------------------
Annotation coordinates live in each page's own native pixel space, and
PubLayNet pages are NOT a uniform size: val.json contains 35 distinct sizes,
of which 601x792 is 30.6% and 612x792 only 9.0%. Yet every image sampled from
this disk measured 612x792.

Two explanations, with opposite consequences:

  a) the train split really is (mostly) 612x792 — splits were rendered
     differently — so the annotations line up and there is nothing to do;
  b) these images came from a mirror that re-rendered every page to a uniform
     size, in which case annotation coordinates are in the ORIGINAL space.
     A pure rescale is recoverable by scaling with (local/json) per axis; a
     re-typeset or cropped page is not, and a different download is needed.

Only the file can settle it. This reports recorded-vs-actual dimensions and —
the part that actually decides it — DRAWS the annotations onto the local pages,
so misalignment is seen rather than inferred.
"""

import argparse
import json
import os
import struct
import sys as _sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coco_stream import CocoFile                                  # noqa: E402

CAT_NAMES = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure'}
CAT_COLORS = {
    1: (60, 180, 75),      # text    green
    2: (255, 140, 0),      # title   orange-blue (BGR)
    3: (200, 60, 220),     # list    magenta
    4: (0, 200, 255),      # table   yellow
    5: (0, 80, 255),       # figure  red
}


class Tee:
    """Write everything to stdout AND to a report file."""

    def __init__(self, path):
        self.fh = open(path, 'w', encoding='utf-8')

    def __call__(self, *args):
        line = " ".join(str(x) for x in args)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def _jpeg_size(path):
    """
    Width/height straight out of the JPEG SOF marker. No PIL, no decode.

    Avoids a dependency gamble: if PIL is missing, the alternative was
    cv2.imread on 48,000 pages, which decodes every pixel to learn two numbers.
    """
    with open(path, 'rb') as fh:
        if fh.read(2) != b'\xff\xd8':
            return None
        while True:
            b = fh.read(1)
            if not b:
                return None
            if b != b'\xff':
                continue
            while b == b'\xff':
                b = fh.read(1)
                if not b:
                    return None
            marker = b[0]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                continue
            ln = fh.read(2)
            if len(ln) < 2:
                return None
            seg = struct.unpack('>H', ln)[0]
            # SOF0..SOF15, excluding DHT/JPG/DAC
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                data = fh.read(5)
                if len(data) < 5:
                    return None
                h, w = struct.unpack('>HH', data[1:5])
                return (w, h)
            fh.seek(seg - 2, 1)


def local_dims(path):
    """Image size from the header only — no full decode."""
    p = str(path)
    if p.lower().endswith(('.jpg', '.jpeg')):
        try:
            d = _jpeg_size(p)
            if d:
                return d
        except Exception:                                         # noqa: BLE001
            pass
    try:
        from PIL import Image
        with Image.open(p) as im:
            return im.size
    except Exception:                                            # noqa: BLE001
        pass
    im = cv2.imread(p)
    return None if im is None else (im.shape[1], im.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--overlays", type=int, default=10)
    ap.add_argument("--census", type=int, default=1,
                    help="1 = measure every local image's size; 0 = skip")
    ap.add_argument("--out_dir", default="annotation_check",
                    help="report.txt, summary.json and the overlays all land here")
    a = ap.parse_args()

    idir = Path(a.images)
    out_dir = Path(a.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    say = Tee(out_dir / 'report.txt')
    summary = {'tool': 'check_annotations.py', 'version': 2,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'json': str(Path(a.json).resolve()),
               'images': str(idir.resolve()),
               'python': _sys.version.split()[0], 'opencv': cv2.__version__}
    fails = []

    try:
        local = {}
        for f in os.listdir(idir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in ('.jpg', '.jpeg', '.png'):
                local[stem] = f
        say("=" * 74)
        say(f"images folder : {idir}   ({len(local):,} images)")
        say(f"annotations   : {a.json}")
        say(f"outputs       : {out_dir}")
        say("=" * 74)
        summary['n_local_images'] = len(local)

        # ---- 1. what sizes are actually on disk -------------------------
        census = Counter()
        if a.census:
            say("\n1. census of local image dimensions (JPEG header read, no decode)")
            t0 = time.time()
            for i, (stem, fn) in enumerate(local.items()):
                d = local_dims(idir / fn)
                if d:
                    census[d] += 1
                if (i + 1) % 10000 == 0:
                    say(f"     ... {i+1:,}")
            tot = sum(census.values())
            say(f"   {len(census)} distinct size(s) across {tot:,} images "
                f"in {time.time()-t0:.1f}s")
            for (w, h), c in census.most_common(12):
                say(f"     {w:5d}x{h:<6d} {c:8,d} ({100*c/max(tot,1):6.2f}%)  "
                    f"aspect {w/h:.4f}   stretch @448x576 {(448/576)/(w/h):.4f}x")
            if len(census) > 12:
                say(f"     ... and {len(census)-12} more")
            summary['local_size_census'] = [
                {'w': w, 'h': h, 'count': c} for (w, h), c in census.most_common()]
        else:
            say("\n1. census skipped (--census 0)")

        # ---- 2. locate the top-level arrays ----------------------------
        say("\n2. annotation file structure")
        with CocoFile(a.json) as coco:
            for line in coco.report().splitlines():
                say("   " + line)
            summary['toplevel_ranges'] = {
                k: (list(v) if v else None) for k, v in coco.ranges.items()}
            summary['key_textual_occurrences'] = dict(coco.candidate_counts)
            if coco.ranges['images'] is None or coco.ranges['annotations'] is None:
                say("\n   !! top-level arrays not found — is this a COCO file?")
                fails.append("top-level arrays missing")
                raise SystemExit(0)

            # ---- 3. stem matching --------------------------------------
            say("\n3. matching local stems against the annotation file")
            kept = {}
            n_img = nested = 0
            json_sizes = Counter()
            for rec in coco.stream('images'):
                n_img += 1
                if 'annotations' in rec:
                    nested += 1
                json_sizes[(rec['width'], rec['height'])] += 1
                stem = rec['file_name'].rsplit('.', 1)[0]
                if stem in local:
                    kept[rec['id']] = rec
                if n_img % 100000 == 0:
                    say(f"     ... {n_img:,} image records, {len(kept):,} matched")
            say(f"   image records in file : {n_img:,}")
            if nested:
                say(f"   records with a NESTED 'annotations' array: {nested:,}")
                say(f"     (human-corrected subset merged into the same file;")
                say(f"      handled by coco_stream, not a problem)")
            say(f"   matched local stems   : {len(kept):,} of {len(local):,} "
                f"({100*len(kept)/max(len(local),1):.2f}%)")
            summary.update(n_image_records=n_img, n_nested_annotation_records=nested,
                           n_matched_stems=len(kept))
            if not kept:
                say("\n   !! NO OVERLAP — this file does not describe these images.")
                say("      If the images are the train split, use train.json.")
                fails.append("no stem overlap")
                raise SystemExit(0)
            say("   sizes recorded in the file (top 6):")
            tj = sum(json_sizes.values())
            for (w, h), c in json_sizes.most_common(6):
                say(f"     {w:5d}x{h:<6d} {c:8,d} ({100*c/tj:6.2f}%)")
            summary['json_size_census_top'] = [
                {'w': w, 'h': h, 'count': c} for (w, h), c in json_sizes.most_common(12)]

            # ---- 4. recorded vs actual ---------------------------------
            say("\n4. recorded vs actual dimensions, for the matched images")
            agree = Counter()
            scales = []
            examples = []
            for rec in kept.values():
                stem = rec['file_name'].rsplit('.', 1)[0]
                d = local_dims(idir / local[stem])
                if not d:
                    continue
                aw, ah = d
                jw, jh = rec['width'], rec['height']
                if (aw, ah) == (jw, jh):
                    agree['exact'] += 1
                else:
                    agree['differs'] += 1
                    scales.append((aw / jw, ah / jh))
                    if len(examples) < 8:
                        examples.append((rec['file_name'], [jw, jh], [aw, ah]))
            tot_cmp = max(sum(agree.values()), 1)
            for k in ('exact', 'differs'):
                say(f"     {k:8s} : {agree[k]:,} ({100*agree[k]/tot_cmp:.2f}%)")
            exact_frac = agree['exact'] / tot_cmp
            if examples:
                say("   examples (recorded -> actual):")
                for fn, j, act in examples:
                    say(f"     {fn}: {j[0]}x{j[1]} -> {act[0]}x{act[1]}")
            sc = {}
            if scales:
                sx = np.array([p[0] for p in scales])
                sy = np.array([p[1] for p in scales])
                sc = {'x_min': float(sx.min()), 'x_max': float(sx.max()),
                      'x_std': float(sx.std()), 'y_min': float(sy.min()),
                      'y_max': float(sy.max()), 'y_std': float(sy.std())}
                say(f"   implied scale x: min {sx.min():.4f} max {sx.max():.4f} "
                    f"std {sx.std():.5f}")
                say(f"   implied scale y: min {sy.min():.4f} max {sy.max():.4f} "
                    f"std {sy.std():.5f}")
                say("   -> tight and consistent means a pure rescale fixes them;")
                say("      scattered means pages were rendered differently.")
            summary.update(dims_exact=agree['exact'], dims_differ=agree['differs'],
                           dims_exact_fraction=exact_frac,
                           implied_scale=sc, dim_mismatch_examples=examples)

            if exact_frac > 0.99:
                verdict = "MATCH"
                say("   VERDICT: dimensions match. Annotations are in the same")
                say("            pixel space as these images.")
            elif exact_frac > 0.5:
                verdict = "MOSTLY_MATCH"
                say("   VERDICT: most match but not all — per-image scaling by")
                say("            (actual/recorded) is required.")
            else:
                verdict = "MISMATCH"
                say("   VERDICT: dimensions differ. Per-image scaling required,")
                say("            and the overlays must be inspected before it is")
                say("            trusted at all.")
            summary['dimension_verdict'] = verdict

            # ---- 5. annotations ----------------------------------------
            say("\n5. annotations for the matched images")
            want = set(list(kept.keys())[:max(a.overlays, 0)])
            per_img = defaultdict(list)
            cats = Counter()
            poly_pts = []
            n_ann = n_hit = 0
            for rec in coco.stream('annotations'):
                n_ann += 1
                iid = rec['image_id']
                if iid in kept:
                    n_hit += 1
                    cats[rec['category_id']] += 1
                    if len(poly_pts) < 5000:
                        for poly in rec.get('segmentation', []):
                            poly_pts.append(len(poly) // 2)
                    if iid in want:
                        per_img[iid].append(rec)
                if n_ann % 1000000 == 0:
                    say(f"     ... {n_ann:,} annotation records")
            say(f"   annotation records in file : {n_ann:,}")
            say(f"   for the matched images     : {n_hit:,} "
                f"({n_hit/max(len(kept),1):.1f} per image)")
            for cid, c in sorted(cats.items()):
                say(f"     {cid} {CAT_NAMES.get(cid,'?'):7s} {c:10,d} "
                    f"({100*c/max(n_hit,1):5.2f}%)")
            if poly_pts:
                pp = np.array(poly_pts)
                say(f"   segmentation polygons: {pp.mean():.1f} points mean, "
                    f"max {pp.max()}  (bboxes present on all)")
            summary.update(n_annotation_records=n_ann, n_annotations_matched=n_hit,
                           annotations_per_image=n_hit/max(len(kept), 1),
                           category_counts={int(k): int(v) for k, v in cats.items()})
            if n_hit == 0:
                fails.append("no annotations for the matched images")

            # ---- 6. overlays -------------------------------------------
            say("\n6. overlays — the part that actually decides it")
            written = []
            for iid, anns in per_img.items():
                rec = kept[iid]
                stem = rec['file_name'].rsplit('.', 1)[0]
                im = cv2.imread(str(idir / local[stem]))
                if im is None:
                    continue
                ah, aw = im.shape[:2]
                sx, sy = aw / rec['width'], ah / rec['height']
                vis = im.copy()
                for an in anns:
                    col = CAT_COLORS.get(an['category_id'], (255, 255, 255))
                    x, y, w, h = an['bbox']
                    cv2.rectangle(vis, (int(x*sx), int(y*sy)),
                                  (int((x+w)*sx), int((y+h)*sy)), col, 2)
                    for poly in an.get('segmentation', []):
                        pts = np.asarray(poly, np.float32).reshape(-1, 2)
                        pts[:, 0] *= sx
                        pts[:, 1] *= sy
                        cv2.polylines(vis, [pts.astype(np.int32)], True, col, 1)
                    cv2.putText(vis, CAT_NAMES.get(an['category_id'], '?'),
                                (int(x*sx)+3, int(y*sy)+14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
                bar = np.full((24, vis.shape[1], 3), 20, np.uint8)
                cv2.putText(bar, f"{stem}  recorded {rec['width']}x{rec['height']}"
                                 f"  actual {aw}x{ah}  scale {sx:.3f},{sy:.3f}"
                                 f"  {len(anns)} anns",
                            (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                            (225, 225, 225), 1)
                fn = f'overlay_{stem}.png'
                cv2.imwrite(str(out_dir / fn), np.vstack([bar, vis]))
                written.append(fn)
            say(f"   wrote {len(written)} overlay(s):")
            for fn in written:
                say(f"     {fn}")
            summary['overlays'] = written
            say("\n   OPEN THEM. Boxes sitting exactly on the text blocks means the")
            say("   annotations are usable. Shifted or mis-scaled boxes mean they")
            say("   are not, and a different image download is needed.")

    except SystemExit:
        pass
    except Exception as exc:                                     # noqa: BLE001
        import traceback
        say("\n!! ERROR\n" + traceback.format_exc())
        fails.append(f"exception: {exc}")
    finally:
        summary['problems'] = fails
        say("\n" + "=" * 74)
        if fails:
            say("PROBLEMS: " + "; ".join(fails))
        else:
            say("Structural checks passed. The overlays are the final word.")
        say(f"\nreport  : {out_dir / 'report.txt'}")
        say(f"summary : {out_dir / 'summary.json'}")
        with open(out_dir / 'summary.json', 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        say.close()


if __name__ == "__main__":
    main()
