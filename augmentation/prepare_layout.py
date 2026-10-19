"""
prepare_layout.py
=================
Builds the layout annotation store from PubLayNet's train.json, keeping only
the pages present on disk.

    python augmentation/prepare_layout.py \\
        --json    data/publaynet/train.json \\
        --images  data/publaynet/train \\
        --out_dir dataset \\
        --overlays 8

Writes <out_dir>/layout/ per layout_format.py, plus report.txt, summary.json
and overlay PNGs rendered in the TRAINING coordinate space (448x576) rather
than native page space — because native space is already known to line up
(Session 13), and what still needs proving is that the normalize-then-rescale
round trip lands the regions correctly at training resolution.

Runs independently of the image build: rows are keyed by page stem, so this can
be built while build_dataset.py is still writing, and survives a rebuild.

TWO PASSES, BY NECESSITY
------------------------
COCO orders `annotations` arbitrarily with respect to `images`, so annotations
for one page are scattered through a 1.67 GB array. Buffering all 466,221 in
RAM then sorting is the simple option (~200 MB), which is what this does — the
alternative, a second streaming pass per page, would re-read the file 47,958
times. Memory stays bounded because only bbox, category and polygon points are
kept, not the source JSON objects.
"""

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout_format as lf                                        # noqa: E402
from coco_stream import CocoFile                                  # noqa: E402


class Tee:
    def __init__(self, path):
        self.fh = open(path, 'w', encoding='utf-8')

    def __call__(self, *args):
        line = " ".join(str(x) for x in args)
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def jpeg_size(path):
    """Width/height from the JPEG SOF marker. No decode, no PIL."""
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
            m = b[0]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                continue
            ln = fh.read(2)
            if len(ln) < 2:
                return None
            seg = struct.unpack('>H', ln)[0]
            if m in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                     0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                d = fh.read(5)
                if len(d) < 5:
                    return None
                h, w = struct.unpack('>HH', d[1:5])
                return (w, h)
            fh.seek(seg - 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="dataset root; the store goes in <out_dir>/layout/")
    ap.add_argument("--train_w", type=int, default=448)
    ap.add_argument("--train_h", type=int, default=576)
    ap.add_argument("--overlays", type=int, default=8)
    ap.add_argument("--drop_landscape", type=int, default=1,
                    help="1 = exclude pages wider than tall (issue 14)")
    a = ap.parse_args()

    idir = Path(a.images)
    root = Path(a.out_dir)
    out = root / 'layout'
    out.mkdir(parents=True, exist_ok=True)
    say = Tee(out / 'report.txt')
    summary = {'tool': 'prepare_layout.py', 'layout_version': lf.LAYOUT_VERSION,
               'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'json': str(Path(a.json).resolve()),
               'train_w': a.train_w, 'train_h': a.train_h}
    problems = []

    try:
        say("=" * 74)
        say(f"annotations : {a.json}")
        say(f"images      : {idir}")
        say(f"store       : {out}")
        say(f"train space : {a.train_w} x {a.train_h}")
        say("=" * 74)

        # ---- 1. local pages and their native sizes -------------------
        say("\n1. local pages")
        t0 = time.time()
        local = {}
        for f in os.listdir(idir):
            stem, ext = os.path.splitext(f)
            if ext.lower() in ('.jpg', '.jpeg', '.png'):
                local[stem] = f
        say(f"   {len(local):,} images on disk")

        native = {}
        landscape = []
        for stem, fn in local.items():
            d = jpeg_size(idir / fn) if fn.lower().endswith(('.jpg', '.jpeg')) else None
            if d is None:
                im = cv2.imread(str(idir / fn))
                if im is None:
                    continue
                d = (im.shape[1], im.shape[0])
            if d[0] > d[1]:
                landscape.append(stem)
                if a.drop_landscape:
                    continue
            native[stem] = d
        say(f"   measured {len(native):,} pages in {time.time()-t0:.1f}s")
        say(f"   landscape pages (w > h): {len(landscape):,}"
            + ("  -> EXCLUDED (issue 14)" if a.drop_landscape else "  -> kept"))
        summary.update(n_local=len(local), n_landscape=len(landscape),
                       drop_landscape=bool(a.drop_landscape),
                       n_pages_considered=len(native))

        # ---- 2. stream images, then annotations ----------------------
        say("\n2. streaming train.json")
        with CocoFile(a.json) as coco:
            for line in coco.report().splitlines():
                say("   " + line)

            id2stem = {}
            recorded = {}          # stem -> (w, h) AS RECORDED IN THE FILE
            json_dim_mismatch = 0
            n_img = 0
            for rec in coco.stream('images'):
                n_img += 1
                stem = rec['file_name'].rsplit('.', 1)[0]
                if stem in native:
                    id2stem[rec['id']] = stem
                    recorded[stem] = (rec['width'], rec['height'])
                    if (rec['width'], rec['height']) != native[stem]:
                        json_dim_mismatch += 1
                if n_img % 100000 == 0:
                    say(f"     ... {n_img:,} image records, {len(id2stem):,} matched")
            say(f"   image records {n_img:,}; matched {len(id2stem):,} "
                f"of {len(native):,} local pages")
            if json_dim_mismatch:
                problems.append(f"{json_dim_mismatch} pages disagree on size")
                say(f"   !! {json_dim_mismatch:,} pages disagree on recorded size")
            else:
                say("   recorded sizes agree with the files on disk for all matched pages")
            missing = len(native) - len(id2stem)
            if missing:
                say(f"   !! {missing:,} local pages have NO record in this file")
                problems.append(f"{missing} local pages unannotated")

            # annotations
            t0 = time.time()
            buckets = defaultdict(list)
            cats = Counter()
            n_ann = n_hit = 0
            n_clamped = 0
            poly_pt_total = 0
            degenerate = 0
            for rec in coco.stream('annotations'):
                n_ann += 1
                stem = id2stem.get(rec['image_id'])
                if stem is None:
                    continue
                n_hit += 1
                cats[rec['category_id']] += 1
                # Normalize by the RECORDED size, not the file's actual size.
                # Annotation coordinates live in the space the file declares.
                # On this dataset the two are identical for all 47,958 pages
                # (Session 13), but dividing by the actual size would be
                # silently wrong the moment they diverge — and a re-rendered
                # mirror is exactly the case this has to survive.
                W, H = recorded[stem]
                x, y, w, h = rec['bbox']
                if w <= 0 or h <= 0:
                    degenerate += 1
                    continue
                bx, c1 = lf.quantize([x, x + w], W)
                by, c2 = lf.quantize([y, y + h], H)
                n_clamped += c1 + c2
                polys = rec.get('segmentation') or []
                # COCO allows several polygons per region; concatenating them
                # would invent edges between disjoint parts, so keep the
                # largest and record how often that happens.
                pts = np.zeros((0, 2), np.uint16)
                if polys:
                    best = max(polys, key=len)
                    arr = np.asarray(best, dtype=np.float64).reshape(-1, 2)
                    qx, c3 = lf.quantize(arr[:, 0], W)
                    qy, c4 = lf.quantize(arr[:, 1], H)
                    n_clamped += c3 + c4
                    pts = np.stack([qx, qy], axis=-1)
                    poly_pt_total += len(pts)
                buckets[stem].append((
                    np.uint8(rec['category_id']),
                    np.array([bx[0], by[0], bx[1], by[1]], np.uint16),
                    pts,
                    len(polys),
                ))
                if n_ann % 1000000 == 0:
                    say(f"     ... {n_ann:,} annotation records, {n_hit:,} kept")
            say(f"   annotation records {n_ann:,}; kept {n_hit:,} in "
                f"{time.time()-t0:.1f}s")
            if degenerate:
                say(f"   dropped {degenerate:,} with a non-positive bbox")
            multi = sum(1 for v in buckets.values() for it in v if it[3] > 1)
            say(f"   regions with >1 polygon (largest kept): {multi:,}")
            say(f"   coordinate values clamped to the page: {n_clamped:,} "
                f"of {2*n_hit + 2*poly_pt_total:,} "
                f"({100*n_clamped/max(2*n_hit+2*poly_pt_total,1):.3f}%)")
            say("     (clamping means an annotation extended past the declared")
            say("      page bounds; a few are normal, many would be suspicious)")
            for cid, c in sorted(cats.items()):
                say(f"     {cid} {lf.CAT_NAMES.get(cid,'?'):7s} {c:10,d} "
                    f"({100*c/max(n_hit,1):5.2f}%)")
            summary.update(n_annotation_records=n_ann, n_kept=n_hit,
                           n_degenerate_bbox=degenerate, n_multi_polygon=multi,
                           n_clamped_coords=n_clamped,
                           category_counts={int(k): int(v) for k, v in cats.items()})

        # ---- 3. pack -------------------------------------------------
        say("\n3. packing the store")
        stems_sorted = sorted(buckets.keys())
        N = sum(len(buckets[s]) for s in stems_sorted)
        cat_arr = np.zeros(N, np.uint8)
        bbox_arr = np.zeros((N, 4), np.uint16)
        poly_off = np.zeros(N + 1, np.int64)
        poly_all = np.zeros((poly_pt_total, 2), np.uint16)

        rows = []
        ai = 0
        pi = 0
        for stem in stems_sorted:
            start = ai
            # Sort each page's regions top-to-bottom then left-to-right, so
            # region order is deterministic and not COCO's arbitrary order.
            items = sorted(buckets[stem], key=lambda it: (int(it[1][1]), int(it[1][0])))
            for cid, bb, pts, _ in items:
                cat_arr[ai] = cid
                bbox_arr[ai] = bb
                poly_off[ai] = pi
                if len(pts):
                    poly_all[pi:pi + len(pts)] = pts
                    pi += len(pts)
                ai += 1
            poly_off[ai] = pi
            rw, rh = recorded[stem]
            aw, ah = native[stem]
            rows.append({'stem': stem,
                         'recorded_w': rw, 'recorded_h': rh,
                         'actual_w': aw, 'actual_h': ah,
                         'ann_start': start, 'ann_end': ai})
        poly_off[N] = pi

        np.save(out / lf.FILES['cat'], cat_arr)
        np.save(out / lf.FILES['bbox'], bbox_arr)
        np.save(out / lf.FILES['poly_off'], poly_off)
        np.save(out / lf.FILES['poly_xy'], poly_all[:pi])
        with open(out / lf.FILES['stems'], 'w') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')
        meta = {'layout_version': lf.LAYOUT_VERSION,
                'n_pages': len(rows), 'n_annotations': int(N),
                'n_polygon_points': int(pi),
                'coord_encoding': 'uint16, normalized to the page, 0..65535',
                'categories': lf.CAT_NAMES,
                'drop_landscape': bool(a.drop_landscape),
                'built_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        with open(out / lf.FILES['meta'], 'w') as fh:
            json.dump(meta, fh, indent=2)

        total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
        say(f"   pages       : {len(rows):,}")
        say(f"   annotations : {N:,}  ({N/max(len(rows),1):.1f} per page)")
        say(f"   poly points : {pi:,}  ({pi/max(N,1):.1f} per region)")
        say(f"   on disk     : {total/1e6:.1f} MB")
        summary.update(n_pages=len(rows), n_annotations=int(N),
                       n_polygon_points=int(pi), bytes_on_disk=int(total))

        # ---- 4. verify by re-reading ---------------------------------
        say("\n4. verifying the store by re-reading it")
        store = lf.LayoutStore(root)
        say(f"   reader sees {len(store):,} pages, {store.n_annotations():,} annotations")
        ok = (len(store) == len(rows) and store.n_annotations() == N)
        say(f"   [{'PASS' if ok else 'FAIL'}] counts round-trip")
        if not ok:
            problems.append("count mismatch after re-read")

        # quantisation error, measured against the original float bboxes
        rng = np.random.default_rng(0)
        pick = [stems_sorted[i] for i in
                rng.choice(len(stems_sorted), min(300, len(stems_sorted)),
                           replace=False)]
        errs = []
        for stem in pick:
            W, H = recorded[stem]
            b, _ = store.bboxes_array(stem, W, H)          # back to recorded px
            orig = sorted(buckets[stem], key=lambda it: (int(it[1][1]), int(it[1][0])))
            for k, (cid, bb, pts, _) in enumerate(orig):
                ref = np.array([lf.dequantize(bb[0], W), lf.dequantize(bb[1], H),
                                lf.dequantize(bb[2], W), lf.dequantize(bb[3], H)])
                errs.append(np.abs(b[k] - ref).max())
        errs = np.array(errs) if errs else np.zeros(1)
        say(f"   bbox re-read error: max {errs.max():.6f} px (should be ~0)")
        q_px = (1.0 / lf.U16_MAX) * max(a.train_w, a.train_h)
        say(f"   uint16 quantisation at {a.train_w}x{a.train_h}: "
            f"{q_px:.5f} px per step")

        # ---- 5. overlays in TRAINING space ---------------------------
        say("\n5. overlays in training space "
            f"({a.train_w}x{a.train_h}, the resize the model sees)")
        written = []
        for stem in stems_sorted[:max(a.overlays, 0)]:
            im = cv2.imread(str(idir / local[stem]))
            if im is None:
                continue
            clean = cv2.resize(im, (a.train_w, a.train_h),
                               interpolation=cv2.INTER_AREA)
            regs = store.regions_for(stem, a.train_w, a.train_h)
            vis = clean.copy()
            for r in regs:
                col = lf.CAT_COLORS.get(r['category_id'], (255, 255, 255))
                x0, y0, x1, y1 = [int(round(v)) for v in r['bbox']]
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 2)
                p = r.get('polygon')
                if p is not None and len(p):
                    cv2.polylines(vis, [p.astype(np.int32)], True, col, 1)
                cv2.putText(vis, r['category'], (x0 + 3, y0 + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            bar = np.full((22, vis.shape[1], 3), 20, np.uint8)
            cv2.putText(bar, f"{stem}  recorded {store.stems[store.by_stem[stem]]['recorded_w']}x{store.stems[store.by_stem[stem]]['recorded_h']}"
                             f" -> {a.train_w}x{a.train_h}  {len(regs)} regions",
                        (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (225, 225, 225), 1)
            fn = f'layout_{stem}.png'
            cv2.imwrite(str(out / fn), np.vstack([bar, vis]))
            written.append(fn)
        say(f"   wrote {len(written)} overlay(s)")
        summary['overlays'] = written

    except Exception as exc:                                      # noqa: BLE001
        import traceback
        say("\n!! ERROR\n" + traceback.format_exc())
        problems.append(f"exception: {exc}")
    finally:
        summary['problems'] = problems
        say("\n" + "=" * 74)
        say("PROBLEMS: " + "; ".join(problems) if problems
            else "Layout store built and verified.")
        with open(out / 'summary.json', 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        say(f"report  : {out / 'report.txt'}")
        say(f"summary : {out / 'summary.json'}")
        say.close()


if __name__ == "__main__":
    main()
