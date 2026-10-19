"""
verify_dataset.py
=================
Checks a built dataset is actually correct and actually reproducible.

    python augmentation/verify_dataset.py --root dataset

Run it after every build. It verifies:

  1. meta / table / memmap sizes agree with each other
  2. every sample decodes (page, clean, flow, mask) at the right shape+dtype
  3. REPRODUCIBILITY: regenerating a sample from its manifest row alone
     (seed + source stem) reproduces the stored JPEG byte for byte
  4. flow fidelity: decoded stored flow vs freshly generated full-resolution
     flow, in pixels, inside the page mask
  5. the flow really does relate the two stored images — resampling the clean
     page with the decoded flow reproduces the stored warped page
  6. profile mix and page-coverage distribution look as intended
  7. the stored flow is invertible, so it can actually rectify an image
     (it maps warped -> clean; rectifying needs the inverse)
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset_format as fmt                                        # noqa: E402
from warp_augment import (augment_document, apply_flow,             # noqa: E402
                          DISTORTION_PROFILES, FLAT_PROFILE)

PROFILE_BY_NAME = {p['name']: p for p in DISTORTION_PROFILES}
PROFILE_BY_NAME[FLAT_PROFILE['name']] = FLAT_PROFILE


class _Tee:
    """Duplicate stdout into a file, so the run leaves a readable artefact."""

    def __init__(self, path, stream):
        self.fh = open(path, 'w', encoding='utf-8')
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:                                        # noqa: BLE001
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source_dir", default=None,
                    help="Original page folder; enables checks 3-5")
    ap.add_argument("--n_check", type=int, default=24,
                    help="How many samples to deep-check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None,
                    help="where to write the run log "
                         "(default: <root>/verify_report.txt)")
    a = ap.parse_args()

    root = Path(a.root)
    fails = []

    report_path = Path(a.report) if a.report else (root / 'verify_report.txt')
    real_stdout = sys.stdout
    tee = _Tee(report_path, real_stdout)
    sys.stdout = tee
    t_start = time.time()

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
        if not cond:
            fails.append(name)

    print("=" * 74)
    print("1. structural consistency")
    print("=" * 74)
    ds = fmt.DewarpDataset(root)
    meta = ds.meta
    n = len(ds)
    print(f"      {n:,} samples, {len(ds.clean_index['clean_id']):,} sources, "
          f"page {meta['page_w']}x{meta['page_h']}, flow "
          f"{meta['flow_w']}x{meta['flow_h']} {meta['flow_dtype']}")
    check("manifest length matches meta", n == meta['n_samples'],
          f"{n} vs {meta['n_samples']}")
    check("flows memmap length matches", ds.flows.shape[0] == n)
    check("masks memmap length matches", ds.masks.shape[0] == n)
    exp_flow = n * fmt.FLOW_ITEM_BYTES
    got_flow = (root / 'flows.f16').stat().st_size
    check("flows file size exact", exp_flow == got_flow,
          f"{got_flow:,} B == {exp_flow:,} B")
    check("masks file size exact",
          n * fmt.MASK_ITEM_BYTES == (root / 'masks.u8').stat().st_size)
    ids = np.asarray(ds.manifest['sample_id'])
    check("sample_id is 0..n-1 contiguous",
          bool(np.array_equal(ids, np.arange(n))))
    cids = np.asarray(ds.manifest['clean_id'])
    check("every clean_id resolves",
          int(cids.max()) < len(ds.clean_index['clean_id']),
          f"max clean_id {cids.max()}")

    print()
    print("=" * 74)
    print("2. decode every sampled record")
    print("=" * 74)
    rng = random.Random(a.seed)
    idxs = rng.sample(range(n), min(a.n_check, n))
    ok_shape = True
    for i in idxs:
        r = ds[i]
        if (r['warped'].shape != (meta['page_h'], meta['page_w'], 3)
                or r['clean'].shape != (meta['page_h'], meta['page_w'], 3)
                or r['flow'].shape != (meta['page_h'], meta['page_w'], 2)
                or r['mask'].shape != (meta['page_h'], meta['page_w'])):
            ok_shape = False
    check(f"{len(idxs)} records decode at correct shapes", ok_shape)
    r0 = ds[idxs[0]]
    check("flow decodes float32", r0['flow'].dtype == np.float32)
    check("mask in [0,1]", 0.0 <= r0['mask'].min() and r0['mask'].max() <= 1.0)

    # pairing: all variants of one source must share one clean image
    by_src = {}
    for k, (sid, cid) in enumerate(zip(ds.manifest['source_index'], cids)):
        by_src.setdefault(sid, set()).add(int(cid))
    check("each source maps to exactly one clean image",
          all(len(v) == 1 for v in by_src.values()))

    if not a.source_dir:
        print("\n  (--source_dir not given: skipping reproducibility and "
              "flow-fidelity checks)")
    else:
        src_dir = Path(a.source_dir)
        stems = {p.stem: p for p in src_dir.rglob('*')
                 if p.suffix.lower() in ('.jpg', '.jpeg', '.png')}

        print()
        print("=" * 74)
        print("3. REPRODUCIBILITY — regenerate from the manifest row alone")
        print("=" * 74)
        n_same = n_tried = 0
        for i in idxs:
            stem = ds.manifest['stem'][i]
            if stem not in stems:
                continue
            img = cv2.imread(str(stems[stem]))
            clean = cv2.resize(img, (meta['page_w'], meta['page_h']),
                               interpolation=cv2.INTER_AREA)
            p = dict(PROFILE_BY_NAME[ds.manifest['profile'][i]])
            p.pop('name')
            s = int(ds.manifest['sample_seed'][i])
            random.seed(s)
            np.random.seed(s)
            warped, _, _, _ = augment_document(clean, **p)
            regen = fmt.encode_jpeg(warped, meta['jpeg_quality'])
            stored = ds._read(ds._page_fh, 'pages',
                              int(ds._p_shard[i]), int(ds._p_off[i]),
                              int(ds._p_len[i]))
            n_tried += 1
            n_same += int(regen == stored)
        check(f"{n_tried} samples regenerate byte-identical",
              n_tried > 0 and n_same == n_tried, f"{n_same}/{n_tried}")

        print()
        print("=" * 74)
        print("4. flow fidelity — stored vs freshly generated full-resolution")
        print("=" * 74)
        errs = []
        for i in idxs:
            stem = ds.manifest['stem'][i]
            if stem not in stems:
                continue
            img = cv2.imread(str(stems[stem]))
            clean = cv2.resize(img, (meta['page_w'], meta['page_h']),
                               interpolation=cv2.INTER_AREA)
            p = dict(PROFILE_BY_NAME[ds.manifest['profile'][i]])
            p.pop('name')
            s = int(ds.manifest['sample_seed'][i])
            random.seed(s)
            np.random.seed(s)
            _, flow_true, mask_true, _ = augment_document(clean, **p)
            rec = ds[i]
            e = np.linalg.norm(rec['flow'] - flow_true, axis=-1)
            errs.append(e[mask_true > 0.999])
        if errs:
            e = np.concatenate(errs)
            print(f"      mean {e.mean():.4f} px   p95 {np.percentile(e,95):.4f}   "
                  f"p99 {np.percentile(e,99):.4f}   max {e.max():.4f}")
            check("flow mean error < 0.20 px", float(e.mean()) < 0.20)
            check("flow p99 error < 2.0 px", float(np.percentile(e, 99)) < 2.0)

        print()
        print("=" * 74)
        print("5. the stored flow actually relates the two stored images")
        print("=" * 74)
        # The stored page has the lighting gradient baked in, so a raw
        # intensity comparison measures the SHADOW, not the geometry (it comes
        # out ~37 grey levels, which is just add_shadow's 0.4-1.0 multiplier).
        # The shadow is low-frequency by construction, so divide out a blurred
        # gain field first: that removes the shading without being able to hide
        # a geometric error, which is high-frequency.
        raw, corr = [], []
        for i in idxs:
            rec = ds[i]
            rebuilt = apply_flow(rec['clean'], rec['flow'])
            inside = rec['mask'] > 0.999
            if inside.sum() < 1000:
                continue
            a = cv2.cvtColor(rec['warped'], cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(rebuilt, cv2.COLOR_BGR2GRAY).astype(np.float32)
            raw.append(float(np.abs(a - b)[inside].mean()))
            ga = cv2.GaussianBlur(a, (0, 0), 25)
            gb = cv2.GaussianBlur(b, (0, 0), 25)
            gain = np.clip(ga / np.maximum(gb, 1.0), 0.2, 3.0)
            corr.append(float(np.abs(a - b * gain)[inside].mean()))
        if corr:
            raw = np.array(raw)
            corr = np.array(corr)
            print(f"      raw difference (dominated by the baked-in shadow):"
                  f"      mean {raw.mean():5.2f} grey levels")
            print(f"      after dividing out the low-frequency gain field:"
                  f"    mean {corr.mean():5.2f}, worst {corr.max():5.2f}")
            print("      (residual is JPEG q90 on both images plus the sub-pixel")
            print("       flow reconstruction error from check 4)")
            check("resampling clean with the flow reproduces the warped page",
                  float(corr.mean()) < 12.0, f"{corr.mean():.2f} grey levels")

        print()
        print("=" * 74)
        print("7. the stored flow is invertible — i.e. it can actually rectify")
        print("=" * 74)
        # The stored flow maps WARPED -> clean. Rectifying needs its inverse.
        res_m, res_p, psnrs = [], [], []
        for i in idxs:
            rec = ds[i]
            g = fmt.invert_flow(rec['flow'])
            vis = apply_flow(rec['mask'].astype(np.float32), g) > 0.999
            if vis.sum() < 5000:
                continue
            H, W = rec['flow'].shape[:2]
            gx, gy = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            sx = (gx + g[..., 0]).astype(np.float32)
            sy = (gy + g[..., 1]).astype(np.float32)
            fx = cv2.remap(rec['flow'][..., 0], sx, sy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(rec['flow'][..., 1], sx, sy, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
            r = np.sqrt((fx + g[..., 0]) ** 2 + (fy + g[..., 1]) ** 2)
            res_m.append(float(r[vis].mean()))
            res_p.append(float(np.percentile(r[vis], 99)))

            # Rectified vs clean, with the baked-in shadow divided out.
            rect = apply_flow(rec['warped'], g)
            a = cv2.cvtColor(rec['clean'], cv2.COLOR_BGR2GRAY).astype(np.float32)
            b = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gain = np.clip(cv2.GaussianBlur(a, (0, 0), 25)
                           / np.maximum(cv2.GaussianBlur(b, (0, 0), 25), 1.0), 0.2, 3.0)
            d = (a - b * gain)[vis]
            psnrs.append(10 * np.log10(255.0 ** 2 / max(float((d ** 2).mean()), 1e-9)))
        if res_m:
            print(f"      inversion residual: mean {np.mean(res_m):.4f} px, "
                  f"p99 {np.mean(res_p):.4f} px, worst {max(res_m):.4f} px")
            print(f"      rectified vs clean (gain-corrected): "
                  f"PSNR {np.mean(psnrs):.2f} dB")
            check("flow inverts accurately", float(np.mean(res_m)) < 0.05,
                  f"{np.mean(res_m):.4f} px")
            check("rectification recovers the page", float(np.mean(psnrs)) > 20.0,
                  f"{np.mean(psnrs):.2f} dB")

    print()
    print("=" * 74)
    print("6. dataset composition")
    print("=" * 74)
    cnt = Counter(ds.manifest['profile'])
    for k, v in sorted(cnt.items()):
        print(f"      {k:12s} {v:8,d}  ({100*v/n:5.2f}%)")
    cov = np.array([float(m.mean()) / 255.0 for m in ds.masks[::max(1, n // 400)]])
    print(f"      page coverage: mean {cov.mean()*100:.2f}%  "
          f"min {cov.min()*100:.2f}%  max {cov.max()*100:.2f}%")
    zero = 0
    for i in range(0, n, max(1, n // 400)):
        if float(np.abs(ds.flows[i]).max()) < 1e-4:
            zero += 1
    print(f"      exact zero-flow samples in sample: {zero}")

    print()
    print("=" * 74)
    ok = not fails
    if fails:
        print(f"{len(fails)} FAILURE(S): " + ", ".join(fails))
    else:
        print("DATASET OK")
    print(f"\nelapsed {time.time()-t_start:.1f}s")
    print(f"report  : {report_path}")
    print(f"summary : {root / 'verify_summary.json'}")

    with open(root / 'verify_summary.json', 'w', encoding='utf-8') as fh:
        json.dump({'tool': 'verify_dataset.py',
                   'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   'root': str(root.resolve()),
                   'n_samples': int(n),
                   'n_checked': len(idxs),
                   'ok': ok, 'failures': fails,
                   'meta': meta}, fh, indent=2)

    sys.stdout = real_stdout
    tee.close()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
