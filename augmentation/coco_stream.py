"""
coco_stream.py
==============
Streaming reader for large COCO JSON files. Built for PubLayNet's train.json
(~1.7 GB), which `json.load` cannot open without many gigabytes of RAM.

TWO TRAPS THIS HANDLES
----------------------
1. The naive approach — search the text for `"annotations"` and start parsing
   at the following `[` — is WRONG on these files. PubLayNet's val.json
   contains 84 image records that carry their own nested `"annotations"` array
   plus a `"corrected": false` flag (the human-verified subset, merged in). The
   first textual occurrence of `"annotations"` is therefore ~2.7 KB into the
   file, nested inside an image record, and a naive parser silently reads 19
   annotations instead of 120,761.

   So the key must be located at TOP LEVEL: brace depth 1, bracket depth 0.
   Depth is computed in one chunked numpy pass, which is fast enough for a
   multi-gigabyte file.

2. A per-character Python loop over 1.7 GB is unusably slow. Object boundaries
   are found instead with `mmap.find`, which is C-level: two calls per object
   rather than one Python iteration per byte.

ASSUMPTION: no `{`, `}`, `[` or `]` appears inside a string value. True for
COCO — the only string fields here are file names like `PMC4055390_00006.jpg`.
`scan_toplevel_arrays` reports the keys it found so a violation shows up as a
wrong count rather than silent corruption.
"""

import json
import mmap

import numpy as np

OPEN_BRACE, CLOSE_BRACE = ord('{'), ord('}')
OPEN_BRACK, CLOSE_BRACK = ord('['), ord(']')


def _depth_at(mm, positions, chunk=1 << 24):
    """
    Brace depth and bracket depth immediately before each byte offset in
    `positions`. One chunked numpy pass over the whole file.
    """
    positions = np.asarray(sorted(positions), dtype=np.int64)
    out = np.zeros((len(positions), 2), dtype=np.int64)
    n = len(mm)
    brace = brack = 0
    pos = 0
    pi = 0
    while pos < n and pi < len(positions):
        end = min(pos + chunk, n)
        a = np.frombuffer(mm[pos:end], dtype=np.uint8)
        d_brace = np.cumsum((a == OPEN_BRACE).astype(np.int64)
                            - (a == CLOSE_BRACE).astype(np.int64))
        d_brack = np.cumsum((a == OPEN_BRACK).astype(np.int64)
                            - (a == CLOSE_BRACK).astype(np.int64))
        while pi < len(positions) and positions[pi] < end:
            off = positions[pi] - pos
            out[pi, 0] = brace + (d_brace[off - 1] if off > 0 else 0)
            out[pi, 1] = brack + (d_brack[off - 1] if off > 0 else 0)
            pi += 1
        brace += int(d_brace[-1])
        brack += int(d_brack[-1])
        pos = end
    return {int(p): (int(b), int(k)) for p, (b, k) in zip(positions, out)}


def find_toplevel_array(mm, key):
    """
    Byte range (start, end) of the top-level array `key`, exclusive of the
    enclosing brackets. Returns None if the key has no top-level array.
    """
    needle = f'"{key}"'.encode()
    cands = []
    s = 0
    while True:
        p = mm.find(needle, s)
        if p == -1:
            break
        cands.append(p)
        s = p + 1
    if not cands:
        return None, len(cands)

    depths = _depth_at(mm, cands)
    top = [p for p in cands if depths[p] == (1, 0)]
    if not top:
        return None, len(cands)
    p = top[0]

    br = mm.find(b'[', p + len(needle))
    if br == -1:
        return None, len(cands)

    # Matching close bracket, found with C-level scans.
    depth = 0
    pos = br
    n = len(mm)
    while pos < n:
        o = mm.find(b'[', pos)
        c = mm.find(b']', pos)
        if c == -1:
            break
        if o != -1 and o < c:
            depth += 1
            pos = o + 1
        else:
            depth -= 1
            pos = c + 1
            if depth == 0:
                return (br + 1, c), len(cands)
    return (br + 1, n), len(cands)


def iter_objects(mm, start, end):
    """
    Yield each brace-delimited object between byte offsets start and end,
    honouring nesting. Boundaries are found with mmap.find (C-level).
    """
    pos = start
    depth = 0
    obj_start = None
    while pos < end:
        o = mm.find(b'{', pos, end)
        c = mm.find(b'}', pos, end)
        if o == -1 and c == -1:
            return
        if o != -1 and (c == -1 or o < c):
            if depth == 0:
                obj_start = o
            depth += 1
            pos = o + 1
        else:
            depth -= 1
            pos = c + 1
            if depth == 0 and obj_start is not None:
                yield json.loads(mm[obj_start:c + 1])
                obj_start = None


class CocoFile:
    """
    Context manager exposing the top-level arrays of a COCO file as streams.

        with CocoFile(path) as coco:
            print(coco.report())
            for img in coco.stream('images'):
                ...
    """

    KEYS = ('images', 'annotations', 'categories')

    def __init__(self, path):
        self.path = str(path)
        self._fh = None
        self._mm = None
        self.ranges = {}
        self.candidate_counts = {}

    def __enter__(self):
        self._fh = open(self.path, 'rb')
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        for k in self.KEYS:
            rng, n_cand = find_toplevel_array(self._mm, k)
            self.ranges[k] = rng
            self.candidate_counts[k] = n_cand
        return self

    def __exit__(self, *exc):
        if self._mm is not None:
            self._mm.close()
        if self._fh is not None:
            self._fh.close()

    def stream(self, key):
        rng = self.ranges.get(key)
        if rng is None:
            return iter(())
        return iter_objects(self._mm, rng[0], rng[1])

    def report(self):
        lines = [f"file: {self.path}  ({len(self._mm)/1e9:.3f} GB)"]
        for k in self.KEYS:
            rng = self.ranges[k]
            nc = self.candidate_counts[k]
            if rng is None:
                lines.append(f"  {k:12s} : NOT FOUND at top level "
                             f"({nc} textual occurrence(s))")
            else:
                extra = (f"   [{nc-1} further occurrence(s) are NESTED and "
                         f"correctly ignored]" if nc > 1 else "")
                lines.append(f"  {k:12s} : bytes {rng[0]:,}..{rng[1]:,}{extra}")
        return "\n".join(lines)
