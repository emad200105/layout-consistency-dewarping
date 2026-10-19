"""
train.py
========
Training loop for the dewarping network. Built for Kaggle: mixed precision,
a wall-clock budget, and resume that actually resumes.

    # smoke test
    python training/train.py --root ds_pilot --steps 20 --batch 2 --out runs/smoke

    # Kaggle
    python training/train.py --root /kaggle/input/publaynet-dewarp/dataset ^
                             --out /kaggle/working/run1 --batch 8 --max_hours 11.0

    # the ablation
    python training/train.py ... --w_layout 0 --out runs/no_layout

SURVIVING KAGGLE'S 12-HOUR CAP
------------------------------
A session is killed at 12 hours whatever it is doing, and the full dataset is
~10 epochs of work per session, so training MUST span sessions. `--max_hours`
stops cleanly before the cap and writes a checkpoint; `--resume auto` picks the
newest one up and continues.

Resume restores the optimizer, the AMP scaler, the LR schedule position, the
epoch, the global step, AND the RNG states for python/numpy/torch. Without the
RNG states a resumed run silently reshuffles and re-augments differently, which
makes a resumed curve incomparable to an uninterrupted one — the sort of thing
that is invisible until an ablation disagrees with itself.

Checkpoints are pruned to the last two plus the best, because /kaggle/working
is 20 GB and a checkpoint is ~78 MB (weights plus Adam's two moments).

WHAT TO WATCH
-------------
`epe_px` — mean end-point error in full-resolution pixels — is the headline. It
is the Local Distortion proxy and it is directly comparable to published
numbers. The identity warp ("predict nothing") scores about 35 px on this data,
so that is the bar to clear before anything else matters.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'augmentation'))

from dewarp_dataset import DewarpTrainingSet, collate          # noqa: E402


from model import DewarpNet, count_params                      # noqa: E402
from losses import DewarpLoss                                  # noqa: E402


class ResumableSampler(torch.utils.data.Sampler):
    """
    A shuffle that can be resumed exactly in the middle of an epoch.

    torch's default shuffling draws from the global RNG when the loader is
    iterated, so restoring RNG state is not enough: a run resumed mid-epoch
    restarts that epoch with a DIFFERENT order, seeing some samples twice and
    skipping others. The damage is invisible — the loss curve looks fine — but
    a resumed run is then not comparable to an uninterrupted one, which is
    fatal for an ablation that will be resumed several times across Kaggle
    sessions.

    Here the permutation for epoch N is a pure function of (seed, N), so it can
    be reproduced from scratch and sliced. `start` skips the already-consumed
    indices without loading them.
    """

    def __init__(self, n, seed, epoch=0, start=0, shuffle=True):
        self.n, self.seed, self.shuffle = n, seed, shuffle
        self.epoch, self.start = epoch, start

    def set_epoch(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def _perm(self):
        if not self.shuffle:
            return np.arange(self.n)
        return np.random.RandomState(self.seed + self.epoch).permutation(self.n)

    def __iter__(self):
        return iter(self._perm()[self.start:].tolist())

    def __len__(self):
        return max(self.n - self.start, 0)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rng_state():
    return {'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def load_rng(st):
    if not st:
        return
    random.setstate(st['python'])
    np.random.set_state(st['numpy'])
    torch.set_rng_state(st['torch'].cpu() if torch.is_tensor(st['torch']) else st['torch'])
    if st.get('cuda') and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in st['cuda']])


def lr_at(step, total, base_lr, warmup):
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    import math
    p = (step - warmup) / max(total - warmup, 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def to_device(batch, dev):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(dev, non_blocking=True) if torch.is_tensor(v) else v
    return out


def ckpt_step_of(p):
    """Global step encoded in the filename. -1 if it is not a step checkpoint."""
    name = Path(p).stem
    if not name.startswith('ckpt_step'):
        return -1
    try:
        return int(name[len('ckpt_step'):])
    except ValueError:
        return -1


def prune_checkpoints(out_dir, keep=2):
    cks = sorted(Path(out_dir).glob('ckpt_step*.pt'), key=ckpt_step_of)
    for p in cks[:-keep]:
        try:
            p.unlink()
        except OSError:
            pass


def find_checkpoint(out_dir, extra_dirs):
    """
    Newest checkpoint across the output dir and any attached input dirs.

    Ordered by the step in the FILENAME, never by mtime: a Kaggle dataset mount
    rewrites modification times, so mtime would happily pick a stale checkpoint
    and silently rewind training.
    """
    cands = []
    for d in [out_dir] + list(extra_dirs):
        d = Path(d)
        if not d.exists():
            continue
        for p in list(d.glob('ckpt_step*.pt')) + list(d.glob('last.pt')):
            if p.name == 'last.pt':
                try:
                    st = int(torch.load(p, map_location='cpu',
                                        weights_only=False)['gstep'])
                except Exception:                                # noqa: BLE001
                    continue
            else:
                st = ckpt_step_of(p)
            if st >= 0:
                cands.append((st, p))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    return cands[-1][1]


@torch.no_grad()
def validate(net, loader, crit, dev, amp, max_batches=None):
    net.eval()
    agg, n = {}, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = to_device(batch, dev)
        with torch.autocast(device_type=dev.type, enabled=amp):
            out = net(batch['warped'])
            _, logs = crit(out, batch)
        for k, v in logs.items():
            agg[k] = agg.get(k, 0.0) + v
        n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def find_under(base, marker, depth=7):
    """
    Every directory under `base` that directly contains a file named `marker`.

    WHY THIS EXISTS
    ---------------
    Kaggle does not guarantee where an attached dataset mounts. The old
    notebooks used /kaggle/input/<slug>; newer ones nest it, e.g.
    /kaggle/input/datasets/<owner>/<slug>/versions/N. Hard-coding either one
    produces a FileNotFoundError that looks like a missing dataset when in
    fact the dataset is attached and fine.

    Searching for a file we know is inside each store is immune to all of it,
    costs a directory walk of a few hundred entries, and fails loudly with the
    candidates listed when it is ambiguous.
    """
    base = Path(base)
    if not base.exists():
        return []
    hits = []
    for root, dirs, files in os.walk(base):
        if len(Path(root).relative_to(base).parts) > depth:
            dirs[:] = []
            continue
        if marker in files:
            hits.append(root)
    return sorted(hits)


def resolve_input(spec, marker, label, base='/kaggle/input'):
    """Pass a path through unchanged; resolve the literal 'auto' by search."""
    if spec is None or spec != 'auto':
        return spec
    hits = find_under(base, marker)
    if not hits:
        raise SystemExit(
            f"--{label} auto: nothing under {base} contains {marker!r}.\n"
            f"  The dataset is probably not attached yet. Add Data -> your "
            f"dataset, wait for the Input panel to list it, and rerun.")
    if len(hits) > 1:
        print(f"[paths] {label}: {len(hits)} candidates, using the first")
        for h in hits:
            print(f"        {h}")
    print(f"[paths] {label} = {hits[0]}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Dataset root, or 'auto' to locate it under "
                         "/kaggle/input by searching for flows.f16")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout_root", default=None,
                    help="Layout store location. Default <root>/layout. On "
                         "Kaggle the layout store is a separate dataset, so "
                         "point this at its mount — or pass 'auto' to find it "
                         "by searching for stems.jsonl.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=0,
                    help="STOP after N global steps. Does not change the LR "
                         "schedule — see --total_steps.")
    ap.add_argument("--total_steps", type=int, default=0,
                    help="LR schedule horizon. Default: epochs x steps/epoch. "
                         "Keep this FIXED across resumed sessions, otherwise "
                         "each session restarts the cosine and the run is not "
                         "the run you think it is.")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--max_hours", type=float, default=0.0,
                    help="stop cleanly after this many hours (Kaggle's cap is 12)")
    ap.add_argument("--val_every", type=int, default=1000, help="steps")
    ap.add_argument("--ckpt_every", type=int, default=500, help="steps")
    ap.add_argument("--val_batches", type=int, default=60)
    ap.add_argument("--resume", default="auto", help="'auto', 'none', or a path")
    ap.add_argument("--resume_dirs", default="",
                    help="Comma-separated EXTRA directories to search for a "
                         "checkpoint, newest step wins. This is how Kaggle "
                         "continuation works: /kaggle/working is wiped between "
                         "sessions, so the previous session's output has to be "
                         "attached as a read-only dataset and named here. "
                         "'auto' searches /kaggle/input for a last.pt.")
    ap.add_argument("--val_at_start", type=int, default=1,
                    help="validate before training, so a broken run shows up "
                         "in minutes instead of at the 12-hour mark")
    ap.add_argument("--w_flow", type=float, default=1.0)
    ap.add_argument("--w_photo", type=float, default=0.5)
    ap.add_argument("--w_layout", type=float, default=0.2)
    ap.add_argument("--w_aux", type=float, default=0.2)
    ap.add_argument("--layout_mode", default="global")
    a = ap.parse_args()

    # Resolve any 'auto' paths before anything touches the filesystem, so a
    # missing dataset is one clear message rather than a stack trace 200 lines
    # into startup.
    a.root = resolve_input(a.root, 'flows.f16', 'root')
    a.layout_root = resolve_input(a.layout_root, 'stems.jsonl', 'layout_root')
    if a.resume_dirs.strip() == 'auto':
        hits = find_under('/kaggle/input', 'last.pt')
        if not hits:
            raise SystemExit(
                "--resume_dirs auto: no last.pt under /kaggle/input.\n"
                "  Attach the PREVIOUS session's saved output as an input.\n"
                "  If this really is session 1, pass --resume_dirs '' instead "
                "so the intent is explicit.")
        for h in hits:
            print(f"[paths] resume candidate {h}")
        a.resume_dirs = ','.join(hits)

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp = bool(a.amp) and dev.type == 'cuda'
    set_seed(a.seed)

    train_set = DewarpTrainingSet(a.root, split='train',
                                 layout_root=a.layout_root)
    val_set = DewarpTrainingSet(a.root, split='val',
                                layout_root=a.layout_root)

    # An empty split is ALWAYS a configuration error, and without this check it
    # surfaces ~200 lines later as a KeyError inside the validation aggregator,
    # which says nothing about the real cause. The overwhelmingly likely cause
    # on Kaggle is a layout store that does not match the image dataset (wrong
    # dataset attached, or a stale version), which drops 100% of samples.
    if len(train_set) == 0 or len(val_set) == 0:
        man = set(train_set._stems)
        lay = set(train_set.layout.by_stem)
        both = man & lay
        raise SystemExit(
            f"\nEMPTY SPLIT — train {len(train_set)}, val {len(val_set)}.\n"
            f"  dataset root : {a.root}\n"
            f"  layout root  : {a.layout_root}\n"
            f"  manifest stems {len(man):,}, layout stems {len(lay):,}, "
            f"overlap {len(both):,}\n"
            + ("  The two stores share NO page stems: the layout dataset does "
               "not belong to this image dataset.\n"
               f"    manifest e.g. {sorted(man)[:3]}\n"
               f"    layout   e.g. {sorted(lay)[:3]}\n"
               if not both else
               "  Stems overlap, so this is a split-fraction problem, not a "
               "mismatch.\n"))
    sampler = ResumableSampler(len(train_set), a.seed)

    def make_train_loader(epoch, start):
        sampler.set_epoch(epoch, start)
        return DataLoader(train_set, batch_size=a.batch, sampler=sampler,
                          num_workers=a.workers, collate_fn=collate,
                          pin_memory=(dev.type == 'cuda'), drop_last=True)

    val_ld = DataLoader(val_set, batch_size=a.batch, shuffle=False,
                        num_workers=max(a.workers // 2, 0), collate_fn=collate,
                        pin_memory=(dev.type == 'cuda'))

    net = DewarpNet(base=a.base).to(dev)
    crit = DewarpLoss(page_w=train_set.page_w, page_h=train_set.page_h,
                      w_flow=a.w_flow, w_photo=a.w_photo,
                      w_layout=a.w_layout, w_aux=a.w_aux,
                      layout_mode=a.layout_mode)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler(enabled=amp)

    steps_per_epoch = max(len(train_set) // a.batch, 1)
    # The schedule horizon is deliberately independent of where this session
    # happens to stop: a run split across five Kaggle sessions must follow one
    # continuous cosine, not five truncated ones.
    total_steps = a.total_steps or (steps_per_epoch * a.epochs)

    start_epoch, gstep, best, start_in_epoch = 0, 0, float('inf'), 0
    extra = [d.strip() for d in a.resume_dirs.split(',') if d.strip()]
    ck_path = None
    if a.resume == 'auto':
        ck_path = find_checkpoint(out_dir, extra)
        if ck_path is None and extra:
            print(f"[resume] nothing found in {out_dir} or {extra} — "
                  f"STARTING FROM SCRATCH")
    elif a.resume not in ('none', ''):
        ck_path = Path(a.resume)
    if ck_path and ck_path.exists():
        ck = torch.load(ck_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        scaler.load_state_dict(ck['scaler'])
        start_epoch, gstep, best = ck['epoch'], ck['gstep'], ck.get('best', float('inf'))
        start_in_epoch = ck.get('seen_in_epoch', 0)
        load_rng(ck.get('rng'))
        print(f"[resume] {ck_path.name}: epoch {start_epoch}, step {gstep:,}, "
              f"{start_in_epoch:,} samples into the epoch, best epe {best:.4f}")

    cfg = vars(a) | {'params_M': count_params(net) / 1e6, 'device': str(dev),
                     'amp': amp, 'train': len(train_set), 'val': len(val_set),
                     'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps}
    with open(out_dir / 'config.json', 'w') as fh:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                  fh, indent=2)
    log_fh = open(out_dir / 'log.jsonl', 'a')

    print(f"device {dev}  amp {amp}  params {count_params(net)/1e6:.2f}M")
    print(f"train {len(train_set):,}  val {len(val_set):,}  "
          f"{steps_per_epoch:,} steps/epoch")
    print(f"LR schedule horizon {total_steps:,} steps"
          + (f"; this session stops at step {a.steps:,}" if a.steps else ""))
    print(f"loss weights: flow {a.w_flow}  photo {a.w_photo}  "
          f"layout {a.w_layout}  aux {a.w_aux}"
          + ("   [LAYOUT ABLATED]" if a.w_layout == 0 else ""))

    if a.val_at_start:
        v0 = validate(net, val_ld, crit, dev, amp, min(a.val_batches, 20))
        print(f"  [val @ start] epe {v0['epe_px']:.3f}px  flow {v0['flow_l1']:.3f}  "
              f"layout {v0.get('layout', 0):.4f}")
        print(f"                identity warp scores ~35 px; if this run is "
              f"resumed correctly it should already be below that")
        log_fh.write(json.dumps({'step': gstep, 'split': 'val_at_start', **v0}) + '\n')
        log_fh.flush()

    t0 = time.time()
    stop = False
    epoch = start_epoch
    for epoch in range(start_epoch, a.epochs):
        seen = start_in_epoch if epoch == start_epoch else 0
        train_ld = make_train_loader(epoch, seen)
        for batch in train_ld:
            if a.steps and gstep >= a.steps:
                stop = True
                break
            if a.max_hours and (time.time() - t0) / 3600.0 >= a.max_hours:
                print(f"[budget] {a.max_hours}h reached — checkpointing and exiting")
                stop = True
                break
            # Counted only once the batch is actually going to be trained on:
            # incrementing before the break checks would make resume skip a
            # batch the interrupted run never saw.
            seen += a.batch

            lr = lr_at(gstep, total_steps, a.lr, a.warmup)
            for g in opt.param_groups:
                g['lr'] = lr

            batch = to_device(batch, dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                out = net(batch['warped'])
                loss, logs = crit(out, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            gstep += 1

            if gstep % 20 == 0 or gstep == 1:
                el = time.time() - t0
                rec = {'t': round(el, 1), 'epoch': epoch, 'step': gstep, 'lr': lr,
                       'grad_norm': float(gn), **logs}
                log_fh.write(json.dumps(rec) + '\n')
                log_fh.flush()
                print(f"  e{epoch} s{gstep:>6,} lr {lr:.2e} "
                      f"loss {logs['total']:8.4f} epe {logs['epe_px']:7.3f}px "
                      f"flow {logs['flow_l1']:7.3f} "
                      f"lay {logs.get('layout', 0):6.3f} "
                      f"ph {logs.get('photometric', 0):6.4f} "
                      f"[{el/60:.1f}m]")

            if a.ckpt_every and gstep % a.ckpt_every == 0:
                save(out_dir, 'ckpt_step%09d.pt' % gstep, net, opt, scaler,
                     epoch, gstep, best, seen)
                prune_checkpoints(out_dir)

            if a.val_every and gstep % a.val_every == 0:
                v = validate(net, val_ld, crit, dev, amp, a.val_batches)
                rec = {'t': round(time.time() - t0, 1), 'step': gstep,
                       'split': 'val', **v}
                log_fh.write(json.dumps(rec) + '\n')
                log_fh.flush()
                print(f"  [val] step {gstep:,}  epe {v['epe_px']:.3f}px  "
                      f"flow {v['flow_l1']:.3f}  layout {v.get('layout', 0):.4f}")
                if v['epe_px'] < best:
                    best = v['epe_px']
                    save(out_dir, 'best.pt', net, opt, scaler, epoch, gstep, best, seen)
                    print(f"        new best {best:.4f} px")
        if stop:
            break

        # End of a clean epoch: checkpoint and validate unconditionally, so
        # there is always a resumable point on an epoch boundary regardless of
        # where the step counters happened to land.
        save(out_dir, 'ckpt_step%09d.pt' % gstep, net, opt, scaler,
             epoch + 1, gstep, best, 0)
        prune_checkpoints(out_dir)
        v = validate(net, val_ld, crit, dev, amp, a.val_batches)
        print(f"  [epoch {epoch} done] step {gstep:,}  val epe {v['epe_px']:.3f}px  "
              f"flow {v['flow_l1']:.3f}  layout {v.get('layout', 0):.4f}")
        log_fh.write(json.dumps({'t': round(time.time() - t0, 1), 'step': gstep,
                                 'epoch': epoch, 'split': 'val_epoch', **v}) + '\n')
        log_fh.flush()
        if v['epe_px'] < best:
            best = v['epe_px']
            save(out_dir, 'best.pt', net, opt, scaler, epoch + 1, gstep, best, 0,
                 also_last=False)
            print(f"        new best {best:.4f} px")

    save(out_dir, 'ckpt_step%09d.pt' % gstep, net, opt, scaler, epoch, gstep,
         best, seen)
    prune_checkpoints(out_dir)
    v = validate(net, val_ld, crit, dev, amp, a.val_batches)
    print(f"\nfinished at step {gstep:,} after {(time.time()-t0)/60:.1f} min")
    print(f"final val epe {v['epe_px']:.3f} px   best {best:.3f} px")
    print(f"\nTO CONTINUE: save this notebook's output as a dataset, attach it to")
    print(f"the next session, and add:  --resume_dirs auto")
    print(f"(or name the mount explicitly if more than one run is attached)")
    print(f"resume point: {out_dir/'last.pt'}  (epoch {epoch}, step {gstep:,})")
    log_fh.write(json.dumps({'step': gstep, 'split': 'final', **v}) + '\n')
    log_fh.close()


def save(out_dir, name, net, opt, scaler, epoch, gstep, best, seen=0,
         also_last=True):
    """
    Write a checkpoint, and mirror it to `last.pt`.

    `last.pt` exists so the next Kaggle session can point at one stable
    filename instead of guessing which step the previous session reached.
    """
    blob = {'model': net.state_dict(), 'opt': opt.state_dict(),
            'scaler': scaler.state_dict(), 'epoch': epoch, 'gstep': gstep,
            'best': best, 'seen_in_epoch': seen, 'rng': rng_state()}
    torch.save(blob, Path(out_dir) / name)
    if also_last and name != 'last.pt':
        torch.save(blob, Path(out_dir) / 'last.pt')


if __name__ == "__main__":
    main()
