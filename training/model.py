"""
model.py
========
The dewarping network: a U-Net encoder over the full 448 x 576 page whose
decoder stops at 112 x 144, where the ground-truth flow lives.

WHY THE DECODER STOPS EARLY
---------------------------
The stored flow is 112 x 144 — quarter resolution — because that is where the
label is accurate to 0.02 px and decoding further would invent detail the
supervision cannot check. Running the decoder back to 448 x 576 would add the
three most expensive stages of the whole network to produce a prediction no
loss could constrain. At inference the predicted flow is upsampled and applied
once at full resolution, exactly as the dataset's own `decode_flow` does.

WHAT IT PREDICTS
----------------
Normalized backward-flow displacement, grid_sample convention, the same units
the dataset stores. Two consequences worth stating:

  * zero output == the identity warp, so the network starts from "do nothing"
    rather than having to learn an identity mapping first;
  * the prediction is resolution-independent, so the same weights apply to a
    full-resolution page at inference without rescaling anything.

GLOBAL CONTEXT
--------------
Page curvature is a global property: the bow of a text line near the spine
depends on where the spine is, which can be 400 px away. A plain U-Net
bottleneck at 14 x 18 has a wide but still local receptive field, so a global
pooling branch is concatenated there — cheap (one 1x1 per channel) and it gives
every decoder position access to page-level statistics.

DEEP SUPERVISION
----------------
Flow regression converges faster when intermediate decoder stages are also
supervised, so the /8 and /16 stages carry their own flow heads. They are
train-time only; `forward` returns them as `aux` and inference ignores them.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(cin, cout, n=2):
    layers = []
    for i in range(n):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout),
                   nn.SiLU(inplace=True)]
    return nn.Sequential(*layers)


class GlobalContext(nn.Module):
    """Squeeze the bottleneck to a page-level vector and broadcast it back."""

    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch), nn.SiLU(inplace=True), nn.Linear(ch, ch))
        self.merge = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.SiLU(inplace=True))

    def forward(self, x):
        g = F.adaptive_avg_pool2d(x, 1).flatten(1)
        g = self.fc(g)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.merge(torch.cat([x, g], 1))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.reduce = nn.Conv2d(cin, cout, 1, bias=False)
        self.block = conv_block(cout + cskip, cout)

    def forward(self, x, skip):
        x = self.reduce(x)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                          align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class DewarpNet(nn.Module):
    """
    Input : (B, 3, 576, 448) warped page, values in [0, 1]
    Output: (B, 2, 144, 112) normalized flow, plus aux heads while training
    """

    def __init__(self, base=32, in_ch=3, deep_supervision=True):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 12
        self.deep_supervision = deep_supervision

        # One cheap full-resolution conv keeps text-line detail available
        # before the first downsample; everything heavier runs at /2 or below.
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base // 2), nn.SiLU(inplace=True))

        self.enc1 = conv_block(base // 2, c1)        # /2    224x288
        self.enc2 = conv_block(c1, c2)               # /4    112x144
        self.enc3 = conv_block(c2, c3)               # /8     56x72
        self.enc4 = conv_block(c3, c4)               # /16    28x36
        self.enc5 = conv_block(c4, c5)               # /32    14x18
        self.pool = nn.MaxPool2d(2)

        self.context = GlobalContext(c5)

        self.up4 = Up(c5, c4, c4)                    # -> /16
        self.up3 = Up(c4, c3, c3)                    # -> /8
        self.up2 = Up(c3, c2, c2)                    # -> /4, the output grid

        self.head = nn.Conv2d(c2, 2, 3, padding=1)
        if deep_supervision:
            self.head8 = nn.Conv2d(c3, 2, 3, padding=1)
            self.head16 = nn.Conv2d(c4, 2, 3, padding=1)

        # Start at the identity warp: zero output means "no displacement", so
        # the network begins from a sane prediction instead of noise.
        for h in [self.head] + ([self.head8, self.head16] if deep_supervision else []):
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    def forward(self, x):
        s = self.stem(x)
        e1 = self.enc1(self.pool(s))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))
        e5 = self.context(e5)

        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)

        out = {'flow': self.head(d2)}
        if self.deep_supervision and self.training:
            out['aux'] = [self.head8(d3), self.head16(d4)]
        return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def apply_flow_norm(img, flow_norm):
    """
    Resample `img` with a normalized backward flow, matching the dataset's
    convention: out[y, x] = img at (base_grid + flow)[y, x].

    The flow is upsampled to the image resolution first, which is exactly what
    inference does — predict at 112 x 144, apply once at full size.
    """
    B, _, H, W = img.shape
    if flow_norm.shape[-2:] != (H, W):
        flow_norm = F.interpolate(flow_norm, size=(H, W), mode='bilinear',
                                  align_corners=True)
    ys = torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype)
    xs = torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype)
    base_y, base_x = torch.meshgrid(ys, xs, indexing='ij')
    gx = base_x.unsqueeze(0) + flow_norm[:, 0]
    gy = base_y.unsqueeze(0) + flow_norm[:, 1]
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border',
                         align_corners=True)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    net = DewarpNet()
    x = torch.randn(2, 3, 576, 448)
    net.train()
    t = time.time()
    o = net(x)
    fwd = time.time() - t
    print(f"params        : {count_params(net)/1e6:.2f} M")
    print(f"input         : {tuple(x.shape)}")
    print(f"flow output   : {tuple(o['flow'].shape)}")
    print(f"aux outputs   : {[tuple(a.shape) for a in o['aux']]}")
    print(f"initial |flow|: {o['flow'].abs().max().item():.3e}  (zero-init head)")
    print(f"fwd (b=2, cpu): {fwd*1000:.0f} ms")
