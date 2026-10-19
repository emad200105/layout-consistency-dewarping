# Trained weights

| file | λ (layout loss weight) | training | MD5 |
|---|---|---|---|
| `dewarpnet_layout_lambda0.2.pt` | 0.2 (ours) | 30 epochs, 531,990 steps | `b4cd3ce3b76f1faec01e5472846f30e6` |
| `dewarpnet_ablation_lambda0.pt` | 0 (ablation) | 30 epochs, 531,990 steps | `b70d7ebde8f55a39ee9dd9184fcdaf96` |

Both are the final checkpoints of the paper's two runs (`ckpt_step000531990.pt`),
reduced to the model weights. The optimizer state was removed: 26 MB instead of
78 MB. A forward pass gives exactly the same output as the full checkpoint
(max |difference| = 0).

The architecture is `DewarpNet(base=32)` from `training/model.py`, with 6.45 M
parameters:

- **Input:** an RGB image resized to 448×576 (W×H), values in [0, 1].
- **Output:** a backward flow on a 112×144 grid in normalized `grid_sample`
  units.

Each file is a `torch.save` dict with the keys `model` (state_dict), `epoch`,
`gstep`, `w_layout`, `base` and `note`. Load it with:

```python
import sys; sys.path[:0] = ['training', 'augmentation']
from inference import load_checkpoint, dewarp
import cv2
net, info = load_checkpoint('weights/dewarpnet_layout_lambda0.2.pt')   # CPU by default
rectified, valid, flow = dewarp(net, cv2.imread('photo.jpg'))
```

or from the command line:

```bash
python training/inference.py --ckpt weights/dewarpnet_layout_lambda0.2.pt --image photo.jpg --out fixed.png
```

The files are pickled PyTorch objects: load only copies you obtained from a
source you trust. `torch.load(..., weights_only=False)` is used because the dict
holds a few Python scalars next to the tensors.
