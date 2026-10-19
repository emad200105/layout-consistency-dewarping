"""
Where the figure / analysis scripts find their inputs. Everything is relative to
the repository root, and the two large external inputs can be redirected with
environment variables:

  DEWARP_OUTPUTS  folder for generated, non-versioned files   (default: <repo>/outputs)
                    ocr_subset.npz              training/extract_ocr_subset.py
                    subset_boxes_flowgrid.json  paper/scripts/export_subset_boxes.py
                    docunet/{distorted,ours,ablation}/<i>_<k>_rec.png
                                                evaluation/docunet/{make_input_set,rectify}.py
  DOCUNET_DATA    the unpacked DocUNet benchmark, with crop/ and scan/
                                                              (default: <repo>/data/docunet)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / 'paper' / 'figs'
RESULTS = ROOT / 'results'
WEIGHTS = {'layout': ROOT / 'weights' / 'dewarpnet_layout_lambda0.2.pt',
           'ablation': ROOT / 'weights' / 'dewarpnet_ablation_lambda0.pt'}
OUTPUTS = Path(os.environ.get('DEWARP_OUTPUTS', ROOT / 'outputs'))
OCR_SUBSET = OUTPUTS / 'ocr_subset.npz'
SUBSET_BOXES = OUTPUTS / 'subset_boxes_flowgrid.json'
DOCUNET_OUT = OUTPUTS / 'docunet'
DOCUNET_DATA = Path(os.environ.get('DOCUNET_DATA', ROOT / 'data' / 'docunet'))

for p in (ROOT / 'training', ROOT / 'augmentation', Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def need(path, how):
    if not Path(path).exists():
        raise SystemExit(f'missing {path}\n  make it with: {how}')
    return path
