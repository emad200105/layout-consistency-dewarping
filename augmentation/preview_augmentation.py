"""
preview_augmentation.py
========================
Quick visual test of the augmentation pipeline.

Downloads a sample document image (or uses one you provide)
and shows the warped output + flow field visualization side by side.

Usage:
    python preview_augmentation.py --image path/to/your/image.png

If no --image is given, a synthetic blank document is generated for testing.
"""

import cv2
import numpy as np
import argparse
import sys
import os

# Add augmentation folder to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'augmentation'))
from warp_augment import (
    cylindrical_warp, perspective_warp, wrinkle_warp, add_shadow,
    augment_document
)


def make_dummy_document(H=1024, W=768):
    """Create a simple synthetic document image for testing."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 255

    # Simulate text lines
    for y in range(80, H - 60, 35):
        line_width = np.random.randint(int(W * 0.5), int(W * 0.85))
        x_start = np.random.randint(40, 80)
        cv2.line(img, (x_start, y), (x_start + line_width, y), (30, 30, 30), 2)

    # Simulate a title
    cv2.rectangle(img, (60, 30), (W - 60, 65), (10, 10, 10), -1)

    # Simulate a figure box
    cv2.rectangle(img, (60, 400), (W - 200, 650), (180, 200, 220), -1)
    cv2.rectangle(img, (60, 400), (W - 200, 650), (100, 100, 100), 2)

    return img


def visualize_flow(flow: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 2) flow field to an HSV color visualization."""
    fx, fy = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)

    # Normalize
    mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang_norm = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang_norm     # Hue = direction
    hsv[..., 1] = 255          # Saturation = full
    hsv[..., 2] = mag_norm     # Value = magnitude

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def run_preview(image_path: str = None):
    # Load or create a dummy image
    if image_path and os.path.exists(image_path):
        image = cv2.imread(image_path)
        print(f"Loaded: {image_path}")
    else:
        print("No valid image provided — using synthetic dummy document.")
        image = make_dummy_document()

    H, W = image.shape[:2]
    print(f"Image size: {W}x{H}")

    # ── Run each warp individually for visual inspection ──
    results = {}

    print("Applying cylindrical warp...")
    cyl, flow_cyl = cylindrical_warp(image, strength=0.00025)
    results['Cylindrical'] = (cyl, flow_cyl)

    print("Applying perspective warp...")
    persp, flow_persp = perspective_warp(image, max_shift_frac=0.10)
    results['Perspective'] = (persp, flow_persp)

    print("Applying wrinkle warp...")
    wrink, flow_wrink = wrinkle_warp(image, n_control=6, max_disp_frac=0.025)
    results['Wrinkle'] = (wrink, flow_wrink)

    print("Applying shadow...")
    shadow_img = add_shadow(image)
    results['Shadow only'] = (shadow_img, np.zeros((H, W, 2), np.float32))

    print("Applying FULL combined augmentation...")
    full, flow_full, params = augment_document(image)
    results['Full Combined'] = (full, flow_full)
    print(f"  Params: {params}")

    # ── Build visualization grid ──
    thumb_h = 400
    panels = []
    for name, (warped, flow) in results.items():
        # Resize warped image
        scale = thumb_h / H
        tw = int(W * scale)
        warped_thumb = cv2.resize(warped, (tw, thumb_h))
        flow_thumb = cv2.resize(visualize_flow(flow), (tw, thumb_h))

        # Add label
        label = np.zeros((30, tw * 2 + 4, 3), dtype=np.uint8)
        cv2.putText(label, name, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        pair = np.hstack([warped_thumb, np.ones((thumb_h, 4, 3), dtype=np.uint8) * 80, flow_thumb])
        panel = np.vstack([label, pair])
        panels.append(panel)

    # Combine all panels horizontally
    max_h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    grid = np.hstack([p for p in padded])

    # Add column header
    header = np.zeros((28, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, "LEFT = Warped Image | RIGHT = Flow Field (direction=hue, magnitude=brightness)",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 220, 150), 1)
    final = np.vstack([header, grid])

    # Save and display
    out_path = os.path.join(os.path.dirname(__file__), 'preview_output.png')
    cv2.imwrite(out_path, final)
    print(f"\nPreview saved to: {out_path}")

    cv2.imshow("Augmentation Preview — press any key to close", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview augmentation pipeline")
    parser.add_argument("--image", default=None, help="Path to a document image to augment")
    args = parser.parse_args()
    run_preview(args.image)
