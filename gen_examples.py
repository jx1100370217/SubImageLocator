"""Generate example sub-images by cropping from source images."""

import cv2
import os
import numpy as np
import shutil


def generate_examples(
    src_dir: str = "/media/ubuntu/Data/codes/jianxiong/GIM-online/datasets/gim",
    out_dir: str = "datasets/examples",
):
    """Generate example pairs (original + cropped sub-image)."""
    os.makedirs(out_dir, exist_ok=True)

    # Copy source images and create crops
    pairs = [("0a.png", "0b.png"), ("1a.png", "1b.png"), ("2a.png", "2b.png"), ("3a.png", "3b.png")]

    examples = []
    for i, (fa, fb) in enumerate(pairs):
        src_a = os.path.join(src_dir, fa)
        if not os.path.exists(src_a):
            print(f"[warn] {src_a} not found, skipping", flush=True)
            continue

        img = cv2.imread(src_a)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Save original
        orig_path = os.path.join(out_dir, f"orig_{i}.png")
        cv2.imwrite(orig_path, img)

        # Create cropped sub-image (random region ~30-50% of image)
        np.random.seed(i + 42)
        crop_w = int(w * np.random.uniform(0.3, 0.5))
        crop_h = int(h * np.random.uniform(0.3, 0.5))
        x0 = np.random.randint(0, w - crop_w)
        y0 = np.random.randint(0, h - crop_h)
        crop = img[y0:y0+crop_h, x0:x0+crop_w]

        crop_path = os.path.join(out_dir, f"crop_{i}.png")
        cv2.imwrite(crop_path, crop)

        # Also copy the b image as a "different view" sub-image for feature matching test
        src_b = os.path.join(src_dir, fb)
        if os.path.exists(src_b):
            view_path = os.path.join(out_dir, f"view_{i}.png")
            shutil.copy2(src_b, view_path)
            examples.append((orig_path, view_path, "SuperPoint+LightGlue"))

        examples.append((orig_path, crop_path, "Template"))

        print(f"[examples] Generated pair {i}: orig={orig_path}, crop={crop_path}", flush=True)

    return examples


if __name__ == "__main__":
    generate_examples()
