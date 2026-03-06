"""Feature matching using SuperPoint + LightGlue with model caching."""

import cv2
import numpy as np
import time
import torch
from typing import Optional, Tuple

from .template_matcher import MatchResult

device = "cuda" if torch.cuda.is_available() else "cpu"

# Global model cache
_extractor = None
_matcher = None


def _load_models():
    """Load and cache SuperPoint + LightGlue models."""
    global _extractor, _matcher
    if _extractor is None:
        from lightglue import LightGlue, SuperPoint
        print("[cache] Loading SuperPoint...", flush=True)
        _extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
        print("[cache] Loading LightGlue...", flush=True)
        _matcher = LightGlue(features="superpoint").eval().to(device)
        print("[cache] Models loaded!", flush=True)
    return _extractor, _matcher


def preload_models():
    """Preload models at startup."""
    _load_models()


def _extract_features(extractor, image: np.ndarray) -> dict:
    """Extract SuperPoint features from an image."""
    from lightglue.utils import numpy_image_to_torch
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    tensor = numpy_image_to_torch(gray).to(device)
    with torch.no_grad():
        feats = extractor.extract(tensor)
    return feats


def match_features(
    image: np.ndarray,
    template: np.ndarray,
    min_matches: int = 8,
    confidence_threshold: float = 0.3,
) -> Tuple[MatchResult, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Locate template in image using SuperPoint + LightGlue.

    Returns:
        (MatchResult, matched_keypoints_in_image, matched_keypoints_in_template)
    """
    t0 = time.perf_counter()

    extractor, matcher = _load_models()
    img_h, img_w = image.shape[:2]

    # Extract features
    feats0 = _extract_features(extractor, image)
    feats1 = _extract_features(extractor, template)

    # Match
    with torch.no_grad():
        matches_result = matcher({"image0": feats0, "image1": feats1})

    # Get matched keypoints
    # LightGlue returns matches0: [1, N0] where value is index into kpts1 or -1
    kpts0 = feats0["keypoints"][0].cpu().numpy()  # [N0, 2]
    kpts1 = feats1["keypoints"][0].cpu().numpy()  # [N1, 2]
    matches0 = matches_result["matches0"][0].cpu().numpy()  # [N0]

    valid = matches0 > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches0[valid]]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    n_matches = len(mkpts0)

    if n_matches < min_matches:
        return MatchResult(
            found=False,
            confidence=0.0,
            x_min=0, y_min=0, x_max=0, y_max=0,
            x_min_pct=0, y_min_pct=0, x_max_pct=0, y_max_pct=0,
            elapsed_ms=round(elapsed_ms, 1),
            method=f"SuperPoint+LightGlue ({n_matches} matches)",
        ), mkpts0, mkpts1

    # Compute homography to find template region in image
    H, mask = cv2.findHomography(mkpts1, mkpts0, cv2.USAC_MAGSAC, 5.0)

    if H is None:
        return MatchResult(
            found=False,
            confidence=0.0,
            x_min=0, y_min=0, x_max=0, y_max=0,
            x_min_pct=0, y_min_pct=0, x_max_pct=0, y_max_pct=0,
            elapsed_ms=round(elapsed_ms, 1),
            method=f"SuperPoint+LightGlue ({n_matches} matches, H failed)",
        ), mkpts0, mkpts1

    tpl_h, tpl_w = template.shape[:2]
    corners = np.float32([
        [0, 0], [tpl_w, 0], [tpl_w, tpl_h], [0, tpl_h]
    ]).reshape(-1, 1, 2)

    projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x_min = max(0, int(projected[:, 0].min()))
    y_min = max(0, int(projected[:, 1].min()))
    x_max = min(img_w, int(projected[:, 0].max()))
    y_max = min(img_h, int(projected[:, 1].max()))

    # Confidence: ratio of inliers
    inlier_ratio = mask.sum() / len(mask) if mask is not None else 0
    confidence = float(inlier_ratio)

    return MatchResult(
        found=confidence >= confidence_threshold,
        confidence=round(confidence, 4),
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        x_min_pct=round(x_min / img_w * 100, 2),
        y_min_pct=round(y_min / img_h * 100, 2),
        x_max_pct=round(x_max / img_w * 100, 2),
        y_max_pct=round(y_max / img_h * 100, 2),
        elapsed_ms=round(elapsed_ms, 1),
        method=f"SuperPoint+LightGlue ({n_matches} matches, {int(inlier_ratio*100)}% inliers)",
    ), mkpts0, mkpts1
