"""Template matching using cv2.matchTemplate with multi-scale support."""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class MatchResult:
    """Result of sub-image localization."""
    found: bool
    confidence: float
    # Bounding box in pixels
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    # Bounding box as percentage of original image
    x_min_pct: float
    y_min_pct: float
    x_max_pct: float
    y_max_pct: float
    # Timing
    elapsed_ms: float
    # Method used
    method: str
    # Scale at which best match was found
    best_scale: float = 1.0


METHODS = {
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
}


def match_template(
    image: np.ndarray,
    template: np.ndarray,
    method_name: str = "TM_CCOEFF_NORMED",
    multi_scale: bool = True,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    scale_steps: int = 20,
    confidence_threshold: float = 0.5,
) -> MatchResult:
    """
    Locate template in image using cv2.matchTemplate.

    Args:
        image: Original image (BGR or RGB)
        template: Sub-image to find (BGR or RGB)
        method_name: OpenCV matching method name
        multi_scale: Whether to try multiple scales
        scale_range: (min_scale, max_scale) for multi-scale
        scale_steps: Number of scales to try
        confidence_threshold: Minimum confidence to consider a match

    Returns:
        MatchResult with location and confidence
    """
    t0 = time.perf_counter()

    method = METHODS.get(method_name, cv2.TM_CCOEFF_NORMED)
    is_sqdiff = method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)

    # Convert to grayscale for matching
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    if len(template.shape) == 3:
        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_tpl = template.copy()

    img_h, img_w = gray_img.shape[:2]
    tpl_h, tpl_w = gray_tpl.shape[:2]

    best_val = -1 if not is_sqdiff else float('inf')
    best_loc = (0, 0)
    best_scale = 1.0
    best_tpl_size = (tpl_w, tpl_h)

    if multi_scale:
        scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    else:
        scales = [1.0]

    for scale in scales:
        new_w = int(tpl_w * scale)
        new_h = int(tpl_h * scale)

        # Skip if template is larger than image
        if new_w > img_w or new_h > img_h or new_w < 10 or new_h < 10:
            continue

        resized_tpl = cv2.resize(gray_tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(gray_img, resized_tpl, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if is_sqdiff:
            if min_val < best_val:
                best_val = min_val
                best_loc = min_loc
                best_scale = scale
                best_tpl_size = (new_w, new_h)
        else:
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
                best_tpl_size = (new_w, new_h)

    # Normalize confidence
    if is_sqdiff:
        confidence = max(0, 1.0 - best_val)
    else:
        confidence = max(0, best_val)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    x_min = best_loc[0]
    y_min = best_loc[1]
    x_max = x_min + best_tpl_size[0]
    y_max = y_min + best_tpl_size[1]

    found = confidence >= confidence_threshold

    return MatchResult(
        found=found,
        confidence=float(confidence),
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        x_min_pct=round(x_min / img_w * 100, 2),
        y_min_pct=round(y_min / img_h * 100, 2),
        x_max_pct=round(x_max / img_w * 100, 2),
        y_max_pct=round(y_max / img_h * 100, 2),
        elapsed_ms=round(elapsed_ms, 1),
        method=f"Template ({method_name})",
        best_scale=round(best_scale, 3),
    )
