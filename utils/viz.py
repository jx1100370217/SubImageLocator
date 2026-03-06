"""Visualization utilities for sub-image localization."""

import cv2
import numpy as np
from typing import Optional, Tuple
from matchers.template_matcher import MatchResult

# Max output dimension to keep Gradio transfer fast
MAX_OUTPUT_DIM = 800


def _limit_size(img: np.ndarray, max_dim: int = MAX_OUTPUT_DIM) -> np.ndarray:
    """Resize image if any dimension exceeds max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_match_result(
    image: np.ndarray,
    template: np.ndarray,
    result: MatchResult,
    mkpts0: Optional[np.ndarray] = None,
    mkpts1: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Draw match results on images. Returns downsized images for fast transfer."""
    vis_img = image.copy()

    if result.found:
        cv2.rectangle(
            vis_img,
            (result.x_min, result.y_min),
            (result.x_max, result.y_max),
            (0, 255, 0), 3
        )
        label = f"Conf: {result.confidence:.3f}"
        cv2.putText(
            vis_img, label,
            (result.x_min, max(result.y_min - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
    else:
        cv2.putText(
            vis_img, "No match found",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
        )

    vis_img = _limit_size(vis_img)

    matches_vis = None
    if mkpts0 is not None and mkpts1 is not None and len(mkpts0) > 0:
        matches_vis = draw_matches_side_by_side(image, template, mkpts0, mkpts1)
        matches_vis = _limit_size(matches_vis, max_dim=1200)

    return vis_img, matches_vis


def draw_matches_side_by_side(
    img0: np.ndarray,
    img1: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    max_matches: int = 200,
) -> np.ndarray:
    """Draw matched keypoints side by side."""
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]

    target_h = max(h0, h1)
    scale0 = target_h / h0
    scale1 = target_h / h1

    new_w0 = int(w0 * scale0)
    new_w1 = int(w1 * scale1)

    resized0 = cv2.resize(img0, (new_w0, target_h))
    resized1 = cv2.resize(img1, (new_w1, target_h))

    canvas = np.concatenate([resized0, resized1], axis=1)

    kpts0_scaled = kpts0 * scale0
    kpts1_scaled = kpts1 * scale1
    kpts1_scaled[:, 0] += new_w0

    n = min(len(kpts0_scaled), max_matches)
    colors = np.random.RandomState(42).randint(0, 255, (n, 3))

    for i in range(n):
        pt0 = tuple(kpts0_scaled[i].astype(int))
        pt1 = tuple(kpts1_scaled[i].astype(int))
        color = tuple(int(c) for c in colors[i])
        cv2.line(canvas, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, pt0, 4, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt1, 4, color, -1, cv2.LINE_AA)

    return canvas


def format_position_text(result: MatchResult) -> str:
    """Format position as readable text."""
    if not result.found:
        return "❌ 未找到匹配区域"

    return f"""✅ 匹配成功！

📍 **子图在原图中的位置（百分比）：**
  - 左上角: ({result.x_min_pct}%, {result.y_min_pct}%)
  - 右下角: ({result.x_max_pct}%, {result.y_max_pct}%)
  - 宽度范围: {result.x_min_pct}% ~ {result.x_max_pct}%
  - 高度范围: {result.y_min_pct}% ~ {result.y_max_pct}%

📍 **像素坐标：**
  - 左上角: ({result.x_min}, {result.y_min})
  - 右下角: ({result.x_max}, {result.y_max})

📊 **匹配信息：**
  - 方法: {result.method}
  - 置信度: {result.confidence:.4f}
  - 耗时: {result.elapsed_ms:.1f}ms"""
