"""Visualization utilities for sub-image localization."""

import cv2
import numpy as np
from typing import Optional, Tuple
from matchers.dinov3_matcher import MatchResult

MAX_OUTPUT_DIM = 800


def _limit_size(img: np.ndarray, max_dim: int = MAX_OUTPUT_DIM) -> np.ndarray:
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
    return vis_img, None   # DINOv3 无关键点可视化


def format_position_text(result: MatchResult) -> str:
    if not result.found:
        return f"❌ 未找到匹配区域\n\n📊 **方法:** {result.method}\n**置信度:** {result.confidence:.4f}\n**耗时:** {result.elapsed_ms:.1f}ms"

    return f"""✅ 匹配成功！

📍 **子图在原图中的位置（百分比）：**
  - 左上角: ({result.x_min_pct}%, {result.y_min_pct}%)
  - 右下角: ({result.x_max_pct}%, {result.y_max_pct}%)
  - 中心: ({result.center_x_pct:.2f}%, {result.center_y_pct:.2f}%)

📍 **像素坐标：**
  - 左上角: ({result.x_min}, {result.y_min})
  - 右下角: ({result.x_max}, {result.y_max})

📊 **匹配信息：**
  - 方法: {result.method}
  - 置信度: {result.confidence:.4f}
  - 耗时: {result.elapsed_ms:.1f}ms"""
