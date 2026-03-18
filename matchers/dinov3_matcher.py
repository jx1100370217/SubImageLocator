"""DINOv3 dense patch feature matching for sub-image localization."""

import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchResult:
    """Result of sub-image localization."""
    found: bool
    confidence: float
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    x_min_pct: float
    y_min_pct: float
    x_max_pct: float
    y_max_pct: float
    elapsed_ms: float
    method: str

    @property
    def center_x_pct(self) -> float:
        return (self.x_min_pct + self.x_max_pct) / 2.0

    @property
    def center_y_pct(self) -> float:
        return (self.y_min_pct + self.y_max_pct) / 2.0


# Global model cache
_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_patch_size = 16
_feature_dim = 768
_model_name = "vit_base_patch16_dinov3"


def _load_model():
    global _model, _patch_size, _feature_dim
    if _model is not None:
        return _model
    import timm
    print(f"[DINOv3] Loading {_model_name} via timm ...", flush=True)
    _model = timm.create_model(_model_name, pretrained=True, num_classes=0)
    _model = _model.eval().to(_device)
    if hasattr(_model, "patch_embed"):
        ps = getattr(_model.patch_embed, "patch_size", None)
        if ps is not None:
            _patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
    if hasattr(_model, "embed_dim"):
        _feature_dim = _model.embed_dim
    print(f"[DINOv3] Loaded (device={_device}, patch_size={_patch_size}, dim={_feature_dim})", flush=True)
    return _model


def preload_models():
    """Preload DINOv3 model at startup."""
    _load_model()


def _prepare_tensor(image: np.ndarray, target_size: int = 518):
    """Resize + normalize image → tensor. Returns (tensor, resized_h, resized_w)."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = target_size / max(h, w)
    new_h = max(_patch_size, (int(h * scale) // _patch_size) * _patch_size)
    new_w = max(_patch_size, (int(w * scale) // _patch_size) * _patch_size)
    img_res = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_res.astype(np.float32) / 255.0 - mean) / std
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(_device)
    return tensor, new_h, new_w


def _extract_features(image: np.ndarray, target_size: int = 518):
    """Extract DINOv3 patch token features. Returns (feats [n_ph*n_pw, dim], n_ph, n_pw)."""
    model = _load_model()
    tensor, rh, rw = _prepare_tensor(image, target_size)
    n_ph = rh // _patch_size
    n_pw = rw // _patch_size
    with torch.no_grad():
        feats = model.forward_features(tensor)      # [1, prefix+n_patches, dim]
        n_patches = n_ph * n_pw
        n_prefix = feats.shape[1] - n_patches
        patch_tokens = feats[:, n_prefix:, :] if n_prefix > 0 else feats
    return patch_tokens[0], n_ph, n_pw  # [n_patches, dim]


def match_features(
    image: np.ndarray,
    template: np.ndarray,
    min_matches: int = 50,          # kept for API compat, unused in DINOv3
    confidence_threshold: float = 0.6,
):
    """
    Locate template in image using DINOv3 dense patch features + sliding window.

    Returns:
        (MatchResult, None, None)   # No keypoint arrays for DINOv3
    """
    t0 = time.perf_counter()

    orig_h, orig_w = image.shape[:2]
    crop_h, crop_w = template.shape[:2]

    cam_feats, cam_ph, cam_pw = _extract_features(image, target_size=518)
    crop_target = max(_patch_size * 2,
                      min(280, int(518 * max(crop_h, crop_w) / max(orig_h, orig_w))))
    crop_feats, crop_ph, crop_pw = _extract_features(template, target_size=crop_target)

    dim = cam_feats.shape[-1]
    cam_grid  = F.normalize(cam_feats.reshape(cam_ph, cam_pw, dim),   dim=-1)
    crop_grid = F.normalize(crop_feats.reshape(crop_ph, crop_pw, dim), dim=-1)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if crop_ph > cam_ph or crop_pw > cam_pw:
        return MatchResult(
            found=False, confidence=0.0,
            x_min=0, y_min=0, x_max=0, y_max=0,
            x_min_pct=0, y_min_pct=0, x_max_pct=0, y_max_pct=0,
            elapsed_ms=round(elapsed_ms, 1),
            method=f"DINOv3 (crop {crop_ph}x{crop_pw} > cam {cam_ph}x{cam_pw})",
        ), None, None

    # Sliding window via unfold
    cam_4d   = cam_grid.permute(2, 0, 1).unsqueeze(0)          # [1, dim, cam_ph, cam_pw]
    crop_flat = crop_grid.reshape(-1, dim)                       # [crop_ph*crop_pw, dim]
    out_h = cam_ph - crop_ph + 1
    out_w = cam_pw - crop_pw + 1

    windows = cam_4d.unfold(2, crop_ph, 1).unfold(3, crop_pw, 1)
    windows = windows[0].permute(1, 2, 3, 4, 0)                 # [out_h, out_w, crop_ph, crop_pw, dim]
    windows = windows.reshape(out_h, out_w, -1, dim)
    sim = (windows * crop_flat.unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean(dim=-1)  # [out_h, out_w]

    sim_np = sim.cpu().numpy()
    by, bx = np.unravel_index(sim_np.argmax(), sim_np.shape)
    best_score = float(sim_np[by, bx])

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Patch → pixel coords
    cam_res_h = cam_ph * _patch_size
    cam_res_w = cam_pw * _patch_size
    sx = orig_w / cam_res_w
    sy = orig_h / cam_res_h
    x_min = max(0, min(int(bx * _patch_size * sx), orig_w))
    y_min = max(0, min(int(by * _patch_size * sy), orig_h))
    x_max = max(0, min(int((bx + crop_pw) * _patch_size * sx), orig_w))
    y_max = max(0, min(int((by + crop_ph) * _patch_size * sy), orig_h))

    confidence = max(0.0, min(1.0, best_score))
    found = confidence >= confidence_threshold

    return MatchResult(
        found=found,
        confidence=round(confidence, 4),
        x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max,
        x_min_pct=round(x_min / orig_w * 100, 2),
        y_min_pct=round(y_min / orig_h * 100, 2),
        x_max_pct=round(x_max / orig_w * 100, 2),
        y_max_pct=round(y_max / orig_h * 100, 2),
        elapsed_ms=round(elapsed_ms, 1),
        method=f"DINOv3 (score={best_score:.4f}, cam={cam_ph}x{cam_pw}, crop={crop_ph}x{crop_pw})",
    ), None, None
