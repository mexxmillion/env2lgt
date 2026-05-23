# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""HDR -> 8-bit display conversion for the viewer.

This is for *display only* — the underlying float buffer is never modified.
Operators implemented: linear exposure + ACES Filmic approximation (Krzysztof
Narkowicz) + sRGB encode. Also a turbo-colormap helper for the depth view.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QImage


def aces_filmic(x: np.ndarray) -> np.ndarray:
    """Narkowicz 2015 ACES fit (sRGB output)."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)


def encode_srgb(x: np.ndarray) -> np.ndarray:
    """Linear -> sRGB encode (IEC 61966-2-1 piecewise)."""
    lo = 12.92 * x
    hi = 1.055 * np.power(np.maximum(x, 1e-9), 1.0 / 2.4) - 0.055
    return np.where(x <= 0.0031308, lo, hi)


def to_display_qimage(hdr: np.ndarray, exposure: float = 0.0, use_aces: bool = True) -> QImage:
    """Tonemap an HDR (H, W, 3) float buffer and return a QImage (RGB888).

    `exposure` is in stops (2 ** exposure multiplier).
    """
    if hdr.ndim != 3 or hdr.shape[-1] != 3:
        raise ValueError(f"Expected (H, W, 3), got {hdr.shape}")
    scaled = hdr * (2.0 ** exposure)
    if use_aces:
        ldr = aces_filmic(scaled)
    else:
        ldr = encode_srgb(np.clip(scaled, 0.0, 1.0))
    u8 = (ldr * 255.0 + 0.5).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    h, w, _ = u8.shape
    return QImage(u8.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()


def depth_to_display_qimage(distance: np.ndarray, invert: bool = False) -> QImage:
    """Turbo-colormap a per-pixel distance map for visual inspection.

    `distance` is (H, W) float32 from DA-2. Range is per-image normalized
    (DA-2's output is scale-invariant). `invert` flips so near=hot when True.
    """
    import cv2

    if distance.ndim != 2:
        raise ValueError(f"Expected (H, W), got {distance.shape}")
    d = distance.astype(np.float32)
    lo, hi = float(d.min()), float(d.max())
    span = max(1e-6, hi - lo)
    norm = (d - lo) / span
    if invert:
        norm = 1.0 - norm
    u8 = (np.clip(norm, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    cm_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)
    cm_rgb = np.ascontiguousarray(cm_rgb)
    h, w, _ = cm_rgb.shape
    return QImage(cm_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
