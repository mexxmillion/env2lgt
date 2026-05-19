"""Colour-checker chart sampling + least-squares colour matching.

The user places a 4-corner quad over a 24-patch colour chart (X-Rite
ColorChecker layout, 6x4) on the panorama. We perspective-rectify the chart,
sample the 24 patches, and solve a colour correction that best maps the
measured swatches onto a reference — either the built-in CC24 values, a
custom JSON target, or swatches sampled from a second (reference) image.

The correction is always expressed as a 3x3 matrix (row-vector convention:
``corrected = rgb @ M``) so exposure-only, white-balance-only and full-matrix
fits share one apply path. This mirrors the MMColorTarget gizmo idea.

Swatch math is ported from the hdr_cal reference (colorchecker_erp.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# CC24 reference, linear sRGB / Rec.709 primaries, D65. Row-major from the
# dark-skin patch (top-left), 6 columns x 4 rows.
CC24_LINEAR_SRGB = np.array([
    [0.4000, 0.3176, 0.2745],  # 01 dark skin
    [0.7608, 0.5804, 0.4941],  # 02 light skin
    [0.3451, 0.4314, 0.5686],  # 03 blue sky
    [0.3373, 0.4196, 0.2706],  # 04 foliage
    [0.5059, 0.4863, 0.6863],  # 05 blue flower
    [0.3098, 0.6627, 0.6196],  # 06 bluish green
    [0.7490, 0.3608, 0.0667],  # 07 orange
    [0.2549, 0.3020, 0.6627],  # 08 purplish blue
    [0.6314, 0.2196, 0.2471],  # 09 moderate red
    [0.2000, 0.1333, 0.2471],  # 10 purple
    [0.5765, 0.6863, 0.1020],  # 11 yellow green
    [0.8471, 0.5608, 0.0471],  # 12 orange yellow
    [0.1529, 0.1882, 0.5961],  # 13 blue
    [0.2510, 0.4902, 0.2078],  # 14 green
    [0.5412, 0.0980, 0.0980],  # 15 red
    [0.9020, 0.7882, 0.0314],  # 16 yellow
    [0.6314, 0.2078, 0.4510],  # 17 magenta
    [0.0353, 0.4706, 0.6314],  # 18 cyan
    [0.9412, 0.9412, 0.9412],  # 19 white
    [0.6196, 0.6196, 0.6196],  # 20 neutral 8
    [0.3647, 0.3647, 0.3647],  # 21 neutral 6.5
    [0.1882, 0.1882, 0.1882],  # 22 neutral 5
    [0.0902, 0.0902, 0.0902],  # 23 neutral 3.5
    [0.0314, 0.0314, 0.0314],  # 24 black
], dtype=np.float32)

# Colorspace the built-in reference values are authored in (an OCIO config
# colorspace name) — used to bring them into the working space.
CC24_REFERENCE_COLORSPACE = "Utility - Linear - sRGB"

_CC_COLS = 6
_CC_ROWS = 4
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)

TARGET_FORMAT = "env2lgt-cc-target"


# ---------- swatch sampling ----------

def _swatch_masks(w: int, h: int, samples: int) -> np.ndarray:
    """(24, 4) [y0, y1, x0, x1] sample windows for a 6x4 grid."""
    masks = []
    off_x = w / _CC_COLS / 2.0
    off_y = h / _CC_ROWS / 2.0
    for j in np.linspace(off_y, h - off_y, _CC_ROWS):
        for i in np.linspace(off_x, w - off_x, _CC_COLS):
            masks.append([int(j - samples), int(j + samples),
                          int(i - samples), int(i + samples)])
    return np.array(masks, dtype=np.int32)


def _bilinear_erp(erp: np.ndarray, x: float, y: float) -> np.ndarray:
    """Bilinear sample of an equirect image — wraps in X, clamps in Y."""
    H, W = erp.shape[:2]
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    fx, fy = x - x0, y - y0
    x0m, x1m = x0 % W, (x0 + 1) % W
    y0c = min(max(y0, 0), H - 1)
    y1c = min(max(y0 + 1, 0), H - 1)
    return (
        erp[y0c, x0m] * (1 - fx) * (1 - fy) + erp[y0c, x1m] * fx * (1 - fy)
        + erp[y1c, x0m] * (1 - fx) * fy + erp[y1c, x1m] * fx * fy
    )


def sample_swatches_spherical(
    erp: np.ndarray, corner_dirs: np.ndarray, sub: int = 5
) -> np.ndarray:
    """Sample the 24 chart patches from an equirect panorama.

    The chart corners are 4 unit directions (TL, TR, BR, BL). Each patch
    centre is found by spherical-bilinear blend of the corners, so the swatch
    grid follows the equirect warp exactly (no flat-perspective error). A
    `sub` x `sub` neighbourhood is averaged per patch. Returns (24, 3)."""
    from env2lgt.proj import angles_from_dir, angles_to_pix, spherical_bilinear

    erp = np.asarray(erp, dtype=np.float32)
    H, W = erp.shape[:2]
    corners = np.asarray(corner_dirs, dtype=np.float64).reshape(4, 3)
    # Sample window ~0.22 of a cell, in parametric units.
    hw_u = 0.22 / _CC_COLS
    hw_v = 0.22 / _CC_ROWS
    offs = np.linspace(-1.0, 1.0, max(1, sub))
    swatches = np.zeros((24, 3), dtype=np.float32)
    for j in range(_CC_ROWS):
        for i in range(_CC_COLS):
            cu = (i + 0.5) / _CC_COLS
            cv = (j + 0.5) / _CC_ROWS
            acc = np.zeros(3, dtype=np.float64)
            for du in offs:
                for dv in offs:
                    d = spherical_bilinear(corners, cu + du * hw_u, cv + dv * hw_v)
                    yaw, pitch = angles_from_dir(d)
                    x, y = angles_to_pix(np.asarray(yaw), np.asarray(pitch), W, H)
                    acc += _bilinear_erp(erp, float(x), float(y))
            swatches[j * _CC_COLS + i] = acc / (offs.size * offs.size)
    return swatches


def rectify_swatches(
    img: np.ndarray, corners_px: np.ndarray, rect_w: int = 600, rect_h: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-rectify a chart and sample its 24 patches — for a *flat*
    (regular 2D) image, e.g. a reference photograph.

    `corners_px` is (4, 2) — the chart corners in image pixels, ordered
    TL, TR, BR, BL (TL = the dark-skin patch). Returns (swatches (24,3),
    rectified image)."""
    src = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [rect_w, 0], [rect_w, rect_h], [0, rect_h]], dtype=np.float32
    )
    M = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(
        img, M, (rect_w, rect_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    # Sample window ~half a quarter-cell so it stays clear of patch borders.
    cell = min(rect_w / _CC_COLS, rect_h / _CC_ROWS)
    samples = max(2, int(cell * 0.22))
    masks = _swatch_masks(rect_w, rect_h, samples)
    swatches = np.zeros((24, 3), dtype=np.float32)
    for i in range(24):
        y0, y1, x0, x1 = masks[i]
        y0, x0 = max(0, y0), max(0, x0)
        y1, x1 = min(rect_h, y1), min(rect_w, x1)
        if y1 > y0 and x1 > x0:
            swatches[i] = rect[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
    return swatches, rect


def _neutral_corr(measured: np.ndarray, reference: np.ndarray) -> float:
    """Pearson correlation of the 6 neutral patches' log-luma — used to
    detect a 180-degree placement flip."""
    lm = np.log(np.clip(measured[18:24] @ _LUMA, 1e-6, None))
    lr = np.log(np.clip(reference[18:24] @ _LUMA, 1e-6, None))
    lm = lm - lm.mean()
    lr = lr - lr.mean()
    denom = float(np.sqrt((lm * lm).sum() * (lr * lr).sum()))
    return float((lm * lr).sum() / denom) if denom > 1e-8 else 0.0


# ---------- solve ----------

def solve_correction(
    measured: np.ndarray, reference: np.ndarray, mode: str = "matrix"
) -> tuple[np.ndarray, float, bool]:
    """Solve a 3x3 colour correction so ``measured @ M`` best matches reference.

    mode:
      "exposure" — single scalar gain (M = s * I)
      "wb"       — per-channel diagonal from the neutral patches
      "matrix"   — full 3x3 least-squares fit over all 24 patches

    Returns (M (3,3) float32, RMSE, flipped) where `flipped` is True if the
    chart was detected as placed upside-down and the swatches were reversed.
    """
    m = np.asarray(measured, dtype=np.float64).reshape(24, 3)
    r = np.asarray(reference, dtype=np.float64).reshape(24, 3)

    # The chart can be placed rotated 180 degrees — the 6x4 grid then reverses.
    flipped = _neutral_corr(m[::-1], r) > _neutral_corr(m, r)
    if flipped:
        m = m[::-1]

    if mode == "exposure":
        num = float((r * m).sum())
        den = float((m * m).sum())
        s = num / den if den > 1e-12 else 1.0
        M = np.eye(3) * s
    elif mode == "wb":
        # Neutrals 19-23 (skip pure black, patch 24, which is noise-prone).
        neu_m = np.clip(m[18:23], 1e-6, None)
        neu_r = r[18:23]
        d = (neu_r / neu_m).mean(axis=0)
        M = np.diag(d)
    else:
        M, _, _, _ = np.linalg.lstsq(m, r, rcond=None)

    rmse = float(np.sqrt(np.mean((m @ M - r) ** 2)))
    return M.astype(np.float32), rmse, flipped


def apply_matrix(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 3x3 colour matrix (row-vector convention) to an image."""
    a = np.asarray(img, dtype=np.float32)
    out = a.reshape(-1, 3) @ np.asarray(M, dtype=np.float32)
    return np.clip(out, 0.0, None).reshape(a.shape).astype(np.float32)


# ---------- JSON targets ----------

def load_target(path: str | Path) -> tuple[np.ndarray, str, str]:
    """Load a custom 24-swatch target. Returns (swatches (24,3), name,
    colorspace). The colorspace names the OCIO space the values are in."""
    raw = json.loads(Path(path).read_text())
    if raw.get("format") != TARGET_FORMAT:
        raise ValueError(f"{Path(path).name} is not an env2lgt colour target.")
    sw = np.asarray(raw["swatches"], dtype=np.float32)
    if sw.shape != (24, 3):
        raise ValueError(f"Target must have 24 RGB swatches, got {sw.shape}.")
    return sw, str(raw.get("name", Path(path).stem)), str(
        raw.get("colorspace", CC24_REFERENCE_COLORSPACE)
    )


def save_target(
    path: str | Path, swatches: np.ndarray, name: str, colorspace: str
) -> None:
    """Write a 24-swatch target JSON (e.g. swatches sampled from a reference
    image, so they can be reused later)."""
    data = {
        "format": TARGET_FORMAT,
        "name": name,
        "colorspace": colorspace,
        "swatches": [[float(c) for c in row] for row in np.asarray(swatches)],
    }
    Path(path).write_text(json.dumps(data, indent=2))
