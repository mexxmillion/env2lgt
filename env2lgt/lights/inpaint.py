# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Dome residual: zero the rect mask regions in the panorama, fill the holes
with a smooth gradient extending from valid neighbors. Equivalent to Nuke's
EdgeExtend / Mocha's PushPull — way more stable on HDR than cv2 TELEA, which
produces rainbow noise on high-DR values.
"""

from __future__ import annotations

import cv2
import numpy as np


def grow_mask_by_luminance(
    hdr: np.ndarray,
    mask: np.ndarray,
    *,
    max_radius_px: int = 32,
    bright_factor: float = 4.0,
) -> np.ndarray:
    """Expand a binary mask outward to swallow the bright halo around small
    sources (camera PSF / glow bleed). Without this, tiny bulbs in a high-res
    pano leak hot pixels past the geometric quad edge → those pixels survive
    the dome inpaint and double-light the scene.

    Algorithm, per masked region:
      1. Find the mask's bounding box and extract a local crop with a
         `max_radius_px` margin.
      2. Background luminance = median of the *unmasked* pixels in the crop —
         the "what the dome should be here" reference.
      3. Iteratively dilate the local mask one pixel at a time. After each
         step, only keep newly-touched pixels whose luminance is at least
         `bright_factor` × background. Stop when no new pixels qualify or
         we've grown `max_radius_px` steps.

    The result is unioned back into the global mask. Cost is bounded by the
    bounding boxes (small per-quad regions, not the whole pano).
    """
    if not mask.any():
        return mask
    H, W = mask.shape[:2]
    out = mask.copy()
    lum = (0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]).astype(
        np.float32
    )

    # Find connected components of the input mask so each bright source grows
    # against its own local background, not the median of the whole pano.
    num, labels = cv2.connectedComponents((mask > 0).astype(np.uint8))
    pad = int(max_radius_px) + 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for cid in range(1, num):
        ys, xs = np.where(labels == cid)
        if ys.size == 0:
            continue
        y0 = max(0, int(ys.min()) - pad)
        y1 = min(H, int(ys.max()) + pad + 1)
        x0 = max(0, int(xs.min()) - pad)
        x1 = min(W, int(xs.max()) + pad + 1)
        local_lum = lum[y0:y1, x0:x1]
        local_mask = (labels[y0:y1, x0:x1] == cid).astype(np.uint8)
        outside = local_lum[local_mask == 0]
        if outside.size == 0:
            continue
        bg_med = float(np.median(outside))
        threshold = bg_med * float(bright_factor) + 1e-6
        prev = local_mask.copy()
        for _ in range(int(max_radius_px)):
            grown = cv2.dilate(local_mask, kernel)
            new_pixels = (grown > 0) & (local_mask == 0)
            keep = new_pixels & (local_lum > threshold)
            if not keep.any():
                break
            local_mask = local_mask | keep.astype(np.uint8)
            if np.array_equal(local_mask, prev):
                break
            prev = local_mask.copy()
        out[y0:y1, x0:x1] = np.where(local_mask > 0, 255, out[y0:y1, x0:x1])
    return out


def build_mask(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], dilate_px: int = 4) -> np.ndarray:
    """(Legacy axis-aligned-bbox mask builder. New pipeline uses
    `rasterize_spherical_quad` from env2lgt.proj for arbitrary quad shapes.)
    """
    H, W = shape
    m = np.zeros((H, W), dtype=np.uint8)
    for (x, y, w, h) in rects:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        m[y0:y1, x0:x1] = 255
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        m = cv2.dilate(m, kernel)
    return m


def edge_extend(
    hdr: np.ndarray,
    mask: np.ndarray,
    iters: int = 96,
    sigma: float = 3.0,
    feather_px: int = 4,
) -> np.ndarray:
    """Push valid edge pixels inward to fill `mask` (uint8, 255 = hole).

    Algorithm: each iteration we Gaussian-blur the current image weighted by a
    valid-mask, then write the blurred value into pixels that were invalid but
    now have at least one valid neighbor. Hole shrinks by ~sigma px per pass.

    Works in **log domain** for HDR safety: high-DR bright regions don't blow
    out the kernel, gradients reproject smoothly across orders of magnitude.

    `feather_px` softens the transition where the original mask boundary lies
    so the dome doesn't show a hard seam between original pixels and filled
    pixels.
    """
    if hdr.dtype != np.float32:
        hdr = hdr.astype(np.float32)
    H, W = mask.shape
    if hdr.shape[:2] != (H, W):
        raise ValueError(f"hdr {hdr.shape} vs mask {mask.shape} size mismatch")
    invalid = mask > 0
    if not np.any(invalid):
        return hdr.copy()

    log_hdr = np.log1p(np.maximum(hdr, 0.0))
    work = log_hdr.copy()
    work[invalid] = 0.0
    valid = (~invalid).astype(np.float32)

    # Per-iteration kernel size: ~6*sigma. We bias toward small sigma + more
    # iters to keep gradients smooth.
    ksize = max(3, int(2 * round(3 * sigma) + 1))

    for _ in range(iters):
        if not np.any(valid < 0.5):
            break
        blurred_img = cv2.GaussianBlur(
            work * valid[..., None], (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
        )
        blurred_valid = cv2.GaussianBlur(
            valid, (ksize, ksize), sigma, borderType=cv2.BORDER_REPLICATE
        )
        denom = np.maximum(blurred_valid, 1e-6)
        new_vals = blurred_img / denom[..., None]
        # Pixels still invalid that now have any blurred-valid neighbor → fill
        fill = (valid < 0.5) & (blurred_valid > 1e-3)
        if not np.any(fill):
            # No more reachable pixels (e.g. mask is the entire image); break.
            break
        work[fill] = new_vals[fill]
        valid[fill] = 1.0

    result = np.expm1(work).astype(np.float32)

    if feather_px > 0:
        # Soft blend between the original (outside mask) and the filled
        # (inside mask) so there's no hard ring at the boundary.
        k = 2 * feather_px + 1
        alpha = (mask > 0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (k, k), feather_px)
        result = alpha[..., None] * result + (1.0 - alpha[..., None]) * hdr

    return result


# Back-compat shim so callers that imported the old name still work.
def feathered_inpaint(hdr, mask, inpaint_radius=6, feather_px=8):
    """Deprecated TELEA path — kept only so external code doesn't break.
    Redirects to `edge_extend` which is HDR-safe."""
    return edge_extend(hdr, mask, iters=96, sigma=3.0, feather_px=feather_px)
