"""Per-mask light extraction: rectilinear texture sample + rect-from-quad."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from env2lgt.proj import angles_from_dir, angles_to_pix


@dataclass
class RectFit:
    """A rectangle in world space, ready to author as UsdLuxRectLight."""

    center: np.ndarray   # (3,) world-space center
    normal: np.ndarray   # (3,) plane normal, pointing toward camera origin
    u_axis: np.ndarray   # (3,) "width" basis (unit)
    v_axis: np.ndarray   # (3,) "height" basis (unit)
    width: float         # extent along u_axis
    height: float        # extent along v_axis
    inlier_ratio: float  # 0..1, RANSAC quality
    mean_distance: float # mean depth of inlier points (for fallback / debug)


def crop_rect(hdr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Slice a rectangle out of the panorama; clamps to image bounds."""
    H, W = hdr.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    return np.ascontiguousarray(hdr[y0:y1, x0:x1])


def crop_with_mask(hdr: np.ndarray, mask_full: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    """Crop hdr to the tight bbox of `mask_full` (uint8, full pano res).

    Returns (crop_rgba, mask_crop_u8, bbox_xywh). `crop_rgba` is float32
    (H', W', 4) where alpha is mask/255. Pixels outside the mask are zeroed
    in RGB but kept in the buffer so we can write an EXR with alpha.
    """
    ys, xs = np.where(mask_full > 0)
    if ys.size == 0:
        return np.zeros((0, 0, 4), np.float32), np.zeros((0, 0), np.uint8), (0, 0, 0, 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = hdr[y0:y1, x0:x1].astype(np.float32)
    mcrop = mask_full[y0:y1, x0:x1]
    alpha = (mcrop > 0).astype(np.float32)
    rgb = crop * alpha[..., None]
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return np.ascontiguousarray(rgba), mcrop, (x0, y0, x1 - x0, y1 - y0)


def unproject_mask(
    mask_full: np.ndarray,
    distance: np.ndarray,
    scene_scale: float = 1.0,
    luminance_full: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift all pano pixels where mask>0 to world space.

    Returns (points (N,3), luminances (N,)). Equivalent to the old
    `unproject_rect` but driven by an arbitrary mask shape, not a bbox.
    """
    H, W = distance.shape
    ys, xs = np.where(mask_full > 0)
    if ys.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0,), np.float32)
    dirs = _equirect_dirs(xs, ys, W, H)
    d = distance[ys, xs].astype(np.float64) * float(scene_scale)
    points = (dirs * d[..., None]).astype(np.float32)
    if luminance_full is not None:
        lum = luminance_full[ys, xs].astype(np.float32)
    else:
        lum = np.ones(points.shape[0], dtype=np.float32)
    return points, lum


def _equirect_dirs(uu: np.ndarray, vv: np.ndarray, W: int, H: int) -> np.ndarray:
    """Vectorized (u, v) pixel -> unit direction. Returns (..., 3)."""
    phi = (uu.astype(np.float64) + 0.5) / W * 2.0 * np.pi - np.pi
    theta = np.pi * 0.5 - (vv.astype(np.float64) + 0.5) / H * np.pi
    cos_theta = np.cos(theta)
    x = cos_theta * np.sin(phi)
    y = np.sin(theta)
    z = -cos_theta * np.cos(phi)
    return np.stack([x, y, z], axis=-1)


def unproject_rect(
    rect_xywh: tuple[int, int, int, int],
    distance: np.ndarray,
    scene_scale: float = 1.0,
    luminance_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the pixels inside `rect_xywh` to world space.

    Returns (points (N,3), luminances (N,)).

    `distance` is the DA-2 distance map for the whole panorama (scale-invariant).
    `scene_scale` is the meters-per-unit multiplier the user chose.
    `luminance_mask` is an optional per-pixel luminance buffer (panorama res) to
    return alongside points (used to weight the photometry).
    """
    H, W = distance.shape
    x, y, w, h = rect_xywh
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    uu, vv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    dirs = _equirect_dirs(uu, vv, W, H)
    d = distance[y0:y1, x0:x1].astype(np.float64) * float(scene_scale)
    points = dirs * d[..., None]
    points = points.reshape(-1, 3)
    if luminance_mask is not None:
        lum = luminance_mask[y0:y1, x0:x1].reshape(-1)
    else:
        lum = np.ones(points.shape[0], dtype=np.float64)
    return points.astype(np.float32), lum.astype(np.float32)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane fit. Returns (centroid (3,), normal (3,))."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    return centroid, normal / (np.linalg.norm(normal) + 1e-12)


def ransac_plane(
    points: np.ndarray,
    iters: int = 64,
    inlier_thresh: float | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC plane fit. Returns (centroid, normal, inlier_mask).

    `inlier_thresh` defaults to 1% of the point-cloud bbox diagonal.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = points.shape[0]
    if n < 3:
        c, nrm = _fit_plane_svd(points if n else np.zeros((1, 3), dtype=np.float32))
        return c, nrm, np.ones(n, dtype=bool)
    if inlier_thresh is None:
        bbox = points.max(axis=0) - points.min(axis=0)
        inlier_thresh = max(1e-6, 0.01 * float(np.linalg.norm(bbox)))
    best_inliers = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        tri = points[idx]
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        nrm = np.cross(v1, v2)
        nlen = np.linalg.norm(nrm)
        if nlen < 1e-9:
            continue
        nrm = nrm / nlen
        d = np.abs((points - tri[0]) @ nrm)
        inliers = d < inlier_thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers
    if best_inliers is None or best_count < 3:
        c, nrm = _fit_plane_svd(points)
        return c, nrm, np.ones(n, dtype=bool)
    centroid, normal = _fit_plane_svd(points[best_inliers])
    return centroid, normal, best_inliers


def _orient_normal_toward_origin(centroid: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Flip normal so it faces the camera at (0,0,0)."""
    if np.dot(centroid, normal) > 0.0:
        return -normal
    return normal


def _orthobasis_for_plane(
    centroid: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a stable in-plane basis (u, v). u tries to be world-horizontal."""
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(normal, world_up))) > 0.95:
        # plane is near-horizontal; use a different "up"
        world_up = np.array([1.0, 0.0, 0.0])
    u = np.cross(world_up, normal)
    u /= np.linalg.norm(u) + 1e-12
    v = np.cross(normal, u)
    v /= np.linalg.norm(v) + 1e-12
    return u, v


def fit_rect(points: np.ndarray) -> RectFit:
    """Full rect fit: RANSAC plane, orthobasis, axis-aligned extents on plane."""
    centroid, normal, inliers = ransac_plane(points)
    inlier_pts = points[inliers] if inliers.any() else points
    normal = _orient_normal_toward_origin(centroid, normal)
    u, v = _orthobasis_for_plane(centroid, normal)
    rel = inlier_pts - centroid
    pu = rel @ u
    pv = rel @ v
    # robust extent: 1st/99th percentile to ignore stragglers
    u_lo, u_hi = np.percentile(pu, [1, 99])
    v_lo, v_hi = np.percentile(pv, [1, 99])
    width = float(u_hi - u_lo)
    height = float(v_hi - v_lo)
    # recenter on the actual midpoint of the inlier extent (not just centroid)
    center = centroid + 0.5 * (u_lo + u_hi) * u + 0.5 * (v_lo + v_hi) * v
    return RectFit(
        center=center.astype(np.float32),
        normal=normal.astype(np.float32),
        u_axis=u.astype(np.float32),
        v_axis=v.astype(np.float32),
        width=max(1e-4, width),
        height=max(1e-4, height),
        inlier_ratio=float(inliers.mean()),
        mean_distance=float(np.linalg.norm(inlier_pts, axis=1).mean()),
    )


# ---------- photometry ----------

def luminance(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luminance, shape (..., 3) -> (...)."""
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def equirect_solid_angle(H: int, W: int) -> np.ndarray:
    """Per-pixel solid angle for an equirectangular grid (H, W) sr."""
    j = np.arange(H, dtype=np.float64)
    theta = np.pi * 0.5 - (j + 0.5) / H * np.pi
    # dphi = 2pi/W, dtheta = pi/H
    row_sa = (2.0 * np.pi / W) * (np.pi / H) * np.cos(theta)
    return np.broadcast_to(row_sa[:, None], (H, W)).astype(np.float32)


def mean_color(crop_rgb: np.ndarray) -> np.ndarray:
    """Solid-angle-weighted mean color of an EXR crop, normalized so max=1.
    Returns (3,) float32 — the *color* of the light. Intensity is computed
    separately.
    """
    flat = crop_rgb.reshape(-1, 3).astype(np.float64)
    lum = luminance(flat)
    # weight by luminance so dim border pixels don't drag the color
    w = lum + 1e-6
    color = (flat * w[:, None]).sum(axis=0) / w.sum()
    mx = float(color.max()) + 1e-9
    return (color / mx).astype(np.float32)


def total_emitted_power(
    crop_rgb: np.ndarray, rect_xywh: tuple[int, int, int, int], pano_HW: tuple[int, int]
) -> float:
    """Integrate Y * dOmega over the rectangle on the panorama.

    Returns a scalar power proxy (not strictly watts, but consistent across
    rects and a sensible relative scale for UsdLuxRectLight intensity).
    """
    x, y, w, h = rect_xywh
    H, W = pano_HW
    sa = equirect_solid_angle(H, W)
    sa_crop = sa[y : y + crop_rgb.shape[0], x : x + crop_rgb.shape[1]]
    lum = luminance(crop_rgb.astype(np.float64))
    return float((lum * sa_crop).sum())


def total_emitted_power_masked(
    hdr_full: np.ndarray, mask_full: np.ndarray, pano_HW: tuple[int, int]
) -> float:
    """Like total_emitted_power but integrates only where mask>0 (any pixel
    pattern, not a bbox slice)."""
    H, W = pano_HW
    sa = equirect_solid_angle(H, W)
    lum_full = luminance(hdr_full.astype(np.float64))
    sel = mask_full > 0
    return float((lum_full[sel] * sa[sel]).sum())


def mean_color_masked(hdr_full: np.ndarray, mask_full: np.ndarray) -> np.ndarray:
    """Luminance-weighted mean color of the masked region. Returns (3,) normalized so max=1."""
    sel = mask_full > 0
    if not np.any(sel):
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)
    flat = hdr_full[sel].astype(np.float64)
    lum = luminance(flat)
    w = lum + 1e-6
    color = (flat * w[:, None]).sum(axis=0) / w.sum()
    mx = float(color.max()) + 1e-9
    return (color / mx).astype(np.float32)


# ---------- rectilinear texture sample from quad corners ----------

def _output_size_for_quad(width: float, height: float, max_dim: int = 1024) -> tuple[int, int]:
    """Aspect-matched output (out_h, out_w) capped at max_dim, min 64."""
    if width <= 0 or height <= 0:
        return max_dim, max_dim
    if width >= height:
        out_w = max_dim
        out_h = max(64, int(round(max_dim * height / width)))
    else:
        out_h = max_dim
        out_w = max(64, int(round(max_dim * width / height)))
    # Round to multiples of 8 (cleanliness for compositing)
    out_w = (out_w // 8) * 8 or 64
    out_h = (out_h // 8) * 8 or 64
    return out_h, out_w


def sample_rect_texture(
    hdr_pano: np.ndarray,
    corners_dirs: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Bake an HDR rectilinear texture for a `UsdLuxRectLight`.

    Corners must be the 4 unit directions, ordered consistently with the
    output's UV space — corner index meaning (TL, TR, BR, BL):
        (s=0, t=0) -> corner[0]      top-left
        (s=1, t=0) -> corner[1]      top-right
        (s=1, t=1) -> corner[2]      bottom-right
        (s=0, t=1) -> corner[3]      bottom-left

    Sampling: for each output pixel, bilinear-interpolate the 4 unit-vector
    corners, renormalize, convert to equirect pixel, and remap via
    `cv2.BORDER_WRAP` so seam-spanning quads sample cleanly.
    """
    H, W = hdr_pano.shape[:2]
    corners = np.asarray(corners_dirs, dtype=np.float64).reshape(4, 3)
    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    j, i = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    s = (i + 0.5) / out_w
    t = (j + 0.5) / out_h
    s = s[..., None]
    t = t[..., None]
    # Bilinear interp of the 4 corner dirs over the (s, t) parameterization.
    dirs = (
        (1.0 - s) * (1.0 - t) * tl
        + s * (1.0 - t) * tr
        + s * t * br
        + (1.0 - s) * t * bl
    )
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12

    yaw, pitch = angles_from_dir(dirs)
    u_eq, v_eq = angles_to_pix(yaw, pitch, W, H)

    return cv2.remap(
        hdr_pano,
        u_eq.astype(np.float32),
        v_eq.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


# ---------- rect-light geometry directly from quad corners ----------

def _depth_at_dir(distance: np.ndarray, corner_dir: np.ndarray, patch: int = 3) -> float:
    """Sample the depth map along `corner_dir`. Returns the median of a small
    pixel patch (noise-robust). Horizontal wrap handled for seam-spanning
    corners; vertical clamped at the poles."""
    H, W = distance.shape
    yaw, pitch = angles_from_dir(np.asarray(corner_dir, dtype=np.float64))
    u, v = angles_to_pix(yaw, pitch, W, H)
    ui = int(round(float(np.asarray(u))))
    vi = int(round(float(np.asarray(v))))
    us = [(ui + dx) % W for dx in range(-patch, patch + 1)]
    vs = [min(max(vi + dy, 0), H - 1) for dy in range(-patch, patch + 1)]
    return float(np.median(distance[np.ix_(vs, us)]))


def _bright_region_depth(
    mask_full: np.ndarray,
    distance: np.ndarray,
    lum_full: np.ndarray,
    min_px: int = 16,
) -> float | None:
    """Median depth of the *bright* (light-emitting) pixels inside the quad.

    The user draws the quad to engulf the light, so its corners sit on the
    surrounding wall — sampling depth there places the rect on the wall
    *behind* the light. The light itself is the bright sub-region of the
    quad; an Otsu split of the in-mask log-luminance separates it from the
    dim border. Returns the median distance over that bright region, or
    None if it can't be isolated (caller falls back to the corner fit).
    """
    sel = mask_full > 0
    n = int(sel.sum())
    if n < min_px * 2:
        return None
    lum_sel = lum_full[sel].astype(np.float64)
    dist_sel = distance[sel].astype(np.float64)
    loglum = np.log(np.maximum(lum_sel, 1e-6))
    lo, hi = float(loglum.min()), float(loglum.max())
    if hi - lo < 1e-6:
        return None  # uniform luminance — no light/wall split to make
    u8 = ((loglum - lo) / (hi - lo) * 255.0).astype(np.uint8).reshape(-1, 1)
    # Use cv2's own thresholded output, not the returned threshold value:
    # for a degenerate 2-value histogram Otsu can report threshold 0, and a
    # `>= 0` test would then select every pixel.
    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (otsu.ravel() > 0) & np.isfinite(dist_sel) & (dist_sel > 1e-6)
    if int(bright.sum()) < min_px:
        return None
    return float(np.median(dist_sel[bright]))


def rect_from_quad(
    corners_dirs: np.ndarray,
    mask_full: np.ndarray,
    distance: np.ndarray,
    scene_scale: float = 1.0,
    lum_full: np.ndarray | None = None,
    treat_as_window: bool = False,
) -> RectFit:
    """Build a UsdLuxRectLight transform/size from the user's quad.

    Per-corner depth: each of the 4 corner rays is lifted to world space at
    *its own* depth (sampled at that corner's pano pixel). Because the depths
    differ, the 4 points follow the actual scene surface rather than sitting
    on a sphere of constant radius — so a ceiling light's 4 corners land in a
    horizontal plane and the fitted normal points straight down, instead of
    radially back toward the camera origin.

    This matches the depth-displaced validation mesh (env2lgt.usd.mesh): the
    rect light's plane now coincides with the mesh surface under it.

    Light vs. wall depth: the corners land on the wall *around* the light,
    so the corner fit sits at wall depth. When `lum_full` is given and the
    quad isn't flagged `treat_as_window`, the rect is slid inward along the
    view rays to the bright region's depth (the light itself), keeping the
    corner-derived orientation. `treat_as_window=True` keeps the wall depth
    — correct for windows/skylights, which are flush with the wall and where
    the bright pixels are distant sky.
    """
    corners = np.asarray(corners_dirs, dtype=np.float64).reshape(4, 3)
    corners = corners / (np.linalg.norm(corners, axis=1, keepdims=True) + 1e-12)

    # Mask-median depth — used only as a fallback for corners that read a
    # bad (non-positive / NaN) depth value.
    sel = mask_full > 0
    mask_med = float(np.median(distance[sel])) if np.any(sel) else 1.0

    # Per-corner depth.
    depths = np.empty(4, dtype=np.float64)
    for i in range(4):
        d = _depth_at_dir(distance, corners[i])
        if not np.isfinite(d) or d <= 1e-6:
            d = mask_med
        depths[i] = d
    depths *= float(scene_scale)

    # Lift each corner to its own depth → points on the real surface.
    pts = corners * depths[:, None]          # (4, 3)
    center = pts.mean(axis=0)

    # Parallelogram fit: average opposite edges. Corner order is TL, TR, BR, BL.
    u_edge = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
    v_edge = 0.5 * ((pts[3] - pts[0]) + (pts[2] - pts[1]))
    width = float(np.linalg.norm(u_edge))
    height = float(np.linalg.norm(v_edge))
    u_axis = u_edge / (width + 1e-12)
    v_axis = v_edge / (height + 1e-12)

    # Surface normal = u x v (perpendicular to the fitted plane through the
    # 4 real-depth corners). Flip so it points back toward the scene interior
    # (the camera/origin side) — that's the emission direction for the light.
    normal = np.cross(u_axis, v_axis)
    nlen = np.linalg.norm(normal)
    if nlen < 1e-9:
        # Degenerate quad (collinear corners) — fall back to radial normal.
        normal = -corners.mean(axis=0)
        normal /= np.linalg.norm(normal) + 1e-12
    else:
        normal /= nlen
    if float(center @ normal) > 0.0:
        normal = -normal
        # Keep the (u, v, normal) frame right-handed after the flip.
        v_axis = -v_axis

    # Slide the rect from wall depth to the light's depth (see docstring).
    # The corner fit gives wall-depth geometry; the bright region gives the
    # light's true distance. Same view rays → linear size scales with depth.
    mean_depth = float(depths.mean())
    if not treat_as_window and lum_full is not None:
        d_light = _bright_region_depth(mask_full, distance, lum_full)
        if d_light is not None:
            d_light *= float(scene_scale)
            d_wall = float(np.linalg.norm(center))
            if d_wall > 1e-6:
                # Clamp the slide so a noisy depth read can't fling the rect
                # to infinity (or onto the camera).
                s = min(2.0, max(0.1, d_light / d_wall))
                center = center * s
                width *= s
                height *= s
                mean_depth *= s

    return RectFit(
        center=center.astype(np.float32),
        normal=normal.astype(np.float32),
        u_axis=u_axis.astype(np.float32),
        v_axis=v_axis.astype(np.float32),
        width=max(1e-4, width),
        height=max(1e-4, height),
        inlier_ratio=1.0,                    # by construction
        mean_distance=mean_depth,
    )
