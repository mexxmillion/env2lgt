"""Per-mask light extraction: rectilinear texture sample + rect-from-quad."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from env2lgt.proj import angles_from_dir, angles_to_pix, dir_from_angles, pix_to_angles


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
    fit: "RectFit",
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Bake an HDR rectilinear texture for a `UsdLuxRectLight` via *true*
    perspective projection onto the fitted rect's plane.

    For each output pixel (s, t) in [0, 1]², compute the 3D point on the
    rect's plane (`center + (s-0.5)·width·u_axis + (0.5-t)·height·v_axis`),
    cast a ray from the camera origin through it, convert that direction to
    an equirect pixel, and remap. This is exact rectilinear projection —
    what the camera at origin actually sees of the rect's emissive surface.

    The previous bilinear-of-4-corner-dirs implementation was an
    approximation that warped the interior of the texture relative to true
    perspective: the bilinear blend of unit vectors only matches rectilinear
    on the corners, and the deviation grows with the rect's angular extent.
    After `_fit_from_4points` produces a rigid 3D rectangle (orthonormal
    `(u, v, n)` + width / height), we can sample the panorama exactly, so
    when the texture maps 1:1 onto the `UsdLuxRectLight` it reproduces what
    the original panorama shows of that fixture, regardless of its angular
    size.

    Texture orientation matches the corners returned by `rect_to_corner_dirs`
    (TL/TR/BR/BL): `(s=0, t=0)` is top-left, `(s=1, t=1)` is bottom-right.
    """
    H, W = hdr_pano.shape[:2]
    j, i = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    s = (i.astype(np.float64) + 0.5) / out_w
    t = (j.astype(np.float64) + 0.5) / out_h

    # Rect-frame offsets. Image convention: t=0 (top of texture) → +v_axis
    # side of the rect, matching corner[0] = TL = -hw·u + +hh·v.
    pu = (s - 0.5) * float(fit.width)
    pv = (0.5 - t) * float(fit.height)

    center = np.asarray(fit.center, dtype=np.float64)
    u_axis = np.asarray(fit.u_axis, dtype=np.float64)
    v_axis = np.asarray(fit.v_axis, dtype=np.float64)
    # P(s, t) on the rect plane in world space (output shape: H_out, W_out, 3).
    P = center[None, None, :] + pu[..., None] * u_axis + pv[..., None] * v_axis
    # Ray from camera at origin → unit direction.
    dirs = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)

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


def _fit_from_4points(pts: np.ndarray) -> tuple:
    """*Rigid* (no-shear) rectangle from 4 corner points (any order).

    The 4 points are assumed to lie (approximately) on a plane — either the
    light's bright-region plane or a per-corner-depth plane, both fit upstream
    in `rect_from_quad`. Their best-fit plane is taken via SVD, and the
    rectangle's in-plane rotation is recovered by the **diagonal-bisector**
    method (rect axes = angle bisectors of the two diagonals, equivalently
    the averaged opposite-edge directions). Bbox of the projected points in
    that orthonormal (u, v) frame gives width × height.

    Why diagonal-bisector and not PCA? PCA fails silently on near-square
    inputs — degenerate eigenvalues let `eigh` return an arbitrary basis
    (typically ~45° off the rect's true edges). The diagonal-bisector method
    uses edge directions explicitly and stays correct for squares too.

    Why not the previous "averaged opposite edges" formulation? That
    produced a *parallelogram* (sheared) — fine for spherical rasterisation
    but illegal as a UsdLuxRectLight transform. Renderers that decompose
    the xform into TRS (Arnold, V-Ray, Redshift, RenderMan, Karma, every
    DCC area light) silently drop the shear and the rect lands wrong.

    Returns (center, normal, u_axis, v_axis, width, height) with the
    (u_axis, v_axis, normal) basis guaranteed orthonormal.
    """
    p = np.asarray(pts, dtype=np.float64).reshape(4, 3)
    centroid = p.mean(axis=0)
    # Plane fit: SVD of centred points; normal = smallest singular vector.
    _, _, vt = np.linalg.svd(p - centroid, full_matrices=False)
    n = vt[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    # Emission direction faces the camera (origin).
    if float(centroid @ n) > 0.0:
        n = -n

    # Temporary in-plane basis (e1, e2) for 2D bookkeeping; the real rect
    # axes come from the diagonal-bisector below. e2 follows world-up
    # projected onto the plane (falls back to world-Z on horizontal planes).
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(n @ world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    e2 = world_up - (world_up @ n) * n
    e2 /= np.linalg.norm(e2) + 1e-12
    e1 = np.cross(e2, n)
    e1 /= np.linalg.norm(e1) + 1e-12

    rel = p - centroid
    p2 = np.stack([rel @ e1, rel @ e2], axis=1)        # (4, 2)

    # CCW-order around the centroid, then diagonal-bisector.
    angles = np.arctan2(p2[:, 1], p2[:, 0])
    order = np.argsort(angles)
    cc = p2[order]
    e_u_sum = (cc[1] - cc[0]) + (cc[2] - cc[3])
    e_v_sum = (cc[2] - cc[1]) + (cc[3] - cc[0])
    nu, nv = float(np.linalg.norm(e_u_sum)), float(np.linalg.norm(e_v_sum))
    if nu >= nv:
        u2 = e_u_sum / (nu + 1e-12)
    else:
        u2 = e_v_sum / (nv + 1e-12)
    if u2[0] < 0:                                      # stable direction
        u2 = -u2
    v2 = np.array([-u2[1], u2[0]])

    u_axis = u2[0] * e1 + u2[1] * e2
    v_axis = v2[0] * e1 + v2[1] * e2

    pu = rel @ u_axis
    pv = rel @ v_axis
    u_lo, u_hi = float(pu.min()), float(pu.max())
    v_lo, v_hi = float(pv.min()), float(pv.max())
    center = centroid + 0.5 * (u_lo + u_hi) * u_axis + 0.5 * (v_lo + v_hi) * v_axis
    width = max(1e-4, u_hi - u_lo)
    height = max(1e-4, v_hi - v_lo)
    return center, n, u_axis, v_axis, width, height


def rect_to_corner_dirs(fit: "RectFit") -> np.ndarray:
    """Cast a `RectFit`'s 4 rigid corners back to unit direction vectors.

    Used by the "Fit to rect light" viewer button to update a quad's stored
    `corners_dirs` to the snapped rigid-rect corners — what you see on the
    panorama = what gets authored. Corner order is TL/TR/BR/BL, matching
    `sample_rect_texture`'s UV convention.
    """
    hw = float(fit.width) * 0.5
    hh = float(fit.height) * 0.5
    c = np.asarray(fit.center, dtype=np.float64)
    u = np.asarray(fit.u_axis, dtype=np.float64)
    v = np.asarray(fit.v_axis, dtype=np.float64)
    pts = np.stack([
        c + (-hw) * u + (+hh) * v,   # TL
        c + (+hw) * u + (+hh) * v,   # TR
        c + (+hw) * u + (-hh) * v,   # BR
        c + (-hw) * u + (-hh) * v,   # BL
    ], axis=0)
    return pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-12)


def _corner_depths(
    corners: np.ndarray, mask_full: np.ndarray, distance: np.ndarray, scene_scale: float
) -> np.ndarray:
    """Per-corner depth (× scene_scale), with a mask-median fallback for
    corners that read a bad (non-positive / NaN) depth."""
    sel = mask_full > 0
    mask_med = float(np.median(distance[sel])) if np.any(sel) else 1.0
    depths = np.empty(4, dtype=np.float64)
    for i in range(4):
        d = _depth_at_dir(distance, corners[i])
        if not np.isfinite(d) or d <= 1e-6:
            d = mask_med
        depths[i] = d
    return depths * float(scene_scale)


def _bright_region_plane(
    mask_full: np.ndarray,
    distance: np.ndarray,
    lum_full: np.ndarray,
    scene_scale: float,
    min_px: int = 64,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit a plane to the 3D points of the *bright* (light) pixels in the quad.

    The user engulfs the light, so the quad corners sit on a mix of background
    surfaces (wall, floor, the stand) — a fit through them is both too far and
    mis-tilted. The light itself is the bright sub-region: an Otsu split of the
    in-mask log-luminance isolates it, those pixels are lifted to 3D by their
    depth, and a RANSAC plane is fit. Returns (centroid, unit normal), or None
    if the bright region can't be isolated / fit.
    """
    sel = mask_full > 0
    if int(sel.sum()) < min_px * 2:
        return None
    ys, xs = np.where(sel)
    lum_sel = lum_full[sel].astype(np.float64)
    dist_sel = distance[sel].astype(np.float64)
    loglum = np.log(np.maximum(lum_sel, 1e-6))
    lo, hi = float(loglum.min()), float(loglum.max())
    if hi - lo < 1e-6:
        return None  # uniform luminance — no light/wall split to make
    u8 = ((loglum - lo) / (hi - lo) * 255.0).astype(np.uint8).reshape(-1, 1)
    # Use cv2's thresholded output, not the returned threshold: for a
    # degenerate 2-value histogram Otsu can report threshold 0, and a `>= 0`
    # test would then select every pixel.
    _, otsu = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (otsu.ravel() > 0) & np.isfinite(dist_sel) & (dist_sel > 1e-6)
    if int(bright.sum()) < min_px:
        return None
    by, bx, bd = ys[bright], xs[bright], dist_sel[bright]
    if by.size > 4000:  # cap point count for the RANSAC fit
        idx = np.random.default_rng(0).choice(by.size, 4000, replace=False)
        by, bx, bd = by[idx], bx[idx], bd[idx]
    H, W = distance.shape
    yaw, pitch = pix_to_angles(bx.astype(np.float64), by.astype(np.float64), W, H)
    dirs = dir_from_angles(yaw, pitch)                       # (N, 3)
    pts = dirs * (bd[:, None] * float(scene_scale))
    centroid, normal, inliers = ransac_plane(pts)
    if int(inliers.sum()) < min_px:
        return None
    return centroid, normal / (np.linalg.norm(normal) + 1e-12)


def rect_from_quad(
    corners_dirs: np.ndarray,
    mask_full: np.ndarray,
    distance: np.ndarray,
    scene_scale: float = 1.0,
    lum_full: np.ndarray | None = None,
    treat_as_window: bool = False,
) -> RectFit:
    """Build a UsdLuxRectLight transform/size from the user's quad.

    The user draws the quad to *engulf* a light, so its 4 corners sit on
    whatever is behind/around it — wall, floor, the light stand. Fitting the
    rect directly through those corner points is unreliable: it lands at the
    background's depth and, when the corners straddle different surfaces,
    picks up a spurious tilt.

    So for an ordinary light, the rect plane (position *and* orientation)
    comes from the light itself: the bright sub-region of the quad is fit
    with a RANSAC plane (`_bright_region_plane`), and the 4 corner rays are
    projected onto that plane to give the rect's extent. The quad still
    defines the angular footprint; the light surface defines where/how it sits.

    `treat_as_window=True` (or a missing `lum_full`, or a bright region too
    small to fit) falls back to the per-corner-depth fit — correct for
    windows/skylights, which are flush with the wall and whose bright pixels
    are distant sky.
    """
    corners = np.asarray(corners_dirs, dtype=np.float64).reshape(4, 3)
    corners = corners / (np.linalg.norm(corners, axis=1, keepdims=True) + 1e-12)

    # --- light-plane fit: orientation + position from the light surface ---
    if not treat_as_window and lum_full is not None:
        plane = _bright_region_plane(mask_full, distance, lum_full, scene_scale)
        if plane is not None:
            centroid, pn = plane
            plane_d = float(centroid @ pn)
            denom = corners @ pn
            if np.all(np.abs(denom) > 1e-6):
                t = plane_d / denom
                if np.all(t > 0):
                    # Project each corner ray onto the light plane.
                    proj = corners * t[:, None]
                    center, normal, u_axis, v_axis, width, height = _fit_from_4points(proj)
                    return RectFit(
                        center=center.astype(np.float32),
                        normal=normal.astype(np.float32),
                        u_axis=u_axis.astype(np.float32),
                        v_axis=v_axis.astype(np.float32),
                        width=width,
                        height=height,
                        inlier_ratio=1.0,
                        mean_distance=float(np.linalg.norm(center)),
                    )

    # --- fallback / window: per-corner depth fit ---
    depths = _corner_depths(corners, mask_full, distance, scene_scale)
    pts = corners * depths[:, None]
    center, normal, u_axis, v_axis, width, height = _fit_from_4points(pts)
    return RectFit(
        center=center.astype(np.float32),
        normal=normal.astype(np.float32),
        u_axis=u_axis.astype(np.float32),
        v_axis=v_axis.astype(np.float32),
        width=width,
        height=height,
        inlier_ratio=1.0,
        mean_distance=float(depths.mean()),
    )
