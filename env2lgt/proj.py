"""Projection math: sphere <-> equirect <-> rectilinear.

Single source of truth for the geometry. All other modules import from here.

Conventions
-----------
- Right-handed, Y-up.
- Yaw (longitude, phi) in [-pi, pi]; 0 looks toward -Z.
- Pitch (latitude, theta) in [-pi/2, pi/2]; +pi/2 looks up.
- Pixel coords are origin top-left, +X right, +Y down. Pixel (u, v) maps as:
      phi   = (u + 0.5) / W * 2pi - pi
      theta = pi/2 - (v + 0.5) / H * pi
- A unit direction `d = (sin(phi)*cos(theta), sin(theta), -cos(phi)*cos(theta))`.
"""

from __future__ import annotations

import numpy as np


# ---------- direction <-> spherical angle ----------

def dir_from_angles(yaw: np.ndarray | float, pitch: np.ndarray | float) -> np.ndarray:
    """(yaw, pitch) -> unit dir. Yaw, pitch can be scalars or same-shape arrays."""
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    c = np.cos(pitch)
    x = np.sin(yaw) * c
    y = np.sin(pitch)
    z = -np.cos(yaw) * c
    return np.stack([x, y, z], axis=-1)


def angles_from_dir(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """unit dir -> (yaw, pitch). Works on arrays of shape (..., 3)."""
    d = np.asarray(d, dtype=np.float64)
    yaw = np.arctan2(d[..., 0], -d[..., 2])
    pitch = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return yaw, pitch


# ---------- equirect pixel <-> spherical angle ----------

def pix_to_angles(u: np.ndarray, v: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    yaw = (u + 0.5) / W * 2.0 * np.pi - np.pi
    pitch = np.pi * 0.5 - (v + 0.5) / H * np.pi
    return yaw, pitch


def angles_to_pix(yaw: np.ndarray, pitch: np.ndarray, W: int, H: int) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.asarray(yaw, dtype=np.float64)
    pitch = np.asarray(pitch, dtype=np.float64)
    u = (yaw + np.pi) / (2.0 * np.pi) * W - 0.5
    v = (np.pi * 0.5 - pitch) / np.pi * H - 0.5
    return u, v


# ---------- rectilinear (perspective) projection of the panorama ----------

def view_basis(yaw_c: float, pitch_c: float, roll: float = 0.0) -> np.ndarray:
    """Camera basis (3x3): rows are world-space (right, up, forward).

    Camera looks along its +forward (which corresponds to the view direction
    given by yaw_c, pitch_c).
    """
    # forward
    cp, sp = np.cos(pitch_c), np.sin(pitch_c)
    cy, sy = np.cos(yaw_c), np.sin(yaw_c)
    forward = np.array([sy * cp, sp, -cy * cp], dtype=np.float64)
    # right = forward x world_up; if camera looks straight up/down, use a fallback
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    n = np.linalg.norm(right)
    if n < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= n
    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-12
    if roll != 0.0:
        c, s = np.cos(roll), np.sin(roll)
        new_right = c * right + s * up
        new_up = -s * right + c * up
        right, up = new_right, new_up
    return np.stack([right, up, forward], axis=0)


def rectilinear_remap(
    pano: np.ndarray,
    out_h: int,
    out_w: int,
    yaw_c: float,
    pitch_c: float,
    hfov_rad: float,
    roll: float = 0.0,
) -> np.ndarray:
    """Sample `pano` (latlong, H x W x C) into a rectilinear view.

    Returns (out_h, out_w, C) same dtype as input. Uses OpenCV remap (bilinear).
    """
    import cv2

    H, W = pano.shape[:2]
    aspect = out_w / out_h
    half_w = np.tan(hfov_rad * 0.5)
    half_h = half_w / aspect

    yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    ndc_x = (xx + 0.5) / out_w * 2.0 - 1.0
    ndc_y = 1.0 - (yy + 0.5) / out_h * 2.0
    cam_x = ndc_x * half_w
    cam_y = ndc_y * half_h
    cam_z = np.ones_like(cam_x)
    cam = np.stack([cam_x, cam_y, cam_z], axis=-1)  # (h, w, 3)
    norms = np.linalg.norm(cam, axis=-1, keepdims=True)
    cam /= norms

    basis = view_basis(yaw_c, pitch_c, roll)
    # world = right * cam_x + up * cam_y + forward * cam_z
    world = (
        cam[..., 0:1] * basis[0]
        + cam[..., 1:2] * basis[1]
        + cam[..., 2:3] * basis[2]
    )
    yaw, pitch = angles_from_dir(world)
    u, v = angles_to_pix(yaw, pitch, W, H)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    return cv2.remap(
        pano, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
    )


def rectilinear_pixel_to_dir(
    px: float,
    py: float,
    out_w: int,
    out_h: int,
    yaw_c: float,
    pitch_c: float,
    hfov_rad: float,
    roll: float = 0.0,
) -> np.ndarray:
    """Rectilinear (pixel) -> world unit dir."""
    aspect = out_w / out_h
    half_w = np.tan(hfov_rad * 0.5)
    half_h = half_w / aspect
    ndc_x = (px + 0.5) / out_w * 2.0 - 1.0
    ndc_y = 1.0 - (py + 0.5) / out_h * 2.0
    cam_x = ndc_x * half_w
    cam_y = ndc_y * half_h
    cam_z = 1.0
    cam = np.array([cam_x, cam_y, cam_z], dtype=np.float64)
    cam /= np.linalg.norm(cam)
    basis = view_basis(yaw_c, pitch_c, roll)
    world = cam[0] * basis[0] + cam[1] * basis[1] + cam[2] * basis[2]
    return world


def dir_to_rectilinear_pixel(
    d: np.ndarray,
    out_w: int,
    out_h: int,
    yaw_c: float,
    pitch_c: float,
    hfov_rad: float,
    roll: float = 0.0,
) -> tuple[float, float, bool]:
    """world unit dir -> (px, py, in_front). in_front is False if the dir
    points behind the camera (sample is outside the rectilinear view)."""
    aspect = out_w / out_h
    half_w = np.tan(hfov_rad * 0.5)
    half_h = half_w / aspect
    basis = view_basis(yaw_c, pitch_c, roll)
    cam = basis @ d  # project world dir into camera basis
    if cam[2] <= 1e-6:
        return 0.0, 0.0, False
    cam = cam / cam[2]  # divide by forward
    ndc_x = cam[0] / half_w
    ndc_y = cam[1] / half_h
    px = (ndc_x + 1.0) * 0.5 * out_w - 0.5
    py = (1.0 - ndc_y) * 0.5 * out_h - 0.5
    return float(px), float(py), True


# ---------- quad-on-sphere -> equirect pixel mask ----------

def rasterize_spherical_quad(
    corners_dirs: np.ndarray,
    pano_H: int,
    pano_W: int,
    pad_px: int = 0,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Given 4 unit dirs (corner_dirs, shape (4,3)) defining a planar quad in 3D,
    rasterize the quad onto the equirect pano grid.

    The quad's supporting plane is fit by SVD of the 4 corners' centroid.
    A pano pixel is "inside" if its ray intersects the plane within the quad's
    convex hull (in plane coords).

    Returns:
        mask: uint8 (H, W), 255 inside, 0 outside
        bbox: (x, y, w, h) tight bounding box on the pano (with pad_px applied)
    """
    corners = np.asarray(corners_dirs, dtype=np.float64).reshape(4, 3)
    centroid = corners.mean(axis=0)
    # plane normal = SVD smallest singular vector of centered corners
    _, _, vt = np.linalg.svd(corners - centroid, full_matrices=False)
    normal = vt[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    # in-plane basis
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(normal @ world_up)) > 0.95:
        world_up = np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(world_up, normal)
    u_axis /= np.linalg.norm(u_axis) + 1e-12
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis) + 1e-12

    # plane passes through `centroid`. Intersect each pano-pixel ray with plane.
    # The pano "camera" sits at origin; rays are unit dirs.
    # Ray: r(t) = t * d.  Plane: (p - centroid) . normal = 0.
    # t = (centroid . normal) / (d . normal).
    H, W = pano_H, pano_W
    j = np.arange(H, dtype=np.float64)
    i = np.arange(W, dtype=np.float64)
    yaw = (i + 0.5) / W * 2.0 * np.pi - np.pi
    pitch = np.pi * 0.5 - (j + 0.5) / H * np.pi
    # Build dirs row-by-row to limit peak memory on big panos.
    mask = np.zeros((H, W), dtype=np.uint8)

    # Project corners into (u, v) plane coords
    corner_uv = np.zeros((4, 2), dtype=np.float64)
    for k in range(4):
        rel = corners[k] - centroid
        corner_uv[k] = [rel @ u_axis, rel @ v_axis]
    # Order the 4 corners consistently (CCW around centroid in plane).
    angles = np.arctan2(corner_uv[:, 1], corner_uv[:, 0])
    order = np.argsort(angles)
    corner_uv = corner_uv[order]

    # Plane equation: (p - centroid) . normal == 0. Ray r(t) = t * d, so
    #   t = (centroid . normal) / (d . normal).
    # `cn` is the signed distance from origin to plane along normal; it can be
    # negative if the normal points away from the camera, which is fine — we
    # just need t > 0 for a forward hit.
    cn = float(centroid @ normal)

    # Fully vectorized: build the entire dir grid at once. 4K x 8K x 3 floats
    # is ~800 MB — acceptable but tight. We chunk by row-block to stay safe.
    sin_yaw = np.sin(yaw)
    neg_cos_yaw = -np.cos(yaw)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)

    row_block = max(1, 2_000_000 // max(W, 1))  # ~2M pixels per chunk

    for r0 in range(0, H, row_block):
        r1 = min(H, r0 + row_block)
        cp = cos_pitch[r0:r1, None]                  # (rb, 1)
        sp = sin_pitch[r0:r1, None]                  # (rb, 1)
        dx = sin_yaw[None, :] * cp                   # (rb, W)
        dy = np.broadcast_to(sp, (r1 - r0, W))       # (rb, W)
        dz = neg_cos_yaw[None, :] * cp               # (rb, W)
        dn = dx * normal[0] + dy * normal[1] + dz * normal[2]
        # forward hit: |dn| above eps AND resulting t > 0
        nonparallel = np.abs(dn) > 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(nonparallel, cn / np.where(nonparallel, dn, 1.0), -1.0)
        forward = t > 0.0
        if not np.any(forward):
            continue
        # plane hit points (only used where forward; we still compute everywhere)
        hx = t * dx
        hy = t * dy
        hz = t * dz
        rx = hx - centroid[0]
        ry = hy - centroid[1]
        rz = hz - centroid[2]
        pu = rx * u_axis[0] + ry * u_axis[1] + rz * u_axis[2]
        pv = rx * v_axis[0] + ry * v_axis[1] + rz * v_axis[2]
        inside = _point_in_quad_vec(pu, pv, corner_uv) & forward
        if np.any(inside):
            mask[r0:r1][inside] = 255

    if pad_px > 0:
        import cv2

        k = 2 * pad_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.dilate(mask, kernel)

    # bbox
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return mask, (0, 0, 0, 0)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return mask, (x0, y0, x1 - x0, y1 - y0)


def _point_in_quad_vec(pu: np.ndarray, pv: np.ndarray, quad_uv: np.ndarray) -> np.ndarray:
    """Vectorized point-in-(convex)quad test. quad_uv ordered CCW, shape (4, 2)."""
    inside = np.ones_like(pu, dtype=bool)
    for k in range(4):
        a = quad_uv[k]
        b = quad_uv[(k + 1) % 4]
        edge_x = b[0] - a[0]
        edge_y = b[1] - a[1]
        # left of edge (CCW): cross > 0
        cross = edge_x * (pv - a[1]) - edge_y * (pu - a[0])
        inside &= cross >= -1e-9
    return inside
