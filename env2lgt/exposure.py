"""Exposure + white-balance metering for the HDRI baseline.

These adjustments shift the HDRI's baseline exposure (in stops) and white
balance before anything else happens — they get baked into the dome / rect
textures on export. Display-only viewer exposure is separate (see app.py).

Algorithms are ported from the reference calibrator in W:\\git\\hdr_cal
(hdri_cal.py): Tanner-Helland kelvin curve, grey-world neutralisation, and a
cosine-weighted Lambertian "convolve the dome" gray-ball render for auto
exposure + auto WB.
"""

from __future__ import annotations

import numpy as np

# Middle-grey target for spot metering — an 18% card exposed correctly.
GREY_TARGET = 0.18

# Rec.709 luma weights.
_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Rec.709 luminance of an (..., 3) linear array."""
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


# ---------- white balance: kelvin / tint <-> rgb scale ----------

def kelvin_to_rgb_scale(kelvin: float) -> np.ndarray:
    """Tanner-Helland blackbody approximation -> luminance-neutral RGB scale.

    The returned vector, when multiplied into a linear image, warms it for
    low kelvin and cools it for high kelvin (camera-WB convention).
    """
    t = float(kelvin) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * np.log(max(t, 1e-6)) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10.0) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
        b = 255.0
    rgb = np.clip(np.array([r, g, b], dtype=np.float32) / 255.0, 1e-4, None)
    scale = 1.0 / rgb
    scale /= np.mean(scale)
    return scale.astype(np.float32)


def temp_tint_to_scale(kelvin: float, tint: float) -> np.ndarray:
    """Combine a kelvin curve with a green<->magenta tint into one RGB scale.

    `tint` is in [-1, 1]: negative pushes green, positive pushes magenta.
    Result is mean-normalised so it is roughly luminance-preserving.
    """
    kelvin_scale = kelvin_to_rgb_scale(kelvin)
    t = float(np.clip(tint, -1.0, 1.0))
    tint_scale = np.array(
        [1.0 + 0.35 * t, 1.0 - 0.70 * t, 1.0 + 0.35 * t], dtype=np.float32
    )
    tint_scale = np.clip(tint_scale, 0.05, None)
    scale = kelvin_scale * tint_scale
    scale /= max(float(np.mean(scale)), 1e-8)
    return scale.astype(np.float32)


def scale_to_temp_tint(scale: np.ndarray) -> tuple[float, float]:
    """Inverse of `temp_tint_to_scale` — back-solve the Temp/Tint sliders.

    Used when WB is set by sampling an area or by the auto-meter. The tint
    factor scales R and B equally, so it cancels in the R/B ratio: kelvin is
    found by bisection on R/B (monotonic in K), then tint closes the residual.
    """
    s = np.asarray(scale, dtype=np.float64)
    target_rb = float(s[0] / (s[2] + 1e-8))

    # Clamp the target into the achievable R/B range so bisection converges
    # to an endpoint rather than diverging.
    rb_lo = float(kelvin_to_rgb_scale(2000.0)[0] / kelvin_to_rgb_scale(2000.0)[2])
    rb_hi = float(kelvin_to_rgb_scale(15000.0)[0] / kelvin_to_rgb_scale(15000.0)[2])
    target_rb = float(np.clip(target_rb, rb_lo, rb_hi))

    lo, hi = 2000.0, 15000.0
    for _ in range(40):  # R/B is monotonically increasing in kelvin
        mid = 0.5 * (lo + hi)
        k = kelvin_to_rgb_scale(mid)
        if float(k[0] / (k[2] + 1e-8)) < target_rb:
            lo = mid
        else:
            hi = mid
    kelvin = 0.5 * (lo + hi)

    # Residual after removing the kelvin component matches the tint model
    # [1+0.35t, 1-0.70t, 1+0.35t] up to a scalar. q = G/R closes for t.
    k = kelvin_to_rgb_scale(kelvin).astype(np.float64)
    r = s / np.clip(k, 1e-8, None)
    q = r[1] / (r[0] + 1e-8)
    tint = (1.0 - q) / (0.70 + 0.35 * q + 1e-8)
    return float(kelvin), float(np.clip(tint, -1.0, 1.0))


def grey_world_scale(mean_rgb: np.ndarray) -> np.ndarray:
    """Neutralising RGB scale for a region whose mean should read achromatic.

    `scale_c = L / mean_rgb_c` makes every channel equal the region's luma;
    the result is luminance-neutral so applying it preserves brightness.
    Large corrections (> 3x relative to green) are clamped.
    """
    m = np.clip(np.asarray(mean_rgb, dtype=np.float64), 1e-8, None)
    L = float(m @ _LUMA)
    if L < 1e-8:
        return np.ones(3, dtype=np.float32)
    raw = L / m
    rel = raw / (raw[1] + 1e-8)
    MAX = 3.0
    if np.any(rel > MAX) or np.any(rel < 1.0 / MAX):
        rel = np.clip(rel, 1.0 / MAX, MAX)
        raw = rel * raw[1]
    raw = raw / max(float(raw @ _LUMA), 1e-8)
    return raw.astype(np.float32)


# ---------- spot metering ----------

def spot_meter_offset_ev(region_rgb: np.ndarray, target: float = GREY_TARGET) -> float:
    """EV offset that brings a sampled region's mean luminance to `target`.

    Camera spot-meter behaviour: average the patch to middle grey. Returns an
    absolute baseline-exposure offset in stops (replaces, not adds to, any
    current offset).
    """
    px = np.asarray(region_rgb, dtype=np.float64).reshape(-1, 3)
    if px.size == 0:
        return 0.0
    L = float(luminance(px.mean(axis=0)))
    if L < 1e-8:
        return 0.0
    return float(np.log2(target / L))


# ---------- "convolve the dome": cosine-weighted gray-ball render ----------

def _latlong_dirs(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction per ERP pixel (y-up) + per-pixel solid angle [sr]."""
    ys = (np.arange(h) + 0.5) / h
    xs = (np.arange(w) + 0.5) / w
    theta = ys[:, None] * np.pi
    phi = (xs[None, :] * 2.0 - 1.0) * np.pi
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = np.broadcast_to(np.cos(theta), (h, w))
    z = sin_t * np.sin(phi)
    dirs = np.stack([x, np.broadcast_to(y, (h, w)), z], axis=-1).astype(np.float32)
    d_omega = (sin_t * (np.pi / h) * (2.0 * np.pi / w)).astype(np.float32)
    d_omega = np.broadcast_to(d_omega, (h, w)).astype(np.float32)
    return dirs, d_omega


def _sphere_normals(res: int) -> tuple[np.ndarray, np.ndarray]:
    """Camera-facing hemisphere of unit normals for a `res`x`res` gray ball."""
    g = (np.arange(res) + 0.5) / res * 2.0 - 1.0
    xx, yy = np.meshgrid(g, -g)
    r2 = xx * xx + yy * yy
    mask = r2 <= 1.0
    zz = np.sqrt(np.clip(1.0 - r2, 0.0, None))
    normals = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
    return normals, mask


def convolve_dome_meter(
    hdr: np.ndarray, albedo: float = GREY_TARGET, res: int = 48, env_max_w: int = 256
) -> dict:
    """Render a low-res Lambertian gray ball lit by the HDRI and meter it.

    This is the "fancy" auto mode: it convolves the whole dome with a
    cosine kernel, so every direction contributes by its solid angle x N.L —
    physically correct unlike a flat pixel average. The lit ball's mean RGB
    drives both auto exposure (luminance -> middle grey) and auto WB
    (neutralise the colour cast).

    Returns a dict with `offset_ev`, `wb_scale`, `mean_rgb`.
    """
    import cv2

    env = np.asarray(hdr, dtype=np.float32)
    h, w = env.shape[:2]
    if w > env_max_w:
        new_w = env_max_w
        new_h = max(2, (new_w * h) // w)
        env = cv2.resize(env, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = env.shape[:2]

    dirs, d_omega = _latlong_dirs(h, w)
    normals, mask = _sphere_normals(res)

    env_dirs = dirs.reshape(-1, 3)
    env_rgb = np.clip(env.reshape(-1, 3), 0.0, None)
    env_omega = d_omega.reshape(-1)
    norms = normals.reshape(-1, 3)

    n_env = env_dirs.shape[0]
    out = np.zeros((norms.shape[0], 3), dtype=np.float32)
    # Chunk over ball pixels to cap the transient N.L matrix at ~64 MB.
    chunk = max(1, int(64 * 1024 * 1024 / (n_env * 4)))
    for i in range(0, norms.shape[0], chunk):
        cn = norms[i:i + chunk]
        ndl = np.clip(cn @ env_dirs.T, 0.0, None)
        out[i:i + chunk] = (ndl * env_omega[None, :]) @ env_rgb

    ball = out.reshape(res, res, 3) * (float(albedo) / np.pi)
    lit = ball.reshape(-1, 3)[mask.reshape(-1)]
    mean_rgb = lit.mean(axis=0).astype(np.float32) if lit.size else np.zeros(3, np.float32)

    L = float(luminance(mean_rgb))
    offset_ev = float(np.log2(GREY_TARGET / L)) if L > 1e-8 else 0.0
    wb_scale = grey_world_scale(mean_rgb)
    return {
        "offset_ev": offset_ev,
        "wb_scale": wb_scale,
        "mean_rgb": mean_rgb,
    }
