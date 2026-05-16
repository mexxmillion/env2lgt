"""End-to-end bake: latlong EXR + LightQuads -> USD light rig + textures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from env2lgt.depth import get_backend
from env2lgt.io.exr import load_latlong, save_exr
from env2lgt.lights.extract import (
    _output_size_for_quad,
    luminance,
    mean_color_masked,
    rect_from_quad,
    sample_rect_texture,
    total_emitted_power_masked,
)
from env2lgt.lights.inpaint import edge_extend
from env2lgt.proj import rasterize_spherical_quad
from env2lgt.usd.lightrig import RectLightSpec, write_light_rig
from env2lgt.usd.mesh import write_panorama_mesh


@dataclass
class QuadSpec:
    """Bake-time input: name + 4 spherical corner dirs (unit vectors)."""

    name: str
    corners_dirs: np.ndarray  # (4, 3) float64
    # Window/portal: keep the rect at wall depth instead of sliding it in to
    # the bright region (the "light" is distant sky through the opening).
    is_window: bool = False


@dataclass
class BakeOptions:
    write_dome: bool = True
    write_rects: bool = True
    write_usd: bool = True
    write_depth_exr: bool = False
    write_depth_mesh: bool = False
    write_mask_json: bool = True
    # Depth backend: "da2" (scale-invariant) or "dap" (metric). For metric
    # backends `scene_scale` is a fine-tune multiplier rather than the
    # primary meters-per-unit knob.
    depth_backend: str = "da2"
    scene_scale: float = 1.0
    intensity_normalization: float = 1.0
    mask_dilate_px: int = 4
    inpaint_radius: int = 6
    inpaint_feather_px: int = 8
    # Display-only metadata recorded in masks.json. The bake itself doesn't use
    # this — quad corners are absolute.
    yaw_offset_deg: float = 0.0
    # Dome light azimuthal compensation around Y, in degrees. Bridges between
    # our equirect convention and USD/Hydra's dome shader. -180 (= +180) has
    # been the working value in recent versions of Storm; other renderers
    # (Karma, RenderMan) may differ. Configurable via the UI so the user
    # doesn't need a code edit to dial it in.
    dome_rotate_y_deg: float = -180.0


def bake(
    exr_path: str | Path,
    out_dir: str | Path,
    quads: list[QuadSpec],
    options: BakeOptions | None = None,
    progress_cb=None,
) -> dict:
    opts = options or BakeOptions()
    exr_path = Path(exr_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _p(stage: str, frac: float) -> None:
        if progress_cb is not None:
            progress_cb(stage, frac)

    _p("loading panorama", 0.05)
    hdr = load_latlong(exr_path)
    H, W, _ = hdr.shape

    backend = get_backend(opts.depth_backend)
    _p(f"estimating depth ({backend.name.upper()})", 0.10)
    distance = backend.estimate_depth(exr_path, cache_dir=out_dir)
    if distance.shape != (H, W):
        import cv2

        distance = cv2.resize(distance, (W, H), interpolation=cv2.INTER_LINEAR)

    if opts.write_depth_exr:
        save_exr(out_dir / "depth.exr", distance[..., None].repeat(3, axis=-1), half=False)

    lum_full = luminance(hdr)

    # ---------- per-quad extract ----------
    rasterized_masks: list[np.ndarray] = []
    specs: list[RectLightSpec] = []
    summaries: list[dict] = []
    _p("extracting lights", 0.20)
    for i, q in enumerate(quads):
        _p(f"  rasterizing {q.name}", 0.20 + 0.4 * (i + 0.2) / max(1, len(quads)))
        mask_full, _bbox = rasterize_spherical_quad(
            q.corners_dirs, H, W, pad_px=opts.mask_dilate_px
        )
        if not mask_full.any():
            summaries.append({"name": q.name, "skipped": "mask empty (quad missed pano?)"})
            continue
        rasterized_masks.append(mask_full)

        _p(f"  fitting {q.name}", 0.20 + 0.4 * (i + 0.5) / max(1, len(quads)))
        # Geometry from quad corners + median mask depth. No RANSAC on noisy
        # per-pixel depth.
        fit = rect_from_quad(
            q.corners_dirs,
            mask_full,
            distance,
            opts.scene_scale,
            lum_full=lum_full,
            treat_as_window=q.is_window,
        )

        # Texture: rectilinear projection onto the quad surface so that, when
        # mapped 1:1 onto the flat UsdLuxRectLight, it reproduces what the
        # camera sees in the panorama.
        tex_path: Path | None = None
        if opts.write_rects:
            out_h, out_w = _output_size_for_quad(fit.width, fit.height, max_dim=1024)
            tex = sample_rect_texture(hdr, q.corners_dirs, out_h, out_w)
            tex_path = out_dir / f"{q.name}.exr"
            save_exr(tex_path, tex)

        # UsdLuxRectLight emits intensity * color * textureFile. With a texture,
        # the texture already carries the scene-linear HDR radiance, so leave
        # intensity at 1 and color white — otherwise exposure is double-applied
        # and the (tinted) mean color remaps the texture's chromaticity. Without
        # a texture, fall back to the integrated-power proxy + mean color.
        if tex_path is not None:
            color = np.ones(3, dtype=np.float32)
            intensity = 1.0
        else:
            color = mean_color_masked(hdr, mask_full)
            power = total_emitted_power_masked(hdr, mask_full, (H, W))
            intensity = float(power) * float(opts.intensity_normalization)

        specs.append(
            RectLightSpec(
                name=q.name,
                center=fit.center,
                normal=fit.normal,
                u_axis=fit.u_axis,
                v_axis=fit.v_axis,
                width=fit.width,
                height=fit.height,
                color=color,
                intensity=intensity,
                texture_path=str(tex_path) if tex_path else None,
            )
        )
        summaries.append({
            "name": q.name,
            "center": [float(v) for v in fit.center],
            "normal": [float(v) for v in fit.normal],
            "size": [fit.width, fit.height],
            "inlier_ratio": fit.inlier_ratio,
            "intensity": intensity,
        })

    # ---------- dome residual ----------
    dome_tex_path: Path | None = None
    if opts.write_dome:
        _p("filling dome residual (edge extend)", 0.78)
        if rasterized_masks:
            combined = np.zeros((H, W), dtype=np.uint8)
            for m in rasterized_masks:
                combined |= m
            dome_hdr = edge_extend(
                hdr,
                combined,
                iters=96,
                sigma=3.0,
                feather_px=opts.inpaint_feather_px,
            )
        else:
            dome_hdr = hdr.copy()
        dome_tex_path = out_dir / "dome.exr"
        save_exr(dome_tex_path, dome_hdr)

    # ---------- USD ----------
    usd_path: Path | None = None
    if opts.write_usd:
        _p("writing USD light rig", 0.92)
        usd_path = write_light_rig(
            out_dir / "lightrig.usda",
            dome_texture=dome_tex_path,
            rect_lights=specs,
            meters_per_unit=1.0,
            dome_intensity=1.0,
            dome_rotate_y_deg=opts.dome_rotate_y_deg,
        )

    # ---------- depth mesh USD (validation) ----------
    mesh_path: Path | None = None
    if opts.write_depth_mesh:
        _p("writing depth mesh USD", 0.96)
        mesh_path = write_panorama_mesh(
            out_dir / "panorama_geo.usda",
            distance=distance,
            dome_texture=dome_tex_path,
            scene_scale=opts.scene_scale,
        )

    # ---------- mask sidecar ----------
    if opts.write_mask_json:
        import json

        (out_dir / "masks.json").write_text(
            json.dumps(
                {
                    "yaw_offset_deg": float(opts.yaw_offset_deg),
                    "scene_scale": float(opts.scene_scale),
                    "depth_backend": backend.name,
                    "is_metric": bool(backend.is_metric),
                    "quads": [
                        {
                            "name": q.name,
                            "is_window": bool(q.is_window),
                            "corners_dirs": [
                                [float(c) for c in row] for row in q.corners_dirs.tolist()
                            ],
                        }
                        for q in quads
                    ],
                },
                indent=2,
            )
        )

    _p("done", 1.0)
    return {
        "panorama": str(exr_path),
        "output_dir": str(out_dir),
        "depth_backend": backend.name,
        "is_metric": bool(backend.is_metric),
        "usd": str(usd_path) if usd_path else None,
        "dome": str(dome_tex_path) if dome_tex_path else None,
        "mesh": str(mesh_path) if mesh_path else None,
        "rect_lights": [s.name for s in specs],
        "rect_fits": summaries,
    }
