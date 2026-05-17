"""Depth-displaced UV-sphere mesh for panorama-as-geometry validation.

Each vertex of a latlong sphere is moved radially by the DA-2 distance at its
(u, v) pano pixel, multiplied by the user's scene scale. Spherical UVs map
`dome.exr` directly onto the surface via UsdPreviewSurface emissiveColor —
opening the resulting USD in usdview shows the panorama wrapped onto the
estimated 3D structure of the room. Rect lights authored in the matching
`lightrig.usda` line up with the actual lights visible in the mesh.

UV convention (matches OpenGL / standard USD):
    (u=0, v=0)   = bottom-left  → pano (col=0,   row=H-1) → -π yaw, -π/2 pitch (south pole)
    (u=1, v=1)   = top-right    → pano (col=W,   row=0)   → +π yaw,  +π/2 pitch (north pole)
    (u=0.5, v=0.5) = center     → pano (col=W/2, row=H/2) → 0 yaw,    0 pitch  (-Z forward)
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def write_panorama_mesh(
    out_path: str | Path,
    distance: np.ndarray,
    dome_texture: str | Path | None,
    scene_scale: float = 1.0,
    segments_lon: int = 256,
    segments_lat: int = 128,
    radius_floor: float = 0.0,
    use_emissive: bool = True,
    geom_inflation: float = 1.0,
    open_sky: bool = True,
    sky_depth_frac: float = 0.45,
) -> Path:
    """Author the depth-displaced sphere mesh USD.

    Parameters
    ----------
    out_path : USDA path to write to.
    distance : (H, W) float32 map (DA-2 raw output; scale-invariant).
    dome_texture : path to dome.exr — bound as the mesh's emissive texture.
        If None, the mesh is written without a material.
    scene_scale : meters per DA-2 unit (the same value used for the rect
        lights so geometry stays consistent).
    segments_lon, segments_lat : tesselation. 256×128 ≈ 33k verts, fine for
        usdview at interactive rates.
    radius_floor : minimum world-space radius for any vertex (clamp small
        depth values so the mesh doesn't pinch at the origin). 0 disables.
    use_emissive : when True, the dome texture drives emissiveColor so the
        mesh self-lights in usdview without needing the dome light to render.
        When False, the texture drives diffuseColor (mesh needs lighting).
    geom_inflation : radial scale (>= 1.0) pushing every vertex outward, so the
        mesh sits just behind the rect lights instead of coplanar with them.
        1.0 disables; 1.025 ≈ 2.5% bigger.
    open_sky : when True, the sky faces are dropped, leaving the mesh open so
        the real dome / 3D sky shows through. The sky is found as the far
        region (depth above `sky_depth_frac` of the normalised depth range)
        that connects to the zenith — so a far building that doesn't reach
        straight up is not mistaken for sky.
    sky_depth_frac : normalised-depth cutoff (0..1) for the far/sky split.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = distance.shape

    # ---------- vertex grid (vectorized) ----------
    n_lat = segments_lat + 1
    n_lon = segments_lon + 1
    # i = latitude index (0 = north pole / top), j = longitude index
    i_idx = np.arange(n_lat, dtype=np.float64)
    j_idx = np.arange(n_lon, dtype=np.float64)
    ii, jj = np.meshgrid(i_idx, j_idx, indexing="ij")  # (n_lat, n_lon)

    # Spherical angles.
    pitch = np.pi * 0.5 - np.pi * (ii / segments_lat)            # +pi/2 .. -pi/2
    yaw = -np.pi + 2.0 * np.pi * (jj / segments_lon)             # -pi .. +pi

    # UV (st): u increases left→right (yaw), v=1 at top of image (north pole).
    u_st = jj / segments_lon
    v_st = 1.0 - ii / segments_lat

    # World-direction per vertex (our convention: forward = -Z).
    cos_p = np.cos(pitch)
    dirs = np.stack(
        [np.sin(yaw) * cos_p, np.sin(pitch), -np.cos(yaw) * cos_p],
        axis=-1,
    )

    # Sample depth at corresponding pano pixels (bilinear, wrap horizontally).
    # pano_col in [0, W], pano_row in [0, H].
    pano_col = u_st * (W - 1)
    pano_row = (1.0 - v_st) * (H - 1)
    depths = cv2.remap(
        distance.astype(np.float32),
        pano_col.astype(np.float32),
        pano_row.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    radius = depths * float(scene_scale) * float(geom_inflation)
    if radius_floor > 0:
        radius = np.maximum(radius, float(radius_floor))

    points = (dirs * radius[..., None]).astype(np.float32)
    uvs = np.stack([u_st, v_st], axis=-1).astype(np.float32)

    points_flat = points.reshape(-1, 3)
    uvs_flat = uvs.reshape(-1, 2)

    # ---------- faces (quads) ----------
    i_face = np.arange(segments_lat, dtype=np.int64)
    j_face = np.arange(segments_lon, dtype=np.int64)
    ifg, jfg = np.meshgrid(i_face, j_face, indexing="ij")
    v0 = (ifg * n_lon + jfg)
    v1 = (ifg * n_lon + jfg + 1)
    v2 = ((ifg + 1) * n_lon + jfg + 1)
    v3 = ((ifg + 1) * n_lon + jfg)
    quads = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)

    # Open-sky: drop the sky faces so the mesh has a hole there and the real
    # dome shows through. The sky is the far region that connects to the
    # zenith (top row of the equirect) — a far building that doesn't reach
    # straight up stays. A face goes only if all 4 vertices are sky, keeping
    # the horizon rim intact.
    if open_sky:
        d = depths.astype(np.float64)
        lo = float(np.percentile(d, 1.0))
        hi = float(np.percentile(d, 99.9))
        norm = np.clip((d - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
        far = (norm > float(sky_depth_frac)).astype(np.uint8)
        _, labels = cv2.connectedComponents(far)
        top = [int(v) for v in np.unique(labels[0, :]) if v != 0]
        sky_v = np.isin(labels, top) if top else far.astype(bool)
        face_sky = (
            sky_v[ifg, jfg] & sky_v[ifg, jfg + 1]
            & sky_v[ifg + 1, jfg + 1] & sky_v[ifg + 1, jfg]
        ).reshape(-1)
        quads = quads[~face_sky]

    face_indices = quads.reshape(-1).astype(np.int32)
    face_counts = np.full(quads.shape[0], 4, dtype=np.int32)

    # ---------- write USD ----------
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/geo"))

    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path("/World/geo/panorama"))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points_flat))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(face_counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(face_indices))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    # Backface culling on the inside of the sphere is annoying for inspection.

    # Extent (Gf.Vec3f requires python floats, not numpy float32 scalars).
    # Measure over referenced vertices only — dropped sky verts sit at huge
    # radius and would otherwise blow up the bounding box.
    ext_pts = (
        points_flat[np.unique(face_indices)]
        if open_sky and face_indices.size
        else points_flat
    )
    bbox_min = [float(v) for v in ext_pts.min(axis=0)]
    bbox_max = [float(v) for v in ext_pts.max(axis=0)]
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*bbox_min), Gf.Vec3f(*bbox_max)]))

    # st primvar (per-vertex)
    st_pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    )
    st_pv.Set(Vt.Vec2fArray.FromNumpy(uvs_flat))

    # ---------- material ----------
    if dome_texture is not None:
        UsdGeom.Scope.Define(stage, Sdf.Path("/World/materials"))
        mat_path = Sdf.Path("/World/materials/panorama")
        material = UsdShade.Material.Define(stage, mat_path)

        surf = UsdShade.Shader.Define(stage, mat_path.AppendChild("PreviewSurface"))
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        tex = UsdShade.Shader.Define(stage, mat_path.AppendChild("Texture"))
        tex.CreateIdAttr("UsdUVTexture")
        rel_tex = _relative_to(out_path, Path(dome_texture))
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel_tex))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

        reader = UsdShade.Shader.Define(stage, mat_path.AppendChild("PrimvarReader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.ConnectableAPI(), "result"
        )

        tex_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        if use_emissive:
            surf.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )
        else:
            surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.ConnectableAPI(), "rgb"
            )

        material.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        _ = tex_out  # silence unused-name

    stage.GetRootLayer().Save()
    return out_path


def _relative_to(usda_path: Path, asset_path: Path) -> str:
    try:
        return str(asset_path.resolve().relative_to(usda_path.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(asset_path.resolve()).replace("\\", "/")
