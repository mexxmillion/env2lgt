"""USD light-rig authoring.

Stage layout:

    /World (Xform, defaultPrim)
        /lights (Scope)
            /dome      (UsdLuxDomeLight, dome.exr)
            /rect_NN   (UsdLuxRectLight, light_NN.exr) [* per extracted mask]

Convention: Y-up, meters. The dome's intrinsic forward axis in USD points
toward -Z when un-rotated, which matches our equirect math (see
env2lgt.lights.extract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RectLightSpec:
    name: str
    center: np.ndarray  # (3,)
    normal: np.ndarray  # (3,) unit, facing the camera
    u_axis: np.ndarray  # (3,) unit
    v_axis: np.ndarray  # (3,) unit
    width: float
    height: float
    color: np.ndarray   # (3,) 0..1
    intensity: float
    texture_path: str | None = None


def _rotation_matrix_from_axes(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Build a 4x4 (column-major) transform whose local axes are (u, v, n).

    UsdLuxRectLight is a unit square in the XY plane facing -Z. Our local
    basis must map +X -> u, +Y -> v, -Z -> n  (so light faces along its
    normal). So the local Z axis = -n.
    """
    x = u / (np.linalg.norm(u) + 1e-12)
    y = v / (np.linalg.norm(v) + 1e-12)
    z = -n / (np.linalg.norm(n) + 1e-12)
    R = np.eye(4, dtype=np.float64)
    R[:3, 0] = x
    R[:3, 1] = y
    R[:3, 2] = z
    return R


def write_minimal_stage(path: str | Path) -> None:
    """Tiny smoke-test stage: just a dome light."""
    from pxr import Sdf, Usd, UsdGeom, UsdLux

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/lights"))
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome"))
    dome.CreateIntensityAttr(1.0)
    stage.GetRootLayer().Save()


def write_light_rig(
    out_path: str | Path,
    dome_texture: str | Path | None,
    rect_lights: list[RectLightSpec],
    meters_per_unit: float = 1.0,
    dome_intensity: float = 1.0,
) -> Path:
    """Author a complete light-rig USD.

    Texture paths are stored relative to the USDA file when possible.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux

    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))

    world = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, Sdf.Path("/World/lights"))

    # Dome
    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome"))
    dome.CreateIntensityAttr(float(dome_intensity))
    if dome_texture is not None:
        rel = _relative_to(out_path, Path(dome_texture))
        dome.CreateTextureFileAttr(Sdf.AssetPath(rel))
        dome.CreateTextureFormatAttr("latlong")
    # Convention bridge: USD's `UsdLuxDomeLight` shader has a different
    # azimuthal origin than our equirect. The total compensation rotation
    # required for textures + rect lights to line up in usdview is **270°
    # around Y** (equivalent to -90°). Verified by eyeballing the result:
    # bright fixtures in the dome texture sit exactly under the matching
    # rect-light prims. If you swap USD versions (or test in a non-Storm
    # renderer like Karma) and the lights drift, this is the line.
    dome_xf = UsdGeom.Xformable(dome.GetPrim())
    dome_xf.AddRotateYOp().Set(270.0)

    # Rects
    for lr in rect_lights:
        prim_path = Sdf.Path(f"/World/lights/{lr.name}")
        rect = UsdLux.RectLight.Define(stage, prim_path)
        rect.CreateWidthAttr(float(lr.width))
        rect.CreateHeightAttr(float(lr.height))
        rect.CreateIntensityAttr(float(lr.intensity))
        rect.CreateColorAttr(Gf.Vec3f(*[float(c) for c in lr.color]))
        if lr.texture_path is not None:
            rel = _relative_to(out_path, Path(lr.texture_path))
            rect.CreateTextureFileAttr(Sdf.AssetPath(rel))

        # set transform
        R = _rotation_matrix_from_axes(lr.u_axis, lr.v_axis, lr.normal)
        R[:3, 3] = lr.center  # translation in last column (Gf uses row vectors)
        xf = UsdGeom.Xformable(rect.GetPrim())
        op = xf.AddTransformOp()
        # Gf.Matrix4d is row-major; numpy R is currently column-major
        m = Gf.Matrix4d(*R.T.flatten().tolist())
        op.Set(m)

    stage.GetRootLayer().Save()
    return out_path


def _relative_to(usda_path: Path, asset_path: Path) -> str:
    """Try to make asset path relative to the USDA file, else absolute."""
    try:
        return str(asset_path.resolve().relative_to(usda_path.parent.resolve())).replace("\\", "/")
    except ValueError:
        return str(asset_path.resolve()).replace("\\", "/")
