"""env2lgt project file format (*.env2lgt.json).

A project file captures everything the user has done in a session — the
source EXR reference, drawn quads (as absolute 4-corner spherical dirs),
scene/display settings, and export option toggles — so they can come back
later and pick up exactly where they left off.

By default a project is saved alongside its source EXR with the suffix
`.env2lgt.json` (so `room.exr` -> `room.env2lgt.json`). On EXR open the
app probes for a sibling project and offers to restore it.

The format is plain JSON, version-tagged, human-readable. Forward-
compatible: unknown keys at load time are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np


FILE_VERSION = 1
PROJECT_SUFFIX = ".env2lgt.json"


@dataclass
class QuadState:
    name: str
    corners_dirs: list[list[float]]  # (4, 3) — float-list-of-lists for JSON


@dataclass
class SceneState:
    scene_scale: float = 100.0
    yaw_offset_deg: float = 0.0
    exposure_ev: float = 0.0
    dome_rotate_y_deg: float = -180.0


@dataclass
class ExportState:
    # Defaults match the LightPanel checkbox defaults so a hand-written
    # project (or one missing the `export` block) loads the same way as a
    # freshly-launched UI.
    dome: bool = True
    rect: bool = True
    usd: bool = True
    depth_exr: bool = False
    depth_mesh: bool = False
    masks: bool = True
    output_dir: str = ""


@dataclass
class Project:
    source_exr: str
    scene: SceneState = field(default_factory=SceneState)
    quads: list[QuadState] = field(default_factory=list)
    export: ExportState = field(default_factory=ExportState)
    saved_at: str = ""
    format: str = "env2lgt-project"
    version: int = FILE_VERSION


def default_project_path(source_exr: str | Path) -> Path:
    """Where to auto-save / look for the project for a given EXR."""
    p = Path(source_exr)
    return p.with_name(p.stem + PROJECT_SUFFIX)


def save_project(path: str | Path, project: Project) -> Path:
    """Write a project to disk. Returns the resolved path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    project.saved_at = datetime.now().isoformat(timespec="seconds")
    data = asdict(project)
    out.write_text(json.dumps(data, indent=2))
    return out


def load_project(path: str | Path) -> Project:
    """Read a project from disk. Validates format tag; tolerates unknown keys."""
    p = Path(path)
    raw = json.loads(p.read_text())
    if raw.get("format") != "env2lgt-project":
        raise ValueError(
            f"{p.name} is not an env2lgt project file (missing 'format' tag)."
        )
    # Accept any version <= FILE_VERSION; warn on newer
    ver = int(raw.get("version", 1))
    if ver > FILE_VERSION:
        import warnings
        warnings.warn(
            f"{p.name} is version {ver}, this build supports up to {FILE_VERSION}; "
            "unknown fields will be ignored.",
            stacklevel=2,
        )
    scene_raw = raw.get("scene", {})
    export_raw = raw.get("export", {})

    def _filt(d: dict, allowed_keys: set[str]) -> dict:
        return {k: v for k, v in d.items() if k in allowed_keys}

    scene = SceneState(**_filt(scene_raw, {f.name for f in SceneState.__dataclass_fields__.values()}))  # type: ignore[attr-defined]
    export = ExportState(**_filt(export_raw, {f.name for f in ExportState.__dataclass_fields__.values()}))  # type: ignore[attr-defined]
    quads = [
        QuadState(name=q["name"], corners_dirs=[[float(c) for c in row] for row in q["corners_dirs"]])
        for q in raw.get("quads", [])
    ]
    return Project(
        source_exr=raw["source_exr"],
        scene=scene,
        quads=quads,
        export=export,
        saved_at=raw.get("saved_at", ""),
        version=ver,
    )


def project_from_app_state(
    source_exr: str | Path,
    quads: list,                # list of LightQuad-like (.name, .corners_dirs)
    scene_scale: float,
    yaw_offset_deg: float,
    exposure_ev: float,
    dome_rotate_y_deg: float,
    export_opts: dict | None = None,
) -> Project:
    """Convenience constructor used by the app on save."""
    e = export_opts or {}
    return Project(
        source_exr=str(source_exr),
        scene=SceneState(
            scene_scale=float(scene_scale),
            yaw_offset_deg=float(yaw_offset_deg),
            exposure_ev=float(exposure_ev),
            dome_rotate_y_deg=float(dome_rotate_y_deg),
        ),
        quads=[
            QuadState(
                name=q.name,
                corners_dirs=[[float(c) for c in row] for row in np.asarray(q.corners_dirs).tolist()],
            )
            for q in quads
        ],
        export=ExportState(
            dome=bool(e.get("dome", True)),
            rect=bool(e.get("rect", True)),
            usd=bool(e.get("usd", True)),
            depth_exr=bool(e.get("depth_exr", False)),
            depth_mesh=bool(e.get("depth_mesh", True)),
            masks=bool(e.get("masks", True)),
            output_dir=str(e.get("output_dir", "")),
        ),
    )
