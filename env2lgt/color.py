"""OCIO colour management.

env2lgt works internally in a single scene-linear working space (the OCIO
config's `scene_linear` role — ACEScg in an ACES config). The pipeline is:

    source EXR --input transform-->  WORKING (ACEScg)
        ...all metering / WB / chart / bake math happens in WORKING...
    WORKING --output transform--> exported dome / rect EXRs
    WORKING --display+view-->     the viewport

The OCIO config is taken from the `$OCIO` environment variable. If OCIO is
unavailable the app falls back to a fixed ACES-filmic display (see app.py)
and the colour-management UI is disabled.
"""

from __future__ import annotations

import numpy as np

try:
    import PyOpenColorIO as ocio

    _OCIO_IMPORTED = True
except Exception:  # noqa: BLE001
    ocio = None
    _OCIO_IMPORTED = False

# The working space — the config's scene-linear role.
WORKING = "scene_linear"

_config = None
_config_err: str | None = None
_proc_cache: dict = {}


def _cfg():
    global _config, _config_err
    if _config is None and _config_err is None:
        try:
            _config = ocio.Config.CreateFromEnv()
        except Exception as e:  # noqa: BLE001
            _config_err = str(e)
    if _config is None:
        raise RuntimeError(_config_err or "OCIO config unavailable")
    return _config


def ocio_available() -> bool:
    """True if PyOpenColorIO is importable and $OCIO points at a valid config."""
    if not _OCIO_IMPORTED:
        return False
    try:
        _cfg()
        return True
    except Exception:  # noqa: BLE001
        return False


def config_path() -> str:
    import os

    return os.environ.get("OCIO", "(unset)")


# ---------- introspection (for the UI dropdowns) ----------

def colorspace_names() -> list[str]:
    return sorted(_cfg().getColorSpaceNames())


def working_colorspace() -> str:
    """Resolved name of the working space (the scene_linear role target)."""
    return _cfg().getColorSpace(WORKING).getName()


def displays() -> list[str]:
    return list(_cfg().getDisplays())


def views(display: str) -> list[str]:
    return list(_cfg().getViews(display))


def default_display() -> str:
    return _cfg().getDefaultDisplay()


def default_view(display: str) -> str:
    return _cfg().getDefaultView(display)


# ---------- processing ----------

def _apply(cpu, img: np.ndarray) -> np.ndarray:
    """Run a CPU processor over an (H, W, 3+) array. Returns a new (H, W, 3)
    float32 array — the input is never mutated."""
    a = np.array(img[..., :3], dtype=np.float32, copy=True, order="C")
    h, w = a.shape[:2]
    desc = ocio.PackedImageDesc(a, w, h, ocio.ChannelOrdering.CHANNEL_ORDERING_RGB)
    cpu.apply(desc)
    return a


def _convert_cpu(src: str, dst: str):
    key = ("cs", src, dst)
    cpu = _proc_cache.get(key)
    if cpu is None:
        cpu = _cfg().getProcessor(src, dst).getDefaultCPUProcessor()
        _proc_cache[key] = cpu
    return cpu


def convert(img: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Convert an image between two colorspaces (no-op-safe if src == dst)."""
    if src == dst:
        return np.asarray(img[..., :3], dtype=np.float32)
    return _apply(_convert_cpu(src, dst), img)


def to_working(img: np.ndarray, src: str) -> np.ndarray:
    """Input transform: source colorspace -> WORKING."""
    return convert(img, src, WORKING)


def from_working(img: np.ndarray, dst: str) -> np.ndarray:
    """Output transform: WORKING -> destination colorspace."""
    return convert(img, WORKING, dst)


def make_display_cpu(display: str, view: str):
    """CPU processor for WORKING -> (display, view). Cached per pair."""
    key = ("dv", display, view)
    cpu = _proc_cache.get(key)
    if cpu is None:
        tr = ocio.DisplayViewTransform()
        tr.setSrc(WORKING)
        tr.setDisplay(display)
        tr.setView(view)
        cpu = _cfg().getProcessor(tr).getDefaultCPUProcessor()
        _proc_cache[key] = cpu
    return cpu


def display_to_u8(img_working: np.ndarray, display_cpu) -> np.ndarray:
    """Apply a display+view processor and encode to contiguous (H, W, 3) uint8."""
    disp = _apply(display_cpu, img_working)
    u8 = (np.clip(disp, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(u8)
