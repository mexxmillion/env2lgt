"""Colour management.

env2lgt always works internally in ACEScg linear. Source EXRs are input-
transformed into ACEScg on load; bake outputs are output-transformed on
write; the viewport runs an ACEScg -> display+view transform.

Two backends provide that pipeline:

  * **ocio**     — full PyOpenColorIO + the config in ``$OCIO``. Rich
                   colorspace / display / view lists, accurate transforms.
  * **builtin**  — always available, no dependency on OCIO. Working space
                   is ACEScg; input/output transforms cover ACEScg,
                   sRGB-Linear and sRGB-encoded; a single "sRGB" display
                   with "ACES Filmic" and "Linear" views. Common OCIO
                   colorspace names (``Utility - Linear - sRGB`` etc.) are
                   aliased so projects authored against an OCIO config
                   still load cleanly.

Callers don't branch on the backend — they call the module-level functions
below. Switching backends at runtime is supported via :func:`set_backend`.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import PyOpenColorIO as ocio

    _OCIO_IMPORTED = True
except Exception:  # noqa: BLE001
    ocio = None
    _OCIO_IMPORTED = False

# OCIO role used for the working space (also accepted as a colorspace name
# by the builtin backend, where it's an alias for ACEScg).
WORKING = "scene_linear"


# ---------- builtin colour-space constants ----------

# sRGB(D65) linear -> ACEScg (AP1, D60; Bradford-adapted). Row-vector
# convention — apply via `px @ M.T`.
_M_SRGB_TO_ACESCG = np.array([
    [0.61319, 0.33951, 0.04737],
    [0.07021, 0.91634, 0.01345],
    [0.02062, 0.10957, 0.86961],
], dtype=np.float32)

_M_ACESCG_TO_SRGB = np.array([
    [ 1.70505, -0.62179, -0.08326],
    [-0.13026,  1.14080, -0.01055],
    [-0.02400, -0.12897,  1.15297],
], dtype=np.float32)

# Canonical builtin colorspace names + aliases that match what OCIO configs
# commonly call the same thing (so projects round-trip across backends).
_BUILTIN_ACESCG = "ACEScg"
_BUILTIN_SRGB_LINEAR = "sRGB - Linear"
_BUILTIN_SRGB_ENCODED = "sRGB"

_ACESCG_ALIASES = {
    _BUILTIN_ACESCG, "ACES - ACEScg", "ACEScg - linear", "acescg",
    WORKING, "",
}
_SRGB_LINEAR_ALIASES = {
    _BUILTIN_SRGB_LINEAR, "Utility - Linear - sRGB", "Linear sRGB",
    "lin_srgb", "Linear - sRGB",
}
_SRGB_ENCODED_ALIASES = {
    _BUILTIN_SRGB_ENCODED, "Utility - sRGB - Texture", "sRGB - Texture",
    "srgb_tx", "Output - sRGB", "sRGB - Display",
}


# ---------- processors ----------

class _Proc:
    """Abstract image processor. Numpy in, numpy out, never mutates input."""

    def apply(self, img: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _IdentityProc(_Proc):
    def apply(self, img):
        return np.ascontiguousarray(img[..., :3].astype(np.float32, copy=False))


class _MatrixProc(_Proc):
    def __init__(self, M: np.ndarray):
        self._Mt = np.ascontiguousarray(M.astype(np.float32).T)

    def apply(self, img):
        a = img[..., :3].astype(np.float32, copy=False)
        return np.ascontiguousarray(a @ self._Mt)


class _SRGBDecodeProc(_Proc):
    """sRGB-encoded (0..1) -> sRGB linear."""

    def apply(self, img):
        a = img[..., :3].astype(np.float32, copy=False)
        thresh = 0.04045
        return np.where(
            a <= thresh,
            a / 12.92,
            ((np.maximum(a, thresh) + 0.055) / 1.055) ** 2.4,
        ).astype(np.float32)


class _SRGBEncodeProc(_Proc):
    """sRGB linear -> sRGB-encoded (clamped 0..1)."""

    def apply(self, img):
        a = np.clip(img[..., :3].astype(np.float32, copy=False), 0.0, 1.0)
        thresh = 0.0031308
        return np.where(
            a <= thresh,
            12.92 * a,
            1.055 * np.power(np.maximum(a, thresh), 1.0 / 2.4) - 0.055,
        ).astype(np.float32)


class _ChainProc(_Proc):
    def __init__(self, *procs: _Proc):
        self._procs = procs

    def apply(self, img):
        out = img
        for p in self._procs:
            out = p.apply(out)
        return out


class _ACESFilmicProc(_Proc):
    """ACEScg -> sRGB display via the Narkowicz ACES fit. Output is
    already in sRGB-display encoding (0..1, gamma-baked)."""

    def apply(self, img):
        from env2lgt.io.tonemap import aces_filmic

        return aces_filmic(img[..., :3].astype(np.float32, copy=False))


class _LinearSRGBDisplayProc(_Proc):
    """ACEScg -> sRGB(linear) (matrix) -> sRGB-encoded (clipped). No
    tonemap; HDR values clip hard. Useful for verifying linear data."""

    def __init__(self):
        self._mat = _MatrixProc(_M_ACESCG_TO_SRGB)
        self._enc = _SRGBEncodeProc()

    def apply(self, img):
        return self._enc.apply(self._mat.apply(img))


class _OCIOProc(_Proc):
    def __init__(self, cpu):
        self._cpu = cpu

    def apply(self, img):
        a = np.array(img[..., :3], dtype=np.float32, copy=True, order="C")
        h, w = a.shape[:2]
        desc = ocio.PackedImageDesc(
            a, w, h, ocio.ChannelOrdering.CHANNEL_ORDERING_RGB
        )
        self._cpu.apply(desc)
        return a


# ---------- backends ----------

class _Backend:
    name: str = ""

    def working_colorspace(self) -> str: ...
    def colorspace_names(self) -> list[str]: ...
    def displays(self) -> list[str]: ...
    def views(self, display: str) -> list[str]: ...
    def default_display(self) -> str: ...
    def default_view(self, display: str) -> str: ...
    def convert(self, img: np.ndarray, src: str, dst: str) -> np.ndarray: ...
    def make_display_cpu(self, display: str, view: str) -> _Proc: ...


class _OCIOBackend(_Backend):
    name = "ocio"

    def __init__(self):
        self._config = ocio.Config.CreateFromEnv()
        self._cache: dict = {}

    def working_colorspace(self):
        cs = self._config.getColorSpace(WORKING)
        return cs.getName() if cs is not None else WORKING

    def colorspace_names(self):
        return sorted(self._config.getColorSpaceNames())

    def displays(self):
        return list(self._config.getDisplays())

    def views(self, display):
        return list(self._config.getViews(display))

    def default_display(self):
        return self._config.getDefaultDisplay()

    def default_view(self, display):
        return self._config.getDefaultView(display)

    def _cs_cpu(self, src: str, dst: str):
        key = ("cs", src, dst)
        cpu = self._cache.get(key)
        if cpu is None:
            cpu = self._config.getProcessor(src, dst).getDefaultCPUProcessor()
            self._cache[key] = cpu
        return cpu

    def convert(self, img, src, dst):
        if src == dst:
            return np.asarray(img[..., :3], dtype=np.float32)
        return _OCIOProc(self._cs_cpu(src, dst)).apply(img)

    def make_display_cpu(self, display, view):
        key = ("dv", display, view)
        cpu = self._cache.get(key)
        if cpu is None:
            tr = ocio.DisplayViewTransform()
            tr.setSrc(WORKING)
            tr.setDisplay(display)
            tr.setView(view)
            cpu = self._config.getProcessor(tr).getDefaultCPUProcessor()
            self._cache[key] = cpu
        return _OCIOProc(cpu)


class _BuiltinBackend(_Backend):
    """Always-available colour manager. Working = ACEScg. Three input
    colorspaces; one display (sRGB) with two views."""

    name = "builtin"

    _ACESCG = _BUILTIN_ACESCG
    _SRGB_LIN = _BUILTIN_SRGB_LINEAR
    _SRGB_ENC = _BUILTIN_SRGB_ENCODED

    def working_colorspace(self):
        return self._ACESCG

    def colorspace_names(self):
        return [self._ACESCG, self._SRGB_LIN, self._SRGB_ENC]

    def displays(self):
        return ["sRGB"]

    def views(self, display):
        return ["ACES Filmic", "Linear"]

    def default_display(self):
        return "sRGB"

    def default_view(self, display):
        return "ACES Filmic"

    @staticmethod
    def _resolve(cs: str) -> str:
        """Map any known alias to one of the three canonical builtin names.
        Unknown names fall through unchanged — callers treat as identity."""
        if cs in _ACESCG_ALIASES:
            return _BUILTIN_ACESCG
        if cs in _SRGB_LINEAR_ALIASES:
            return _BUILTIN_SRGB_LINEAR
        if cs in _SRGB_ENCODED_ALIASES:
            return _BUILTIN_SRGB_ENCODED
        return cs

    def _to_acescg(self, cs: str) -> _Proc:
        r = self._resolve(cs)
        if r == _BUILTIN_ACESCG:
            return _IdentityProc()
        if r == _BUILTIN_SRGB_LINEAR:
            return _MatrixProc(_M_SRGB_TO_ACESCG)
        if r == _BUILTIN_SRGB_ENCODED:
            return _ChainProc(_SRGBDecodeProc(), _MatrixProc(_M_SRGB_TO_ACESCG))
        return _IdentityProc()

    def _from_acescg(self, cs: str) -> _Proc:
        r = self._resolve(cs)
        if r == _BUILTIN_ACESCG:
            return _IdentityProc()
        if r == _BUILTIN_SRGB_LINEAR:
            return _MatrixProc(_M_ACESCG_TO_SRGB)
        if r == _BUILTIN_SRGB_ENCODED:
            return _LinearSRGBDisplayProc()
        return _IdentityProc()

    def convert(self, img, src, dst):
        if src == dst:
            return np.asarray(img[..., :3], dtype=np.float32)
        working = self._to_acescg(src).apply(img)
        return self._from_acescg(dst).apply(working)

    def make_display_cpu(self, display, view):
        if view == "Linear":
            return _LinearSRGBDisplayProc()
        return _ACESFilmicProc()


# ---------- module-level state ----------

_active: _Backend | None = None
_ocio_err: str | None = None


def _try_make_ocio() -> _OCIOBackend | None:
    global _ocio_err
    if not _OCIO_IMPORTED:
        _ocio_err = "PyOpenColorIO is not installed"
        return None
    try:
        return _OCIOBackend()
    except Exception as e:  # noqa: BLE001
        _ocio_err = str(e)
        return None


def _ensure_active() -> _Backend:
    """Lazy-pick a backend on first use. Prefers OCIO when available."""
    global _active
    if _active is None:
        b = _try_make_ocio()
        _active = b if b is not None else _BuiltinBackend()
    return _active


def available_backends() -> list[str]:
    """Backends the user can switch to. `builtin` is always present."""
    out = ["builtin"]
    if _OCIO_IMPORTED:
        out.append("ocio")
    return out


def ocio_available() -> bool:
    """True iff PyOpenColorIO is importable AND $OCIO points at a valid
    config. Reflects feasibility, not the active backend."""
    if not _OCIO_IMPORTED:
        return False
    if isinstance(_active, _OCIOBackend):
        return True
    return _try_make_ocio() is not None


def set_backend(name: str) -> None:
    """Switch the active backend. Raises if `name` is OCIO but OCIO is
    unavailable, so callers can fall back gracefully."""
    global _active
    if name == "ocio":
        b = _try_make_ocio()
        if b is None:
            raise RuntimeError(_ocio_err or "OCIO is unavailable")
        _active = b
    elif name == "builtin":
        _active = _BuiltinBackend()
    else:
        raise ValueError(f"unknown colour-management backend: {name!r}")


def active_backend() -> str:
    return _ensure_active().name


def config_path() -> str:
    return os.environ.get("OCIO", "(unset)")


# ---------- public routing functions ----------

def working_colorspace() -> str:
    return _ensure_active().working_colorspace()


def colorspace_names() -> list[str]:
    return _ensure_active().colorspace_names()


def displays() -> list[str]:
    return _ensure_active().displays()


def views(display: str) -> list[str]:
    return _ensure_active().views(display)


def default_display() -> str:
    return _ensure_active().default_display()


def default_view(display: str) -> str:
    return _ensure_active().default_view(display)


def convert(img: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Colorspace conversion (no-op-safe if src == dst)."""
    if src == dst:
        return np.asarray(img[..., :3], dtype=np.float32)
    return _ensure_active().convert(img, src, dst)


def to_working(img: np.ndarray, src: str) -> np.ndarray:
    """Input transform: source colorspace -> working."""
    return convert(img, src, WORKING)


def from_working(img: np.ndarray, dst: str) -> np.ndarray:
    """Output transform: working -> destination colorspace."""
    return convert(img, WORKING, dst)


def make_display_cpu(display: str, view: str) -> _Proc:
    """Display+view processor for working -> display. Cached internally."""
    return _ensure_active().make_display_cpu(display, view)


def display_to_u8(img_working: np.ndarray, processor: _Proc) -> np.ndarray:
    """Apply a display processor and encode to contiguous (H, W, 3) uint8."""
    disp = processor.apply(img_working)
    u8 = (np.clip(disp, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return np.ascontiguousarray(u8)
