# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""EXR I/O using OpenImageIO.

OIIO is preferred over the pip `OpenEXR` wrapper because it handles arbitrary
channel layouts, multi-part EXRs, half/float promotion, and colorspace metadata
without ceremony — the same pattern used in nuke/houdini/maya pipelines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import OpenImageIO as oiio  # type: ignore[import-not-found]


def load_latlong(path: str | Path) -> np.ndarray:
    """Load an equirectangular EXR as float32 RGB (H, W, 3).

    The image is validated to be 2:1 aspect ratio (latlong). Alpha is dropped.
    """
    path = str(path)
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise IOError(f"OpenImageIO could not open {path}: {oiio.geterror()}")
    try:
        spec = inp.spec()
        pixels = inp.read_image(format="float")
    finally:
        inp.close()

    h, w = spec.height, spec.width
    nchans = spec.nchannels
    arr = np.asarray(pixels, dtype=np.float32).reshape(h, w, nchans)

    if nchans >= 3:
        arr = arr[..., :3]
    elif nchans == 1:
        arr = np.repeat(arr, 3, axis=-1)
    else:
        raise ValueError(f"Unexpected channel count {nchans} in {path}")

    if w != 2 * h:
        # Soft warning rather than hard fail — some panoramas have weird crops.
        # Caller may resample.
        import warnings

        warnings.warn(f"{path}: aspect {w}x{h} is not 2:1 latlong", stacklevel=2)
    return arr


def save_exr(path: str | Path, rgb: np.ndarray, half: bool = True) -> None:
    """Write an HDR float buffer as EXR.

    `half` writes float16 (smaller, default for textures); set False for float32.
    """
    path = str(path)
    if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
        raise ValueError(f"Expected (H, W, 3|4), got {rgb.shape}")
    h, w, c = rgb.shape
    fmt = oiio.HALF if half else oiio.FLOAT
    spec = oiio.ImageSpec(w, h, c, fmt)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(path)
    if out is None:
        raise IOError(f"OpenImageIO cannot create writer for {path}: {oiio.geterror()}")
    try:
        if not out.open(path, spec):
            raise IOError(f"OpenImageIO open-for-write failed: {out.geterror()}")
        out.write_image(np.ascontiguousarray(rgb, dtype=np.float32))
    finally:
        out.close()
