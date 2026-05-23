# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Depth-backend protocol.

env2lgt estimates per-pixel depth from a latlong HDRI to place rect lights
and (optionally) a panorama mesh. Two backends implement this:

- **DA²** (`da2`) — scale-invariant (relative) depth. The user's `scene_scale`
  slider is the primary knob that turns relative depth into meters.
- **DAP** (`dap`) — metric depth (already in meters). `scene_scale` degrades
  to a fine-tune multiplier (default 1.0).

`is_metric` is the flag callers branch on to tell the two apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DepthBackend(Protocol):
    name: str          # "da2" | "dap"
    is_metric: bool     # True -> estimate_depth output is already in meters

    def estimate_depth(
        self, exr_path: str | Path, cache_dir: str | Path | None = None
    ) -> np.ndarray:
        """Return a (H, W) float32 depth map at the EXR's resolution."""
        ...

    def shutdown(self) -> None:
        """Release the backend (e.g. kill its inference daemon). Idempotent."""
        ...
