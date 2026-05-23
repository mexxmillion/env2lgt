# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""env2lgt — HDRI panorama to USD light rig.

A VFX tool that turns a single equirectangular HDRI into a portable USD
light rig: per-light textured `UsdLuxRectLight` prims for practicals, a
`UsdLuxDomeLight` for the global ambient with practicals inpainted out,
and an optional depth mesh for parallax-correct viewport preview.

Highlights
----------
* **Panorama-aware depth.** Swappable backend: Depth Anything 2 (scale
  invariant) or Depth Anything Pano / DAP (metric). Depth is what places
  each light at its world distance — not a wall behind it.
* **Rigid rect-light fit.** `Fit to rect light` snaps each quad to a
  shear-free rectangle on its surface plane so the USD authors what
  every renderer (Arnold, V-Ray, Redshift, RenderMan, Karma, …) actually
  consumes. No silent shear-drop.
* **OCIO colour management.** Input transform on load, output transform
  on bake, viewport display + view selectable. ACES-aware defaults.
* **Three colour-match modes** against a flat reference image:
  *Chart* (24-patch CC24), *Regions* (paired sample rectangles with a
  log-space gain/gamma solve), and *Auto* (NLE-style whole-image match
  with percentile clipping so suns and lamps don't blow the gains).
* **Exposure mode.** Spot meter, WB eyedropper, and convolve-the-dome
  auto-meter — all bake into the exported rig.

Author
------
Maung Maung Hla Win <mexxmillion@gmail.com>

Released under the Apache License 2.0 in good faith for the wider VFX
community. If this tool helps your work, attribution is appreciated.
"""

__version__ = "0.1.0"
__author__ = "Maung Maung Hla Win"
__email__ = "mexxmillion@gmail.com"
__license__ = "Apache-2.0"
