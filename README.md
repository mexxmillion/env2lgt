# env2lgt

**Convert a single equirectangular HDRI into a physically-positioned USD light rig.**

Built for VFX lighting pipelines: take an HDRI latlong EXR, mark the practical lights with 4 clicks each, and bake out a USD scene containing a `UsdLuxDomeLight` for the environment, one `UsdLuxRectLight` per marked light positioned in world space via monocular panorama depth (DA²), and a depth-displaced `UsdGeomMesh` of the scene for validation. Drop the result into Karma, RenderMan, Storm, or any other USD-aware renderer.

![env2lgt GUI](docs/preview.png)

> *Above: the env2lgt main window. A 4K HDRI is loaded; three light quads have been placed on ceiling fixtures (the large yellow one is selected, with its 4 corner handles visible). The right panel shows the quad list, output path, and per-rig export toggles.*

> **Status:** working tool, not feature-complete. End-to-end pipeline is solid; UX still being iterated on.

---

## What it does

VFX lighting work often starts with an HDRI captured on set or generated from a CG scene. To use those panoramas as **practical lighting** in a render — i.e., lights that actually exist in 3D space and cast shadows correctly — you need three things:

1. **The bright sources isolated** so a renderer can sample them at high resolution (a dome light alone undersamples small bright sources, producing noisy specular reflections).
2. **Each source positioned in world space** so it occludes correctly behind walls / props.
3. **The rest of the environment** preserved as a dome / IBL so global indirect bounce still matches the captured scene.

env2lgt does all three with one tool, in minutes, for a single HDRI. The user's only manual work is clicking 4 corners around each practical light on the equirect panorama. Everything else — depth estimation, world-space placement, light texture extraction, dome inpainting, USD authoring — is automatic.

### Inputs

- **One** equirectangular `.exr` panorama (any resolution; 2K–8K tested).
- Optionally: a scene-scale value (meters per DA-2 unit) and a yaw-offset for seam-straddling lights.

### Outputs

For each bake, in a user-chosen output directory:

| File | Purpose |
|---|---|
| `lightrig.usda` | Composed scene: `UsdLuxDomeLight` (with `dome.exr` texture) + N × `UsdLuxRectLight` with per-light textures, sized + positioned in world space. |
| `dome.exr` | Original panorama with the marked light regions removed and **edge-extended** so the dome integrates cleanly with no dark holes. |
| `<lightname>.exr` | Per-light **rectilinear** texture sampled from the panorama through the quad surface. Maps 1:1 onto its `UsdLuxRectLight`. |
| `panorama_geo.usda` *(optional)* | Depth-displaced `UsdGeomMesh` sphere with the dome texture as `emissiveColor` — used to validate the depth + light placement visually in `usdview`. |
| `*.distance.exr` | DA-2 depth map cache (next to source EXR or in `.env2lgt_cache/`). Reused across bakes of the same HDRI. |
| `masks.json` | Sidecar with the 4 spherical corners of each quad + the yaw offset and scene scale at bake time. Lets you re-author future bakes deterministically. |

---

## GUI tour

Refer to the screenshot above.

### Toolbar (top)

- **Exposure** — log2 stops, display-only. Doesn't affect what's written.
- **Scene scale** — meters per DA-2 unit. DA-2 returns scale-invariant relative distance, so this is the user's only knob for "how big is this room actually." Default **100 m/u**; slider range `0.001 .. 1000`.
- **Yaw offset** — rolls the displayed panorama horizontally (in degrees). Use this to place lights that straddle the seam at u=0/u=W. Quad data stays in **absolute** spherical coords, so changing the offset doesn't move the lights — it only changes what's under the mouse. **Reset** zeroes it.
- **Show depth** (hotkey **D**) — toggle the equirect view between HDR and a turbo-colormap of the DA-2 distance map. Quad outlines + handles render on top either way, so you can verify each quad sits on a region of consistent depth before baking. First press runs DA-2 in a background thread; subsequent toggles are instant.

### Light quads panel (right)

- **List** of all defined quads. Double-click to rename — input is sanitized to USD/Maya/Houdini-friendly identifiers (`spaces → _`, non-`[A-Za-z0-9_]` stripped, no leading digits).
- **Add quad (click 4 corners)** — hotkey **A**. Enters add mode: cursor becomes a crosshair, four left-clicks place the four corners of a new quad. The 4 corners are auto-sorted CCW so click order doesn't matter. The newly-committed quad is selected and gets draggable vertex handles.
- **Delete selected** — hotkey **Delete**.

### Output panel

- **Output path** — pre-populated to `<exr_dir>/<exr_stem>_lightrig/` on every EXR load; editable; **Browse…** for a folder picker. The bake will create the directory if it doesn't exist, and warn before overwriting existing `lightrig.usda` / `dome.exr` / `rect_*.exr`.

### Export panel

Per-file output checkboxes. All are independent, so you can bake just the dome, just the rect textures, or any subset:

- `dome.exr` — the inpainted dome panorama
- `light_<i>.exr` — per-rect rectilinear textures
- `lightrig.usda` — the composed light scene
- `depth.exr` (debug) — write the raw DA-2 distance map as a 3-channel EXR
- `panorama_geo.usda` — the depth-displaced UV sphere mesh
- `masks.json` — the mask sidecar

### Preview vs Bake

- **Preview (no files)** — runs the full pipeline with every `write_*` set to False. Produces an in-memory table of per-quad fit results (center, size, normal, intensity, RANSAC inlier ratio). Useful for sanity-checking before committing files to disk. Distance cache lands at `<exr_dir>/.env2lgt_cache/` so the first real bake hits it for free.
- **Bake light rig** — full pipeline, writes everything checked.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  env2lgt   (conda env, python 3.11)                                │
│  ├─ PySide6 6.10                       UI                          │
│  ├─ OpenUSD 26.03 (with usdview)       authoring                   │
│  ├─ OpenEXR 3.4 / OIIO 3.1             EXR I/O                     │
│  ├─ OpenCV 4.13                        image ops                   │
│  └─ numpy / scipy                                                  │
│                                                                    │
│        │   JSON over stdin/stdout (one daemon per session)         │
│        ▼                                                           │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  env2lgt-da2   (conda env, python 3.12)                    │    │
│  │  ├─ torch 2.5 + CUDA 12.4                                  │    │
│  │  ├─ xformers 0.0.28 + triton-windows 3.1                   │    │
│  │  └─ DA-2 (Depth Anything in Any Direction)                 │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

**Why two envs?** DA-2 pins `torch==2.5.0 + cu124 + xformers==0.0.28.post2`, which conflicts with what the openusd / pyside6 stack wants. Keeping them separate keeps each environment cleanly pip-resolvable and lets the UI release independently of the depth model. The UI talks to a single long-lived DA-2 worker over line-delimited JSON; cold start is ~12 s, every subsequent bake on the same EXR hits the file-hash distance cache and runs in ~1 s.

---

## Install

### Prerequisites

- Windows 10/11 x64 (Linux untested but should work — replace path separators)
- NVIDIA GPU (RTX 3090-class, ≥ 24 GB VRAM recommended; smaller works but DA-2 has to downscale)
- conda (miniconda or miniforge)
- Disk: ~15 GB on `E:` for the two envs + DA-2 model weights cache

### Setup

```cmd
:: 1. UI / USD / EXR environment
conda create -p E:\conda\envs\env2lgt -c conda-forge python=3.11 ^
    openusd pyside6 pyopengl ^
    numpy scipy opencv openexr py-openimageio
conda activate E:\conda\envs\env2lgt
pip install -e .

:: 2. DA-2 inference environment
conda create -p E:\conda\envs\env2lgt-da2 -c conda-forge python=3.12 pip git
git clone https://github.com/EnVision-Research/DA-2 E:\models\DA-2
E:\conda\envs\env2lgt-da2\Scripts\pip install ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
E:\conda\envs\env2lgt-da2\Scripts\pip install -e E:\models\DA-2\src
:: triton enables xformers' fused attention (~17% faster):
E:\conda\envs\env2lgt-da2\Scripts\pip install triton-windows==3.1.0.post17
:: OpenEXR for reading the input panorama inside the DA-2 process:
E:\conda\envs\env2lgt-da2\Scripts\pip install OpenEXR
```

### Cache locations (one-time)

```cmd
setx HF_HOME             "E:\models\huggingface"
setx TORCH_HOME          "E:\models\torch"
setx HF_TOKEN            "hf_xxxxxxxxxxxxxxxxxxxxxx"
setx ENV2LGT_DA2_ENV     "E:\conda\envs\env2lgt-da2"
setx ENV2LGT_DA2_REPO    "E:\models\DA-2"
```

The two `ENV2LGT_*` vars are optional — the runner falls back to `E:\conda\envs\env2lgt-da2` and `E:\models\DA-2` if not set.

### Run

```cmd
conda activate E:\conda\envs\env2lgt
python -m env2lgt.app
```

Or the bundled launcher:

```cmd
launch.cmd
```

---

## Pipeline (what happens when you click Bake)

1. **Load EXR** — `OpenImageIO` reads the latlong as float32 RGB.
2. **DA-2 inference** — IPC to the daemon. First time per EXR: ~150 ms of GPU work + ~2 s of file I/O and Python overhead. Cached as `<stem>.<hash>.distance.exr` and reused across bakes.
3. **For each quad**:
   - **Rasterize spherical quad** → `(H, W)` uint8 mask, via ray/plane intersection on every panorama pixel (handles seam-wrap and near-pole quads correctly, see [env2lgt/proj.py](env2lgt/proj.py)).
   - **Rect-light geometry** from quad corners × median mask depth: center, normal, u-axis, v-axis, width, height — all in world space ([env2lgt/lights/extract.py::rect_from_quad](env2lgt/lights/extract.py)).
   - **Color** = luminance-weighted mean of the masked region, normalized to max=1.
   - **Intensity** = sum of luminance × per-pixel solid angle over the mask.
   - **Texture** = `sample_rect_texture()`: rectilinear projection of the quad surface via `cv2.remap`, so the resulting EXR maps 1:1 onto a flat `UsdLuxRectLight`. Aspect-matched to the world-space rect, max 1024 on the long side.
4. **Dome residual** — union of all quad masks, then `edge_extend()` ([env2lgt/lights/inpaint.py](env2lgt/lights/inpaint.py)): iterative Gaussian push-fill in log-domain (Nuke `EdgeExtend` / Mocha PushPull style). Produces a smooth, HDR-safe fill — no rainbow noise like `cv2.INPAINT_TELEA` on bright values.
5. **USD authoring** — `lightrig.usda` with `/World/lights/dome` (DomeLight + 90° Y compensation for USD's azimuth convention) and `/World/lights/<name>` per quad.
6. **Optional mesh** — `panorama_geo.usda`: 256×128 UV sphere displaced radially by depth, with `dome.exr` bound as emissive texture through `UsdPreviewSurface` + `UsdUVTexture` + `UsdPrimvarReader_float2`. The mesh and the rect lights agree on world space, so any drift between dome and rect lights is a renderer convention issue, not pipeline math.

---

## Validation workflow

1. **Place quads** on the panorama.
2. **Press D** — flip to depth view. Confirm each quad sits on a region of plausible, consistent depth. Lights at very different depths than their immediate surroundings (e.g. a window with the sky visible through it) may need manual scale-tweaking later.
3. **Preview (no files)** — get the fit table. Sanity-check sizes (a wall sconce should be ~0.3 m, a ceiling panel ~1–2 m). If a quad reports a 50 m × 30 m fit, the depth under that mask is unreliable.
4. **Check Depth mesh USD**, **Bake**.
5. **Tools → Open last bake in usdview**. The rect lights should sit directly on the bright fixtures in the dome texture; the panorama mesh shows them in 3D space.
6. **Adjust scene scale** in the toolbar and re-bake if everything is too small / too big. Cache hit → ~1 s per re-bake.

---

## Repository layout

```
env2lgt/
├── app.py                   PySide6 entry point (MainWindow)
├── bake.py                  End-to-end pipeline orchestration
├── cli.py                   `env2lgt-bake` headless command (stub)
├── proj.py                  Sphere ↔ equirect ↔ rectilinear math (single source of truth)
├── depth/
│   └── da2_runner.py        Persistent DA-2 daemon manager
├── io/
│   ├── exr.py               OIIO-based latlong load/save
│   └── tonemap.py           ACES + sRGB display tonemap, turbo depth colormap
├── lights/
│   ├── extract.py           sample_rect_texture, rect_from_quad, photometry
│   └── inpaint.py           Iterative gaussian edge-extend (HDR-safe)
├── ui/
│   ├── viewer.py            Equirect viewer + 4-click placement + drag handles
│   └── light_panel.py       Quad list, name sanitizer, export options, paths
└── usd/
    ├── lightrig.py          UsdLuxDomeLight + UsdLuxRectLight authoring
    └── mesh.py              Depth-displaced UV sphere with emissive dome
scripts/
├── da2_infer.py             Lives inside env2lgt-da2; one-shot OR --serve daemon mode
└── batch_bake.py            Batch-bake every EXR in a directory (auto top-N brightness blobs)
```

---

## Limitations & known issues

- **Depth is relative.** DA-2 outputs scale-invariant distance. The scene-scale slider is the user's only mechanism to make it metric. There's no auto-estimation of absolute scale.
- **Plane fit assumes user's quad is genuinely planar in 3D.** A quad drawn around a curved fixture or a multi-window arrangement will fit a single plane through the average, which may look wrong.
- **Rectilinear texture works best for small angular extents.** Quads spanning > 90° in any direction will show distortion in the sampled rectangular texture (the bilinear-of-4-corners interpolation deviates from true rectilinear projection for wide quads).
- **`triton-windows` is required for the xformers fused-attention path.** Without it inference still works, just ~17 % slower.
- **Windows-only paths.** Forward-slash equivalents would be a small refactor in `da2_runner.py` and the launcher.

---

## Roadmap

- [ ] Auto-detect mode — propose quads from brightness-thresholded connected components, with K-means grouping by brightness so the user can include/exclude "windows" vs "lamps" with one click.
- [ ] **Load masks.json** — reproduce a previous bake's quad layout for a new render.
- [ ] Disk + portal lights (in addition to rect).
- [ ] Per-quad intensity multiplier in the panel.
- [ ] Compose `lightrig.usda` + `panorama_geo.usda` into a single `validation.usda` reference.
- [ ] Drag-drop EXR file load (currently dependent on Windows OLE behaviour that doesn't play with `QGraphicsView`).
- [ ] Linux build of the conda envs.

---

## Acknowledgements

- **DA-2** — [EnVision-Research/DA-2](https://github.com/EnVision-Research/DA-2) ("Depth Anything in Any Direction", arXiv:2509.26618). The SOTA monocular panorama depth model this pipeline relies on. © Tencent Hunyuan / HKUST.
- **OpenUSD** — Pixar / Apple / the OpenUSD community.
- **PySide6 / Qt6** — The Qt Company.
- **OpenImageIO / OpenEXR / OpenColorIO** — the VFX foundation libraries every pipeline is built on.
- **Nuke** — the inspiration for the dome `edge_extend` fill behaviour.

---

## License

Apache-2.0.

Note: DA-2 weights are governed by their own license — see the [DA-2 repo](https://github.com/EnVision-Research/DA-2) for terms.
