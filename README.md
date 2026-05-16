# env2lgt

Convert a single equirectangular HDRI (EXR) into a USD light rig:

- **Rect area lights** extracted from user-drawn rectangle masks, positioned in world space via panorama depth estimation.
- **Dome light** for the residual environment, with extracted regions Gaussian-inpainted to remove black holes.
- **Optional depth mesh** of the panorama as a `UsdGeomMesh` for sanity-checking 3D placement.

Built around VFX-foundation tooling: **OpenUSD**, **OpenEXR / Imath**, **OpenImageIO**, **OpenColorIO**, **PySide6**, and **DA² (Depth Anything in Any Direction)** for monocular panorama depth.

> Status: **early development.** UI scaffold + USD writer working; DA² integration in progress.

## Requirements

- Windows 10/11 x64 (Linux support planned)
- NVIDIA GPU with ≥ 8 GB VRAM (24 GB recommended for full-resolution DA² inference)
- Two conda environments (see Install)

## Install

```cmd
:: 1. UI / USD / EXR environment
conda create -p E:\conda\envs\env2lgt -c conda-forge python=3.11 ^
    openusd pyside6 pyopengl ^
    numpy scipy opencv openexr py-openimageio
pip install -e .

:: 2. DA-2 inference environment (separate; called via subprocess)
conda create -p E:\conda\envs\env2lgt-da2 -c conda-forge python=3.12 pip git
git clone https://github.com/EnVision-Research/DA-2 E:\models\DA-2
E:\conda\envs\env2lgt-da2\Scripts\pip install ^
    --index-url https://download.pytorch.org/whl/cu124 ^
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0
E:\conda\envs\env2lgt-da2\Scripts\pip install -e E:\models\DA-2\src
```

Set up cache locations (cache + token; never commit these):

```cmd
setx HF_HOME       "E:\models\huggingface"
setx TORCH_HOME    "E:\models\torch"
setx HF_TOKEN      "hf_xxxxxxxxxxxxxxxxxxxxxx"
setx ENV2LGT_DA2_ENV  "E:\conda\envs\env2lgt-da2"
setx ENV2LGT_DA2_REPO "E:\models\DA-2"
```

## Run

```cmd
conda activate E:\conda\envs\env2lgt
env2lgt
```

Or headless:

```cmd
env2lgt-bake input.exr -o out\
```

## Pipeline

1. Load EXR latlong panorama (open dialog or future drag-drop).
2. **Add quad** (button or `A`): click the 4 corners of a light directly on the equirect view. Corners are stored as 4 spherical directions; great-circle arcs render the edges (seam-wrap handled).
3. While selected, drag the yellow vertex handles to refine — the quad updates live.
4. Repeat for each light source. Set scene scale in the toolbar.
5. **Bake**: DA² runs (subprocess into `env2lgt-da2`) to get a per-pixel relative-distance map. Cached next to the input.
6. Per quad: rasterize the spherical-quad mask onto the pano → crop with alpha → unproject mask pixels with depth → RANSAC plane fit → `UsdLuxRectLight` (center, normal, basis, width, height, color, solid-angle-weighted intensity).
7. Zero + Telea-inpaint the masked regions in the original EXR → `dome.exr` → `UsdLuxDomeLight`.
8. Author `lightrig.usda`. (Optional) launch `usdview` from Tools menu.

## License

Apache-2.0
