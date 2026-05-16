# Plan: swappable depth backend (DA² → DAP)

> Status: **planning doc for a future session.** DA² work is parked. This
> document is the spec for adding a depth-backend abstraction and a DAP
> backend.

## Why

env2lgt currently hard-depends on **DA²** ([EnVision-Research/DA-2](https://github.com/EnVision-Research/DA-2),
Tencent Hunyuan + HKUST + UCSD). Some studios restrict ML weights of
Chinese origin in shipped pipeline code. **DAP** ([Insta360-Research-Team/DAP](https://github.com/Insta360-Research-Team/DAP),
CVPR 2026) is a studio-approvable alternative — MIT-licensed, DINOv3 (Meta)
backbone, mixed authorship (Insta360 + UCSD + Wuhan + UC Merced).

We want both available behind a common interface so the model is a
config switch, not a fork.

## DA² vs DAP — reference table

| | DA² (current) | DAP (target) |
|---|---|---|
| Repo | https://github.com/EnVision-Research/DA-2 | https://github.com/Insta360-Research-Team/DAP |
| License | Apache-2.0 | MIT |
| Weights | `haodongli/DA-2` (HF) | `Insta360-Research/DAP-weights` (HF) |
| Backbone | SphereViT (custom) | DINOv3-Large (Meta) |
| Depth output | **scale-invariant** (relative) | **metric** (adaptive range) |
| Released | Sept 2025 | Dec 2025 |
| torch | 2.5.0 + cu124 | ~2.7.1 (untested pin) |
| VRAM @ 4K | ~3 GB fp16 | ~16 GB est. (3090 OK) |
| Python API | none — fork `infer.py` | none — fork `infer.py` |
| Paper | arXiv 2509.26618 | arXiv 2512.16913 |

## Requirements

### R1 — Depth-backend protocol
A `DepthBackend` interface in `env2lgt/depth/base.py`:

```python
class DepthBackend(Protocol):
    name: str                       # "da2" | "dap"
    is_metric: bool                 # True -> output already in meters
    def estimate_depth(self, exr_path: Path, cache_dir: Path) -> np.ndarray: ...
    def shutdown(self) -> None: ...
```

- `estimate_depth` returns a `(H, W)` float32 map at the EXR's resolution.
- `is_metric` distinguishes DAP (metric) from DA² (scale-invariant). When
  True, `bake.py` skips the `scene_scale` multiplication — rect lights and
  the depth mesh take world positions straight from the map.

### R2 — Refactor DA² to fit the protocol
Wrap the current module-level daemon state in `da2_runner.py` into a
`Da2Backend(DepthBackend)` class. `is_metric = False`. No behaviour change.
The persistent-daemon + JSON-IPC + file-hash cache logic is reused as-is.

### R3 — DAP backend
- New conda env `env2lgt-dap` (`environment-dap.yml`): Python, torch ~2.7,
  DINOv3 deps, DAP's `requirements`.
- `git clone https://github.com/Insta360-Research-Team/DAP E:\models\DAP`.
- `scripts/dap_infer.py` — fork of DAP's `test/infer.py`, refactored to the
  same one-shot **and** `--serve` daemon contract as `scripts/da2_infer.py`
  (read newline-JSON requests from stdin, write `{"ok": ..., "output": ...}`).
- `env2lgt/depth/dap_runner.py` — `DapBackend(DepthBackend)`, mirrors
  `Da2Backend`'s daemon manager. `is_metric = True`.
- EXR → model-input bridge: DAP (like DA²) wants an LDR/sRGB image, not
  HDR EXR. Reuse the Reinhard tone-flatten already in `da2_infer.py`.
- Output handling: DAP returns metric depth with an adaptive range mask —
  capture the mask if present, treat masked-out pixels (sky/invalid) as
  "very far" so the depth mesh + unprojection don't pull them to origin.

### R4 — Backend selection
- Env var `ENV2LGT_DEPTH_BACKEND` ∈ `{da2, dap}`, default `da2` (don't
  break existing installs).
- Registry in `env2lgt/depth/__init__.py`:
  `get_backend(name) -> DepthBackend`.
- UI: a dropdown in the bake/export panel ("Depth backend: DA² | DAP")
  so per-bake switching needs no restart. The selected name is persisted
  into the project file (`*.env2lgt.json`, add a `depth_backend` key).

### R5 — Metric-aware bake
`bake.py`:
- If `backend.is_metric`: rect-light + mesh positions come straight from
  the depth map. The `scene_scale` slider becomes a *fine-tune multiplier*
  (default 1.0) rather than the primary control.
- If not metric (DA²): current behaviour — `scene_scale` is the primary knob.
- Record `is_metric` + `depth_backend` in `masks.json`.

### R6 — Docs
- README: add a "Depth backends" section, install steps for `env2lgt-dap`.
- `requirements-dap.txt` + `environment-dap.yml`.

## Suggested commit sequence

1. `base.py` protocol + registry + `ENV2LGT_DEPTH_BACKEND` env var.
2. Refactor `da2_runner.py` → `Da2Backend` (pure refactor, no behaviour change).
3. `bake.py` routes through the registry; metric branch added (DA² path
   unchanged since `is_metric=False`).
4. UI dropdown + project-file `depth_backend` key.
5. `environment-dap.yml`, `scripts/dap_infer.py`, `dap_runner.py` —
   the actual DAP integration. Largest step (~1 day).
6. Validation pass: A/B DA² vs DAP on the 5 sample EXRs, eyeball depth
   quality at window / ceiling-fixture regions, document findings.

## Effort estimate

- Steps 1–4 (abstraction + DA² refit + UI): ~2–3 hours.
- Step 5 (DAP backend): ~1 day — wrangling DAP's setup, the EXR→PNG
  bridge, daemon mode, the metric-range mask, VRAM tuning.
- Step 6 (validation): ~half a day.

## Open questions for the implementing session

- DAP VRAM at native 4K — if it OOMs on 24 GB, add a cube-face or
  downscale path.
- DAP's metric scale accuracy on synthetic vs real HDRIs — does the
  `scene_scale` fine-tune knob still earn its place?
- Whether to keep DA² installed at all once DAP is validated, or drop it
  to `da2` being an opt-in legacy backend.
