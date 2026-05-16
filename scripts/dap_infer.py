"""DAP inference helper.

Forked from `scripts/da2_infer.py`. Same two modes and the same daemon
contract, so `env2lgt.depth.dap_runner.DapBackend` can drive it exactly like
the DA-2 daemon:

- **One-shot**: `--input X.exr --output Y.exr --repo <DAP repo>` — run once.
- **Serve** (`--serve`): newline-delimited JSON requests on stdin, responses
  on stdout. Model loaded once.

This script runs *inside* the `env2lgt-dap` conda env, with the DAP repo
(https://github.com/Insta360-Research-Team/DAP) on disk.

It mirrors DAP's own reference path (`test/infer.py`): build the model via
`networks.models.make`, load `model.pth`, feed a [0,1] CHW tensor, take the
`pred_depth` / `pred_mask` dict back. DAP emits depth normalised so that
`1.0 == 100 m` (see `pred_to_vis` "100m" in DAP's `test/infer.py`); we
multiply back to metres. Masked-out pixels (sky / invalid) are pushed to
"very far" so the depth mesh + unprojection don't collapse them to origin.

Weights: download from https://huggingface.co/Insta360-Research/DAP-weights
and point `ENV2LGT_DAP_WEIGHTS` at the directory holding `model.pth`
(otherwise the `load_weights_dir` in DAP's `config/infer.yaml` is used).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# DAP's normalised depth output: 1.0 == 100 m. From `pred_to_vis` "100m" in
# DAP's test/infer.py (clip to [0,1], the full range maps to 100 m).
DAP_METRIC_SCALE = 100.0


# ---------- EXR -> model input (reused verbatim from da2_infer.py) ----------
# DAP, like DA-2, wants an LDR/sRGB image, not an HDR EXR. This Reinhard
# tone-flatten is the same bridge da2_infer.py uses.

def _load_exr_as_uint8(path: Path) -> np.ndarray:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]

    f = OpenEXR.InputFile(str(path))
    try:
        dw = f.header()["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        chans = [
            np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
            for c in ("R", "G", "B")
        ]
    finally:
        f.close()
    rgb = np.maximum(np.stack(chans, axis=-1), 0.0)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    key = float(np.exp(np.mean(np.log(y + 1e-6))))
    scaled = rgb / (key + 1e-6) * 0.18
    ldr = scaled / (1.0 + scaled)
    lo = 12.92 * ldr
    hi = 1.055 * np.power(np.maximum(ldr, 1e-9), 1.0 / 2.4) - 0.055
    srgb = np.where(ldr <= 0.0031308, lo, hi)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _save_distance_exr(path: Path, distance: np.ndarray) -> None:
    import OpenEXR  # type: ignore[import-not-found]
    import Imath  # type: ignore[import-not-found]

    h, w = distance.shape
    header = OpenEXR.Header(w, h)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {"Y": Imath.Channel(pt)}
    out = OpenEXR.OutputFile(str(path), header)
    try:
        out.writePixels({"Y": distance.astype(np.float32).tobytes()})
    finally:
        out.close()


# ---------- model loader ----------

def _load_model(repo: Path):
    """Build the DAP net and load weights. Returns (model, device, input_hw).

    Mirrors `load_model` in DAP's test/infer.py. DAP's model build resolves
    its DINOv3 repo via a *cwd-relative* path, so we chdir into the repo.
    """
    import time

    import torch
    import torch.nn as nn
    import yaml

    repo = repo.resolve()
    os.chdir(repo)  # DAP's dinov3_repo_dir is "./depth_anything_v2_metric/..."
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from networks.models import make  # type: ignore[import-not-found]

    config_path = repo / "config" / "infer.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)

    weights_dir = os.environ.get("ENV2LGT_DAP_WEIGHTS") or config.get("load_weights_dir", "")
    model_path = Path(weights_dir) / "model.pth"
    if not model_path.is_file():
        raise RuntimeError(
            f"DAP weights not found at {model_path}. Download from "
            "https://huggingface.co/Insta360-Research/DAP-weights and set "
            "ENV2LGT_DAP_WEIGHTS to the directory containing model.pth."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    print(f"[dap] loading weights from {model_path}", file=sys.stderr, flush=True)
    state = torch.load(str(model_path), map_location=device)

    model = make(config["model"])
    # Weights trained under nn.DataParallel carry a "module." prefix.
    if any(k.startswith("module") for k in state.keys()):
        model = nn.DataParallel(model)
    model = model.to(device)
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
    model.eval()

    in_cfg = config.get("input", {}) or {}
    input_hw = (int(in_cfg.get("height", 512)), int(in_cfg.get("width", 1024)))
    print(
        f"[dap] model loaded on {device} in {time.time() - t0:.2f}s "
        f"(input {input_hw[1]}x{input_hw[0]})",
        file=sys.stderr,
        flush=True,
    )
    return model, device, input_hw


# ---------- inference (model already loaded) ----------

def _run_one(model, device, input_hw, input_path: Path, output_path: Path) -> dict:
    """Run inference for one EXR. Caller holds model + device."""
    import time

    import cv2
    import torch

    t0 = time.time()
    rgb_full = _load_exr_as_uint8(input_path)  # (H, W, 3) uint8 RGB — Reinhard bridge
    H_in, W_in, _ = rgb_full.shape
    in_h, in_w = input_hw
    rgb_resized = cv2.resize(rgb_full, (in_w, in_h), interpolation=cv2.INTER_CUBIC)

    # DAP's reference path (test/infer.py::infer_raw) feeds a plain [0,1] CHW
    # tensor — no ImageNet normalisation.
    img = rgb_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    t_inf = time.time()
    with torch.inference_mode():
        outputs = model(img_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_s = time.time() - t_inf

    # DAP returns {"pred_depth", "pred_mask"}: normalised depth + a validity
    # mask. Masked-out pixels (sky / invalid) get pushed to the far value so
    # bake.py's depth mesh + unprojection don't pull them onto the origin.
    if isinstance(outputs, dict) and "pred_depth" in outputs:
        pred_depth = outputs["pred_depth"]
        if "pred_mask" in outputs:
            invalid = (1 - outputs["pred_mask"]) > 0.5
            pred_depth[invalid] = 1.0  # 1.0 normalised == far (100 m)
        pred = pred_depth[0].detach().float().cpu().squeeze().numpy()
    else:
        pred = outputs[0].detach().float().cpu().squeeze().numpy()
    if pred.ndim != 2:
        raise RuntimeError(f"unexpected DAP output shape: {pred.shape}")

    distance = pred.astype(np.float32) * DAP_METRIC_SCALE  # normalised -> metres
    if distance.shape != (H_in, W_in):
        distance = cv2.resize(distance, (W_in, H_in), interpolation=cv2.INTER_LINEAR)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_distance_exr(output_path, distance)
    total_s = time.time() - t0
    return {
        "ok": True,
        "input": str(input_path),
        "output": str(output_path),
        "size": [W_in, H_in],
        "model_size": [in_w, in_h],
        "distance_range": [float(distance.min()), float(distance.max())],
        "inference_s": inf_s,
        "total_s": total_s,
    }


# ---------- daemon (server) mode ----------

def _serve(repo: Path) -> int:
    model, device, input_hw = _load_model(repo)
    # Ready marker — one line of JSON on stdout.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    print("[dap] serve: ready, waiting for requests on stdin", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            cmd = req.get("cmd", "infer")
            if cmd == "ping":
                resp = {"ok": True, "pong": True}
            elif cmd == "shutdown":
                sys.stdout.write(json.dumps({"ok": True, "shutdown": True}) + "\n")
                sys.stdout.flush()
                break
            elif cmd == "infer":
                resp = _run_one(model, device, input_hw, Path(req["input"]), Path(req["output"]))
            else:
                resp = {"ok": False, "error": f"unknown cmd: {cmd}"}
        except Exception as e:  # noqa: BLE001
            import traceback

            resp = {"ok": False, "error": str(e), "trace": traceback.format_exc()}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


# ---------- one-shot mode ----------

def _one_shot(args) -> int:
    model, device, input_hw = _load_model(args.repo)
    resp = _run_one(model, device, input_hw, args.input, args.output)
    print(json.dumps(resp), file=sys.stderr)
    return 0 if resp.get("ok") else 1


# ---------- entry ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--repo", required=True, type=Path, help="Path to the DAP repo checkout.")
    ap.add_argument("--serve", action="store_true", help="Daemon mode: read JSON requests from stdin.")
    args = ap.parse_args()
    if args.serve:
        return _serve(args.repo)
    if args.input is None or args.output is None:
        ap.error("--input and --output are required when --serve is not used")
    return _one_shot(args)


if __name__ == "__main__":
    raise SystemExit(main())
