"""DA-2 inference helper.

Two modes:

- **One-shot** (default): `--input X.exr --output Y.exr` — runs inference once
  and exits. Each invocation pays ~9s Python/torch import + ~3s model load,
  so use sparingly.

- **Serve** (`--serve`): reads newline-delimited JSON requests from stdin and
  writes responses to stdout. Model is loaded once. The actual GPU inference
  is ~150ms per 4K EXR on an RTX 3090. Used by env2lgt.depth.da2_runner.

This script runs *inside* the env2lgt-da2 conda env.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


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


def _target_size(h: int, w: int, min_px: int, max_px: int) -> tuple[int, int]:
    px = h * w
    if min_px <= px <= max_px:
        return h, w
    target = (min_px + max_px) // 2
    scale = (target / px) ** 0.5
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = (new_w // 2) * 2
    new_h = new_w // 2
    return new_h, new_w


# ---------- inference (model already loaded) ----------

def _run_one(model, device, cfg, input_path: Path, output_path: Path) -> dict:
    """Run inference for one EXR. Caller holds model + device."""
    import time
    import cv2
    import torch

    t0 = time.time()
    rgb_full = _load_exr_as_uint8(input_path)
    H_in, W_in, _ = rgb_full.shape
    min_px = int(cfg["inference"]["min_pixels"])
    max_px = int(cfg["inference"]["max_pixels"])
    H_m, W_m = _target_size(H_in, W_in, min_px, max_px)
    rgb_resized = cv2.resize(rgb_full, (W_m, H_m), interpolation=cv2.INTER_AREA)

    img = rgb_resized.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_t = torch.from_numpy(img).unsqueeze(0).to(device)

    t_inf = time.time()
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
        out = model(img_t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_s = time.time() - t_inf

    distance_t = out[0] if isinstance(out, (tuple, list)) else out
    distance = distance_t.detach().float().cpu().numpy().squeeze()
    if distance.ndim != 2:
        raise RuntimeError(f"unexpected model output shape: {distance.shape}")
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
        "model_size": [W_m, H_m],
        "distance_range": [float(distance.min()), float(distance.max())],
        "inference_s": inf_s,
        "total_s": total_s,
    }


# ---------- model loader (used by both modes) ----------

def _load_model(config_path: Path):
    import time
    import torch
    from da2.model.spherevit import SphereViT  # type: ignore[import-not-found]

    with config_path.open() as f:
        cfg = json.load(f)
    cfg["env"]["verbose"] = False
    cfg["env"]["logger"] = type("L", (), {"info": lambda self, *a, **k: None})()

    t0 = time.time()
    print("[da2] loading model from HuggingFace (haodongli/DA-2)...", file=sys.stderr, flush=True)
    model = SphereViT.from_pretrained("haodongli/DA-2", config=cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    cfg["spherevit"]["dtype"] = torch.float32  # autocast handles fp16 at op level
    print(
        f"[da2] model loaded fp32 on {device} (autocast fp16) in {time.time()-t0:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    return model, device, cfg


# ---------- daemon (server) mode ----------

def _serve(config_path: Path) -> int:
    import time

    model, device, cfg = _load_model(config_path)
    # Ready marker — one line of JSON on stdout.
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()
    print("[da2] serve: ready, waiting for requests on stdin", file=sys.stderr, flush=True)

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
                resp = _run_one(model, device, cfg, Path(req["input"]), Path(req["output"]))
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
    model, device, cfg = _load_model(args.config)
    resp = _run_one(model, device, cfg, args.input, args.output)
    print(json.dumps(resp), file=sys.stderr)
    return 0 if resp.get("ok") else 1


# ---------- entry ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--serve", action="store_true", help="Daemon mode: read JSON requests from stdin.")
    args = ap.parse_args()
    if args.serve:
        return _serve(args.config)
    if args.input is None or args.output is None:
        ap.error("--input and --output are required when --serve is not used")
    return _one_shot(args)


if __name__ == "__main__":
    raise SystemExit(main())
