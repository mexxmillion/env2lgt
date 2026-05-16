"""DAP inference helper — SCAFFOLD.

Forked from `scripts/da2_infer.py`. Same two modes and the same daemon
contract, so `env2lgt.depth.dap_runner.DapBackend` can drive it exactly like
the DA-2 daemon:

- **One-shot**: `--input X.exr --output Y.exr --repo <DAP repo>` — run once.
- **Serve** (`--serve`): newline-delimited JSON requests on stdin, responses
  on stdout. Model loaded once.

This script runs *inside* the `env2lgt-dap` conda env.

UNFINISHED — the daemon/IPC plumbing is complete, but the three TODO blocks
below must be wired against a real DAP checkout + weights:
  1. `_load_model`     — import DAP, build the net, load weights.
  2. `_run_one`        — the forward pass + metric-output extraction.
  3. metric-range mask — DAP emits an adaptive range mask; masked-out pixels
     (sky / invalid) must be pushed to "very far", not left at 0, or the
     depth mesh + unprojection collapse them onto the origin.

Until then, `--serve` reports `ok: false` for `infer` requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


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
    """Build the DAP net and load weights. Returns (model, device).

    TODO(dap-1): wire against the real DAP checkout. Expect something like:
        sys.path.insert(0, str(repo))
        from dap.model import DAP            # exact import TBD
        model = DAP.from_pretrained("Insta360-Research/DAP-weights")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
    DAP uses a DINOv3-Large backbone; ~16 GB VRAM at 4K is the doc's estimate.
    """
    raise NotImplementedError(
        "DAP model loading is not wired yet — see TODO(dap-1) in scripts/dap_infer.py"
    )


# ---------- inference (model already loaded) ----------

def _run_one(model, device, input_path: Path, output_path: Path) -> dict:
    """Run inference for one EXR. Caller holds model + device.

    TODO(dap-2): the forward pass + metric-output extraction.
      - rgb = _load_exr_as_uint8(input_path)            # LDR bridge, ready
      - resize to DAP's expected input size (mind ~16 GB VRAM at 4K)
      - run the net under torch.no_grad() + autocast
      - DAP returns METRIC depth (meters) — no scale-invariant normalization
      - resize the depth back to the EXR's native (H, W)
    TODO(dap-3): metric-range mask.
      - DAP emits an adaptive range mask; capture it if present
      - set masked-out pixels (sky / invalid) to a large "very far" value
        rather than 0, so bake.py's depth mesh + unprojection don't pull
        them to the origin. A finite sentinel (e.g. 1e4 m) is safer than inf.
      - then: _save_distance_exr(output_path, distance)
    """
    raise NotImplementedError(
        "DAP inference is not wired yet — see TODO(dap-2)/(dap-3) in scripts/dap_infer.py"
    )


# ---------- daemon (server) mode ----------

def _serve(repo: Path) -> int:
    model, device = _load_model(repo)
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
                resp = _run_one(model, device, Path(req["input"]), Path(req["output"]))
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
    model, device = _load_model(args.repo)
    resp = _run_one(model, device, args.input, args.output)
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
