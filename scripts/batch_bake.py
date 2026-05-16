"""Batch the MVP across every EXR in a directory.

For each input, auto-detect the top-N brightest blobs and run the full bake.
Useful for visually scanning many scenes to validate the pipeline.

Run inside the env2lgt conda env, with ENV2LGT_DA2_ENV / ENV2LGT_DA2_REPO set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from env2lgt.bake import BakeOptions, MaskRect, bake
from env2lgt.io.exr import load_latlong


def auto_masks(hdr: np.ndarray, top_n: int = 6, min_area_frac: float = 5e-5) -> list[MaskRect]:
    H, W, _ = hdr.shape
    lum = 0.2126 * hdr[..., 0] + 0.7152 * hdr[..., 1] + 0.0722 * hdr[..., 2]
    thresh = float(np.percentile(lum, 99.5))
    mask = (lum > thresh).astype(np.uint8)
    nlbl, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(50, int(min_area_frac * H * W))
    blobs: list[tuple[int, int, int, int, int]] = []
    for i in range(1, nlbl):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        blobs.append((area, x, y, w, h))
    blobs.sort(reverse=True)
    blobs = blobs[:top_n]
    return [MaskRect(name=f"rect_{j:02d}", x=int(x), y=int(y), w=int(w), h=int(h))
            for j, (_, x, y, w, h) in enumerate(blobs)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--top-n", type=int, default=6)
    ap.add_argument("--scale", type=float, default=10.0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exrs = sorted(args.input_dir.glob("*.exr"))
    print(f"found {len(exrs)} EXR(s) in {args.input_dir}")

    summary = []
    for exr in exrs:
        scene = exr.stem
        out = args.output_dir / scene
        print(f"\n=== {scene} ===")
        t0 = time.time()
        hdr = load_latlong(exr)
        masks = auto_masks(hdr, top_n=args.top_n)
        print(f"  {hdr.shape[1]}x{hdr.shape[0]} -> {len(masks)} mask(s)")
        for m in masks:
            print(f"    {m.name}: {m.w}x{m.h} @ {m.x},{m.y}")
        try:
            s = bake(
                exr,
                out,
                masks,
                BakeOptions(scene_scale=args.scale),
                progress_cb=lambda stage, frac: None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {e}")
            summary.append({"scene": scene, "error": str(e)})
            continue
        dt = time.time() - t0
        rect_summary = [
            {
                "name": r["name"],
                "size": r["size"],
                "center": r["center"],
                "inliers": round(r["inlier_ratio"], 2),
            }
            for r in s["rect_fits"]
        ]
        print(f"  done in {dt:.1f}s -> {s['usd']}")
        for r in rect_summary:
            c = r["center"]
            print(
                f"    {r['name']:7s}  size={r['size'][0]:5.2f}x{r['size'][1]:5.2f}  "
                f"pos=[{c[0]:6.2f}, {c[1]:5.2f}, {c[2]:6.2f}]  inl={r['inliers']:.2f}"
            )
        summary.append({"scene": scene, "seconds": round(dt, 1), "rects": rect_summary})

    (args.output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nbatch_summary.json -> {args.output_dir / 'batch_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
