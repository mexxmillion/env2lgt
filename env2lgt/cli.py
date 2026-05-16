"""Headless / CLI entry — `env2lgt-bake`. Stub until the pipeline lands."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake a USD light rig from an EXR panorama.")
    ap.add_argument("input", type=Path, help="Input EXR latlong panorama.")
    ap.add_argument("-o", "--output", type=Path, default=Path("out"), help="Output directory.")
    ap.add_argument("--masks", type=Path, help="Mask JSON sidecar (optional).")
    args = ap.parse_args()
    print(f"[env2lgt-bake] stub: would bake {args.input} -> {args.output} (masks={args.masks})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
