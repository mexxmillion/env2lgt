# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Auto-detect candidate light quads from a latlong HDR panorama.

Pipeline (matches the UI's "Auto-detect lights" panel):

1. Blur the luminance a little so a light split by window mullions / ceiling
   beams reads as one blob instead of fragmenting.
2. Threshold relative to the scene: `mean + t*(bright - mean)`, where `bright`
   is a robust near-max luminance and `t` is the user's threshold knob.
3. Label the bright blobs (seam-wrap aware), drop ones too small to be a light,
   optionally merge blobs that sit close together on the sphere (so a cluster
   of small lights — ceiling spots, a lit tree — becomes one quad).
4. Give each blob an axis-aligned bounding box *in equirect pixel space* and
   lift its 4 corners to direction vectors.

The proposer returns geometry only; naming, dedup against existing quads, and
the lock policy live in the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from env2lgt.lights.extract import equirect_solid_angle, luminance
from env2lgt.proj import (
    angles_from_dir,
    dir_from_angles,
    pix_to_angles,
    rasterize_spherical_quad,
    view_basis,
)


@dataclass
class DetectParams:
    """Knobs for `propose_quads`, surfaced in the UI's auto-detect panel."""

    # Luma-key threshold as a fraction (0..1) of the scene's brightest
    # luminance. Lower engulfs more of each light (down its gradient toward
    # the dim edges) and catches dimmer fixtures; higher keeps only the
    # hottest cores. Defaults low because one very bright light makes the
    # near-max reference large, so even modest lights need a low fraction.
    threshold: float = 0.03
    # Gaussian blur applied to luminance before thresholding, in degrees.
    # Bridges the dark mullions / beams / sheer-curtain gaps that would
    # otherwise split one light into pieces.
    blur_deg: float = 1.0
    # Hard cap on proposals — the N highest-power blobs survive.
    max_quads: int = 12
    # Blobs whose angular diameter is below this are discarded as noise.
    min_diameter_deg: float = 1.0
    # Blobs bigger than this are discarded — a "light" spanning most of the
    # view is the threshold flooding a bright wall/ceiling, not a fixture.
    max_diameter_deg: float = 90.0
    # Blobs whose centroids are within this angular distance are merged into a
    # single quad. 0 keeps every light separate.
    merge_distance_deg: float = 1.0
    # Drop blobs centred well below the horizon — they are almost always sun /
    # light reflections on the floor, not real lights. Wall mirrors and other
    # near-horizon reflections are kept (the user can delete extras).
    suppress_floor: bool = True


@dataclass
class DetectedQuad:
    corners_dirs: np.ndarray  # (4, 3) unit direction vectors, ordered TL/TR/BR/BL
    power: float              # integrated luminance * solid angle (relative)
    solid_angle: float        # steradians covered by the bright blob


# ---------- union-find ----------

class _UF:
    def __init__(self, n: int):
        self._p = list(range(n))

    def find(self, a: int) -> int:
        while self._p[a] != a:
            self._p[a] = self._p[self._p[a]]
            a = self._p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[ra] = rb


def _pixel_dirs(ys: np.ndarray, xs: np.ndarray, W: int, H: int) -> np.ndarray:
    """(row, col) equirect pixels -> unit direction vectors, shape (N, 3)."""
    yaw, pitch = pix_to_angles(xs.astype(np.float64), ys.astype(np.float64), W, H)
    return dir_from_angles(yaw, pitch)


def _seam_aware_components(bright: np.ndarray) -> np.ndarray:
    """Connected-component label the bright mask, then stitch labels that meet
    across the left/right seam (equirect wraps horizontally). Label 0 is
    background."""
    n, labels = cv2.connectedComponents(bright.astype(np.uint8), connectivity=8)
    W = bright.shape[1]
    uf = _UF(n)
    left = labels[:, 0]
    right = labels[:, W - 1]
    seam = (left > 0) & (right > 0)
    if seam.any():
        for la, lb in zip(left[seam].tolist(), right[seam].tolist()):
            uf.union(la, lb)
        roots = np.array([uf.find(i) for i in range(n)], dtype=np.int32)
        labels = roots[labels]
    return labels


def _order_box(box: np.ndarray) -> np.ndarray:
    """Order 4 box corners as TL, TR, BR, BL (image coords, +y down)."""
    c = box.mean(axis=0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    box = box[np.argsort(ang)]                       # consistent cyclic order
    start = int(np.argmin(box[:, 0] + box[:, 1]))    # top-left-most corner
    box = np.roll(box, -start, axis=0)
    # argsort(atan2) winds clockwise in image coords -> TL, TR, BR, BL.
    return box


def _blob_quad(
    ys: np.ndarray, xs: np.ndarray, W: int, H: int, pad_frac: float = 0.04
) -> np.ndarray:
    """Fit an oriented quad to a blob in a gnomonic tangent plane.

    A flat rectangular fixture has straight 3D edges, so on the sphere each
    edge is a great-circle arc — *curved* in equirect pixel space. Fitting a
    rectangle directly to the equirect blob therefore comes out skewed and
    loose. Instead we project the blob's pixel directions into a gnomonic
    (rectilinear) tangent plane centred on the blob: there great circles
    become straight lines, so the fixture reads as a true straight-edged
    quad. A minimum-area rotated rectangle is fit in that plane, padded
    slightly, and its 4 corners lifted back to direction vectors (ordered
    TL, TR, BR, BL).

    Working in directions rather than pixel columns makes the seam wrap a
    non-issue — the tangent projection handles it implicitly."""
    dirs = _pixel_dirs(ys, xs, W, H)                       # (N, 3) unit dirs
    centroid = dirs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    yaw, pitch = angles_from_dir(centroid.reshape(1, 3))
    basis = view_basis(float(yaw[0]), float(pitch[0]))     # rows: right/up/fwd

    # Gnomonic projection onto the tangent plane at the centroid. y is negated
    # so the plane uses image convention (+y down), matching `_order_box`.
    cam = dirs @ basis.T                                   # (N, 3) camera frame
    front = cam[:, 2] > 1e-3
    if int(front.sum()) >= 3:
        cam = cam[front]
    tx = cam[:, 0] / cam[:, 2]
    ty = -cam[:, 1] / cam[:, 2]
    pts = np.stack([tx, ty], axis=1).astype(np.float32)

    (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
    grow = 1.0 + 2.0 * pad_frac
    box = cv2.boxPoints(((cx, cy), (rw * grow, rh * grow), ang))
    box = _order_box(np.asarray(box, dtype=np.float64))    # TL, TR, BR, BL

    # Tangent-plane corner -> camera ray -> world direction.
    corners = np.empty((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(box):
        ray = u * basis[0] - v * basis[1] + basis[2]
        corners[i] = ray / (np.linalg.norm(ray) + 1e-12)
    return corners


def bright_mask(
    hdr: np.ndarray,
    params: DetectParams | None = None,
    exclude_mask: np.ndarray | None = None,
) -> np.ndarray:
    """The flat black/white luma-key mask the detector works on.

    Blurs the luminance, applies the luma key (`threshold` × the scene's
    brightest), then a morphological close to bridge panes / transoms. Exposed
    so the UI can preview exactly what the Brightness / Blur knobs select.
    Returns a bool array (H, W)."""
    p = params or DetectParams()
    H, W = hdr.shape[:2]
    lum = luminance(hdr.astype(np.float32))

    # --- blur so window panes / beam-split lights read as one blob ---
    sigma = max(0.0, p.blur_deg) / 180.0 * H
    if sigma > 0.3:
        lum = cv2.GaussianBlur(lum, (0, 0), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REFLECT)

    # --- luma key: threshold is a fraction of the scene's brightest
    # luminance. Pixels above it become a flat white blob, everything else
    # black — the blob's shape no longer depends on the light's internal
    # gradient, so a quad engulfs the whole light surface above that level.
    bright_ref = float(np.percentile(lum, 99.9))
    t = float(np.clip(p.threshold, 0.0, 1.0))
    bright = lum > t * bright_ref
    if exclude_mask is not None:
        bright &= ~(np.asarray(exclude_mask) > 0)
    if not bright.any():
        return bright

    # Morphological close fills the leftover holes a blur can't fully bridge —
    # window panes, sheer-curtain gaps, transom bars, beam shadows — so the
    # blob engulfs the whole light as one solid region. The kernel scales
    # generously with the blur knob: raising Blur both smooths and bridges
    # wider gaps, so a window split by a transom merges into one quad.
    close_px = max(3, int(round(3.5 * sigma)) | 1)
    return cv2.morphologyEx(
        bright.astype(np.uint8), cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px)),
    ).astype(bool)


def propose_quads(
    hdr: np.ndarray,
    params: DetectParams | None = None,
    exclude_mask: np.ndarray | None = None,
) -> list[DetectedQuad]:
    """Propose light quads for a latlong HDR panorama.

    Parameters
    ----------
    hdr : (H, W, 3) float32 latlong panorama.
    params : detection knobs (see `DetectParams`).
    exclude_mask : optional (H, W) bool/uint8 — bright pixels here are ignored
        (used to skip regions already covered by locked quads).

    Returns a list of `DetectedQuad`, brightest first, capped at
    `params.max_quads`.
    """
    p = params or DetectParams()
    H, W = hdr.shape[:2]

    bright = bright_mask(hdr, p, exclude_mask)
    if not bright.any():
        return []

    labels = _seam_aware_components(bright)
    sa = equirect_solid_angle(H, W)
    lum = luminance(hdr.astype(np.float32))  # for per-blob power ranking

    # Min / max blob solid angle from the angular-diameter bounds: treat each
    # bound as the diameter of a spherical cap, omega = 2*pi*(1 - cos(rho)).
    rho_lo = np.deg2rad(max(0.0, p.min_diameter_deg) * 0.5)
    rho_hi = np.deg2rad(min(180.0, p.max_diameter_deg) * 0.5)
    min_sa = 2.0 * np.pi * (1.0 - np.cos(rho_lo))
    max_sa = 2.0 * np.pi * (1.0 - np.cos(rho_hi))

    # --- per-component stats (one sorted pass, no per-label array scans) ---
    ys_all, xs_all = np.where(labels > 0)
    if ys_all.size == 0:
        return []
    lab_all = labels[ys_all, xs_all]
    order = np.argsort(lab_all, kind="stable")
    ys_all, xs_all, lab_all = ys_all[order], xs_all[order], lab_all[order]
    _uniq, starts = np.unique(lab_all, return_index=True)
    bounds = list(starts) + [lab_all.size]

    comps: list[dict] = []
    for gi in range(len(_uniq)):
        s, e = bounds[gi], bounds[gi + 1]
        ys, xs = ys_all[s:e], xs_all[s:e]
        comp_sa = float(sa[ys, xs].sum())
        if comp_sa < min_sa or comp_sa > max_sa:
            continue
        dirs = _pixel_dirs(ys, xs, W, H)
        power = float((lum[ys, xs] * sa[ys, xs]).sum())
        centroid = dirs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12
        # Floor suppression: a centroid pointing well below the horizon is a
        # reflection on the ground, not a light. -15° keeps low windows.
        if p.suppress_floor and centroid[1] < np.sin(np.deg2rad(-15.0)):
            continue
        comps.append({
            "ys": ys, "xs": xs, "centroid": centroid,
            "solid_angle": comp_sa, "power": power,
        })
    if not comps:
        return []

    # --- merge nearby components (single-linkage, angular) ---
    k = len(comps)
    uf = _UF(k)
    if p.merge_distance_deg > 0.0 and k > 1:
        merge_cos = float(np.cos(np.deg2rad(p.merge_distance_deg)))
        cents = np.stack([c["centroid"] for c in comps], axis=0)
        for a in range(k):
            for b in range(a + 1, k):
                if float(cents[a] @ cents[b]) >= merge_cos:
                    uf.union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(k):
        groups.setdefault(uf.find(i), []).append(i)

    # --- build a candidate quad per group ---
    cands: list[dict] = []
    for members in groups.values():
        ys = np.concatenate([comps[m]["ys"] for m in members])
        xs = np.concatenate([comps[m]["xs"] for m in members])
        cands.append({
            "ys": ys, "xs": xs,
            "power": sum(comps[m]["power"] for m in members),
            "sa": sum(comps[m]["solid_angle"] for m in members),
            "quad": _blob_quad(ys, xs, W, H),
        })

    # --- merge quads that overlap on the sphere (kills nested duplicates) ---
    cands = _merge_overlapping_quads(cands, W, H)

    out = [
        DetectedQuad(corners_dirs=c["quad"], power=c["power"], solid_angle=c["sa"])
        for c in cands
    ]
    out.sort(key=lambda d: d.power, reverse=True)
    return out[: max(1, int(p.max_quads))]


def _merge_overlapping_quads(
    cands: list[dict], W: int, H: int,
    raster_h: int = 256, raster_w: int = 512, min_overlap_frac: float = 0.15,
) -> list[dict]:
    """Merge candidate quads whose footprints overlap on the sphere.

    Each quad is rasterised to a low-res equirect mask; any pair whose
    intersection exceeds `min_overlap_frac` of the smaller quad is fused — the
    underlying blob pixels are combined and a single quad refit. Iterates until
    no overlaps remain, so a stack of nested duplicates collapses to one quad."""
    for _ in range(5):
        n = len(cands)
        if n < 2:
            break
        masks = [
            rasterize_spherical_quad(c["quad"], raster_h, raster_w)[0] > 0
            for c in cands
        ]
        areas = [max(1, int(m.sum())) for m in masks]
        uf = _UF(n)
        for i in range(n):
            for j in range(i + 1, n):
                inter = int(np.logical_and(masks[i], masks[j]).sum())
                if inter > min_overlap_frac * min(areas[i], areas[j]):
                    uf.union(i, j)
        merged: dict[int, list[int]] = {}
        for i in range(n):
            merged.setdefault(uf.find(i), []).append(i)
        if len(merged) == n:
            break  # nothing overlapped — stable
        cands = [
            {
                "ys": np.concatenate([cands[m]["ys"] for m in members]),
                "xs": np.concatenate([cands[m]["xs"] for m in members]),
                "power": sum(cands[m]["power"] for m in members),
                "sa": sum(cands[m]["sa"] for m in members),
                "quad": _blob_quad(
                    np.concatenate([cands[m]["ys"] for m in members]),
                    np.concatenate([cands[m]["xs"] for m in members]),
                    W, H,
                ),
            }
            for members in merged.values()
        ]
    return cands
