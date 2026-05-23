"""Equirect panorama viewer with click-to-place + draggable-vertex editing.

UX
--
- **Add mode** (entered via "Add quad" button or `A`): cursor turns crosshair,
  user clicks 4 times to place 4 vertices. A preview polyline shows the work in
  progress. The 4th click commits a LightQuad and returns to Select mode.
- **Select mode** (default): clicking a quad selects it; clicking empty area
  deselects. The selected quad gets 4 draggable vertex handles — dragging a
  handle moves that corner directly. The 4 great-circle edges between corners
  are redrawn live as curves on the equirect pano (including correct seam-wrap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QImage,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

from env2lgt.proj import (
    angles_from_dir,
    angles_to_pix,
    dir_from_angles,
    pix_to_angles,
)


@dataclass
class LightQuad:
    """A light's region on the sphere, as 4 unit direction vectors (corners)."""

    name: str
    corners_dirs: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    # Window/portal: bake keeps the rect at wall depth (see bake.QuadSpec).
    is_window: bool = False
    # Provenance: "user" (4-click placed) or "auto" (proposed by detect).
    source: str = "user"
    # Locked quads are never removed/replaced by "Propose quads". The lock is
    # set explicitly via the list lock toggle, and automatically the first time
    # an auto quad's vertices are edited (so a refinement survives a re-run).
    locked: bool = False
    # True once the user has pressed "Fit to rect light" — at that point the
    # 4 corner dirs have been snapped (via depth + plane fit + diagonal-
    # bisector) to a *rigid* rectangle on the sphere, so what the viewer
    # shows matches what the bake authors. Any vertex drag flips this back
    # to False (so the user knows a re-fit is needed before the next bake).
    is_rect_fitted: bool = False

    @classmethod
    def from_pano_bbox(cls, name: str, x: int, y: int, w: int, h: int, W: int, H: int) -> "LightQuad":
        """(kept for backwards compatibility / batch tools) Seed from a 2D bbox."""
        cx = x + w / 2.0
        cy = y + h / 2.0
        yaw_c = float((cx + 0.5) / W * 2.0 * np.pi - np.pi)
        pitch_c = float(np.pi * 0.5 - (cy + 0.5) / H * np.pi)
        half_yaw = (w / W) * np.pi
        half_pitch = (h / H) * np.pi * 0.5
        forward = dir_from_angles(yaw_c, pitch_c)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up)
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, forward)
        up /= np.linalg.norm(up) + 1e-12
        tx, ty = np.tan(half_yaw), np.tan(half_pitch)
        corners = np.stack(
            [
                forward - tx * right + ty * up,
                forward + tx * right + ty * up,
                forward + tx * right - ty * up,
                forward - tx * right - ty * up,
            ],
            axis=0,
        )
        corners /= np.linalg.norm(corners, axis=1, keepdims=True) + 1e-12
        return cls(name=name, corners_dirs=corners)

    @classmethod
    def from_corner_dirs(cls, name: str, corners: np.ndarray) -> "LightQuad":
        c = np.asarray(corners, dtype=np.float64).reshape(4, 3)
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
        return cls(name=name, corners_dirs=c)


HANDLE_R = 7  # pixel radius; drawn ignoring view transform so it stays constant


def _cc24_display_colors() -> list:
    """The 24 CC24 reference patches, sRGB-encoded for on-screen drawing.
    Painted (semi-transparent) inside the chart cells so the user can line
    the chart up by eye — patch 1 (dark skin) onto the dark-skin patch."""
    from env2lgt.colorchecker import CC24_LINEAR_SRGB

    cols = []
    for rgb in CC24_LINEAR_SRGB:
        enc = [int(round(min(1.0, max(0.0, float(c))) ** (1.0 / 2.2) * 255)) for c in rgb]
        cols.append(QColor(enc[0], enc[1], enc[2]))
    return cols


_CC24_DISPLAY = _cc24_display_colors()

# Reference swatches are drawn inset within each chart cell (as a fraction of
# the cell), leaving a window so the real chart underneath stays visible.
_CC_SWATCH_INSET = 0.26


# ---------- vertex handle (draggable) ----------

class _VertexHandle(QGraphicsEllipseItem):
    """Constant-pixel-size draggable dot. Reports drags to the owning viewer."""

    def __init__(self, index: int, owner: "PanoramaViewer"):
        super().__init__(-HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2)
        self._index = index
        self._owner = owner
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setBrush(QBrush(QColor(255, 220, 80)))
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setZValue(200)
        self._suppress_notify = False

    def set_pano_pos_silent(self, px: float, py: float) -> None:
        self._suppress_notify = True
        self.setPos(QPointF(px, py))
        self._suppress_notify = False

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._suppress_notify
        ):
            self._owner.on_vertex_dragged(self._index, self.scenePos())
        return super().itemChange(change, value)


class _ChartHandle(QGraphicsEllipseItem):
    """Draggable corner handle for the colour-checker chart quad."""

    def __init__(self, index: int, owner: "PanoramaViewer"):
        super().__init__(-HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2)
        self._index = index
        self._owner = owner
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        # White handles, distinct from the yellow light-quad handles.
        self.setBrush(QBrush(QColor(245, 245, 245)))
        self.setPen(QPen(QColor(20, 20, 20), 1))
        self.setZValue(210)
        self._suppress_notify = False

    def set_pano_pos_silent(self, px: float, py: float) -> None:
        self._suppress_notify = True
        self.setPos(QPointF(px, py))
        self._suppress_notify = False

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._suppress_notify
        ):
            self._owner.on_chart_vertex_dragged(self._index, self.scenePos())
        return super().itemChange(change, value)


class _RegionHandle(QGraphicsEllipseItem):
    """Draggable corner handle for a region rect (calibration regions mode).
    Constant-pixel-size; routes drags to the owning viewer."""

    def __init__(self, pair_index: int, slot: str, corner_index: int, owner: "PanoramaViewer"):
        super().__init__(-HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2)
        self._pair_index = int(pair_index)
        self._slot = str(slot)
        self._corner_index = int(corner_index)
        self._owner = owner
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(220)
        self._suppress_notify = False

    def set_scene_pos_silent(self, x: float, y: float) -> None:
        self._suppress_notify = True
        self.setPos(QPointF(x, y))
        self._suppress_notify = False

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._suppress_notify
        ):
            self._owner._on_region_handle_moved(
                self._pair_index, self._slot, self._corner_index, self.scenePos()
            )
        return super().itemChange(change, value)

    def mousePressEvent(self, event):  # noqa: N802
        # Click on a handle also selects the pair.
        self._owner._on_region_selected(self._pair_index)
        super().mousePressEvent(event)


class _RegionMoveHandle(QGraphicsPathItem):
    """A "+" glyph drawn at the centre of a region rect. Drag to move the
    whole rect; constant pixel size so it stays grabbable at any zoom."""

    def __init__(self, pair_index: int, slot: str, owner: "PanoramaViewer"):
        super().__init__()
        self._pair_index = int(pair_index)
        self._slot = str(slot)
        self._owner = owner
        r = HANDLE_R + 2
        path = QPainterPath()
        path.moveTo(-r, 0); path.lineTo(r, 0)
        path.moveTo(0, -r); path.lineTo(0, r)
        self.setPath(path)
        # A small invisible square enlarges the hit area beyond the thin glyph.
        self._hit_radius = r + 2
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        self.setZValue(225)
        self._suppress_notify = False
        # Last known centre (scene coords) so we can compute drag deltas.
        self._last_centre = QPointF(0.0, 0.0)

    def boundingRect(self):  # noqa: N802
        r = self._hit_radius
        return QRectF(-r, -r, 2 * r, 2 * r)

    def shape(self):  # noqa: N802
        # Wider hit shape than the glyph so the user can grab it easily.
        r = self._hit_radius
        path = QPainterPath()
        path.addRect(QRectF(-r, -r, 2 * r, 2 * r))
        return path

    def set_scene_pos_silent(self, x: float, y: float) -> None:
        self._suppress_notify = True
        self.setPos(QPointF(x, y))
        self._last_centre = QPointF(x, y)
        self._suppress_notify = False

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._suppress_notify
        ):
            self._owner._on_region_move_handle_dragged(
                self._pair_index, self._slot, self.scenePos(), self._last_centre
            )
        return super().itemChange(change, value)

    def mousePressEvent(self, event):  # noqa: N802
        self._owner._on_region_selected(self._pair_index)
        super().mousePressEvent(event)


class _RegionBodyItem(QGraphicsRectItem):
    """The rect body of one region in one slot (HDR or REF). Drag-to-move."""

    def __init__(self, pair_index: int, slot: str, owner: "PanoramaViewer"):
        super().__init__(0.0, 0.0, 1.0, 1.0)
        self._pair_index = int(pair_index)
        self._slot = str(slot)
        self._owner = owner
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self._suppress_notify = False

    def set_pos_silent(self, x: float, y: float) -> None:
        self._suppress_notify = True
        self.setPos(QPointF(x, y))
        self._suppress_notify = False

    def set_rect_silent(self, x: float, y: float, w: float, h: float) -> None:
        self._suppress_notify = True
        self.setRect(0.0, 0.0, max(2.0, w), max(2.0, h))
        self.setPos(QPointF(x, y))
        self._suppress_notify = False

    def itemChange(self, change, value):  # noqa: N802
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemScenePositionHasChanged
            and not self._suppress_notify
        ):
            self._owner._on_region_body_moved(self._pair_index, self._slot)
        return super().itemChange(change, value)

    def mousePressEvent(self, event):  # noqa: N802
        # Body click selects the pair (in addition to letting Qt start a drag).
        self._owner._on_region_selected(self._pair_index)
        super().mousePressEvent(event)


# ---------- viewer ----------

class PanoramaViewer(QGraphicsView):
    quad_committed = Signal(LightQuad)  # new quad finished (4-click placement)
    quad_modified = Signal(str)         # name of quad whose vertices were edited
    quad_selected = Signal(str)         # name; empty string to deselect
    quads_delete_requested = Signal(list)  # marquee-selected names, Del key
    quads_marquee_changed = Signal(list)   # marquee selection set changed
    quad_lock_changed = Signal(str, bool)  # name, locked — auto-lock on edit
    quad_fit_changed = Signal(str, bool)   # name, is_rect_fitted — toggled on
                                            # depth-snap + auto-invalidated on edit
    add_mode_changed = Signal(bool)
    pixel_probed = Signal(int, int)     # absolute pano pixel (display res) under cursor
    probe_left = Signal()               # cursor left the panorama
    area_sampled = Signal(int, int, int, int)  # absolute pano px x0,y0,x1,y1
    sample_mode_changed = Signal(bool)
    chart_committed = Signal()          # colour-chart 4 corners finished
    chart_modified = Signal()           # colour-chart corner dragged
    chart_mode_changed = Signal(bool)
    # Region-pair calibration. Pair rects are persistent + draggable; the
    # viewer reports edits live. Selection is shared across HDR / REF views.
    region_pair_selected = Signal(int)            # pair_index, -1 = deselect
    region_pair_modified = Signal(int, str)       # pair_index, "hdri"|"ref"
    region_pair_delete_requested = Signal(int)    # Delete-key on selected pair

    MODE_SELECT = "select"
    MODE_ADD = "add"
    MODE_SAMPLE = "sample"
    MODE_CHART = "chart"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor(20, 20, 20))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Track motion without a pressed button so the pixel probe updates live.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._bg_item = None
        self._W = 0
        self._H = 0
        # Yaw offset (in display pixels) applied to the *display* of the pano.
        # The HDR is rolled by this amount before tonemap (done by the app).
        # All quad data remains in absolute spherical coords; we convert
        # display<->absolute at click / render time.
        self._yaw_offset_px = 0
        # Middle-mouse pan state.
        self._panning = False
        self._pan_last: QPointF | None = None
        self._mode = self.MODE_SELECT
        # Add-mode in-progress vertices and preview.
        self._placing_dirs: list[np.ndarray] = []
        self._placing_dots: list[QGraphicsEllipseItem] = []
        self._placing_path: QGraphicsPathItem | None = None
        # Stored quads.
        self._quads: dict[str, LightQuad] = {}
        self._quad_items: dict[str, QGraphicsPathItem] = {}
        self._selected: str | None = None
        # Multi-selection set built by drag-marquee in select mode. Separate
        # from `_selected` (which still drives the per-corner edit handles).
        self._marquee_selected: set[str] = set()
        self._marquee_active: bool = False
        self._marquee_start: QPointF | None = None
        self._marquee_rect_item: QGraphicsRectItem | None = None
        # Vertex handles for the currently selected quad (rebuilt on selection change).
        self._vertex_handles: list[_VertexHandle] = []
        # Auto-detect key-mask preview overlay (an RGBA QImage, display res).
        self._key_overlay_img: QImage | None = None
        self._key_overlay_item = None
        # Rectangle-sample (exposure / WB / probe) drag state.
        self._sampling = False
        self._sample_start: QPointF | None = None
        self._sample_rect_item: QGraphicsRectItem | None = None
        # Quads hidden in exposure mode; chart + region items hidden outside it.
        self._quads_visible = True
        self._calibration_visible = True
        # Last-rendered yaw, so the fast-path swap can re-project quad/chart/
        # region paths only when yaw actually changed (cheap when it hasn't).
        self._last_rendered_yaw_px = 0
        # Colour-checker chart. Two independent slots so the HDRI chart and a
        # flat reference-image chart both survive a view switch:
        #   _chart_dirs : (4,3) corner dirs   — used on the equirect panorama
        #   _chart_uv   : (4,2) normalised px — used on a flat reference image
        # `_flat_mode` selects which one is active (matches the shown image).
        self._flat_mode: bool = False
        self._chart_dirs: np.ndarray | None = None
        self._chart_uv: np.ndarray | None = None
        self._chart_item: QGraphicsPathItem | None = None
        self._chart_grid_item: QGraphicsPathItem | None = None
        self._chart_cell_items: list = []
        self._chart_handles: list[_ChartHandle] = []
        self._placing_chart: list[np.ndarray] = []
        self._placing_chart_dots: list[QGraphicsEllipseItem] = []
        # Region-pair calibration state. Pairs are persistent + editable;
        # selecting a pair highlights its rect on whichever view is showing.
        #   _region_pairs: list of {"name": str, "hdri_uv": [u0,v0,u1,v1],
        #                           "ref_uv": [u0,v0,u1,v1]}
        #   _region_items: dict[(pair_i, slot)] -> {"body": item, "handles": [..],
        #                                            "label": item}
        self._region_pairs: list[dict] = []
        self._region_items: dict = {}
        self._region_selected: int = -1

    # ---------- mode ----------

    def is_add_mode(self) -> bool:
        return self._mode == self.MODE_ADD

    def start_add_mode(self):
        if self._mode == self.MODE_ADD:
            return
        self._mode = self.MODE_ADD
        self._clear_placing()
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._set_selected(None)
        self.quad_selected.emit("")
        self.add_mode_changed.emit(True)

    def cancel_add_mode(self):
        if self._mode != self.MODE_ADD:
            return
        self._mode = self.MODE_SELECT
        self._clear_placing()
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.add_mode_changed.emit(False)

    def _clear_placing(self):
        self._placing_dirs = []
        for d in self._placing_dots:
            self._scene.removeItem(d)
        self._placing_dots = []
        if self._placing_path is not None:
            self._scene.removeItem(self._placing_path)
            self._placing_path = None

    # ---------- rectangle-sample mode ----------

    def is_sample_mode(self) -> bool:
        return self._mode == self.MODE_SAMPLE

    def start_sample_mode(self):
        """Enter drag-a-rectangle mode. On release the viewer emits
        `area_sampled` with absolute pano pixel bounds."""
        if self._mode == self.MODE_ADD:
            self.cancel_add_mode()
        self._mode = self.MODE_SAMPLE
        self._clear_sample_rect()
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.sample_mode_changed.emit(True)

    def cancel_sample_mode(self):
        if self._mode != self.MODE_SAMPLE:
            return
        self._mode = self.MODE_SELECT
        self._clear_sample_rect()
        self._sampling = False
        self._sample_start = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.sample_mode_changed.emit(False)

    def _clear_sample_rect(self):
        if self._sample_rect_item is not None:
            self._scene.removeItem(self._sample_rect_item)
            self._sample_rect_item = None

    # ---------- quad visibility ----------

    def set_quads_visible(self, visible: bool) -> None:
        """Show/hide quad outlines + vertex handles (exposure mode hides them)."""
        self._quads_visible = bool(visible)
        for item in self._quad_items.values():
            item.setVisible(self._quads_visible)
        for h in self._vertex_handles:
            h.setVisible(self._quads_visible)

    def set_calibration_visible(self, visible: bool) -> None:
        """Show/hide all chart + region-pair items. Used to keep calibration
        overlays confined to exposure mode (so the lights view is uncluttered).
        Data is preserved; only scene visibility changes."""
        self._calibration_visible = bool(visible)
        # Chart items.
        for it in (self._chart_item, self._chart_grid_item):
            if it is not None:
                it.setVisible(self._calibration_visible)
        for it in self._chart_cell_items:
            it.setVisible(self._calibration_visible)
        for h in self._chart_handles:
            h.setVisible(self._calibration_visible)
        for d in self._placing_chart_dots:
            d.setVisible(self._calibration_visible)
        # Region items.
        for entry in self._region_items.values():
            for k in ("body", "label", "move"):
                it = entry.get(k)
                if it is not None:
                    it.setVisible(self._calibration_visible)
            for h in entry.get("handles", []):
                h.setVisible(self._calibration_visible)

    # ---------- colour-checker chart ----------

    def is_chart_mode(self) -> bool:
        return self._mode == self.MODE_CHART

    def start_chart_mode(self):
        """Enter 4-click placement for the colour-checker chart. The user
        clicks corners in order: dark-skin (TL), TR, BR, BL."""
        if self._mode == self.MODE_ADD:
            self.cancel_add_mode()
        if self._mode == self.MODE_SAMPLE:
            self.cancel_sample_mode()
        self._mode = self.MODE_CHART
        self._placing_chart = []
        for d in self._placing_chart_dots:
            self._scene.removeItem(d)
        self._placing_chart_dots = []
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.chart_mode_changed.emit(True)

    def cancel_chart_mode(self):
        if self._mode != self.MODE_CHART:
            return
        self._mode = self.MODE_SELECT
        for d in self._placing_chart_dots:
            self._scene.removeItem(d)
        self._placing_chart_dots = []
        self._placing_chart = []
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.chart_mode_changed.emit(False)

    def _place_chart_vertex(self, scene_pt: QPointF) -> None:
        x = float(np.clip(scene_pt.x(), 0.0, max(1.0, self._W - 1)))
        y = float(np.clip(scene_pt.y(), 0.0, max(1.0, self._H - 1)))
        if self._flat_mode:
            self._placing_chart.append(np.array([x / self._W, y / self._H]))
        else:
            u_abs = self._display_to_abs_x(x)
            yaw, pitch = pix_to_angles(np.array(u_abs), np.array(y), self._W, self._H)
            self._placing_chart.append(
                np.asarray(dir_from_angles(yaw, pitch), dtype=np.float64).reshape(3)
            )
        dot = self._scene.addEllipse(
            -HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2,
            QPen(QColor(20, 20, 20), 1), QBrush(QColor(245, 245, 245)),
        )
        dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        dot.setPos(QPointF(x, y))
        dot.setZValue(212)
        self._placing_chart_dots.append(dot)
        if len(self._placing_chart) >= 4:
            corners = np.stack(self._placing_chart, axis=0)
            if self._flat_mode:
                self._chart_uv = corners
            else:
                self._chart_dirs = corners
            for dd in self._placing_chart_dots:
                self._scene.removeItem(dd)
            self._placing_chart_dots = []
            self._placing_chart = []
            self._mode = self.MODE_SELECT
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._refresh_chart()
            self.chart_mode_changed.emit(False)
            self.chart_committed.emit()

    # ---------- chart accessors ----------

    def set_flat_mode(self, flat: bool) -> None:
        """Select which chart slot is active: a flat reference image (planar
        chart) vs the equirect panorama (spherical chart). Call before
        set_image() for the new view."""
        self._flat_mode = bool(flat)
        # Region overlays are filtered by view; re-render so the inactive
        # view's rects are hidden and the active view's are shown.
        self._refresh_regions()

    def _active_chart(self):
        return self._chart_uv if self._flat_mode else self._chart_dirs

    def set_chart(self, dirs: np.ndarray | None) -> None:
        """Set the equirect (panorama) chart from 4 corner dirs."""
        self._chart_dirs = (
            None if dirs is None else np.asarray(dirs, dtype=np.float64).reshape(4, 3)
        )
        self._refresh_chart()

    def set_chart_uv(self, uv: np.ndarray | None) -> None:
        """Set the flat (reference-image) chart from 4 normalised corners."""
        self._chart_uv = (
            None if uv is None else np.asarray(uv, dtype=np.float64).reshape(4, 2)
        )
        self._refresh_chart()

    def clear_chart(self) -> None:
        """Clear the chart for the active view (HDRI or reference)."""
        if self._flat_mode:
            self._chart_uv = None
        else:
            self._chart_dirs = None
        self._clear_chart_items()

    def has_chart(self) -> bool:
        return self._active_chart() is not None

    def chart_dirs(self) -> np.ndarray | None:
        return None if self._chart_dirs is None else self._chart_dirs.copy()

    def chart_uv(self) -> np.ndarray | None:
        return None if self._chart_uv is None else self._chart_uv.copy()

    # ---------- chart geometry ----------

    def _chart_param_display(self, u: float, v: float) -> tuple[float, float]:
        """Parametric chart coord (u, v) in [0,1]^2 -> display scene pixel.
        Spherical-bilinear on the panorama, plain bilinear on a flat image."""
        if self._flat_mode:
            c = self._chart_uv
            top = c[0] * (1.0 - u) + c[1] * u
            bot = c[3] * (1.0 - u) + c[2] * u
            p = top * (1.0 - v) + bot * v
            return float(p[0] * self._W), float(p[1] * self._H)
        from env2lgt.proj import spherical_bilinear

        return self._dir_to_display_pix(spherical_bilinear(self._chart_dirs, u, v))

    def _chart_corner_display(self, i: int) -> tuple[float, float]:
        if self._flat_mode:
            c = self._chart_uv[i]
            return float(c[0] * self._W), float(c[1] * self._H)
        return self._dir_to_display_pix(self._chart_dirs[i])

    def _clear_chart_items(self) -> None:
        self._clear_chart_shapes()
        for h in self._chart_handles:
            self._scene.removeItem(h)
        self._chart_handles = []

    def _clear_chart_shapes(self) -> None:
        for it in (self._chart_item, self._chart_grid_item):
            if it is not None:
                self._scene.removeItem(it)
        self._chart_item = None
        self._chart_grid_item = None
        for it in self._chart_cell_items:
            self._scene.removeItem(it)
        self._chart_cell_items = []

    def _build_chart_shapes(self) -> None:
        """A 6x4 grid with an inset reference swatch in every cell, plus an
        outer border. Each swatch is drawn smaller than its cell so the real
        chart underneath shows through around it — line the chart up by
        matching each swatch to the patch framing it."""
        cell_pen = QPen(QColor(245, 245, 245, 120), 1)
        cell_pen.setCosmetic(True)
        no_pen = QPen(Qt.PenStyle.NoPen)
        m = _CC_SWATCH_INSET
        for j in range(4):
            for i in range(6):
                # Full cell — grid lines only, no fill.
                cell = QPolygonF()
                for (uu, vv) in (
                    (i / 6.0, j / 4.0), ((i + 1) / 6.0, j / 4.0),
                    ((i + 1) / 6.0, (j + 1) / 4.0), (i / 6.0, (j + 1) / 4.0),
                ):
                    cell.append(QPointF(*self._chart_param_display(uu, vv)))
                cell_item = self._scene.addPolygon(
                    cell, cell_pen, QBrush(Qt.BrushStyle.NoBrush)
                )
                cell_item.setZValue(15)
                self._chart_cell_items.append(cell_item)
                # Inset reference swatch.
                swatch = QPolygonF()
                for (su, sv) in (
                    (i + m, j + m), (i + 1 - m, j + m),
                    (i + 1 - m, j + 1 - m), (i + m, j + 1 - m),
                ):
                    swatch.append(
                        QPointF(*self._chart_param_display(su / 6.0, sv / 4.0))
                    )
                col = _CC24_DISPLAY[j * 6 + i]
                fill = QColor(col.red(), col.green(), col.blue(), 235)
                sw_item = self._scene.addPolygon(swatch, no_pen, QBrush(fill))
                sw_item.setZValue(15)
                self._chart_cell_items.append(sw_item)
        self._chart_item = self._scene.addPath(
            self._chart_border_path(), QPen(QColor(255, 235, 90), 2)
        )
        self._chart_item.setZValue(16)

    def _chart_border_path(self) -> QPainterPath:
        """The outer chart border, sampled along its 4 edges (curves on the
        panorama, straight on a flat image), with seam-wrap breaks."""
        n = 14
        edges = (
            [(t, 0.0) for t in np.linspace(0, 1, n)]
            + [(1.0, t) for t in np.linspace(0, 1, n)]
            + [(t, 1.0) for t in np.linspace(1, 0, n)]
            + [(0.0, t) for t in np.linspace(1, 0, n)]
        )
        pts = [self._chart_param_display(u, v) for (u, v) in edges]
        path = QPainterPath()
        path.moveTo(*pts[0])
        last_u = pts[0][0]
        for (u, v) in pts[1:]:
            if abs(u - last_u) > self._W / 2:
                path.moveTo(u, v)
            else:
                path.lineTo(u, v)
            last_u = u
        return path

    def _refresh_chart(self) -> None:
        self._clear_chart_items()
        if self._active_chart() is None or self._W <= 0:
            return
        self._build_chart_shapes()
        for i in range(4):
            h = _ChartHandle(i, self)
            self._scene.addItem(h)
            h.set_pano_pos_silent(*self._chart_corner_display(i))
            self._chart_handles.append(h)
        # Rebuilt items default to visible — re-apply the calibration-visible
        # flag so the chart stays hidden when we're not in exposure mode.
        if not self._calibration_visible:
            self.set_calibration_visible(False)

    def on_chart_vertex_dragged(self, index: int, scene_pos: QPointF) -> None:
        if self._active_chart() is None:
            return
        x = float(np.clip(scene_pos.x(), 0.0, self._W - 1))
        y = float(np.clip(scene_pos.y(), 0.0, self._H - 1))
        if self._flat_mode:
            self._chart_uv[index] = [x / self._W, y / self._H]
        else:
            u_abs = self._display_to_abs_x(x)
            yaw, pitch = pix_to_angles(np.array(u_abs), np.array(y), self._W, self._H)
            self._chart_dirs[index] = np.asarray(
                dir_from_angles(yaw, pitch), dtype=np.float64
            ).reshape(3)
        # Rebuild cells + border; leave the handles (the dragged one tracks
        # the cursor on its own).
        self._clear_chart_shapes()
        self._build_chart_shapes()
        self.chart_modified.emit()

    # ---------- region-pair calibration ----------

    # A small palette cycled per-pair; mirrors the Lights outliner colour rules.
    _REGION_PALETTE = [
        (120, 220, 255),  # cyan
        (255, 170, 80),   # orange
        (140, 230, 140),  # green
        (240, 130, 240),  # magenta
        (240, 220, 120),  # yellow
        (180, 160, 255),  # lavender
    ]

    @classmethod
    def region_pair_color(cls, pair_index: int) -> tuple:
        return cls._REGION_PALETTE[pair_index % len(cls._REGION_PALETTE)]

    def set_region_pairs(
        self, pairs: list, selected: int = -1
    ) -> None:
        """Replace the region-pair list. Each pair is a dict with "hdri_uv"
        and "ref_uv" (each [u0,v0,u1,v1] normalised, or empty). `selected` is
        the pair index to highlight (-1 = none)."""
        self._region_pairs = [
            {
                "name": str(p.get("name") or f"Region {i + 1}"),
                "hdri_uv": list(p.get("hdri_uv", []) or []),
                "ref_uv": list(p.get("ref_uv", []) or []),
            }
            for i, p in enumerate(pairs or [])
        ]
        self._region_selected = int(selected)
        self._refresh_regions()

    def set_region_selection(self, pair_index: int) -> None:
        """Highlight a pair without rebuilding (panel -> viewer sync)."""
        new = int(pair_index)
        if new == self._region_selected:
            return
        self._region_selected = new
        self._restyle_regions()

    def region_pairs(self) -> list:
        return [dict(p) for p in self._region_pairs]

    def region_selected(self) -> int:
        return self._region_selected

    def _clear_region_items(self) -> None:
        for entry in self._region_items.values():
            for k in ("body", "label", "move"):
                it = entry.get(k)
                if it is not None:
                    self._scene.removeItem(it)
            for h in entry.get("handles", []):
                self._scene.removeItem(h)
        self._region_items = {}

    def _refresh_regions(self) -> None:
        """Rebuild rect items for the currently shown view."""
        self._clear_region_items()
        if self._W <= 0:
            return
        slot = "ref" if self._flat_mode else "hdri"
        for i, pair in enumerate(self._region_pairs):
            uv = pair.get(f"{slot}_uv") or []
            if len(uv) != 4:
                continue
            self._build_region_items(i, slot, uv)
        self._restyle_regions()
        # Rebuilt items default to visible — re-apply the calibration-visible
        # flag so regions stay hidden when we're not in exposure mode (e.g.
        # after a project restore that calls set_region_pairs directly).
        if not self._calibration_visible:
            self.set_calibration_visible(False)

    def _build_region_items(self, pair_index: int, slot: str, uv: list) -> None:
        u0, v0, u1, v1 = uv
        v0, v1 = sorted((float(v0), float(v1)))
        u0, u1 = float(u0), float(u1)
        # For HDR, u1 may be < u0 — that means the rect wraps across the abs
        # equirect seam, which is a valid 2D rect on the cyclic panorama.
        # Use cyclic distance so we never sort-and-stretch.
        if slot == "hdri":
            width_fraction = (u1 - u0) if u1 >= u0 else (1.0 - u0 + u1)
            x0 = self._abs_to_display_x(u0 * self._W)
        else:
            # Flat reference image — no wrap.
            if u1 < u0:
                u0, u1 = u1, u0
            width_fraction = u1 - u0
            x0 = u0 * self._W
        # Sanity cap absurdly wide stored rects (legacy bug data).
        if width_fraction > 0.9:
            width_fraction = 0.15
            half = width_fraction * 0.5
            u0 = 0.5 - half
            u1 = 0.5 + half
            x0 = self._abs_to_display_x(u0 * self._W) if slot == "hdri" else u0 * self._W
            self._region_pairs[pair_index][f"{slot}_uv"] = [
                float(u0), float(v0), float(u1), float(v1)
            ]
        width_px = max(2.0, width_fraction * self._W)
        y0 = v0 * self._H
        y1 = v1 * self._H
        body = _RegionBodyItem(pair_index, slot, self)
        body.setRect(0.0, 0.0, width_px, max(2.0, y1 - y0))
        body.setPos(QPointF(x0, y0))
        body.setZValue(18)
        self._scene.addItem(body)
        handles = []
        for ci in range(4):
            h = _RegionHandle(pair_index, slot, ci, self)
            self._scene.addItem(h)
            handles.append(h)
        move = _RegionMoveHandle(pair_index, slot, self)
        self._scene.addItem(move)
        label = self._scene.addSimpleText(
            self._region_pairs[pair_index].get("name") or f"Region {pair_index + 1}"
        )
        label.setBrush(QBrush(QColor(255, 255, 255)))
        label.setPen(QPen(QColor(0, 0, 0), 1))
        label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        label.setZValue(19)
        self._region_items[(pair_index, slot)] = {
            "body": body, "handles": handles, "move": move, "label": label,
        }
        self._sync_region_handles(pair_index, slot)

    def _sync_region_handles(self, pair_index: int, slot: str) -> None:
        """Reposition the 4 corner handles + centre + label from the body."""
        entry = self._region_items.get((pair_index, slot))
        if entry is None:
            return
        body = entry["body"]
        pos = body.scenePos()
        r = body.rect()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + r.width(), y0 + r.height()
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        for h, (cx, cy) in zip(entry["handles"], corners):
            h.set_scene_pos_silent(cx, cy)
        cx_c, cy_c = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        move = entry.get("move")
        if move is not None:
            move.set_scene_pos_silent(cx_c, cy_c)
        entry["label"].setPos(QPointF(x0 + 4, y0 + 2))

    def _restyle_regions(self) -> None:
        for (pair_i, _slot), entry in self._region_items.items():
            sel = (pair_i == self._region_selected)
            col = QColor(*self.region_pair_color(pair_i))
            stroke = QPen(col, 3 if sel else 1.5)
            stroke.setCosmetic(True)
            fill_col = QColor(col)
            fill_col.setAlpha(80 if sel else 36)
            entry["body"].setPen(stroke)
            entry["body"].setBrush(QBrush(fill_col))
            handle_fill = QColor(col) if sel else QColor(30, 30, 30)
            for h in entry["handles"]:
                h.setBrush(QBrush(handle_fill))
                h.setPen(QPen(col, 2 if sel else 1))
            move = entry.get("move")
            if move is not None:
                move_pen = QPen(col, 3 if sel else 2)
                move_pen.setCosmetic(True)
                move.setPen(move_pen)

    # ---------- region edit callbacks (from items) ----------

    def _on_region_body_moved(self, pair_index: int, slot: str) -> None:
        entry = self._region_items.get((pair_index, slot))
        if entry is None:
            return
        body = entry["body"]
        pos = body.scenePos()
        r = body.rect()
        width, height = r.width(), r.height()
        # Clamp purely in display space so the rect stays inside the image
        # the user is looking at — no snapping. The stored UV is then derived
        # from the display position and may cross the abs equirect seam
        # (u1 < u0); rendering and sampling both handle that case.
        new_x = float(np.clip(pos.x(), 0.0, max(0.0, self._W - width)))
        new_y = float(np.clip(pos.y(), 0.0, max(0.0, self._H - height)))
        if new_x != pos.x() or new_y != pos.y():
            body.set_pos_silent(new_x, new_y)
        if slot == "hdri":
            abs_x0 = self._display_to_abs_x(new_x)
            u0 = abs_x0 / self._W
            u1 = ((abs_x0 + width) / self._W) % 1.0
        else:
            u0 = new_x / self._W
            u1 = (new_x + width) / self._W
        v0 = new_y / self._H
        v1 = (new_y + height) / self._H
        self._region_pairs[pair_index][f"{slot}_uv"] = [
            float(u0), float(v0), float(u1), float(v1)
        ]
        self._sync_region_handles(pair_index, slot)
        self.region_pair_modified.emit(pair_index, slot)

    def _on_region_handle_moved(
        self, pair_index: int, slot: str, corner_index: int, scene_pos: QPointF
    ) -> None:
        entry = self._region_items.get((pair_index, slot))
        if entry is None:
            return
        body = entry["body"]
        pos = body.scenePos()
        r = body.rect()
        x0, y0 = pos.x(), pos.y()
        x1, y1 = x0 + r.width(), y0 + r.height()
        nx = float(np.clip(scene_pos.x(), 0.0, self._W - 1))
        ny = float(np.clip(scene_pos.y(), 0.0, self._H - 1))
        if corner_index == 0:   # TL
            x0, y0 = nx, ny
        elif corner_index == 1:  # TR
            x1, y0 = nx, ny
        elif corner_index == 2:  # BR
            x1, y1 = nx, ny
        else:                    # BL
            x0, y1 = nx, ny
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        x1 = max(x0 + 2.0, x1)
        y1 = max(y0 + 2.0, y1)
        if slot == "hdri":
            abs_x0 = self._display_to_abs_x(x0)
            width = x1 - x0
            u0 = abs_x0 / self._W
            u1 = ((abs_x0 + width) / self._W) % 1.0
        else:
            u0, u1 = x0 / self._W, x1 / self._W
        v0, v1 = y0 / self._H, y1 / self._H
        body.set_rect_silent(x0, y0, x1 - x0, y1 - y0)
        self._region_pairs[pair_index][f"{slot}_uv"] = [
            float(u0), float(v0), float(u1), float(v1)
        ]
        self._sync_region_handles(pair_index, slot)
        self.region_pair_modified.emit(pair_index, slot)

    def _on_region_move_handle_dragged(
        self, pair_index: int, slot: str,
        new_centre: QPointF, last_centre: QPointF,
    ) -> None:
        """Centre "+" handle dragged — translate the body by the delta."""
        entry = self._region_items.get((pair_index, slot))
        if entry is None:
            return
        body = entry["body"]
        r = body.rect()
        pos = body.scenePos()
        # Translate body so its centre tracks the handle.
        new_x = float(np.clip(
            new_centre.x() - 0.5 * r.width(),
            0.0, max(0.0, self._W - r.width()),
        ))
        new_y = float(np.clip(
            new_centre.y() - 0.5 * r.height(),
            0.0, max(0.0, self._H - r.height()),
        ))
        body.set_pos_silent(new_x, new_y)
        # Recompute uv and resync everything (incl. the centre handle itself).
        self._on_region_body_moved(pair_index, slot)

    # ---------- quad marquee multi-selection ----------

    def _clear_marquee_selection(self) -> None:
        if not self._marquee_selected:
            return
        prev = self._marquee_selected
        self._marquee_selected = set()
        for name in prev:
            item = self._quad_items.get(name)
            if item is not None:
                # Restore default pen (color from quad style).
                self._apply_default_quad_pen(name)
        self.quads_marquee_changed.emit([])

    def _restyle_marquee_selection(self) -> None:
        # Mark selected items with a distinct accent stroke.
        pen = QPen(QColor(120, 220, 255), 3)
        pen.setCosmetic(True)
        for name, item in self._quad_items.items():
            if name in self._marquee_selected:
                item.setPen(pen)
            else:
                self._apply_default_quad_pen(name)

    def marquee_selected_quads(self) -> list[str]:
        return list(self._marquee_selected)

    def _apply_default_quad_pen(self, name: str) -> None:
        """Restore a quad's normal outline (used to revert marquee highlight).
        Delegates to the shared `_quad_pen` so user/auto/locked/fitted styles
        stay consistent."""
        item = self._quad_items.get(name)
        q = self._quads.get(name)
        if item is None or q is None:
            return
        item.setPen(self._quad_pen(q, selected=(name == self._selected)))

    def _on_region_selected(self, pair_index: int) -> None:
        if self._region_selected == pair_index:
            return
        self._region_selected = pair_index
        self._restyle_regions()
        self.region_pair_selected.emit(pair_index)

    # ---------- image ----------

    def set_image(self, qimg: QImage) -> None:
        # Hot path: exposure / yaw / wb sliders re-push a QImage every tick at
        # the SAME resolution. Don't tear down the whole scene every time —
        # just swap the pixmap on the existing bg item. This keeps quads,
        # chart, region rects, handles etc. alive (Qt is fast at swapping a
        # pixmap; it's slow at recreating dozens of QGraphicsItems per frame).
        if (
            self._bg_item is not None
            and qimg.width() == self._W and qimg.height() == self._H
            and self._W > 0
        ):
            self._bg_item.setPixmap(QPixmap.fromImage(qimg))
            # Yaw-dependent overlays (quad paths, vertex handles, chart, HDR
            # region rects) need re-projection only when yaw actually moved.
            yaw_changed = self._yaw_offset_px != self._last_rendered_yaw_px
            if yaw_changed:
                # Quads — re-project their great-circle paths.
                for name, q in self._quads.items():
                    if name in self._quad_items:
                        self._refresh_quad_path(q)
                # Selected quad's draggable vertex handles.
                if self._selected and self._selected in self._quads:
                    q = self._quads[self._selected]
                    for h, d in zip(self._vertex_handles, q.corners_dirs):
                        x, y = self._dir_to_display_pix(d)
                        h.set_pano_pos_silent(x, y)
                # Chart — only the HDR-slot chart depends on yaw.
                if self._chart_dirs is not None and not self._flat_mode:
                    self._clear_chart_shapes()
                    self._build_chart_shapes()
                    for i, h in enumerate(self._chart_handles):
                        h.set_pano_pos_silent(*self._chart_corner_display(i))
                # HDR region rects.
                if self._region_pairs:
                    for (pair_i, slot) in list(self._region_items.keys()):
                        if slot == "hdri":
                            self._reposition_region_for_yaw(pair_i)
                self._last_rendered_yaw_px = self._yaw_offset_px
            return

        had_quads = dict(self._quads)
        sel = self._selected
        saved_transform = self.transform()
        saved_h = self.horizontalScrollBar().value()
        saved_v = self.verticalScrollBar().value()
        prev_w, prev_h = self._W, self._H
        keep_view = prev_w == qimg.width() and prev_h == qimg.height() and prev_w > 0
        self._scene.clear()
        self._quad_items.clear()
        self._vertex_handles.clear()
        self._key_overlay_item = None
        self._sample_rect_item = None
        self._chart_item = None
        self._chart_grid_item = None
        self._chart_cell_items = []
        self._chart_handles = []
        self._placing_dots = []
        self._placing_path = None
        pix = QPixmap.fromImage(qimg)
        self._bg_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._W = pix.width()
        self._H = pix.height()
        self._refresh_key_overlay()
        for name, q in had_quads.items():
            self._add_quad_item(q)
        if sel and sel in self._quads:
            self._set_selected(sel)
        if self._active_chart() is not None:
            self._refresh_chart()
        self._region_items = {}
        if self._region_pairs:
            self._refresh_regions()
        if keep_view:
            self.setTransform(saved_transform)
            self.horizontalScrollBar().setValue(saved_h)
            self.verticalScrollBar().setValue(saved_v)
        elif self._W > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if not self._quads_visible:
            self.set_quads_visible(False)
        if not self._calibration_visible:
            self.set_calibration_visible(False)
        self._last_rendered_yaw_px = self._yaw_offset_px

    def _reposition_region_for_yaw(self, pair_index: int) -> None:
        """Cheap re-sync of an HDR-slot region rect after a yaw change — moves
        the existing items instead of recreating them."""
        entry = self._region_items.get((pair_index, "hdri"))
        if entry is None:
            return
        pair = self._region_pairs[pair_index] if pair_index < len(self._region_pairs) else None
        if not pair:
            return
        uv = pair.get("hdri_uv") or []
        if len(uv) != 4:
            return
        u0, v0, u1, v1 = uv
        body = entry["body"]
        x0 = self._abs_to_display_x(float(u0) * self._W)
        y0 = float(v0) * self._H
        body.set_pos_silent(x0, y0)
        self._sync_region_handles(pair_index, "hdri")

    # ---------- auto-detect key-mask overlay ----------

    def set_key_overlay(self, qimg: QImage) -> None:
        """Show an RGBA mask overlay (display-res) above the panorama."""
        self._key_overlay_img = qimg
        self._refresh_key_overlay()

    def clear_key_overlay(self) -> None:
        self._key_overlay_img = None
        self._refresh_key_overlay()

    def _refresh_key_overlay(self) -> None:
        if self._key_overlay_item is not None:
            self._scene.removeItem(self._key_overlay_item)
            self._key_overlay_item = None
        if self._key_overlay_img is not None and self._bg_item is not None:
            item = self._scene.addPixmap(QPixmap.fromImage(self._key_overlay_img))
            item.setZValue(5)  # above the pano, below quads (z=10)
            self._key_overlay_item = item

    def set_yaw_offset_px(self, px: int) -> None:
        """Caller passes the offset they rolled the HDR by. Viewer stores it
        to keep quad-on-display projection in sync. Does not re-render the bg —
        the app should call `set_image()` with the freshly rolled QImage right
        after.
        """
        if self._W > 0:
            self._yaw_offset_px = int(px) % self._W
        else:
            self._yaw_offset_px = 0

    def yaw_offset_px(self) -> int:
        return self._yaw_offset_px

    def _abs_to_display_x(self, u_abs: float) -> float:
        if self._W <= 0:
            return u_abs
        return (u_abs + self._yaw_offset_px) % self._W

    def _display_to_abs_x(self, u_disp: float) -> float:
        if self._W <= 0:
            return u_disp
        return (u_disp - self._yaw_offset_px) % self._W

    def reset_image(self):
        self._scene.clear()
        self._quads.clear()
        self._quad_items.clear()
        self._vertex_handles.clear()
        self._placing_dots = []
        self._placing_path = None
        self._W = self._H = 0
        self._bg_item = None
        self._selected = None
        self._key_overlay_img = None
        self._key_overlay_item = None
        self._chart_dirs = None
        self._chart_uv = None
        self._chart_item = None
        self._chart_grid_item = None
        self._chart_cell_items = []
        self._chart_handles = []
        self._placing_chart = []
        self._placing_chart_dots = []
        self._region_pairs = []
        self._region_items = {}
        self._region_selected = -1

    def fit_view(self) -> None:
        """Fit the whole panorama in the viewport (reset zoom/pan)."""
        if self._W > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    # ---------- mouse ----------

    def mousePressEvent(self, event):  # noqa: N802
        # Middle-mouse drag = pan the viewport.
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or self._bg_item is None:
            return super().mousePressEvent(event)
        scene_pt = self.mapToScene(event.position().toPoint())

        # Add mode: clicks place vertices.
        if self._mode == self.MODE_ADD:
            # If user clicked outside image, ignore.
            if not self._scene.sceneRect().contains(scene_pt):
                return
            self._place_vertex(scene_pt)
            return

        # Chart mode: clicks place the 4 colour-checker corners.
        if self._mode == self.MODE_CHART:
            if not self._scene.sceneRect().contains(scene_pt):
                return
            self._place_chart_vertex(scene_pt)
            return

        # Sample mode: start a rubber-band rectangle.
        if self._mode == self.MODE_SAMPLE:
            self._sampling = True
            self._sample_start = scene_pt
            self._clear_sample_rect()
            pen = QPen(QColor(255, 230, 90), 0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            item = self._scene.addRect(QRectF(scene_pt, scene_pt), pen,
                                       QBrush(QColor(255, 230, 90, 40)))
            item.setZValue(220)
            self._sample_rect_item = item
            event.accept()
            return

        # Let interactive scene items (drag handles, region bodies) handle
        # their own clicks via Qt's item-event flow. Without this early-out
        # the empty-area marquee branch below would swallow clicks that
        # belong to a region rect or chart corner.
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, (
            _VertexHandle,
            _ChartHandle,
            _RegionHandle,
            _RegionMoveHandle,
            _RegionBodyItem,
        )):
            return super().mousePressEvent(event)

        # Select mode: hit-test for selection
        hit = self._hit_test_quad(scene_pt)
        if hit is not None:
            self._set_selected(hit)
            self.quad_selected.emit(hit)
            self._clear_marquee_selection()
            return
        # Empty area click — start a drag-marquee for multi-select. The
        # release handler intersects the final rect against each quad's
        # display-space bounding box.
        if self._quads:
            self._marquee_active = True
            self._marquee_start = scene_pt
            pen = QPen(QColor(120, 220, 255), 0, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            self._marquee_rect_item = self._scene.addRect(
                QRectF(scene_pt, scene_pt), pen,
                QBrush(QColor(120, 220, 255, 40)),
            )
            self._marquee_rect_item.setZValue(230)
            self._set_selected(None)
            self.quad_selected.emit("")
            self._clear_marquee_selection()
            event.accept()
            return
        self._set_selected(None)
        self.quad_selected.emit("")
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._panning and self._pan_last is not None:
            now = event.position()
            delta = now - self._pan_last
            self._pan_last = now
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - int(delta.x()))
            vbar.setValue(vbar.value() - int(delta.y()))
            event.accept()
            return
        # Drag-marquee: grow the rubber-band rectangle.
        if self._marquee_active and self._marquee_start is not None:
            sp = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._marquee_start, sp).normalized()
            if self._marquee_rect_item is not None:
                self._marquee_rect_item.setRect(rect)
            event.accept()
            return
        # Sample mode: grow the rubber-band rectangle.
        if self._sampling and self._sample_start is not None:
            sp = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._sample_start, sp).normalized()
            if self._sample_rect_item is not None:
                self._sample_rect_item.setRect(rect)
            event.accept()
            return
        # Pixel probe: report the absolute pano pixel under the cursor.
        if self._bg_item is not None:
            sp = self.mapToScene(event.position().toPoint())
            if self._scene.sceneRect().contains(sp):
                u_abs = self._display_to_abs_x(float(sp.x()))
                self.pixel_probed.emit(int(u_abs), int(sp.y()))
            else:
                self.probe_left.emit()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self.probe_left.emit()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._marquee_active
            and self._marquee_start is not None
        ):
            self._marquee_active = False
            sp = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._marquee_start, sp).normalized()
            self._marquee_start = None
            if self._marquee_rect_item is not None:
                self._scene.removeItem(self._marquee_rect_item)
                self._marquee_rect_item = None
            # Pick quads whose display-space bbox intersects the marquee.
            hits = []
            if rect.width() > 1.0 and rect.height() > 1.0:
                for name, item in self._quad_items.items():
                    if not item.isVisible():
                        continue
                    if rect.intersects(item.boundingRect()):
                        hits.append(name)
            self._marquee_selected = set(hits)
            self._restyle_marquee_selection()
            self.quads_marquee_changed.emit(list(self._marquee_selected))
            # Keep keyboard focus on the viewer so Delete fires here without
            # the user having to click back into the panorama first.
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            event.accept()
            return
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._sampling
            and self._sample_start is not None
        ):
            self._sampling = False
            sp = self.mapToScene(event.position().toPoint())
            rect = QRectF(self._sample_start, sp).normalized()
            self._sample_start = None
            # Clamp to image, convert display -> absolute pano px.
            x0 = float(np.clip(rect.left(), 0.0, self._W - 1))
            x1 = float(np.clip(rect.right(), 0.0, self._W - 1))
            y0 = float(np.clip(rect.top(), 0.0, self._H - 1))
            y1 = float(np.clip(rect.bottom(), 0.0, self._H - 1))
            if x1 - x0 >= 1.0 and y1 - y0 >= 1.0:
                ax0 = self._display_to_abs_x(x0)
                ax1 = self._display_to_abs_x(x1)
                self.area_sampled.emit(int(ax0), int(y0), int(ax1), int(y1))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        # Marquee-multi-select delete: when a drag-selected group exists,
        # Delete / Backspace fires `quads_delete_requested(list[str])` so the
        # app can remove them all and refresh the panel.
        if (
            event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and self._marquee_selected
        ):
            names = list(self._marquee_selected)
            self._clear_marquee_selection()
            self.quads_delete_requested.emit(names)
            return
        if event.key() == Qt.Key.Key_Escape and self._marquee_selected:
            self._clear_marquee_selection()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_ADD:
            self.cancel_add_mode()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_SAMPLE:
            self.cancel_sample_mode()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_CHART:
            self.cancel_chart_mode()
            return
        # Region pairs: arrow keys nudge the selected rect; Del removes it.
        if self._region_selected >= 0 and self._region_pairs:
            step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            slot = "ref" if self._flat_mode else "hdri"
            entry = self._region_items.get((self._region_selected, slot))
            if entry is not None and event.key() in (
                Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down
            ):
                dx = -step if event.key() == Qt.Key.Key_Left else step if event.key() == Qt.Key.Key_Right else 0.0
                dy = -step if event.key() == Qt.Key.Key_Up else step if event.key() == Qt.Key.Key_Down else 0.0
                body = entry["body"]
                body.setPos(body.scenePos() + QPointF(dx, dy))
                return
            if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
                self.region_pair_delete_requested.emit(self._region_selected)
                return
            if event.key() == Qt.Key.Key_Escape:
                self._on_region_selected(-1)
                return
        super().keyPressEvent(event)

    # ---------- 4-click placement ----------

    def _place_vertex(self, scene_pt: QPointF) -> None:
        # display pixel -> absolute pano pixel -> spherical dir
        u_disp = float(scene_pt.x())
        v = float(scene_pt.y())
        u_abs = self._display_to_abs_x(u_disp)
        yaw, pitch = pix_to_angles(np.array(u_abs), np.array(v), self._W, self._H)
        d = dir_from_angles(yaw, pitch)
        d = np.asarray(d, dtype=np.float64).reshape(3)
        self._placing_dirs.append(d)
        # marker
        dot = self._scene.addEllipse(
            -HANDLE_R, -HANDLE_R, HANDLE_R * 2, HANDLE_R * 2,
            QPen(QColor(20, 20, 20), 1),
            QBrush(QColor(255, 220, 80)),
        )
        dot.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True)
        dot.setPos(QPointF(u_disp, v))
        dot.setZValue(150)
        self._placing_dots.append(dot)
        # path connecting placed points (great-circle approx in equirect)
        self._refresh_placing_path()

        if len(self._placing_dirs) >= 4:
            corners = np.stack(self._placing_dirs, axis=0)
            corners = self._order_corners_ccw(corners)
            name = self._next_name()
            q = LightQuad(name=name, corners_dirs=corners)
            self._quads[name] = q
            self._add_quad_item(q)
            # exit add mode + select the new quad
            self.cancel_add_mode()
            self._set_selected(name)
            self.quad_committed.emit(q)
            self.quad_selected.emit(name)

    def _refresh_placing_path(self) -> None:
        if self._placing_path is not None:
            self._scene.removeItem(self._placing_path)
            self._placing_path = None
        n = len(self._placing_dirs)
        if n < 2:
            return
        # draw open polyline through placed dirs (don't close until 4 placed)
        path = self._great_circle_path(self._placing_dirs, closed=False)
        pen = QPen(QColor(255, 220, 80), 2, Qt.PenStyle.DashLine)
        self._placing_path = self._scene.addPath(path, pen)
        self._placing_path.setZValue(140)

    @staticmethod
    def _order_corners_ccw(corners: np.ndarray) -> np.ndarray:
        """Sort 4 dirs CCW about their centroid direction.

        We project corners onto the tangent plane at the centroid and sort by
        polar angle. Without this, the path connecting click 1..4 could form a
        bow-tie (self-intersecting) if the user clicks in weird order.
        """
        c = corners.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-12
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(float(c @ world_up)) > 0.95:
            world_up = np.array([1.0, 0.0, 0.0])
        right = np.cross(world_up, c)
        right /= np.linalg.norm(right) + 1e-12
        up = np.cross(c, right)
        up /= np.linalg.norm(up) + 1e-12
        # project each corner onto (right, up) plane
        proj_x = corners @ right
        proj_y = corners @ up
        ang = np.arctan2(proj_y, proj_x)
        order = np.argsort(ang)
        return corners[order]

    def _next_name(self) -> str:
        # find the lowest unused rect_NN name (so deleting + re-adding fills holes)
        used = set(self._quads.keys())
        for i in range(100):
            n = f"rect_{i:02d}"
            if n not in used:
                return n
        return f"rect_{len(self._quads):02d}"

    # ---------- public quad ops ----------

    def add_quad(self, q: LightQuad):
        self._quads[q.name] = q
        self._add_quad_item(q)
        self._set_selected(q.name)

    def update_quad(self, q: LightQuad):
        if q.name not in self._quads:
            return self.add_quad(q)
        self._quads[q.name] = q
        item = self._quad_items.pop(q.name, None)
        if item is not None:
            self._scene.removeItem(item)
        self._add_quad_item(q)
        if self._selected == q.name:
            self._set_selected(q.name)

    def remove_quad(self, name: str):
        self._quads.pop(name, None)
        item = self._quad_items.pop(name, None)
        if item is not None:
            self._scene.removeItem(item)
        if self._selected == name:
            self._selected = None
            self._clear_vertex_handles()

    def get_quad(self, name: str) -> LightQuad | None:
        return self._quads.get(name)

    def set_quad_locked(self, name: str, locked: bool) -> None:
        """Update a quad's lock flag and refresh its outline color."""
        q = self._quads.get(name)
        if q is None:
            return
        q.locked = bool(locked)
        self._refresh_quad_outline(name)

    def set_quad_fitted(self, name: str, fitted: bool, new_corners: np.ndarray | None = None) -> None:
        """Mark a quad as rigid-rect-fitted (or not). When new_corners is
        provided, also replace the quad's `corners_dirs` and redraw the path
        — used by the "Fit to rect light" handler to push the snapped corners
        back to the viewer in a single call."""
        q = self._quads.get(name)
        if q is None:
            return
        q.is_rect_fitted = bool(fitted)
        if new_corners is not None:
            q.corners_dirs = np.asarray(new_corners, dtype=np.float64).reshape(4, 3)
            self._refresh_quad_path(q)
            if name == self._selected:
                # Move the handles to the new corner positions.
                self._refresh_vertex_handles()
        self._refresh_quad_outline(name)

    def _refresh_quad_outline(self, name: str) -> None:
        """Re-apply the pen/brush for a single quad based on its current
        state. No-op if the quad is the currently-selected one (its pen is
        the selection highlight)."""
        q = self._quads.get(name)
        item = self._quad_items.get(name)
        if q is None or item is None:
            return
        if name == self._selected:
            item.setPen(self._quad_pen(q, selected=True))
            item.setBrush(QBrush(QColor(255, 200, 80, 80)))
        else:
            col = self._quad_color(q)
            item.setPen(self._quad_pen(q))
            item.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 50)))

    def rename_quad(self, old_name: str, new_name: str) -> str:
        """Rename a quad. If `new_name` collides, suffix _2/_3/... is added.
        Returns the actual new name used. No-ops if old_name doesn't exist."""
        if old_name not in self._quads:
            return old_name
        if new_name == old_name:
            return old_name
        # Resolve collision
        base = new_name
        candidate = base
        i = 2
        while candidate in self._quads and candidate != old_name:
            candidate = f"{base}_{i}"
            i += 1
        q = self._quads.pop(old_name)
        q.name = candidate
        self._quads[candidate] = q
        item = self._quad_items.pop(old_name, None)
        if item is not None:
            self._quad_items[candidate] = item
        if self._selected == old_name:
            self._selected = candidate
        return candidate

    def quads(self) -> list[LightQuad]:
        return list(self._quads.values())

    def selected(self) -> str | None:
        return self._selected

    def select_by_name(self, name: str | None):
        self._set_selected(name)

    # ---------- vertex drag callback (from _VertexHandle) ----------

    def on_vertex_dragged(self, index: int, scene_pos: QPointF) -> None:
        if not self._selected:
            return
        q = self._quads.get(self._selected)
        if q is None:
            return
        # clamp inside image bounds and convert display -> absolute
        u_disp = float(np.clip(scene_pos.x(), 0.0, self._W - 1))
        v = float(np.clip(scene_pos.y(), 0.0, self._H - 1))
        u_abs = self._display_to_abs_x(u_disp)
        yaw, pitch = pix_to_angles(np.array(u_abs), np.array(v), self._W, self._H)
        d = dir_from_angles(yaw, pitch)
        d = np.asarray(d, dtype=np.float64).reshape(3)
        q.corners_dirs[index] = d
        # Editing an auto-proposed quad locks it, so the refinement isn't
        # discarded the next time "Propose quads" runs.
        if q.source == "auto" and not q.locked:
            q.locked = True
            self.quad_lock_changed.emit(q.name, True)
        # Any edit invalidates the rigid-rect fit — the depth-based snap
        # was applied to the *previous* corner positions and no longer
        # represents what the user has now. The "Fit to rect light" button
        # needs another press to bring it back in sync.
        if q.is_rect_fitted:
            q.is_rect_fitted = False
            self.quad_fit_changed.emit(q.name, False)
        # re-render the path (handles already moved); don't recreate handles —
        # we want the dragged handle to stay under the mouse.
        self._refresh_quad_path(q)
        self.quad_modified.emit(q.name)

    # ---------- internals ----------

    @staticmethod
    def _quad_color(q: LightQuad) -> QColor:
        """Outline color by state. is_rect_fitted overrides provenance —
        depth-snapped quads get a bright magenta so the rigid-rect status
        reads at a glance regardless of source/locked. Otherwise:
        cyan = user, orange = unconfirmed auto proposal, green = locked."""
        if q.is_rect_fitted:
            return QColor(230, 90, 220)
        if q.source == "auto" and not q.locked:
            return QColor(255, 150, 40)
        if q.locked:
            return QColor(80, 230, 120)
        return QColor(60, 220, 255)

    @staticmethod
    def _quad_pen(q: LightQuad, selected: bool = False) -> QPen:
        """Pen for a quad. Fitted quads get a slightly thicker outline so
        the rigid-rect status is visible without leaning on colour alone —
        useful for colour-blind users and against busy panorama backgrounds."""
        if selected:
            pen = QPen(QColor(255, 200, 80), 2.25)
        else:
            pen = QPen(PanoramaViewer._quad_color(q), 2.25 if q.is_rect_fitted else 1.5)
        return pen

    def _add_quad_item(self, q: LightQuad):
        if self._W <= 0 or self._H <= 0:
            return
        path = self._great_circle_path(list(q.corners_dirs), closed=True)
        col = self._quad_color(q)
        pen = self._quad_pen(q)
        brush = QBrush(QColor(col.red(), col.green(), col.blue(), 50))
        item = self._scene.addPath(path, pen, brush)
        item.setZValue(10)
        item.setVisible(self._quads_visible)
        self._quad_items[q.name] = item

    def _refresh_quad_path(self, q: LightQuad):
        item = self._quad_items.get(q.name)
        if item is None:
            return self._add_quad_item(q)
        path = self._great_circle_path(list(q.corners_dirs), closed=True)
        item.setPath(path)

    def _great_circle_path(self, dirs: list[np.ndarray], closed: bool) -> QPainterPath:
        """Sample N points along each edge (slerp-ish), project to pano px, and
        emit a path that breaks on seam-wraps."""
        n_samples = 24
        pts: list[tuple[float, float]] = []
        n = len(dirs)
        edges = n if closed else n - 1
        for k in range(edges):
            a = dirs[k]
            b = dirs[(k + 1) % n]
            cosang = float(np.clip(a @ b, -1.0, 1.0))
            omega = float(np.arccos(cosang))
            if omega < 1e-6:
                pts.append(self._dir_to_display_pix(a))
                continue
            sin_om = float(np.sin(omega))
            for t in np.linspace(0.0, 1.0, n_samples, endpoint=False):
                w_a = float(np.sin((1.0 - t) * omega)) / sin_om
                w_b = float(np.sin(t * omega)) / sin_om
                d = w_a * a + w_b * b
                pts.append(self._dir_to_display_pix(d))
        if closed and pts:
            pts.append(self._dir_to_display_pix(dirs[0]))

        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(*pts[0])
        last_u = pts[0][0]
        for (u, v) in pts[1:]:
            if abs(u - last_u) > self._W / 2:
                path.moveTo(u, v)
            else:
                path.lineTo(u, v)
            last_u = u
        return path

    def _dir_to_pix(self, d: np.ndarray) -> tuple[float, float]:
        """Absolute pano pixel coords (no yaw offset applied). Used by callers
        that need geometric truth; for *drawing*, use `_dir_to_display_pix`."""
        d = d / (np.linalg.norm(d) + 1e-12)
        yaw, pitch = angles_from_dir(d)
        u, v = angles_to_pix(yaw, pitch, self._W, self._H)
        return float(np.asarray(u).item()), float(np.asarray(v).item())

    def _dir_to_display_pix(self, d: np.ndarray) -> tuple[float, float]:
        u_abs, v = self._dir_to_pix(d)
        return float(self._abs_to_display_x(u_abs)), v

    def _hit_test_quad(self, scene_pt: QPointF) -> str | None:
        for name in reversed(list(self._quad_items.keys())):
            item = self._quad_items[name]
            if item.shape().contains(scene_pt):
                return name
        return None

    def _set_selected(self, name: str | None):
        self._selected = name
        for n, item in self._quad_items.items():
            q = self._quads.get(n)
            if n == name:
                item.setPen(self._quad_pen(q, selected=True) if q is not None
                            else QPen(QColor(255, 200, 80), 2.25))
                item.setBrush(QBrush(QColor(255, 200, 80, 80)))
            else:
                col = self._quad_color(q) if q is not None else QColor(60, 220, 255)
                item.setPen(self._quad_pen(q) if q is not None else QPen(col, 1.5))
                item.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 50)))
        self._refresh_vertex_handles()

    def _clear_vertex_handles(self):
        for h in self._vertex_handles:
            self._scene.removeItem(h)
        self._vertex_handles = []

    def _refresh_vertex_handles(self):
        self._clear_vertex_handles()
        if self._selected is None or self._mode == self.MODE_ADD:
            return
        q = self._quads.get(self._selected)
        if q is None or self._W <= 0:
            return
        for i in range(4):
            d = q.corners_dirs[i]
            px, py = self._dir_to_display_pix(d)
            h = _VertexHandle(i, self)
            self._scene.addItem(h)
            h.set_pano_pos_silent(px, py)
            h.setVisible(self._quads_visible)
            self._vertex_handles.append(h)
