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


# ---------- viewer ----------

class PanoramaViewer(QGraphicsView):
    quad_committed = Signal(LightQuad)  # new quad finished (4-click placement)
    quad_modified = Signal(str)         # name of quad whose vertices were edited
    quad_selected = Signal(str)         # name; empty string to deselect
    quad_lock_changed = Signal(str, bool)  # name, locked — auto-lock on edit
    add_mode_changed = Signal(bool)
    pixel_probed = Signal(int, int)     # absolute pano pixel (display res) under cursor
    probe_left = Signal()               # cursor left the panorama
    area_sampled = Signal(int, int, int, int)  # absolute pano px x0,y0,x1,y1
    sample_mode_changed = Signal(bool)
    chart_committed = Signal()          # colour-chart 4 corners finished
    chart_modified = Signal()           # colour-chart corner dragged
    chart_mode_changed = Signal(bool)

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
        # Vertex handles for the currently selected quad (rebuilt on selection change).
        self._vertex_handles: list[_VertexHandle] = []
        # Auto-detect key-mask preview overlay (an RGBA QImage, display res).
        self._key_overlay_img: QImage | None = None
        self._key_overlay_item = None
        # Rectangle-sample (exposure / WB / probe) drag state.
        self._sampling = False
        self._sample_start: QPointF | None = None
        self._sample_rect_item: QGraphicsRectItem | None = None
        # Quads hidden in exposure mode.
        self._quads_visible = True
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

    # ---------- image ----------

    def set_image(self, qimg: QImage) -> None:
        had_quads = dict(self._quads)
        sel = self._selected
        # Preserve zoom/pan across a re-render at the same resolution (the
        # exposure / yaw sliders re-push an image every tick — the view must
        # not snap back to fit). Only fit on a genuine resolution change.
        prev_w, prev_h = self._W, self._H
        keep_view = prev_w == qimg.width() and prev_h == qimg.height() and prev_w > 0
        saved_transform = self.transform()
        saved_h = self.horizontalScrollBar().value()
        saved_v = self.verticalScrollBar().value()
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
        if keep_view:
            self.setTransform(saved_transform)
            self.horizontalScrollBar().setValue(saved_h)
            self.verticalScrollBar().setValue(saved_v)
        elif self._W > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if not self._quads_visible:
            self.set_quads_visible(False)

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

        # Let vertex handles handle their own clicks (handled by Qt's item system
        # because the handles have ItemIsMovable + are above the bg).
        item = self.itemAt(event.position().toPoint())
        if isinstance(item, _VertexHandle):
            return super().mousePressEvent(event)

        # Select mode: hit-test for selection
        hit = self._hit_test_quad(scene_pt)
        if hit is not None:
            self._set_selected(hit)
            self.quad_selected.emit(hit)
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
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_ADD:
            self.cancel_add_mode()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_SAMPLE:
            self.cancel_sample_mode()
            return
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_CHART:
            self.cancel_chart_mode()
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
        item = self._quad_items.get(name)
        if item is not None and name != self._selected:
            col = self._quad_color(q)
            item.setPen(QPen(col, 2))
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
        # re-render the path (handles already moved); don't recreate handles —
        # we want the dragged handle to stay under the mouse.
        self._refresh_quad_path(q)
        self.quad_modified.emit(q.name)

    # ---------- internals ----------

    @staticmethod
    def _quad_color(q: LightQuad) -> QColor:
        """Outline color by provenance: cyan = user, orange = unconfirmed
        auto proposal, green = locked (kept across re-propose)."""
        if q.source == "auto" and not q.locked:
            return QColor(255, 150, 40)
        if q.locked:
            return QColor(80, 230, 120)
        return QColor(60, 220, 255)

    def _add_quad_item(self, q: LightQuad):
        if self._W <= 0 or self._H <= 0:
            return
        path = self._great_circle_path(list(q.corners_dirs), closed=True)
        col = self._quad_color(q)
        pen = QPen(col, 2)
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
            if n == name:
                item.setPen(QPen(QColor(255, 200, 80), 3))
                item.setBrush(QBrush(QColor(255, 200, 80, 80)))
            else:
                q = self._quads.get(n)
                col = self._quad_color(q) if q is not None else QColor(60, 220, 255)
                item.setPen(QPen(col, 2))
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
