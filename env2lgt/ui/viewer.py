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


# ---------- viewer ----------

class PanoramaViewer(QGraphicsView):
    quad_committed = Signal(LightQuad)  # new quad finished (4-click placement)
    quad_modified = Signal(str)         # name of quad whose vertices were edited
    quad_selected = Signal(str)         # name; empty string to deselect
    add_mode_changed = Signal(bool)

    MODE_SELECT = "select"
    MODE_ADD = "add"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setBackgroundBrush(QColor(20, 20, 20))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
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

    # ---------- image ----------

    def set_image(self, qimg: QImage) -> None:
        had_quads = dict(self._quads)
        sel = self._selected
        self._scene.clear()
        self._quad_items.clear()
        self._vertex_handles.clear()
        self._placing_dots = []
        self._placing_path = None
        pix = QPixmap.fromImage(qimg)
        self._bg_item = self._scene.addPixmap(pix)
        self._scene.setSceneRect(QRectF(pix.rect()))
        self._W = pix.width()
        self._H = pix.height()
        for name, q in had_quads.items():
            self._add_quad_item(q)
        if sel and sel in self._quads:
            self._set_selected(sel)
        if self._W > 0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

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
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._mode == self.MODE_ADD:
            self.cancel_add_mode()
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
        # re-render the path (handles already moved); don't recreate handles —
        # we want the dragged handle to stay under the mouse.
        self._refresh_quad_path(q)
        self.quad_modified.emit(q.name)

    # ---------- internals ----------

    def _add_quad_item(self, q: LightQuad):
        if self._W <= 0 or self._H <= 0:
            return
        path = self._great_circle_path(list(q.corners_dirs), closed=True)
        pen = QPen(QColor(60, 220, 255), 2)
        brush = QBrush(QColor(60, 220, 255, 50))
        item = self._scene.addPath(path, pen, brush)
        item.setZValue(10)
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
                item.setPen(QPen(QColor(60, 220, 255), 2))
                item.setBrush(QBrush(QColor(60, 220, 255, 50)))
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
            self._vertex_handles.append(h)
