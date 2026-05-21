"""env2lgt — PySide6 application entry point.

Single-view UX: equirect panorama. Add a quad by clicking 4 corners (cursor in
"Add" mode). Drag the yellow vertex handles to refine. Bake.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSettings, QThread, Qt, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSlider,
    QStatusBar,
    QToolBar,
    QToolButton,
    QWidget,
)

import cv2

from env2lgt import color
from env2lgt import colorchecker as ccheck
from env2lgt.bake import BakeOptions, QuadSpec, bake
from env2lgt.depth import AVAILABLE_BACKENDS, get_backend, shutdown_all
from env2lgt.exposure import (
    convolve_dome_meter,
    grey_world_scale,
    scale_to_temp_tint,
    spot_meter_offset_ev,
    temp_tint_to_scale,
)
from env2lgt.io import depth_to_display_qimage, load_latlong, to_display_qimage
from env2lgt.io.tonemap import aces_filmic
from env2lgt.project import (
    Project,
    default_project_path,
    load_project,
    project_from_app_state,
    save_project,
)
from env2lgt.ui.exposure_panel import ExposurePanel
from env2lgt.ui.light_panel import LightPanel
from env2lgt.ui.theme import apply_theme
from env2lgt.ui.viewer import LightQuad, PanoramaViewer


# Cap the in-viewer pano width. The full-res HDR is still kept around for
# bake (which does its own load straight from the EXR file anyway), but the
# tonemap + exposure scrub work on this downsampled copy so the sliders feel
# responsive. 2048 is plenty for click precision on a 4K source.
DISPLAY_MAX_WIDTH = 2048


class BakeWorker(QObject):
    progress = Signal(str, float)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(self, exr_path: str, out_dir: str, quads: list[QuadSpec], options: BakeOptions):
        super().__init__()
        self._exr_path = exr_path
        self._out_dir = out_dir
        self._quads = quads
        self._options = options

    def run(self):
        try:
            summary = bake(
                self._exr_path,
                self._out_dir,
                self._quads,
                self._options,
                progress_cb=lambda stage, frac: self.progress.emit(stage, frac),
            )
            self.finished.emit(summary)
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class DepthWorker(QObject):
    """Background depth run, used by the 'Show depth' toolbar toggle."""
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, exr_path: str, cache_dir: str, backend: str = "da2"):
        super().__init__()
        self._exr_path = exr_path
        self._cache_dir = cache_dir
        self._backend = backend

    def run(self):
        try:
            d = get_backend(self._backend).estimate_depth(
                self._exr_path, cache_dir=self._cache_dir
            )
            self.finished.emit(d)
        except Exception as e:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("env2lgt — HDRI → USD light rig")
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self._hdr: np.ndarray | None = None              # full-res HDR (used for bake metadata only)
        self._hdr_display: np.ndarray | None = None      # downsampled HDR for fast tonemap
        self._distance: np.ndarray | None = None         # full-res depth (for bake)
        self._distance_display: np.ndarray | None = None # downsampled depth for fast preview
        # uint8 LDR cache at the current (view_mode, exposure). Reused across
        # yaw-slider ticks (np.roll on uint8 is ~20x cheaper than rolling
        # float32 + retonemapping each time).
        self._display_cache: np.ndarray | None = None
        self._cache_key: tuple | None = None
        self._exr_path: Path | None = None
        self._exposure: float = 0.0          # display-only viewport exposure
        # Baseline HDRI adjustments — baked into the export. WB follows the
        # Lightroom model: kelvin + tint are the source of truth, _wb_scale is
        # derived.
        self._exposure_offset: float = 0.0
        self._wb_kelvin: float = 6500.0
        self._wb_tint: float = 0.0
        self._wb_scale: np.ndarray = np.ones(3, dtype=np.float32)
        self._exposure_mode: bool = False
        # Pending rectangle-sample purpose: "exposure" | "wb" | "probe" | None.
        self._pending_sample: str | None = None

        # ---- colour management ----
        # Everything internal is the working space (ACEScg). The source EXR is
        # input-transformed on load; the bake output-transforms on write. Two
        # backends provide that: OCIO (config-driven) and a built-in fallback
        # (sRGB display + ACEScg/sRGB-Linear input transforms). Default to
        # OCIO when a valid config is present; otherwise the built-in path is
        # always available.
        self._ocio_available: bool = color.ocio_available()
        self._cm_backend: str = "ocio" if self._ocio_available else "builtin"
        try:
            color.set_backend(self._cm_backend)
        except Exception:  # noqa: BLE001
            color.set_backend("builtin")
            self._cm_backend = "builtin"
        _wcs = color.working_colorspace()
        self._input_cs: str = _wcs
        self._output_cs: str = _wcs
        self._ocio_display: str = ""
        self._ocio_view: str = ""
        self._display_cpu = None
        # Source-space copy of the display buffer, so the input colorspace can
        # be changed without reloading the EXR.
        self._hdr_display_src: np.ndarray | None = None

        # ---- colour-checker chart correction ----
        # Solved 3x3 matrix (row-vector convention) or None. Baked into export.
        self._cc_matrix: np.ndarray | None = None
        self._last_rmse: float | None = None
        self._cc_target_swatches: np.ndarray | None = None  # JSON target (24,3)
        self._cc_target_name: str = "Built-in CC24"
        self._cc_target_mode: str = "builtin"   # builtin | json | reference
        self._cc_fit_mode: str = "matrix"
        # Flat reference image for chart matching. `_ref_src` is the
        # source-encoded display-res copy (so the colorspace can be re-picked
        # without reloading); `_ref_display` is the working-space version.
        self._ref_display: np.ndarray | None = None
        self._ref_src: np.ndarray | None = None
        self._ref_cs: str = ""
        self._ref_path: Path | None = None
        # Scene scale = metres per scene unit. DAP depth is metric, so the
        # default backend ships a 1.0 default; DA² is scale-invariant and
        # needs ~100. Adjusted via the Export panel's Scene scale spinbox.
        self._scene_scale: float = 1.0
        self._depth_backend: str = "dap"
        self._yaw_offset_deg: float = 0.0
        self._view_gamma: float = 1.0   # display-only viewport gamma
        self._last_usd: Path | None = None
        self._last_mesh: Path | None = None
        self._view_mode: str = "hdr"    # "hdr" or "depth"
        self._depth_thread: QThread | None = None
        self._depth_worker: DepthWorker | None = None
        self._worker: BakeWorker | None = None
        self._thread: QThread | None = None
        self._key_preview_on: bool = False
        # Set True while a "Fit to rect light" press is waiting on a
        # not-yet-computed depth field. Cleared in _on_depth_ready /
        # _on_depth_failed; checked there so the fit fires automatically as
        # soon as depth lands instead of forcing the user to press again.
        self._pending_fit_to_rect: bool = False

        self.viewer = PanoramaViewer(self)
        self.setCentralWidget(self.viewer)
        self.viewer.quad_committed.connect(self._on_quad_committed)
        self.viewer.quad_selected.connect(self._on_quad_selected)
        self.viewer.quad_lock_changed.connect(self._on_quad_lock_changed)
        self.viewer.quad_fit_changed.connect(self._on_quad_fit_changed)
        self.viewer.add_mode_changed.connect(self._on_add_mode_changed)
        self.viewer.pixel_probed.connect(self._on_pixel_probed)
        self.viewer.probe_left.connect(self._clear_probe)
        self.viewer.area_sampled.connect(self._on_area_sampled)
        self.viewer.sample_mode_changed.connect(self._on_sample_mode_changed)
        self.viewer.chart_committed.connect(self._on_chart_committed)
        self.viewer.chart_mode_changed.connect(self._on_chart_mode_changed)

        # Lights + Exposure panels live in ONE dock, swapped via a stacked
        # widget. A single fixed-geometry dock means toggling exposure mode
        # never relayouts the central area — the viewport (and its zoom/pan)
        # stays put. The tall exposure panel is wrapped in a scroll area so it
        # can never force the window to grow.
        from PySide6.QtWidgets import QScrollArea, QStackedWidget

        self.panel = LightPanel(self)
        self.exposure_panel = ExposurePanel(self)
        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setWidget(self.panel)
        panel_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        exp_scroll = QScrollArea()
        exp_scroll.setWidgetResizable(True)
        exp_scroll.setWidget(self.exposure_panel)
        exp_scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self._panel_stack = QStackedWidget()
        self._panel_stack.addWidget(panel_scroll)      # index 0 — Lights
        self._panel_stack.addWidget(exp_scroll)        # index 1 — Exposure

        dock = QDockWidget("Lights", self)
        dock.setWidget(self._panel_stack)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._dock = dock
        # Give the tool column a comfortable default width — the side panels
        # are cramped at Qt's size-hint default.
        self._panel_stack.setMinimumWidth(340)
        self.resizeDocks([dock], [400], Qt.Orientation.Horizontal)
        self.exposure_panel.exposure_offset_changed.connect(self._on_exposure_offset)
        self.exposure_panel.wb_changed.connect(self._on_wb_changed)
        self.exposure_panel.sample_exposure_requested.connect(
            lambda: self._begin_sample("exposure")
        )
        self.exposure_panel.sample_wb_requested.connect(
            lambda: self._begin_sample("wb")
        )
        self.exposure_panel.auto_meter_requested.connect(self._on_auto_meter)
        self.exposure_panel.input_cs_changed.connect(self._on_input_cs_changed)
        self.exposure_panel.output_cs_changed.connect(self._on_output_cs_changed)
        self.exposure_panel.pick_chart_requested.connect(self._on_pick_chart)
        self.exposure_panel.clear_chart_requested.connect(self._on_clear_chart)
        self.exposure_panel.use_builtin_target.connect(self._on_use_builtin_target)
        self.exposure_panel.load_json_target_requested.connect(
            self._on_load_json_target
        )
        self.exposure_panel.use_reference_target.connect(self._on_use_reference_target)
        self.exposure_panel.load_reference_image_requested.connect(
            self._on_load_reference_image
        )
        self.exposure_panel.reference_cs_changed.connect(
            self._on_reference_cs_changed
        )
        self.exposure_panel.reference_view_toggled.connect(
            self._on_reference_view_toggled
        )
        self.exposure_panel.fit_mode_changed.connect(self._on_fit_mode_changed)
        self.exposure_panel.solve_chart_requested.connect(self._on_solve_chart)
        self.exposure_panel.save_correction_requested.connect(
            self._on_save_correction
        )
        self.exposure_panel.load_correction_requested.connect(
            self._on_load_correction
        )
        self.exposure_panel.populate_colorspaces(
            color.colorspace_names(), self._input_cs, self._output_cs
        )
        self.panel.delete_quad.connect(self._on_delete_quad)
        self.panel.select_quad.connect(self._on_panel_selected)
        self.panel.rename_quad.connect(self._on_rename_quad)
        self.panel.window_toggled.connect(self._on_window_toggled)
        self.panel.lock_toggled.connect(self._on_lock_toggled)
        self.panel.add_quad_requested.connect(self._on_add_quad_requested)
        self.panel.propose_quads_requested.connect(self._on_propose_quads)
        self.panel.fit_to_rect_requested.connect(self._on_fit_to_rect)
        self.panel.key_preview_changed.connect(self._on_key_preview)
        self.panel.bake_requested.connect(self._on_bake)
        self.panel.preview_requested.connect(self._on_preview)
        self.panel.opt_scene_scale.valueChanged.connect(self._on_scene_scale)

        self._build_menu()
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setFixedWidth(280)
        self._build_probe_widget()
        self.statusBar().addPermanentWidget(self._progress)
        self._set_status("Open or drag an EXR latlong panorama to begin.")

    def _build_probe_widget(self):
        """Nuke-style pixel probe in the status bar: a colour swatch plus the
        scene-linear RGB / HSV values under the cursor."""
        probe = QWidget()
        row = QHBoxLayout(probe)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(6)
        # Eyedropper — area RGB probe, sits right beside the pixel readout.
        self._eyedrop_btn = QToolButton()
        self._eyedrop_btn.setIcon(self._make_eyedropper_icon())
        self._eyedrop_btn.setCheckable(True)
        self._eyedrop_btn.setAutoRaise(True)
        self._eyedrop_btn.setToolTip(
            "Eyedropper — drag a rectangle to read the average scene-linear "
            "RGB of that area (Nuke-style area probe)."
        )
        self._eyedrop_btn.clicked.connect(self._on_eyedropper_clicked)
        row.addWidget(self._eyedrop_btn)
        self._probe_swatch = QLabel()
        self._probe_swatch.setFixedSize(14, 14)
        self._probe_swatch.setStyleSheet("background:#000; border:1px solid #555;")
        row.addWidget(self._probe_swatch)
        self._probe_label = QLabel("")
        self._probe_label.setMinimumWidth(420)
        self._probe_label.setTextFormat(Qt.TextFormat.RichText)
        font = self._probe_label.font()
        font.setStyleHint(font.StyleHint.Monospace)
        font.setFamily("Consolas")
        self._probe_label.setFont(font)
        row.addWidget(self._probe_label)
        # Persistent area-probe (eyedropper) readout — survives cursor moves.
        self._area_label = QLabel("")
        self._area_label.setTextFormat(Qt.TextFormat.RichText)
        self._area_label.setFont(font)
        row.addWidget(self._area_label)
        # Probe sits flush bottom-left. The status message goes in its own
        # normal QLabel right of it — NOT via QStatusBar.showMessage(), which
        # hides every non-permanent widget (the probe included) while a
        # message is up.
        self.statusBar().addWidget(probe)
        self._status_msg = QLabel("")
        self.statusBar().addWidget(self._status_msg, 1)

    def _set_status(self, msg: str) -> None:
        """Set the status-bar message. Replaces QStatusBar.showMessage so the
        bottom-left probe widget is never hidden."""
        self._status_msg.setText(msg)

    @staticmethod
    def _make_eyedropper_icon() -> QIcon:
        """Draw a small eyedropper/pipette glyph for the area-probe button."""
        pm = QPixmap(32, 32)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Stem.
        pen = QPen(QColor(225, 225, 225), 4)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(8, 24), QPointF(21, 11))
        # Bulb.
        p.setPen(QPen(QColor(225, 225, 225), 3))
        p.setBrush(QColor(150, 195, 255))
        p.drawEllipse(QPointF(23.5, 8.5), 5.5, 5.5)
        # Tip.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(225, 225, 225))
        p.drawEllipse(QPointF(6.5, 25.5), 2.6, 2.6)
        p.end()
        return QIcon(pm)

    def _on_key_preview(self, on: bool):
        self._key_preview_on = bool(on)
        self._refresh_key_preview()

    def _refresh_key_preview(self):
        """Recompute and show the luma-key mask overlay, or clear it."""
        if not self._key_preview_on or self._hdr_display is None:
            self.viewer.clear_key_overlay()
            return
        from env2lgt.lights.detect import DetectParams, bright_mask

        pp = self.panel.detect_params()
        dp = DetectParams(threshold=pp["threshold"], blur_deg=pp["blur_deg"])
        # Key off the baseline-adjusted HDRI — the same buffer the bake
        # extracts from — so the mask matches the displayed panorama.
        mask = bright_mask(self._adjusted_working(self._hdr_display), dp)
        H, W = mask.shape
        offset = int(round((self._yaw_offset_deg / 360.0) * W)) % W
        if offset:
            mask = np.roll(mask, offset, axis=1)
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[mask] = (255, 70, 210, 120)  # translucent magenta
        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.data, W, H, W * 4, QImage.Format.Format_RGBA8888).copy()
        self.viewer.set_key_overlay(qimg)

    def _clear_probe(self):
        self._probe_label.setText("")
        self._probe_swatch.setStyleSheet("background:#000; border:1px solid #555;")

    def _on_pixel_probed(self, x: int, y: int):
        """Show the pixel value under the cursor (scene-linear, Nuke-style)."""
        # In depth view, probe the distance map; otherwise the HDR panorama.
        if self._view_mode == "depth" and self._distance_display is not None:
            buf = self._distance_display
            if not (0 <= y < buf.shape[0] and 0 <= x < buf.shape[1]):
                return self._clear_probe()
            self._probe_label.setText(
                f"<span style='color:#aaa'>depth</span> {float(buf[y, x]):.4f}"
                f"  <span style='color:#888'>[{x},{y}]</span>"
            )
            self._probe_swatch.setStyleSheet("background:#000; border:1px solid #555;")
            return
        hdr = self._hdr_display
        if hdr is None or not (0 <= y < hdr.shape[0] and 0 <= x < hdr.shape[1]):
            return self._clear_probe()
        import colorsys

        r, g, b = (float(c) for c in hdr[y, x, :3])
        h, s, v = colorsys.rgb_to_hsv(max(r, 0.0), max(g, 0.0), max(b, 0.0))
        self._probe_label.setText(
            f"<span style='color:#ff6b6b'>{r:9.4f}</span> "
            f"<span style='color:#6bd66b'>{g:9.4f}</span> "
            f"<span style='color:#6b9bff'>{b:9.4f}</span>  "
            f"<span style='color:#aaa'>H</span>{h * 360:5.0f} "
            f"<span style='color:#aaa'>S</span>{s:5.3f} "
            f"<span style='color:#aaa'>V</span>{v:8.4f}  "
            f"<span style='color:#888'>[{x},{y}]</span>"
        )
        # Swatch: clamp + gamma 2.2 for an sRGB-ish preview of the colour.
        def _enc(c: float) -> int:
            return int(round(min(1.0, max(0.0, c)) ** (1.0 / 2.2) * 255))
        self._probe_swatch.setStyleSheet(
            f"background:rgb({_enc(r)},{_enc(g)},{_enc(b)}); border:1px solid #555;"
        )

    # ---------- menus / toolbars ----------

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        act_open = QAction("&Open EXR…", self, shortcut="Ctrl+O")
        act_open.triggered.connect(self._open_exr_dialog)
        m_file.addAction(act_open)
        self._recent_menu = m_file.addMenu("Open &Recent")
        self._recent_menu.setToolTipsVisible(True)
        self._rebuild_recent_menu()
        m_file.addSeparator()
        act_save_proj = QAction("&Save Project…", self, shortcut="Ctrl+S")
        act_save_proj.triggered.connect(self._save_project_dialog)
        m_file.addAction(act_save_proj)
        act_open_proj = QAction("Open &Project…", self, shortcut="Ctrl+P")
        act_open_proj.triggered.connect(self._open_project_dialog)
        m_file.addAction(act_open_proj)
        m_file.addSeparator()
        act_quit = QAction("&Quit", self, shortcut="Ctrl+Q")
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        m_tools = self.menuBar().addMenu("&Tools")
        act_add = QAction("Add quad", self, shortcut="A")
        act_add.triggered.connect(self._on_add_quad_requested)
        m_tools.addAction(act_add)
        act_delete = QAction("Delete selected quad", self, shortcut="Delete")
        act_delete.triggered.connect(self._delete_selected)
        m_tools.addAction(act_delete)
        m_tools.addSeparator()
        act_exposure = QAction("Exposure mode", self)
        act_exposure.setToolTip(
            "Baseline exposure + white balance + colour-checker matching."
        )
        act_exposure.triggered.connect(
            lambda: self._exposure_btn.setChecked(not self._exposure_btn.isChecked())
        )
        m_tools.addAction(act_exposure)
        m_tools.addSeparator()
        m_usdview = m_tools.addMenu("Open last bake in usdview")
        act_uv_light = QAction("Light rig", self)
        act_uv_light.triggered.connect(lambda: self._launch_usdview("light"))
        m_usdview.addAction(act_uv_light)
        act_uv_mesh = QAction("Depth mesh", self)
        act_uv_mesh.triggered.connect(lambda: self._launch_usdview("mesh"))
        m_usdview.addAction(act_uv_mesh)
        act_uv_both = QAction("Light rig + depth mesh (layered)", self)
        act_uv_both.triggered.connect(lambda: self._launch_usdview("both"))
        m_usdview.addAction(act_uv_both)

        # ---- View / Colour management ----
        from PySide6.QtGui import QActionGroup

        m_view = self.menuBar().addMenu("&View")
        m_cm = m_view.addMenu("Colour management")
        self._cm_group = QActionGroup(self)
        self._cm_group.setExclusive(True)
        self._cm_action_ocio = QAction("OCIO ($OCIO config)", self, checkable=True)
        self._cm_action_ocio.setEnabled(self._ocio_available)
        if not self._ocio_available:
            self._cm_action_ocio.setToolTip(
                "PyOpenColorIO is not installed, or $OCIO is unset / invalid."
            )
        self._cm_action_ocio.triggered.connect(lambda: self._set_cm_backend("ocio"))
        self._cm_group.addAction(self._cm_action_ocio)
        m_cm.addAction(self._cm_action_ocio)
        self._cm_action_builtin = QAction(
            "Built-in (sRGB display, ACEScg working)", self, checkable=True
        )
        self._cm_action_builtin.triggered.connect(
            lambda: self._set_cm_backend("builtin")
        )
        self._cm_group.addAction(self._cm_action_builtin)
        m_cm.addAction(self._cm_action_builtin)
        (self._cm_action_ocio if self._cm_backend == "ocio"
         else self._cm_action_builtin).setChecked(True)

    def _set_cm_backend(self, name: str) -> None:
        """Switch the colour-management backend at runtime. Rebuilds the
        colorspace + display/view combos against the new backend, resets the
        active names to the new working space's defaults, and re-runs the
        input transform on the current HDRI + reference image."""
        if name == self._cm_backend:
            return
        try:
            color.set_backend(name)
        except Exception as e:  # noqa: BLE001
            QMessageBox.warning(self, "Colour management", str(e))
            # Snap the checkmark back to whatever's actually active.
            (self._cm_action_ocio if self._cm_backend == "ocio"
             else self._cm_action_builtin).setChecked(True)
            return
        self._cm_backend = name
        wcs = color.working_colorspace()
        self._input_cs = wcs
        self._output_cs = wcs
        self._ref_cs = wcs
        self.exposure_panel.populate_colorspaces(
            color.colorspace_names(), self._input_cs, self._output_cs
        )
        self.exposure_panel.set_reference_cs(wcs)
        # Repopulate display/view combos against the new backend.
        self._display_combo.blockSignals(True)
        self._display_combo.clear()
        for d in color.displays():
            self._display_combo.addItem(d)
        self._display_combo.setCurrentText(color.default_display())
        self._ocio_display = self._display_combo.currentText()
        self._display_combo.blockSignals(False)
        self._refill_view_combo()
        # Re-run the input transform under the new backend.
        if self._hdr_display_src is not None:
            self._hdr_display = self._to_working_display(self._hdr_display_src)
        if self._ref_src is not None:
            self._ref_display = self._reference_to_working(self._ref_src)
        self._invalidate_display_cache()
        self._refresh_view()
        label = "OCIO" if name == "ocio" else "Built-in (sRGB / ACEScg)"
        self._set_status(f"Colour management: {label}")

    def _build_toolbar(self):
        from PySide6.QtWidgets import QPushButton

        tb = QToolBar("View", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        # Viewport display + view — leftmost (Nuke/Maya-style dropdowns).
        # Always shown: the builtin backend exposes "sRGB" / "ACES Filmic"
        # even when OCIO isn't available.
        tb.addWidget(QLabel(" Display "))
        self._display_combo = QComboBox()
        for d in color.displays():
            self._display_combo.addItem(d)
        self._display_combo.setCurrentText(color.default_display())
        self._display_combo.currentTextChanged.connect(self._on_display_changed)
        tb.addWidget(self._display_combo)
        self._view_combo = QComboBox()
        self._view_combo.currentTextChanged.connect(self._on_view_changed)
        tb.addWidget(self._view_combo)
        self._ocio_display = color.default_display()
        self._refill_view_combo()
        tb.addSeparator()

        tb.addWidget(QLabel(" Exposure "))
        self._view_exp_slider = QSlider(Qt.Orientation.Horizontal)
        self._view_exp_slider.setRange(-60, 60)
        self._view_exp_slider.setValue(0)
        self._view_exp_slider.setFixedWidth(160)
        self._view_exp_slider.valueChanged.connect(self._on_exposure)
        tb.addWidget(self._view_exp_slider)
        self._exposure_label = QLabel(" 0.0 EV ")
        tb.addWidget(self._exposure_label)

        tb.addSeparator()
        tb.addWidget(QLabel(" Gamma "))
        self._gamma_slider = QSlider(Qt.Orientation.Horizontal)
        # Hundredths of gamma — range 0.25 .. 4.0, default 1.0.
        self._gamma_slider.setRange(25, 400)
        self._gamma_slider.setValue(100)
        self._gamma_slider.setFixedWidth(140)
        self._gamma_slider.setToolTip(
            "Display-only viewport gamma. Does not affect the bake."
        )
        self._gamma_slider.valueChanged.connect(self._on_gamma)
        tb.addWidget(self._gamma_slider)
        self._gamma_label = QLabel(" 1.00 ")
        tb.addWidget(self._gamma_label)

        tb.addSeparator()
        tb.addWidget(QLabel(" Yaw offset "))
        self._yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self._yaw_slider.setRange(-1800, 1800)  # tenths of a degree, -180.0..+180.0
        self._yaw_slider.setValue(0)
        self._yaw_slider.setFixedWidth(200)
        self._yaw_slider.valueChanged.connect(self._on_yaw_offset)
        tb.addWidget(self._yaw_slider)
        self._yaw_label = QLabel("  0.0° ")
        tb.addWidget(self._yaw_label)

        reset_view_btn = QPushButton("⟲ Reset view")
        reset_view_btn.setToolTip(
            "Reset the viewport: display exposure, gamma, yaw offset, and "
            "zoom/pan back to defaults."
        )
        reset_view_btn.clicked.connect(self._reset_view)
        tb.addWidget(reset_view_btn)

        tb.addSeparator()
        tb.addWidget(QLabel(" Depth "))
        self._backend_combo = QComboBox()
        _backend_labels = {"da2": "DA²", "dap": "DAP"}
        for b in AVAILABLE_BACKENDS:
            self._backend_combo.addItem(_backend_labels.get(b, b.upper()), b)
        self._backend_combo.setCurrentIndex(
            max(0, self._backend_combo.findData(self._depth_backend))
        )
        # Self-correct the backend if the requested default isn't available.
        self._depth_backend = (
            self._backend_combo.currentData() or self._depth_backend
        )
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        tb.addWidget(self._backend_combo)
        self._depth_btn = QPushButton("Show depth")
        self._depth_btn.setCheckable(True)
        self._depth_btn.setShortcut("D")
        self._depth_btn.toggled.connect(self._on_depth_toggle)
        tb.addWidget(self._depth_btn)

        # Exposure mode — rightmost, closest to the right edge.
        tb.addSeparator()
        self._exposure_btn = QPushButton("Exposure mode")
        self._exposure_btn.setCheckable(True)
        self._exposure_btn.setShortcut("E")
        self._exposure_btn.setToolTip(
            "Adjust the HDRI baseline exposure + white balance + colour-checker "
            "match. Hides the light quads while active. Baked into the export. "
            "(shortcut: E)"
        )
        self._exposure_btn.toggled.connect(self._on_exposure_mode)
        tb.addWidget(self._exposure_btn)

    # ---------- recent files ----------

    _MAX_RECENT = 10

    @staticmethod
    def _settings() -> QSettings:
        return QSettings("env2lgt", "env2lgt")

    def _recent_paths(self) -> list[str]:
        val = self._settings().value("recentFiles", [])
        if isinstance(val, str):
            val = [val]
        return [str(p) for p in (val or [])]

    def _push_recent(self, path: Path) -> None:
        p = str(Path(path).resolve())
        recent = [r for r in self._recent_paths() if r != p]
        recent.insert(0, p)
        del recent[self._MAX_RECENT:]
        self._settings().setValue("recentFiles", recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        menu = self._recent_menu
        menu.clear()
        recent = self._recent_paths()
        if not recent:
            act = menu.addAction("(no recent files)")
            act.setEnabled(False)
            return
        for p in recent:
            act = menu.addAction(Path(p).name)
            act.setToolTip(p)
            act.triggered.connect(
                lambda _=False, path=p: self._open_recent(path)
            )
        menu.addSeparator()
        menu.addAction("Clear recent files").triggered.connect(
            self._clear_recent
        )

    def _open_recent(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            QMessageBox.warning(
                self, "Open recent", f"File no longer exists:\n{path}"
            )
            recent = [r for r in self._recent_paths() if r != str(p.resolve())]
            self._settings().setValue("recentFiles", recent)
            self._rebuild_recent_menu()
            return
        self._load_exr(p)

    def _clear_recent(self) -> None:
        self._settings().remove("recentFiles")
        self._rebuild_recent_menu()

    # ---------- file loading ----------

    def _open_exr_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open EXR latlong", "", "EXR (*.exr);;All files (*.*)"
        )
        if path:
            self._load_exr(Path(path))

    def _load_exr(self, path: Path):
        try:
            hdr = load_latlong(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open EXR", f"Failed to load {path.name}:\n{e}")
            return
        # If a file was already loaded with quads, ask before discarding them.
        if self._hdr is not None and self.viewer.quads():
            ret = QMessageBox.question(
                self,
                "Open new EXR",
                f"Discard {len(self.viewer.quads())} drawn quad(s) and load {path.name}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return
        self._hdr = hdr
        # Downsampled display copy so exposure + yaw scrubs stay snappy. Kept
        # in the *source* colorspace so the input transform can be re-chosen
        # without reloading; `_hdr_display` is the working-space version.
        H, W, _ = hdr.shape
        if W > DISPLAY_MAX_WIDTH:
            new_w = DISPLAY_MAX_WIDTH
            new_h = max(2, (new_w * H) // W)
            # Even output size keeps 2:1 latlong aspect well.
            new_h -= new_h % 2
            self._hdr_display_src = cv2.resize(hdr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            self._hdr_display_src = hdr
        self._hdr_display = self._to_working_display(self._hdr_display_src)
        # Invalidate caches — last EXR's tonemap is meaningless now.
        self._display_cache = None
        self._cache_key = None
        self._distance_display = None
        self._exr_path = path
        self._push_recent(path)
        # --- reset everything tied to the previous EXR ---
        # Yaw offset back to 0 (a stale offset would confuse quad placement).
        if hasattr(self, "_yaw_slider"):
            self._yaw_slider.blockSignals(True)
            self._yaw_slider.setValue(0)
            self._yaw_slider.blockSignals(False)
        self._yaw_offset_deg = 0.0
        if hasattr(self, "_yaw_label"):
            self._yaw_label.setText("  0.0° ")
        # Baseline exposure + WB back to neutral for the new HDRI.
        self._exposure_offset = 0.0
        self._wb_kelvin = 6500.0
        self._wb_tint = 0.0
        self._wb_scale = np.ones(3, dtype=np.float32)
        # Drop any colour-checker correction from the previous EXR. The loaded
        # reference image persists (it's an independent match target), but its
        # chart is cleared by viewer.reset_image() below.
        self._cc_matrix = None
        self._cc_target_mode = "builtin"
        self._cc_target_name = "Built-in CC24"
        if hasattr(self, "viewer"):
            self.viewer.set_flat_mode(False)
        if hasattr(self, "exposure_panel"):
            self.exposure_panel.set_exposure_offset(0.0)
            self.exposure_panel.set_wb(6500.0, 0.0)
            self.exposure_panel.set_wb_readout(self._wb_scale)
            self.exposure_panel.set_chart_status(has_chart=False, rmse=None)
            self.exposure_panel.set_target_label("Built-in CC24")
            self.exposure_panel.set_correction_available(False)
            self.exposure_panel.set_reference_view(False)
        # Drop cached depth — it's for the previous EXR.
        self._distance = None
        self._view_mode = "hdr"
        if hasattr(self, "_depth_btn"):
            self._depth_btn.blockSignals(True)
            self._depth_btn.setChecked(False)
            self._depth_btn.setEnabled(True)
            self._depth_btn.setText("Show depth")
            self._depth_btn.blockSignals(False)
        # Last bake's USD is no longer applicable.
        self._last_usd = None
        self._last_mesh = None
        # Clear quads everywhere — viewer state, panel list, viewer's stored dict.
        # IMPORTANT: viewer.reset_image() empties self.viewer._quads, so we
        # can't iterate it afterwards to remove panel entries. Just clear the
        # panel list directly.
        self.viewer.reset_image()
        self.panel.clear_quads()
        # Reset output path to default for this EXR.
        default_out = str(path.parent / f"{path.stem}_lightrig")
        self.panel.force_set_output_path(default_out)
        self._refresh_view()
        self._refresh_key_preview()
        h, w, _ = self._hdr.shape
        self._set_status(f"{path.name}  ·  {w}×{h}  ·  float32")

        # Look for a sibling project file and offer to restore.
        sibling = default_project_path(path)
        if sibling.exists():
            ret = QMessageBox.question(
                self,
                "Restore project?",
                f"Found a saved env2lgt project for this EXR:\n  {sibling.name}\n\n"
                "Restore the quads + settings from it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret == QMessageBox.StandardButton.Yes:
                try:
                    proj = load_project(sibling)
                    self._apply_project_state(proj)
                    self._set_status(
                        f"Restored {len(proj.quads)} quad(s) from {sibling.name}"
                    )
                except Exception as e:  # noqa: BLE001
                    QMessageBox.warning(self, "Load project", f"Failed to read {sibling.name}:\n{e}")

    # drag-and-drop
    def dragEnterEvent(self, e):  # noqa: N802
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.toLocalFile().lower().endswith(".exr"):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):  # noqa: N802
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() == ".exr":
                self._load_exr(p)
                e.acceptProposedAction()
                return
        e.ignore()

    # ---------- view tweaks ----------

    def _refresh_view(self):
        """Push a QImage to the viewer for the current (mode, exposure, yaw).

        Two-stage caching strategy:
        - Tonemap (or depth colormap) -> uint8 RGB array, cached in
          `self._display_cache`. Keyed by (view_mode, exposure) so the
          expensive part runs only when exposure or mode actually changes.
        - Yaw offset rolls the cached uint8 (column-wise) and wraps in a
          QImage. Cheap (~5 ms at 2K) — runs on every yaw-slider tick.
        """
        # Reference-image view (flat 2D) — bypasses the pano cache + yaw roll.
        if self._view_mode == "reference":
            if self._ref_display is None:
                return
            u8 = self._working_to_u8(self._ref_display, 0.0)
            H, W, _ = u8.shape
            qimg = QImage(u8.data, W, H, W * 3, QImage.Format.Format_RGB888).copy()
            self.viewer.set_yaw_offset_px(0)
            self.viewer.set_image(qimg)
            return

        if self._hdr_display is None:
            return

        is_hdr = self._view_mode == "hdr"
        cc_id = None if self._cc_matrix is None else hash(self._cc_matrix.tobytes())
        cache_key = (
            self._view_mode,
            round(self._exposure, 3) if is_hdr else None,
            round(self._exposure_offset, 3) if is_hdr else None,
            tuple(round(float(c), 4) for c in self._wb_scale) if is_hdr else None,
            cc_id if is_hdr else None,
            (self._ocio_display, self._ocio_view) if is_hdr else None,
            round(self._view_gamma, 3),
        )
        if self._display_cache is None or self._cache_key != cache_key:
            if self._view_mode == "depth" and self._distance_display is not None:
                self._display_cache = self._tonemap_depth_uint8(self._distance_display)
            else:
                adjusted = self._adjusted_working(self._hdr_display)
                self._display_cache = self._working_to_u8(adjusted, self._exposure)
            self._cache_key = cache_key

        cache = self._display_cache
        H, W, _ = cache.shape
        offset_px = int(round((self._yaw_offset_deg / 360.0) * W)) % W
        if offset_px != 0:
            rolled = np.roll(cache, offset_px, axis=1)
        else:
            rolled = cache
        rolled = np.ascontiguousarray(rolled)
        qimg = QImage(rolled.data, W, H, W * 3, QImage.Format.Format_RGB888).copy()
        self.viewer.set_yaw_offset_px(offset_px)
        self.viewer.set_image(qimg)

    def _to_working_display(self, src: np.ndarray) -> np.ndarray:
        """Convert a source-colorspace buffer into the working space."""
        if not self._input_cs:
            return np.asarray(src, dtype=np.float32)
        try:
            return color.to_working(src, self._input_cs)
        except Exception:  # noqa: BLE001
            return np.asarray(src, dtype=np.float32)

    def _adjusted_working(self, hdr: np.ndarray) -> np.ndarray:
        """Apply the baked baseline adjustments (colour-checker matrix, white
        balance, exposure offset) to a working-space buffer. Display-only
        viewport exposure is NOT included here."""
        out = hdr
        if self._cc_matrix is not None:
            out = ccheck.apply_matrix(out, self._cc_matrix)
        out = out * self._wb_scale.reshape(1, 1, 3) * (2.0 ** self._exposure_offset)
        return out

    def _working_to_u8(self, working: np.ndarray, exposure: float) -> np.ndarray:
        """Working space -> display uint8: viewport exposure, the OCIO
        display+view transform (or an ACES-filmic fallback), then the
        display-only viewport gamma."""
        scaled = working * (2.0 ** float(exposure))
        if self._display_cpu is not None:
            u8 = color.display_to_u8(scaled, self._display_cpu)
        else:
            u8 = self._tonemap_hdr_uint8(scaled)
        return self._apply_view_gamma(u8)

    def _apply_view_gamma(self, u8: np.ndarray) -> np.ndarray:
        """Apply the display-only viewport gamma via a 256-entry LUT."""
        g = self._view_gamma
        if abs(g - 1.0) < 1e-3:
            return u8
        ramp = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / g)
        lut = np.clip(ramp * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(lut[u8])

    @staticmethod
    def _tonemap_hdr_uint8(scaled: np.ndarray) -> np.ndarray:
        """ACES filmic tonemap + sRGB clip — the fallback when OCIO is
        unavailable. Input is already exposure-scaled."""
        ldr = aces_filmic(scaled)
        u8 = (ldr * 255.0 + 0.5).astype(np.uint8)
        return np.ascontiguousarray(u8)

    @staticmethod
    def _tonemap_depth_uint8(distance: np.ndarray) -> np.ndarray:
        """Per-image normalized turbo colormap of the distance map."""
        d = distance.astype(np.float32)
        lo, hi = float(d.min()), float(d.max())
        span = max(1e-6, hi - lo)
        norm = (np.clip((d - lo) / span, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        cm_bgr = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        cm_rgb = cv2.cvtColor(cm_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(cm_rgb)

    def _on_exposure(self, val: int):
        self._exposure = val / 10.0
        self._exposure_label.setText(f" {self._exposure:+.1f} EV ")
        self._refresh_view()

    def _on_scene_scale(self, value: float):
        self._scene_scale = float(value)

    def _on_gamma(self, val: int):
        self._view_gamma = val / 100.0
        self._gamma_label.setText(f" {self._view_gamma:.2f} ")
        self._refresh_view()

    def _on_backend_changed(self, _idx: int):
        name = self._backend_combo.currentData()
        if name == self._depth_backend:
            return
        self._depth_backend = name
        # Snap scene scale to the backend's natural default: DAP is metric
        # (depth already in metres → 1.0 m/u), DA² is scale-invariant and
        # needs the ~100 m/u working default. The spinbox's valueChanged
        # drives _on_scene_scale, which updates self._scene_scale.
        self.panel.opt_scene_scale.setValue(1.0 if name == "dap" else 100.0)
        # The cached depth belongs to the previous backend — drop it.
        self._distance = None
        self._distance_display = None
        if self._view_mode == "depth":
            # Recompute under the new backend (toggle handler respawns the run).
            self._display_cache = None
            self._cache_key = None
            self._on_depth_toggle(self._depth_btn.isChecked())
        self._set_status(f"Depth backend: {self._backend_combo.currentText()}")

    def _on_yaw_offset(self, val: int):
        self._yaw_offset_deg = val / 10.0
        self._yaw_label.setText(f" {self._yaw_offset_deg:+6.1f}° ")
        self._refresh_view()
        if self._key_preview_on:
            self._refresh_key_preview()

    def _reset_view(self):
        """Reset viewport-only controls: display exposure, yaw, zoom/pan.

        These are display conveniences — none of them affect the bake (unlike
        the exposure-mode baseline adjustments)."""
        self._view_exp_slider.setValue(0)
        self._gamma_slider.setValue(100)
        self._yaw_slider.setValue(0)
        self.viewer.fit_view()

    # ---------- OCIO colour management ----------

    def _invalidate_display_cache(self):
        self._display_cache = None
        self._cache_key = None

    def _refill_view_combo(self):
        """Repopulate the view combo for the current display and rebuild the
        cached display processor."""
        views = color.views(self._ocio_display)
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItems(views)
        default_v = color.default_view(self._ocio_display)
        self._view_combo.setCurrentText(
            default_v if default_v in views else (views[0] if views else "")
        )
        self._view_combo.blockSignals(False)
        self._ocio_view = self._view_combo.currentText()
        self._rebuild_display_cpu()

    def _rebuild_display_cpu(self):
        if not (self._ocio_display and self._ocio_view):
            self._display_cpu = None
            return
        try:
            self._display_cpu = color.make_display_cpu(
                self._ocio_display, self._ocio_view
            )
        except Exception:  # noqa: BLE001
            self._display_cpu = None

    def _on_display_changed(self, name: str):
        self._ocio_display = name
        self._refill_view_combo()
        self._invalidate_display_cache()
        self._refresh_view()

    def _on_view_changed(self, name: str):
        if not name:
            return
        self._ocio_view = name
        self._rebuild_display_cpu()
        self._invalidate_display_cache()
        self._refresh_view()

    def _on_input_cs_changed(self, name: str):
        """Source colorspace changed — re-run the input transform."""
        self._input_cs = name
        if self._hdr_display_src is not None:
            self._hdr_display = self._to_working_display(self._hdr_display_src)
            self._invalidate_display_cache()
            self._refresh_view()

    def _on_output_cs_changed(self, name: str):
        """Bake output colorspace — applied when writing the rig (no preview
        effect)."""
        self._output_cs = name

    # ---------- exposure mode: baseline exposure + white balance ----------

    def _recompute_wb(self) -> None:
        """Rebuild the derived RGB scale from the kelvin/tint source of truth."""
        self._wb_scale = temp_tint_to_scale(self._wb_kelvin, self._wb_tint)
        self.exposure_panel.set_wb_readout(self._wb_scale)

    def _on_exposure_offset(self, ev: float):
        self._exposure_offset = float(ev)
        self._refresh_view()

    def _on_wb_changed(self, kelvin: float, tint: float):
        self._wb_kelvin = float(kelvin)
        self._wb_tint = float(tint)
        self._recompute_wb()
        self._refresh_view()

    def _on_exposure_mode(self, checked: bool):
        self._exposure_mode = bool(checked)
        # The light quads are irrelevant while metering — hide them.
        self.viewer.set_quads_visible(not checked)
        # Swap the panel via the stacked widget — the dock geometry (and so
        # the viewport) is untouched, so zoom/pan never jumps.
        self._panel_stack.setCurrentIndex(1 if checked else 0)
        self._dock.setWindowTitle("Exposure" if checked else "Lights")
        if not checked:
            if self.viewer.is_sample_mode():
                self.viewer.cancel_sample_mode()
            if self.viewer.is_chart_mode():
                self.viewer.cancel_chart_mode()
            # Leaving exposure mode drops the reference view.
            if self._view_mode == "reference":
                self.exposure_panel.set_reference_view(False)
                self._on_reference_view_toggled(False)
        if self._exr_path is not None:
            self._set_status(
                "Exposure mode — adjust the HDRI baseline; baked into export."
                if checked else f"{self._exr_path.name}"
            )

    def _begin_sample(self, purpose: str):
        """Arm the rectangle-sample tool for a given purpose and enter the mode."""
        if self._hdr_display is None:
            QMessageBox.information(self, "Sample", "Open an EXR first.")
            self.exposure_panel.clear_sample_buttons()
            self._eyedrop_btn.setChecked(False)
            return
        self._pending_sample = purpose
        self.viewer.start_sample_mode()
        self._set_status(
            "Drag a rectangle to sample an area. Esc to cancel."
        )

    def _on_eyedropper_clicked(self):
        if self._eyedrop_btn.isChecked():
            self._begin_sample("probe")
        elif self.viewer.is_sample_mode():
            self.viewer.cancel_sample_mode()

    def _on_sample_mode_changed(self, active: bool):
        """Sample mode ended (completed or cancelled) — reset the UI affordances."""
        if not active:
            self._pending_sample = None
            self.exposure_panel.clear_sample_buttons()
            self._eyedrop_btn.blockSignals(True)
            self._eyedrop_btn.setChecked(False)
            self._eyedrop_btn.blockSignals(False)

    def _sample_region(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
        """Slice the display HDR for a sampled rectangle, handling seam wrap."""
        hdr = self._hdr_display
        if hdr is None:
            return None
        H, W = hdr.shape[:2]
        y0 = max(0, min(H - 1, y0))
        y1 = max(0, min(H - 1, y1))
        h = y1 - y0 + 1
        w = (x1 - x0 + 1) if x1 >= x0 else (x1 - x0 + 1 + W)
        if h <= 0 or w <= 0:
            return None
        # Cap the pixel count fed to the meter — a sample over a huge crop of
        # an 8K/16K source would otherwise mean over tens of millions of
        # pixels. Stride is derived *before* indexing so the strided crop is
        # the only copy made; a ~1 Mpx subsample is statistically identical
        # for an average.
        max_px = 1_000_000
        stride = max(1, int(np.ceil(np.sqrt(h * w / max_px))))
        cols = np.arange(x0, x0 + w, stride) % W
        region = hdr[y0:y1 + 1:stride][:, cols, :3]
        return region if region.size else None

    def _on_area_sampled(self, x0: int, y0: int, x1: int, y1: int):
        purpose = self._pending_sample
        region = self._sample_region(x0, y0, x1, y1)
        if region is None or region.size == 0:
            self.viewer.cancel_sample_mode()
            return

        if purpose == "exposure":
            offset = spot_meter_offset_ev(region)
            self._exposure_offset = offset
            self.exposure_panel.set_exposure_offset(offset)
            self._refresh_view()
            self._set_status(
                f"Spot meter — exposure offset set to {offset:+.2f} EV"
            )
        elif purpose == "wb":
            mean = region.reshape(-1, 3).mean(axis=0)
            scale = grey_world_scale(mean)
            kelvin, tint = scale_to_temp_tint(scale)
            self._wb_kelvin, self._wb_tint = kelvin, tint
            self.exposure_panel.set_wb(kelvin, tint)
            self._recompute_wb()
            self._refresh_view()
            self._set_status(
                f"WB sampled — {int(kelvin)} K, tint {tint:+.2f}"
            )
        elif purpose == "probe":
            mean = region.reshape(-1, 3).mean(axis=0)
            self._show_probe_rgb(mean, region.shape[0] * region.shape[1])

        self.viewer.cancel_sample_mode()

    def _show_probe_rgb(self, rgb: np.ndarray, n_px: int):
        """Show an averaged RGB readout in the persistent area-probe label.

        Unlike the live cursor probe this survives subsequent mouse moves, so
        the eyedropper sample stays visible (Nuke-style colour sampler)."""
        r, g, b = (float(c) for c in rgb[:3])
        self._area_label.setText(
            f"<span style='color:#aaa'>▣ avg</span> "
            f"<span style='color:#ff6b6b'>{r:.4f}</span> "
            f"<span style='color:#6bd66b'>{g:.4f}</span> "
            f"<span style='color:#6b9bff'>{b:.4f}</span> "
            f"<span style='color:#888'>[{n_px}px]</span>"
        )

    def _on_auto_meter(self):
        """Convolve-the-dome auto: render a gray ball, set exposure + WB from it."""
        if self._hdr_display is None:
            QMessageBox.information(self, "Auto meter", "Open an EXR first.")
            return
        self._set_status("Auto meter — convolving the dome…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            result = convolve_dome_meter(self._hdr_display)
        finally:
            QApplication.restoreOverrideCursor()
        self._exposure_offset = float(result["offset_ev"])
        kelvin, tint = scale_to_temp_tint(result["wb_scale"])
        self._wb_kelvin, self._wb_tint = kelvin, tint
        self.exposure_panel.set_exposure_offset(self._exposure_offset)
        self.exposure_panel.set_wb(kelvin, tint)
        self._recompute_wb()
        self._refresh_view()
        self._set_status(
            f"Auto meter — {self._exposure_offset:+.2f} EV, "
            f"{int(kelvin)} K, tint {tint:+.2f}"
        )

    # ---------- colour-checker chart ----------

    def _on_pick_chart(self):
        if self._hdr_display is None:
            QMessageBox.information(self, "Colour chart", "Open an EXR first.")
            self.exposure_panel.set_pick_chart_active(False)
            return
        self.viewer.start_chart_mode()
        self._set_status(
            "Click the 4 chart corners — start at the dark-skin patch, "
            "then clockwise. Esc to cancel."
        )

    def _on_chart_mode_changed(self, active: bool):
        self.exposure_panel.set_pick_chart_active(active)

    def _on_chart_committed(self):
        self.exposure_panel.set_chart_status(has_chart=True, rmse=None)
        self._set_status(
            "Chart placed — drag the corners to refine, then Solve & apply."
        )

    def _on_clear_chart(self):
        self.viewer.clear_chart()
        self._cc_matrix = None
        self._last_rmse = None
        self.exposure_panel.set_chart_status(has_chart=False, rmse=None)
        self.exposure_panel.set_correction_available(False)
        self._invalidate_display_cache()
        self._refresh_view()

    def _on_fit_mode_changed(self, mode: str):
        self._cc_fit_mode = mode

    def _builtin_target_working(self) -> np.ndarray:
        """The built-in CC24 reference, converted into the working space."""
        cc = ccheck.CC24_LINEAR_SRGB
        try:
            return color.convert(
                cc.reshape(1, 24, 3),
                ccheck.CC24_REFERENCE_COLORSPACE,
                color.WORKING,
            ).reshape(24, 3)
        except Exception:  # noqa: BLE001
            return cc.astype(np.float32)

    def _current_target_working(self) -> np.ndarray | None:
        """The active match target (24,3) in the working space, or None if it
        cannot be resolved (e.g. reference target with no reference chart)."""
        if self._cc_target_mode == "reference":
            return self._sample_reference_swatches()
        if self._cc_target_mode == "json" and self._cc_target_swatches is not None:
            return self._cc_target_swatches
        return self._builtin_target_working()

    def _on_use_builtin_target(self):
        self._cc_target_mode = "builtin"
        self._cc_target_name = "Built-in CC24"
        self.exposure_panel.set_target_label("Built-in CC24")

    def _on_use_reference_target(self):
        if self.viewer.chart_uv() is None and self._ref_display is None:
            QMessageBox.information(
                self, "Reference target",
                "Load a reference image and place a chart on it first.",
            )
            return
        self._cc_target_mode = "reference"
        self._cc_target_name = "Reference image"
        self.exposure_panel.set_target_label("Reference image chart")

    def _on_load_json_target(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load colour target", "", "Colour target (*.json);;All files (*.*)"
        )
        if not path:
            return
        try:
            sw, name, cs = ccheck.load_target(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load target", str(e))
            return
        try:
            sw = color.convert(sw.reshape(1, 24, 3), cs, color.WORKING).reshape(24, 3)
        except Exception:  # noqa: BLE001
            pass
        self._cc_target_swatches = sw.astype(np.float32)
        self._cc_target_mode = "json"
        self._cc_target_name = name
        self.exposure_panel.set_target_label(f"JSON: {name}")
        self._set_status(f"Colour target loaded: {name}")

    # ---------- reference image ----------

    def _on_load_reference_image(self):
        """Load a regular 2D photo of a colour chart to match against."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load reference image", "",
            "Images (*.exr *.jpg *.jpeg *.png *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        try:
            raw, default_cs = self._read_reference_raw(Path(path))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load reference", str(e))
            return
        # Downsample for display, same width cap as the panorama.
        H, W = raw.shape[:2]
        if W > DISPLAY_MAX_WIDTH:
            new_w = DISPLAY_MAX_WIDTH
            new_h = max(2, (new_w * H) // W)
            raw = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
        self._ref_src = np.ascontiguousarray(raw)
        self._ref_path = Path(path)
        # Reflect the auto-detected colorspace on the panel, then convert the
        # reference into the working space through it.
        self.exposure_panel.set_reference_cs(default_cs)
        self._ref_cs = self.exposure_panel.reference_cs()
        self._ref_display = self._reference_to_working(self._ref_src)
        self.exposure_panel.set_reference_loaded(True)
        self._set_status(f"Reference image loaded: {Path(path).name}")
        # Jump straight to the reference view so the user can place a chart.
        self.exposure_panel.set_reference_view(True)
        self._on_reference_view_toggled(True)

    def _read_reference_raw(self, path: Path) -> tuple[np.ndarray, str]:
        """Read a reference image as a source-encoded float RGB array, plus a
        best-guess source colorspace: 8-bit images are treated as sRGB-encoded;
        float / EXR as the current input colorspace."""
        arr = cv2.imread(
            str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR | cv2.IMREAD_COLOR
        )
        if arr is None:
            raise RuntimeError(f"Could not read {path.name}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0, "Utility - sRGB - Texture"
        return np.ascontiguousarray(arr.astype(np.float32)), self._input_cs

    def _reference_to_working(self, src: np.ndarray) -> np.ndarray:
        """Convert the source-encoded reference into the working space using
        the colorspace currently chosen on the panel."""
        if not self._ref_cs:
            return np.asarray(src, dtype=np.float32)
        try:
            return color.to_working(src, self._ref_cs)
        except Exception:  # noqa: BLE001
            return np.asarray(src, dtype=np.float32)

    def _on_reference_cs_changed(self, name: str):
        """Reference-image colorspace changed — re-run its input transform."""
        self._ref_cs = name
        if self._ref_src is None:
            return
        self._ref_display = self._reference_to_working(self._ref_src)
        if self._view_mode == "reference":
            self._invalidate_display_cache()
            self._refresh_view()

    def _on_reference_view_toggled(self, show_reference: bool):
        if show_reference and self._ref_display is None:
            self.exposure_panel.set_reference_view(False)
            return
        if self.viewer.is_chart_mode():
            self.viewer.cancel_chart_mode()
        self._view_mode = "reference" if show_reference else "hdr"
        self.viewer.set_flat_mode(show_reference)
        self.exposure_panel.set_reference_view(show_reference)
        self._invalidate_display_cache()
        self._refresh_view()
        self._set_status(
            "Reference image — place a chart, then set the target to "
            "'Reference image'." if show_reference else ""
        )

    def _sample_chart_swatches(self) -> np.ndarray | None:
        """Sample the 24 HDRI chart patches from the working-space panorama via
        spherical-bilinear blend of the corner directions — equirect-correct,
        and seam-safe (directions wrap naturally). Returns (24,3) or None."""
        dirs = self.viewer.chart_dirs()
        if dirs is None or self._hdr_display is None:
            return None
        return ccheck.sample_swatches_spherical(self._hdr_display, dirs)

    def _sample_reference_swatches(self) -> np.ndarray | None:
        """Sample the 24 patches from the flat reference image's chart."""
        uv = self.viewer.chart_uv()
        if uv is None or self._ref_display is None:
            return None
        H, W = self._ref_display.shape[:2]
        corners_px = np.asarray(uv, dtype=np.float64) * np.array([W, H])
        sw, _rect = ccheck.rectify_swatches(self._ref_display, corners_px)
        return sw

    def _on_solve_chart(self):
        sw = self._sample_chart_swatches()
        if sw is None:
            QMessageBox.information(
                self, "Solve chart",
                "Place a colour chart on the HDRI panorama first "
                "(switch to the HDRI view, then Pick colour chart).",
            )
            return
        target = self._current_target_working()
        if target is None:
            QMessageBox.information(
                self, "Solve chart",
                "The reference-image target has no chart — switch to the "
                "reference view and place one.",
            )
            return
        M, rmse, flipped = ccheck.solve_correction(sw, target, self._cc_fit_mode)
        self._cc_matrix = M
        self._last_rmse = rmse
        self._invalidate_display_cache()
        self._refresh_view()
        self.exposure_panel.set_chart_status(has_chart=True, rmse=rmse)
        self.exposure_panel.set_correction_available(True)
        msg = (
            f"Chart match applied ({self._cc_fit_mode}) — RMSE {rmse:.4f}"
            f"  ·  target: {self._cc_target_name}"
        )
        if flipped:
            msg += "  ·  chart detected upside-down (auto-corrected)"
        self._set_status(msg)

    def _on_save_correction(self):
        """Export the solved colour-checker correction to a JSON file."""
        if self._cc_matrix is None:
            QMessageBox.information(
                self, "Save correction",
                "Solve a colour-checker correction before saving it.",
            )
            return
        default_name = "cc_correction.json"
        if self._exr_path is not None:
            default_name = f"{self._exr_path.stem}_cc_correction.json"
            start = str(self._exr_path.parent / default_name)
        else:
            start = default_name
        path, _ = QFileDialog.getSaveFileName(
            self, "Save colour correction", start,
            "Colour correction (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            ccheck.save_correction(
                path, self._cc_matrix, self._cc_fit_mode, self._cc_target_name,
                rmse=self._last_rmse,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save correction", str(e))
            return
        self._set_status(f"Correction saved: {Path(path).name}")

    def _on_load_correction(self):
        """Load a saved correction JSON and apply it directly — no chart
        needed. Used to batch-match a set of similar HDRIs."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load colour correction", "",
            "Colour correction (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            corr = ccheck.load_correction(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load correction", str(e))
            return
        self._cc_matrix = corr["matrix"]
        self._cc_fit_mode = corr["fit_mode"]
        self._cc_target_name = corr["target_name"] or corr["name"]
        self._last_rmse = corr["rmse"]
        self.exposure_panel.set_fit_mode(self._cc_fit_mode)
        self.exposure_panel.set_correction_available(True)
        self.exposure_panel.set_chart_status(
            has_chart=True, rmse=self._last_rmse, applied=self._last_rmse is None
        )
        self._invalidate_display_cache()
        self._refresh_view()
        self._set_status(
            f"Correction loaded: {Path(path).name} ({self._cc_fit_mode})"
        )

    def _on_depth_toggle(self, checked: bool):
        if not checked:
            self._view_mode = "hdr"
            self._display_cache = None         # invalidate — different mode
            self._cache_key = None
            self._depth_btn.setText("Show depth")
            self._refresh_view()
            return
        if self._exr_path is None or self._hdr is None:
            self._depth_btn.setChecked(False)
            return
        if self._distance is not None:
            # Already computed (cache hit) — instant switch.
            self._view_mode = "depth"
            self._display_cache = None         # invalidate — different mode
            self._cache_key = None
            self._depth_btn.setText("Show HDR")
            self._refresh_view()
            return
        # Need to compute — kick off background DA-2.
        self._depth_btn.setText("Computing depth…")
        self._depth_btn.setEnabled(False)
        cache_dir = self._exr_path.parent / ".env2lgt_cache"
        self._depth_thread = QThread(self)
        self._depth_worker = DepthWorker(
            str(self._exr_path), str(cache_dir), self._depth_backend
        )
        self._depth_worker.moveToThread(self._depth_thread)
        self._depth_thread.started.connect(self._depth_worker.run)
        self._depth_worker.finished.connect(self._on_depth_ready)
        self._depth_worker.failed.connect(self._on_depth_failed)
        self._depth_worker.finished.connect(self._depth_thread.quit)
        self._depth_worker.failed.connect(self._depth_thread.quit)
        self._depth_thread.finished.connect(self._depth_worker.deleteLater)
        self._depth_thread.finished.connect(self._depth_thread.deleteLater)
        self._set_status(
            f"Running {self._backend_combo.currentText()} for depth view…"
        )
        self._depth_thread.start()

    def _on_depth_ready(self, distance):
        self._distance = distance
        # Build the display-res depth at the same dimensions as _hdr_display
        # so the cache + roll path is symmetric with HDR mode.
        if self._hdr_display is not None and distance.shape != self._hdr_display.shape[:2]:
            h_d, w_d = self._hdr_display.shape[:2]
            self._distance_display = cv2.resize(
                distance, (w_d, h_d), interpolation=cv2.INTER_LINEAR
            )
        else:
            self._distance_display = distance
        # Invalidate any HDR cache so the next refresh picks up depth.
        self._display_cache = None
        self._cache_key = None
        # Only flip the visible view to depth if the user pressed "Show depth".
        # When this run was kicked off implicitly by "Fit to rect light",
        # the user's still looking at the HDR — don't yank them into depth view.
        switching_to_depth = self._depth_btn.isChecked()
        if switching_to_depth:
            self._view_mode = "depth"
            self._depth_btn.setText("Show HDR")
        self._depth_btn.setEnabled(True)
        self._set_status(
            f"Depth ready  ·  range [{distance.min():.3f}, {distance.max():.3f}] (DA-2 scale-invariant)"
        )
        self._refresh_view()
        # Run the deferred fit if the user pressed the rect-fit button while
        # depth was still computing.
        if self._pending_fit_to_rect:
            self._pending_fit_to_rect = False
            self._on_fit_to_rect()

    def _on_depth_failed(self, msg: str):
        self._distance = None
        self._depth_btn.setEnabled(True)
        self._depth_btn.setChecked(False)
        self._depth_btn.setText("Show depth")
        # Drop any deferred fit — without depth, the snap can't run.
        self._pending_fit_to_rect = False
        QMessageBox.critical(self, "Depth failed", msg)

    # ---------- quad lifecycle ----------

    def _quad_by_name(self, name: str) -> LightQuad | None:
        return next((q for q in self.viewer.quads() if q.name == name), None)

    def _refresh_window_checkbox(self, name: str) -> None:
        q = self._quad_by_name(name) if name else None
        self.panel.set_window_checkbox(name if q else "", bool(q.is_window) if q else False)

    def _on_window_toggled(self, name: str, checked: bool):
        q = self._quad_by_name(name)
        if q is not None:
            q.is_window = bool(checked)

    def _on_lock_toggled(self, name: str, locked: bool):
        """User toggled a quad's lock checkbox in the list."""
        self.viewer.set_quad_locked(name, locked)

    def _on_quad_lock_changed(self, name: str, locked: bool):
        """Quad got auto-locked (edited an auto quad) — mirror onto the list."""
        self.panel.set_quad_locked(name, locked)

    def _on_quad_committed(self, q: LightQuad):
        self.panel.add_quad(q)
        self.panel.set_selected(q.name)
        self._refresh_window_checkbox(q.name)

    def _on_propose_quads(self, params: dict):
        """Run auto-detection and add the proposals as quads.

        Locked quads are left untouched and their regions are excluded from
        detection so nothing duplicates them; previously-proposed quads that
        are still unlocked are cleared and regenerated.
        """
        if self._hdr is None or self._hdr_display is None:
            QMessageBox.information(self, "Propose quads", "Open an EXR first.")
            return
        from env2lgt.lights.detect import DetectParams, propose_quads
        from env2lgt.proj import rasterize_spherical_quad

        # Detect on the baseline-adjusted HDRI (colour-checker matrix, white
        # balance, exposure offset) — the bake applies these before extracting
        # the rect / dome textures, so detection must see the same buffer or
        # the proposed quads won't match the baked result.
        hdr = self._adjusted_working(self._hdr_display)
        H, W = hdr.shape[:2]
        existing = self.viewer.quads()
        locked = [q for q in existing if q.locked]

        exclude = None
        if locked:
            exclude = np.zeros((H, W), dtype=np.uint8)
            for q in locked:
                m, _ = rasterize_spherical_quad(q.corners_dirs, H, W)
                exclude |= m

        dp = DetectParams(
            threshold=float(params.get("threshold", 0.03)),
            blur_deg=float(params.get("blur_deg", 1.0)),
            max_quads=int(params.get("max_quads", 12)),
            min_diameter_deg=float(params.get("min_diameter_deg", 1.0)),
            merge_distance_deg=float(params.get("merge_distance_deg", 1.0)),
            suppress_floor=bool(params.get("suppress_floor", True)),
        )
        self._set_status("Detecting lights…")
        QApplication.processEvents()
        detected = propose_quads(hdr, dp, exclude_mask=exclude)

        # Clear stale (unlocked) auto proposals. User + locked quads stay.
        for q in list(existing):
            if q.source == "auto" and not q.locked:
                self.viewer.remove_quad(q.name)
                self.panel.remove_quad(q.name)

        added = 0
        for det in detected:
            name = self.viewer._next_name()
            lq = LightQuad(
                name=name, corners_dirs=det.corners_dirs, source="auto"
            )
            self.viewer.add_quad(lq)
            self.panel.add_quad(lq)
            added += 1
        self.viewer.select_by_name(None)
        self.panel.set_selected(None)
        self._refresh_window_checkbox("")
        msg = f"Proposed {added} quad(s)"
        if locked:
            msg += f"  ·  kept {len(locked)} locked"
        self._set_status(msg)

    def _on_fit_to_rect(self):
        """Depth-snap every unfitted quad to a rigid rectangle on its surface.

        The user draws / drags 4 free corners; this button takes them, projects
        each onto the light's surface plane (estimated by depth — bright-region
        plane fit when there's clear contrast, per-corner depths otherwise),
        and snaps to a rectangle with orthogonal axes via diagonal-bisector.
        What the user sees on the panorama becomes what the bake authors —
        no more "looks right in the viewer, ships sheared in USD".

        Depth is computed lazily: the first press kicks off the depth backend
        (10–30 s for DA-2 / DAP) and defers the fit via `_pending_fit_to_rect`;
        subsequent presses are instant. A vertex drag flips
        `LightQuad.is_rect_fitted` back to False, and the next press fits only
        those that need it. Locked quads are still fitted — locking governs
        auto-detect replacement, not whether the rect is rigid.
        """
        if self._hdr is None or self._exr_path is None:
            QMessageBox.information(self, "Fit to rect", "Open an EXR first.")
            return
        quads = self.viewer.quads()
        pending = [q for q in quads if not q.is_rect_fitted]
        if not pending:
            self._set_status("All quads already fitted to rigid rects.")
            return

        # Need depth — kick off the backend if it's not already loaded.
        if self._distance is None:
            if self._depth_thread is not None and self._depth_thread.isRunning():
                # Already running (probably triggered by "Show depth"); just
                # mark our intent and let _on_depth_ready pick it up.
                self._pending_fit_to_rect = True
                self._set_status("Waiting for depth to finish before fitting…")
                return
            self._pending_fit_to_rect = True
            cache_dir = self._exr_path.parent / ".env2lgt_cache"
            self._depth_thread = QThread(self)
            self._depth_worker = DepthWorker(
                str(self._exr_path), str(cache_dir), self._depth_backend
            )
            self._depth_worker.moveToThread(self._depth_thread)
            self._depth_thread.started.connect(self._depth_worker.run)
            self._depth_worker.finished.connect(self._on_depth_ready)
            self._depth_worker.failed.connect(self._on_depth_failed)
            self._depth_worker.finished.connect(self._depth_thread.quit)
            self._depth_worker.failed.connect(self._depth_thread.quit)
            self._depth_thread.finished.connect(self._depth_worker.deleteLater)
            self._depth_thread.finished.connect(self._depth_thread.deleteLater)
            self._set_status(
                f"Estimating depth ({self._depth_backend.upper()}) for rect fit…"
            )
            self._depth_thread.start()
            return

        # Depth ready — do the fit now.
        self._fit_unfitted_quads(pending)

    def _fit_unfitted_quads(self, quads: list[LightQuad]) -> None:
        """Run rect_from_quad + rect_to_corner_dirs on each pending quad and
        push the snapped corners back to the viewer + panel. Same algorithm
        as the bake, so the displayed result matches the authored RectLight."""
        from env2lgt.lights.extract import (
            luminance, rect_from_quad, rect_to_corner_dirs,
        )
        from env2lgt.proj import rasterize_spherical_quad

        H, W = self._hdr.shape[:2]
        # Apply the same baseline adjustments the bake uses, so the bright-
        # region plane fit sees the same luminance the bake will.
        hdr = self._adjusted_working(self._hdr)
        # Depth shape might not match the HDR (DA-2 runs at a different res);
        # resize to align with the mask before sampling.
        distance = self._distance
        if distance.shape != (H, W):
            distance = cv2.resize(distance, (W, H), interpolation=cv2.INTER_LINEAR)
        lum_full = luminance(hdr)

        fitted = 0
        for q in quads:
            mask, _ = rasterize_spherical_quad(q.corners_dirs, H, W)
            if not mask.any():
                continue
            fit = rect_from_quad(
                q.corners_dirs,
                mask,
                distance,
                self._scene_scale,
                lum_full=lum_full,
                treat_as_window=q.is_window,
            )
            new_corners = rect_to_corner_dirs(fit)
            # Push the snapped corners + fitted flag through the viewer
            # (single setter handles geometry + outline + handle refresh).
            self.viewer.set_quad_fitted(q.name, True, new_corners=new_corners)
            self.panel.set_quad_fitted(q.name, True)
            fitted += 1
        self._set_status(f"Fit {fitted} quad(s) to rigid rect.")

    def _on_quad_fit_changed(self, name: str, fitted: bool) -> None:
        """Viewer auto-invalidated the rect-fit (e.g. user dragged a corner).
        Mirror the new state onto the panel row so the ✓ disappears."""
        self.panel.set_quad_fitted(name, fitted)

    def _on_add_quad_requested(self):
        if self._hdr is None:
            QMessageBox.information(self, "Add quad", "Open an EXR first.")
            return
        if self.viewer.is_add_mode():
            self.viewer.cancel_add_mode()
        else:
            self.viewer.start_add_mode()

    def _on_add_mode_changed(self, active: bool):
        self.panel.set_add_mode_active(active)
        if active:
            self._set_status(
                "Add mode — click 4 corners of the light. Esc to cancel."
            )
        else:
            if self._exr_path is not None:
                self._set_status(
                    f"{self._exr_path.name}  ·  {self._hdr.shape[1]}×{self._hdr.shape[0]}"
                )

    def _on_quad_selected(self, name: str):
        # Coming from viewer click
        self.panel.set_selected(name or None)
        self._refresh_window_checkbox(name)

    def _on_panel_selected(self, name: str):
        # Coming from panel list — sync the visual highlight in viewer
        self.viewer._set_selected(name or None)
        self._refresh_window_checkbox(name)

    def _on_delete_quad(self, name: str):
        self.viewer.remove_quad(name)
        self.panel.remove_quad(name)
        self._refresh_window_checkbox(self.viewer.selected() or "")

    def _on_rename_quad(self, old_name: str, new_name: str):
        actual = self.viewer.rename_quad(old_name, new_name)
        # Reflect the final name back in the panel (might be munged on collision).
        self.panel.rename(old_name, actual)
        if actual != new_name:
            self._set_status(
                f"Name '{new_name}' was taken; using '{actual}'."
            )

    def _delete_selected(self):
        sel = self.viewer.selected()
        if not sel:
            return
        self._on_delete_quad(sel)

    # ---------- bake ----------

    # ---------- project file (save / open / restore) ----------

    def _save_project_dialog(self):
        if self._exr_path is None:
            QMessageBox.information(self, "Save project", "Open an EXR first.")
            return
        default = default_project_path(self._exr_path)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save env2lgt project",
            str(default),
            "env2lgt project (*.env2lgt.json);;JSON (*.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            self._save_project_to(Path(path))
            self._set_status(f"Saved: {path}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Save project", str(e))

    def _open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open env2lgt project",
            "",
            "env2lgt project (*.env2lgt.json *.json);;All files (*.*)",
        )
        if not path:
            return
        try:
            proj = load_project(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open project", f"Could not parse {path}:\n{e}")
            return
        # Load the EXR the project points to (will reset state). If it's not
        # at the recorded path, fall back to looking for the same filename
        # next to the project file.
        exr_path = Path(proj.source_exr)
        if not exr_path.is_file():
            alt = Path(path).parent / exr_path.name
            if alt.is_file():
                exr_path = alt
            else:
                QMessageBox.warning(
                    self,
                    "Open project",
                    f"Source EXR not found:\n  {proj.source_exr}\n"
                    "Move/restore the EXR or save a new project.",
                )
                return
        self._load_exr(exr_path)
        # `_load_exr` also probes for a sibling project and may prompt to
        # restore from it (could be the same file we just opened, or a different
        # one). Either way, force-apply the project the user explicitly opened.
        self._apply_project_state(proj)
        self._set_status(
            f"Project loaded: {Path(path).name}  ·  {len(proj.quads)} quad(s)"
        )

    def _save_project_to(self, path: Path) -> None:
        if self._exr_path is None:
            raise RuntimeError("No EXR loaded.")
        proj = project_from_app_state(
            source_exr=self._exr_path,
            quads=self.viewer.quads(),
            scene_scale=self._scene_scale,
            yaw_offset_deg=self._yaw_offset_deg,
            exposure_ev=self._exposure,
            dome_rotate_y_deg=self.panel.opt_dome_rotate.value(),
            depth_backend=self._depth_backend,
            export_opts=self.panel.export_state(),
            exposure_offset_ev=self._exposure_offset,
            wb_kelvin=self._wb_kelvin,
            wb_tint=self._wb_tint,
            input_colorspace=self._input_cs,
            output_colorspace=self._output_cs,
            ocio_display=self._ocio_display,
            ocio_view=self._ocio_view,
            cc_state=self._chart_state_dict(),
        )
        save_project(path, proj)

    def _chart_state_dict(self) -> dict:
        """Snapshot of the colour-checker chart + solved correction."""
        dirs = self.viewer.chart_dirs()
        return {
            "corners_dirs": [] if dirs is None else dirs.tolist(),
            "matrix": [] if self._cc_matrix is None else self._cc_matrix.tolist(),
            "fit_mode": self._cc_fit_mode,
            "target_name": self._cc_target_name,
        }

    def _apply_project_state(self, proj: Project) -> None:
        """Restore quads + scene + export state from a parsed Project. Assumes
        the matching EXR is already loaded (state cleared by _load_exr)."""
        # Scene state
        self._scene_scale = float(proj.scene.scene_scale)
        self.panel.opt_scene_scale.setValue(self._scene_scale)
        self._yaw_offset_deg = float(proj.scene.yaw_offset_deg)
        self._yaw_label.setText(f" {self._yaw_offset_deg:+6.1f}° ")
        # Reposition the yaw slider to match (signal-blocked to avoid a redundant refresh).
        try:
            val = int(round(self._yaw_offset_deg * 10.0))
            self._yaw_slider.blockSignals(True)
            self._yaw_slider.setValue(val)
            self._yaw_slider.blockSignals(False)
        except Exception:
            pass
        self._exposure = float(proj.scene.exposure_ev)
        self._exposure_label.setText(f" {self._exposure:+.1f} EV ")
        # Baseline exposure + white balance.
        self._exposure_offset = float(proj.scene.exposure_offset_ev)
        self._wb_kelvin = float(proj.scene.wb_kelvin)
        self._wb_tint = float(proj.scene.wb_tint)
        self.exposure_panel.set_exposure_offset(self._exposure_offset)
        self.exposure_panel.set_wb(self._wb_kelvin, self._wb_tint)
        self._recompute_wb()
        # Colour management. Unknown colorspace names from a project authored
        # against a different backend are tolerated — the combos only accept
        # names that exist in the active backend, and conversion functions
        # treat unknown names as identity.
        if proj.scene.input_colorspace:
            self._input_cs = proj.scene.input_colorspace
        if proj.scene.output_colorspace:
            self._output_cs = proj.scene.output_colorspace
        self.exposure_panel.populate_colorspaces(
            color.colorspace_names(), self._input_cs, self._output_cs
        )
        self._hdr_display = self._to_working_display(self._hdr_display_src)
        disp = proj.scene.ocio_display
        view = proj.scene.ocio_view
        if disp and disp in color.displays():
            self._display_combo.blockSignals(True)
            self._display_combo.setCurrentText(disp)
            self._display_combo.blockSignals(False)
            self._ocio_display = disp
            self._refill_view_combo()
            if view and view in color.views(disp):
                self._view_combo.blockSignals(True)
                self._view_combo.setCurrentText(view)
                self._view_combo.blockSignals(False)
                self._ocio_view = view
                self._rebuild_display_cpu()
        # Colour-checker chart + solved correction.
        cc = proj.colorchecker
        if cc.corners_dirs:
            self.viewer.set_chart(np.asarray(cc.corners_dirs, dtype=np.float64))
        self._cc_matrix = (
            np.asarray(cc.matrix, dtype=np.float32) if cc.matrix else None
        )
        self._cc_fit_mode = cc.fit_mode or "matrix"
        self._cc_target_name = cc.target_name or "Built-in CC24"
        self.exposure_panel.set_fit_mode(self._cc_fit_mode)
        self.exposure_panel.set_chart_status(
            has_chart=bool(cc.corners_dirs),
            applied=self._cc_matrix is not None,
        )
        self.exposure_panel.set_correction_available(self._cc_matrix is not None)
        # Depth backend (tolerate an unknown name from a newer/edited project).
        backend = (proj.scene.depth_backend or "da2").strip().lower()
        if backend not in AVAILABLE_BACKENDS:
            backend = "da2"
        self._depth_backend = backend
        idx = self._backend_combo.findData(backend)
        if idx >= 0:
            self._backend_combo.blockSignals(True)
            self._backend_combo.setCurrentIndex(idx)
            self._backend_combo.blockSignals(False)
        # Apply export options (including dome rotation + output path)
        ex = {
            "dome": proj.export.dome,
            "rect": proj.export.rect,
            "usd": proj.export.usd,
            "depth_exr": proj.export.depth_exr,
            "depth_mesh": proj.export.depth_mesh,
            "masks": proj.export.masks,
            "output_dir": proj.export.output_dir,
            "dome_rotate_y_deg": proj.scene.dome_rotate_y_deg,
            "geom_inflation_pct": proj.export.geom_inflation_pct,
            "open_sky": proj.export.open_sky,
        }
        self.panel.apply_export_state(ex)
        # Quads
        for q in proj.quads:
            lq = LightQuad(
                name=q.name,
                corners_dirs=np.asarray(q.corners_dirs, dtype=np.float64),
                is_window=bool(getattr(q, "is_window", False)),
                source=str(getattr(q, "source", "user")),
                locked=bool(getattr(q, "locked", False)),
            )
            self.viewer.add_quad(lq)
            self.panel.add_quad(lq)
        if proj.quads:
            self.viewer.select_by_name(proj.quads[-1].name)
        self._refresh_view()

    def _on_preview(self):
        """Run the pipeline with all writes off, then show a summary dialog.
        Reuses the daemon-cached distance.exr next to the source if it exists,
        so a preview after a bake is near-instant."""
        if self._exr_path is None or self._hdr is None:
            QMessageBox.warning(self, "Preview", "Open an EXR first.")
            return
        quads = self.viewer.quads()
        if not quads:
            QMessageBox.information(self, "Preview", "Add at least one quad first.")
            return
        # Use a scratch dir under the source EXR's folder so the distance.exr
        # cache lands somewhere reusable. We disable every write_* so nothing
        # else gets written.
        cache_dir = self._exr_path.parent / ".env2lgt_cache"
        quad_specs = [
            QuadSpec(name=q.name, corners_dirs=q.corners_dirs, is_window=q.is_window)
            for q in quads
        ]
        opts = BakeOptions(
            write_dome=False,
            write_rects=False,
            write_usd=False,
            write_depth_exr=False,
            write_depth_mesh=False,
            write_mask_json=False,
            depth_backend=self._depth_backend,
            scene_scale=self._scene_scale,
            yaw_offset_deg=self._yaw_offset_deg,
            exposure_offset_ev=self._exposure_offset,
            wb_scale=tuple(float(c) for c in self._wb_scale),
            cc_matrix=self._cc_matrix,
            input_colorspace=self._input_cs,
            output_colorspace=self._output_cs,
        )

        self._thread = QThread(self)
        self._worker = BakeWorker(str(self._exr_path), str(cache_dir), quad_specs, opts)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_bake_progress)
        self._worker.finished.connect(self._on_preview_finished)
        self._worker.failed.connect(self._on_bake_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._set_status("Preview running…")
        self._thread.start()

    def _on_preview_finished(self, summary: dict):
        self._progress.setVisible(False)
        self._set_status("Preview ready.")
        self._show_preview_dialog(summary)

    def _show_preview_dialog(self, summary: dict):
        from PySide6.QtWidgets import (
            QDialog,
            QDialogButtonBox,
            QHeaderView,
            QLabel,
            QTableWidget,
            QTableWidgetItem,
            QVBoxLayout,
        )

        dlg = QDialog(self)
        dlg.setWindowTitle("Bake preview")
        dlg.resize(820, 380)
        lay = QVBoxLayout(dlg)
        n = len(summary.get("rect_lights", []))
        head = QLabel(
            f"<b>{n} rect light(s)</b>  ·  scene scale {self._scene_scale:.3f} m/u  ·  "
            "(no files written)"
        )
        lay.addWidget(head)
        cols = ["name", "center (m)", "size (m)", "normal", "inliers", "intensity"]
        table = QTableWidget(0, len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        for r in summary.get("rect_fits", []):
            row = table.rowCount()
            table.insertRow(row)
            if "skipped" in r:
                table.setItem(row, 0, QTableWidgetItem(r["name"]))
                table.setItem(row, 1, QTableWidgetItem(f"SKIPPED: {r['skipped']}"))
                continue
            c = r["center"]; n_ = r["normal"]; s = r["size"]
            table.setItem(row, 0, QTableWidgetItem(r["name"]))
            table.setItem(row, 1, QTableWidgetItem(f"[{c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}]"))
            table.setItem(row, 2, QTableWidgetItem(f"{s[0]:.2f} × {s[1]:.2f}"))
            table.setItem(row, 3, QTableWidgetItem(f"[{n_[0]:+.2f}, {n_[1]:+.2f}, {n_[2]:+.2f}]"))
            table.setItem(row, 4, QTableWidgetItem(f"{r['inlier_ratio']:.2f}"))
            table.setItem(row, 5, QTableWidgetItem(f"{r['intensity']:.3f}"))
        lay.addWidget(table)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def _on_bake(self, opts: dict):
        if self._exr_path is None or self._hdr is None:
            QMessageBox.warning(self, "Bake", "Open an EXR first.")
            return
        # No quads is a valid, intentional case — bake the mesh + a dome-only
        # rig (handy for outdoor scenes: geometry to catch shadows/reflections,
        # the sun and everything else left in the dome). No confirm needed.
        quads = self.viewer.quads()

        out_dir_str = opts.get("output_dir", "").strip()
        if not out_dir_str:
            QMessageBox.warning(self, "Bake", "Set the output path in the panel first.")
            return
        out_dir = Path(out_dir_str)

        if out_dir.exists():
            conflicts = [
                p.name for p in (out_dir / "lightrig.usda", out_dir / "dome.exr")
                if p.exists()
            ]
            conflicts += [p.name for p in out_dir.glob("rect_*.exr")]
            if conflicts:
                ret = QMessageBox.warning(
                    self,
                    "Output exists",
                    f"{out_dir} already contains:\n  "
                    + "\n  ".join(conflicts[:10])
                    + ("\n  …" if len(conflicts) > 10 else "")
                    + "\n\nOverwrite?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                )
                if ret != QMessageBox.StandardButton.Yes:
                    return
        out_dir.mkdir(parents=True, exist_ok=True)

        quad_specs = [
            QuadSpec(name=q.name, corners_dirs=q.corners_dirs, is_window=q.is_window)
            for q in quads
        ]
        bake_opts = BakeOptions(
            write_dome=opts["dome"],
            write_rects=opts["rect"],
            write_usd=opts["usd"],
            write_depth_exr=opts["depth_exr"],
            write_depth_mesh=opts["depth_mesh"],
            write_mask_json=opts["masks"],
            depth_backend=self._depth_backend,
            scene_scale=self._scene_scale,
            yaw_offset_deg=self._yaw_offset_deg,
            dome_rotate_y_deg=opts.get("dome_rotate_y_deg", -180.0),
            geom_inflation=1.0 + opts.get("geom_inflation_pct", 2.5) / 100.0,
            open_sky=bool(opts.get("open_sky", True)),
            exposure_offset_ev=self._exposure_offset,
            wb_scale=tuple(float(c) for c in self._wb_scale),
            cc_matrix=self._cc_matrix,
            input_colorspace=self._input_cs,
            output_colorspace=self._output_cs,
        )

        self._thread = QThread(self)
        self._worker = BakeWorker(str(self._exr_path), str(out_dir), quad_specs, bake_opts)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_bake_progress)
        self._worker.finished.connect(self._on_bake_finished)
        self._worker.failed.connect(self._on_bake_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._set_status("Baking…")
        self._thread.start()

    def _on_bake_progress(self, stage: str, frac: float):
        self._progress.setValue(int(frac * 100))
        self._set_status(f"Baking: {stage}")

    def _on_bake_finished(self, summary: dict):
        self._progress.setVisible(False)
        usd = summary.get("usd")
        if usd:
            self._last_usd = Path(usd)
        mesh = summary.get("mesh")
        self._last_mesh = Path(mesh) if mesh else None
        n = len(summary.get("rect_lights", []))
        # Auto-save the project next to the source EXR so the user's work
        # survives a crash / future session. Silent — failures are logged
        # but don't block the bake summary dialog.
        if self._exr_path is not None:
            try:
                self._save_project_to(default_project_path(self._exr_path))
            except Exception as e:  # noqa: BLE001
                import sys
                print(f"[env2lgt] autosave failed: {e}", file=sys.stderr)
        self._set_status(f"Bake done: {usd}  ({n} rect lights)")
        QMessageBox.information(
            self,
            "Bake complete",
            f"USD: {usd}\nDome: {summary.get('dome')}\nRect lights: {n}\n\nOpen via Tools → usdview.",
        )

    def _on_bake_failed(self, msg: str):
        self._progress.setVisible(False)
        self._set_status("Bake failed.")
        QMessageBox.critical(self, "Bake failed", msg)

    def closeEvent(self, event):  # noqa: N802
        # Politely kill any depth-backend daemon before Qt exits.
        try:
            shutdown_all()
        except Exception:
            pass
        super().closeEvent(event)

    def _write_combined_layer(self) -> Path | None:
        """Write a tiny stage that sublayers the light rig over the depth mesh,
        so usdview can show both at once. Returns the combined layer's path."""
        if self._last_usd is None or self._last_mesh is None:
            return None
        out = self._last_usd.parent / "usdview_layered.usda"
        out.write_text(
            "#usda 1.0\n"
            "(\n"
            "    subLayers = [\n"
            f"        @./{self._last_usd.name}@,\n"
            f"        @./{self._last_mesh.name}@\n"
            "    ]\n"
            ")\n"
        )
        return out

    def _launch_usdview(self, kind: str = "light"):
        """Open a baked stage in usdview. `kind` is 'light', 'mesh', or 'both'
        (the light rig layered over the depth mesh)."""
        if kind == "mesh":
            target, label = self._last_mesh, "depth mesh"
        elif kind == "both":
            target, label = self._write_combined_layer(), "light rig + depth mesh"
        else:
            target, label = self._last_usd, "light rig"
        if target is None or not Path(target).exists():
            QMessageBox.information(
                self, "usdview",
                f"No baked {label} to open — bake a rig first "
                "(the depth mesh needs its export option enabled).",
            )
            return
        py = Path(sys.executable)
        usdview = py.parent / "Library" / "bin" / "usdview"
        if not usdview.exists():
            QMessageBox.warning(self, "usdview", f"usdview not found at {usdview}")
            return
        try:
            subprocess.Popen([str(py), str(usdview), str(target)], shell=False)
        except OSError as e:
            QMessageBox.critical(self, "usdview", f"Failed to launch:\n{e}")


def main():
    app = QApplication(sys.argv)
    apply_theme(app)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
