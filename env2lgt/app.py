"""env2lgt — PySide6 application entry point.

Single-view UX: equirect panorama. Add a quad by clicking 4 corners (cursor in
"Add" mode). Drag the yellow vertex handles to refine. Bake.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSlider,
    QStatusBar,
    QToolBar,
)

from env2lgt.bake import BakeOptions, QuadSpec, bake
from env2lgt.depth import da2_runner
from env2lgt.io import load_latlong, to_display_qimage
from env2lgt.ui.light_panel import LightPanel
from env2lgt.ui.viewer import LightQuad, PanoramaViewer


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("env2lgt — HDRI → USD light rig")
        self.resize(1600, 900)
        self.setAcceptDrops(True)

        self._hdr: np.ndarray | None = None
        self._exr_path: Path | None = None
        self._exposure: float = 0.0
        # DA-2 returns scale-invariant distance ~[0.3, 1.5]. A median indoor
        # room is ~3-10 m across, so 10 m/u is a sane default. The user can
        # adjust per scene via the toolbar slider.
        self._scene_scale: float = 10.0
        self._yaw_offset_deg: float = 0.0
        self._last_usd: Path | None = None
        self._worker: BakeWorker | None = None
        self._thread: QThread | None = None

        self.viewer = PanoramaViewer(self)
        self.setCentralWidget(self.viewer)
        self.viewer.quad_committed.connect(self._on_quad_committed)
        self.viewer.quad_selected.connect(self._on_quad_selected)
        self.viewer.add_mode_changed.connect(self._on_add_mode_changed)

        self.panel = LightPanel(self)
        dock = QDockWidget("Lights", self)
        dock.setWidget(self.panel)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.panel.delete_quad.connect(self._on_delete_quad)
        self.panel.select_quad.connect(self._on_panel_selected)
        self.panel.rename_quad.connect(self._on_rename_quad)
        self.panel.add_quad_requested.connect(self._on_add_quad_requested)
        self.panel.bake_requested.connect(self._on_bake)
        self.panel.preview_requested.connect(self._on_preview)

        self._build_menu()
        self._build_toolbar()
        self.setStatusBar(QStatusBar(self))
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setFixedWidth(280)
        self.statusBar().addPermanentWidget(self._progress)
        self.statusBar().showMessage("Open or drag an EXR latlong panorama to begin.")

    # ---------- menus / toolbars ----------

    def _build_menu(self):
        m_file = self.menuBar().addMenu("&File")
        act_open = QAction("&Open EXR…", self, shortcut="Ctrl+O")
        act_open.triggered.connect(self._open_exr_dialog)
        m_file.addAction(act_open)
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
        act_usdview = QAction("Open last bake in usdview", self)
        act_usdview.triggered.connect(self._launch_usdview)
        m_tools.addAction(act_usdview)

    def _build_toolbar(self):
        tb = QToolBar("View", self)
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Exposure "))
        exp_slider = QSlider(Qt.Orientation.Horizontal)
        exp_slider.setRange(-60, 60)
        exp_slider.setValue(0)
        exp_slider.setFixedWidth(180)
        exp_slider.valueChanged.connect(self._on_exposure)
        tb.addWidget(exp_slider)
        self._exposure_label = QLabel(" 0.0 EV ")
        tb.addWidget(self._exposure_label)

        tb.addSeparator()
        tb.addWidget(QLabel(" Scene scale "))
        scale_slider = QSlider(Qt.Orientation.Horizontal)
        scale_slider.setRange(-200, 200)        # tenths of log10(m/u); -200 = 0.01, 0 = 1.0, +200 = 100
        scale_slider.setValue(100)              # 10 m/u default — sensible for indoor scenes
        scale_slider.setFixedWidth(180)
        scale_slider.valueChanged.connect(self._on_scale)
        tb.addWidget(scale_slider)
        self._scale_label = QLabel(" 10.00 m/u ")
        tb.addWidget(self._scale_label)

        tb.addSeparator()
        tb.addWidget(QLabel(" Yaw offset "))
        self._yaw_slider = QSlider(Qt.Orientation.Horizontal)
        self._yaw_slider.setRange(-1800, 1800)  # tenths of a degree, -180.0..+180.0
        self._yaw_slider.setValue(0)
        self._yaw_slider.setFixedWidth(220)
        self._yaw_slider.valueChanged.connect(self._on_yaw_offset)
        tb.addWidget(self._yaw_slider)
        self._yaw_label = QLabel("  0.0° ")
        tb.addWidget(self._yaw_label)
        from PySide6.QtWidgets import QPushButton

        yaw_reset_btn = QPushButton("Reset")
        yaw_reset_btn.clicked.connect(lambda: self._yaw_slider.setValue(0))
        tb.addWidget(yaw_reset_btn)

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
        self._exr_path = path
        # Reset yaw offset when loading a new EXR — quad placements are
        # absolute so they wouldn't drift, but a stale offset is confusing.
        if hasattr(self, "_yaw_slider"):
            self._yaw_slider.blockSignals(True)
            self._yaw_slider.setValue(0)
            self._yaw_slider.blockSignals(False)
        self._yaw_offset_deg = 0.0
        if hasattr(self, "_yaw_label"):
            self._yaw_label.setText("  0.0° ")
        self.viewer.reset_image()
        # rebuild panel list cleanly
        for name in list(self.viewer._quads.keys()):  # _quads cleared above; loop is no-op but safe
            self.panel.remove_quad(name)
        # Reset the output path to the default for this EXR (always — every new EXR
        # gets its own default; user can still override after).
        default_out = str(path.parent / f"{path.stem}_lightrig")
        self.panel.force_set_output_path(default_out)
        self._refresh_view()
        h, w, _ = self._hdr.shape
        self.statusBar().showMessage(f"{path.name}  ·  {w}×{h}  ·  float32")

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
        if self._hdr is None:
            return
        H, W, _ = self._hdr.shape
        # Convert yaw offset (deg) into integer pano-pixel shift. Positive
        # offset rolls content to the right (the seam moves left).
        offset_px = int(round((self._yaw_offset_deg / 360.0) * W)) % W
        if offset_px != 0:
            display_hdr = np.roll(self._hdr, offset_px, axis=1)
        else:
            display_hdr = self._hdr
        qimg = to_display_qimage(display_hdr, exposure=self._exposure)
        # Tell the viewer the current display offset BEFORE handing it the image,
        # so when it re-projects existing quads onto the new pixmap it uses the
        # right transform.
        self.viewer.set_yaw_offset_px(offset_px)
        self.viewer.set_image(qimg)

    def _on_exposure(self, val: int):
        self._exposure = val / 10.0
        self._exposure_label.setText(f" {self._exposure:+.1f} EV ")
        self._refresh_view()

    def _on_scale(self, val: int):
        self._scene_scale = float(10 ** (val / 100.0))
        self._scale_label.setText(f" {self._scene_scale:.3f} m/u ")

    def _on_yaw_offset(self, val: int):
        self._yaw_offset_deg = val / 10.0
        self._yaw_label.setText(f" {self._yaw_offset_deg:+6.1f}° ")
        self._refresh_view()

    # ---------- quad lifecycle ----------

    def _on_quad_committed(self, q: LightQuad):
        self.panel.add_quad(q)
        self.panel.set_selected(q.name)

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
            self.statusBar().showMessage(
                "Add mode — click 4 corners of the light. Esc to cancel."
            )
        else:
            if self._exr_path is not None:
                self.statusBar().showMessage(
                    f"{self._exr_path.name}  ·  {self._hdr.shape[1]}×{self._hdr.shape[0]}"
                )

    def _on_quad_selected(self, name: str):
        # Coming from viewer click
        self.panel.set_selected(name or None)

    def _on_panel_selected(self, name: str):
        # Coming from panel list — sync the visual highlight in viewer
        self.viewer._set_selected(name or None)

    def _on_delete_quad(self, name: str):
        self.viewer.remove_quad(name)
        self.panel.remove_quad(name)

    def _on_rename_quad(self, old_name: str, new_name: str):
        actual = self.viewer.rename_quad(old_name, new_name)
        # Reflect the final name back in the panel (might be munged on collision).
        self.panel.rename(old_name, actual)
        if actual != new_name:
            self.statusBar().showMessage(
                f"Name '{new_name}' was taken; using '{actual}'."
            )

    def _delete_selected(self):
        sel = self.viewer.selected()
        if not sel:
            return
        self._on_delete_quad(sel)

    # ---------- bake ----------

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
        quad_specs = [QuadSpec(name=q.name, corners_dirs=q.corners_dirs) for q in quads]
        opts = BakeOptions(
            write_dome=False,
            write_rects=False,
            write_usd=False,
            write_depth_exr=False,
            write_depth_mesh=False,
            write_mask_json=False,
            scene_scale=self._scene_scale,
            yaw_offset_deg=self._yaw_offset_deg,
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
        self.statusBar().showMessage("Preview running…")
        self._thread.start()

    def _on_preview_finished(self, summary: dict):
        self._progress.setVisible(False)
        self.statusBar().showMessage("Preview ready.")
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
        quads = self.viewer.quads()
        if not quads:
            ret = QMessageBox.question(
                self,
                "Bake",
                "No quads drawn. Bake dome-only (no rect lights)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

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

        quad_specs = [QuadSpec(name=q.name, corners_dirs=q.corners_dirs) for q in quads]
        bake_opts = BakeOptions(
            write_dome=opts["dome"],
            write_rects=opts["rect"],
            write_usd=opts["usd"],
            write_depth_exr=opts["depth_exr"],
            write_depth_mesh=opts["depth_mesh"],
            write_mask_json=opts["masks"],
            scene_scale=self._scene_scale,
            yaw_offset_deg=self._yaw_offset_deg,
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
        self.statusBar().showMessage("Baking…")
        self._thread.start()

    def _on_bake_progress(self, stage: str, frac: float):
        self._progress.setValue(int(frac * 100))
        self.statusBar().showMessage(f"Baking: {stage}")

    def _on_bake_finished(self, summary: dict):
        self._progress.setVisible(False)
        usd = summary.get("usd")
        if usd:
            self._last_usd = Path(usd)
        n = len(summary.get("rect_lights", []))
        self.statusBar().showMessage(f"Bake done: {usd}  ({n} rect lights)")
        QMessageBox.information(
            self,
            "Bake complete",
            f"USD: {usd}\nDome: {summary.get('dome')}\nRect lights: {n}\n\nOpen via Tools → usdview.",
        )

    def _on_bake_failed(self, msg: str):
        self._progress.setVisible(False)
        self.statusBar().showMessage("Bake failed.")
        QMessageBox.critical(self, "Bake failed", msg)

    def closeEvent(self, event):  # noqa: N802
        # Politely kill the DA-2 daemon before Qt exits.
        try:
            da2_runner.shutdown()
        except Exception:
            pass
        super().closeEvent(event)

    def _launch_usdview(self):
        if self._last_usd is None or not self._last_usd.exists():
            QMessageBox.information(self, "usdview", "Bake a rig first, or open one manually.")
            return
        py = Path(sys.executable)
        usdview = py.parent / "Library" / "bin" / "usdview"
        if not usdview.exists():
            QMessageBox.warning(self, "usdview", f"usdview not found at {usdview}")
            return
        try:
            subprocess.Popen([str(py), str(usdview), str(self._last_usd)], shell=False)
        except OSError as e:
            QMessageBox.critical(self, "usdview", f"Failed to launch:\n{e}")


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
