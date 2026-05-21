"""Side panel: quad list, add/delete buttons, output path, export options."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from env2lgt.ui.viewer import LightQuad


# Item-data roles used on the quad-list rows.
_ROLE_NAME = 0x0100   # canonical name (tracks renames)
_ROLE_LOCK = 0x0101   # last-seen lock check-state (distinguishes lock vs rename)
_ROLE_FITTED = 0x0102 # rigid-rect fit state, used to re-apply the ✓ on rename

# Default auto-detect parameters — the single source of truth, used both to
# initialise the spinboxes and to restore them via the Reset button.
_DETECT_DEFAULTS = {
    "threshold": 3.0,    # %
    "max_lights": 12,
    "blur": 1.0,         # degrees
    "min_size": 1.0,     # degrees
    "merge": 1.0,        # degrees
    "floor": True,
}


# Match USD prim-name / Maya / Houdini node-name rules: only ASCII letters,
# digits, and underscore. Names can't start with a digit.
_INVALID_CHARS_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_name(s: str) -> str:
    """Coerce user-entered names to USD/Maya/Houdini-friendly identifiers.

    - Trim surrounding whitespace
    - Replace internal whitespace with single underscores
    - Strip any character outside [A-Za-z0-9_]
    - Prepend '_' if the result starts with a digit (USD prim-name rule)
    - Return empty string if nothing valid remains (caller should revert)
    """
    s = s.strip()
    s = re.sub(r"\s+", "_", s)              # any whitespace run → single underscore
    s = _INVALID_CHARS_RE.sub("", s)        # drop everything else
    if not s:
        return ""
    if s[0].isdigit():
        s = "_" + s
    return s


class LightPanel(QWidget):
    delete_quad = Signal(str)
    select_quad = Signal(str)
    rename_quad = Signal(str, str)   # old_name, new_name
    window_toggled = Signal(str, bool)  # quad name, is_window
    lock_toggled = Signal(str, bool)    # quad name, locked
    add_quad_requested = Signal()
    propose_quads_requested = Signal(dict)  # auto-detect params
    fit_to_rect_requested = Signal()        # depth-snap all unfitted quads
    key_preview_changed = Signal(bool)      # show/update the key-mask preview
    bake_requested = Signal(dict)
    preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        list_box = QGroupBox("Light quads", self)
        lb = QVBoxLayout(list_box)
        self._list = QListWidget()
        self._list.setMinimumHeight(200)
        self._list.setEditTriggers(
            QListWidget.EditTrigger.DoubleClicked | QListWidget.EditTrigger.EditKeyPressed
        )
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.itemChanged.connect(self._on_item_changed)
        self._suppress_item_changed = False
        lb.addWidget(self._list)
        self._add_btn = QPushButton("Add quad   (click 4 corners)")
        self._add_btn.setCheckable(True)
        self._add_btn.clicked.connect(self._on_add_clicked)
        lb.addWidget(self._add_btn)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(self._on_delete)
        lb.addWidget(del_btn)

        lb.addWidget(self._build_autodetect_group())

        # "Fit to rect light" — depth-snaps every unfitted quad to a rigid
        # rectangle on its surface plane. See app._on_fit_to_rect for the
        # algorithm; the list shows a ✓ next to fitted quads, and any handle
        # drag drops the ✓ so the user knows a re-fit is needed.
        self._fit_btn = QPushButton("Fit to rect light  (depth-snap)")
        self._fit_btn.setToolTip(
            "Project each quad onto its surface plane (computed from the depth "
            "estimate) and snap it to a true rectangle — orthogonal axes, no "
            "shear. What you see becomes what the bake authors. First press "
            "may pause while depth is estimated; subsequent presses are instant. "
            "Dragging a corner invalidates the fit; press again to refresh."
        )
        self._fit_btn.clicked.connect(self.fit_to_rect_requested)
        lb.addWidget(self._fit_btn)

        # Per-quad window/portal flag — applies to the selected quad.
        self._sel_name = ""
        self._window_cb = QCheckBox("Window / portal — sit on wall depth")
        self._window_cb.setToolTip(
            "For windows / skylights: keep the rect light on the wall plane "
            "instead of sliding it in to the bright region (the bright pixels "
            "are distant sky seen through the opening). Lets the rect double "
            "as a portal."
        )
        self._window_cb.setEnabled(False)
        self._window_cb.toggled.connect(self._on_window_toggled)
        lb.addWidget(self._window_cb)
        layout.addWidget(list_box)

        out_box = QGroupBox("Output", self)
        ob = QVBoxLayout(out_box)
        path_row = QHBoxLayout()
        self._out_path_edit = QLineEdit()
        self._out_path_edit.setPlaceholderText("(open an EXR — defaults to <hdri_dir>/<name>_lightrig)")
        path_row.addWidget(self._out_path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)
        ob.addLayout(path_row)
        layout.addWidget(out_box)

        export_box = QGroupBox("Export", self)
        eb = QVBoxLayout(export_box)
        self.opt_dome = QCheckBox("Dome light texture (dome.exr)")
        self.opt_rect = QCheckBox("Rect light textures (light_<i>.exr)")
        self.opt_usd = QCheckBox("USD light rig (lightrig.usda)")
        self.opt_depth_exr = QCheckBox("Depth panorama (depth.exr)  [debug]")
        self.opt_depth_mesh = QCheckBox("Depth mesh USD (panorama_geo.usda)")
        self.opt_masks = QCheckBox("Mask JSON sidecar")
        for cb, default in (
            (self.opt_dome, True),
            (self.opt_rect, True),
            (self.opt_usd, True),
            (self.opt_depth_exr, False),
            (self.opt_depth_mesh, True),
            (self.opt_masks, True),
        ):
            cb.setChecked(default)
            eb.addWidget(cb)
        # Depth-mesh sub-option: drop the sky faces for outdoor scenes.
        self.opt_open_sky = QCheckBox("    ↳ Open sky (drop far/sky faces)")
        self.opt_open_sky.setChecked(True)
        self.opt_open_sky.setToolTip(
            "For outdoor scenes: delete the depth-mesh faces that sit at "
            "infinity (the sky), leaving the mesh open so the real dome / 3D "
            "sky shows through. Only affects the depth mesh USD."
        )
        eb.addWidget(self.opt_open_sky)
        # Dome rotation knob — depends on the target renderer's convention.
        # Empirically -180° for Storm/usdview. Quick presets buttons next to it.
        dome_row = QHBoxLayout()
        dome_row.addWidget(QLabel("Dome rotateY:"))
        self.opt_dome_rotate = QDoubleSpinBox()
        self.opt_dome_rotate.setRange(-360.0, 360.0)
        self.opt_dome_rotate.setSingleStep(90.0)
        self.opt_dome_rotate.setDecimals(1)
        self.opt_dome_rotate.setSuffix(" °")
        self.opt_dome_rotate.setValue(-180.0)
        self.opt_dome_rotate.setToolTip(
            "Y rotation applied to the dome light prim so its texture lines up "
            "with the rect lights. Renderer-dependent (Storm: -180; Karma/RenderMan "
            "may differ). Use the preset buttons or type a value."
        )
        dome_row.addWidget(self.opt_dome_rotate, stretch=1)
        for preset in (-180.0, -90.0, 0.0, 90.0):
            btn = QPushButton(f"{int(preset)}°")
            btn.setFixedWidth(48)
            btn.clicked.connect(lambda _=False, v=preset: self.opt_dome_rotate.setValue(v))
            dome_row.addWidget(btn)
        eb.addLayout(dome_row)

        # Depth-mesh inflation — scales the estimated geometry slightly outward
        # so it doesn't sit coplanar with (and z-fight / intersect) the rect
        # lights, which are placed on the actual light surfaces.
        geom_row = QHBoxLayout()
        geom_row.addWidget(QLabel("Geometry inflation:"))
        self.opt_geom_inflation = QDoubleSpinBox()
        self.opt_geom_inflation.setRange(0.0, 25.0)
        self.opt_geom_inflation.setSingleStep(0.5)
        self.opt_geom_inflation.setDecimals(1)
        self.opt_geom_inflation.setSuffix(" %")
        self.opt_geom_inflation.setValue(2.5)
        self.opt_geom_inflation.setToolTip(
            "Scale the estimated depth mesh outward by this percentage so it "
            "sits just behind the rect lights instead of intersecting them. "
            "2–3% is usually enough."
        )
        geom_row.addWidget(self.opt_geom_inflation, stretch=1)
        eb.addLayout(geom_row)

        # World scale of the baked rig — metres per scene unit.
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scene scale:"))
        self.opt_scene_scale = QDoubleSpinBox()
        self.opt_scene_scale.setRange(0.001, 1000.0)
        self.opt_scene_scale.setDecimals(3)
        self.opt_scene_scale.setSingleStep(1.0)
        self.opt_scene_scale.setSuffix(" m/u")
        self.opt_scene_scale.setValue(1.0)
        self.opt_scene_scale.setToolTip(
            "World scale of the baked rig — metres per scene unit. DAP depth "
            "is metric (≈1.0); DA² is scale-invariant and typically needs a "
            "larger value (≈100)."
        )
        scale_row.addWidget(self.opt_scene_scale, stretch=1)
        eb.addLayout(scale_row)

        self._preview_btn = QPushButton("Preview (no files)")
        self._preview_btn.clicked.connect(self._on_preview)
        eb.addWidget(self._preview_btn)
        self._bake_btn = QPushButton("Bake light rig")
        self._bake_btn.setObjectName("primary")
        self._bake_btn.clicked.connect(self._on_bake)
        eb.addWidget(self._bake_btn)
        layout.addWidget(export_box)

        layout.addStretch(1)

    # ---------- auto-detect ----------

    def _build_autodetect_group(self) -> QGroupBox:
        """'Auto-detect lights' group — just the action buttons. The detection
        parameters live in a separate settings window."""
        box = QGroupBox("Auto-detect lights", self)
        row = QHBoxLayout(box)
        self._propose_btn = QPushButton("Detect lights")
        self._propose_btn.setToolTip(
            "Detect lights and add them as quads. Locked quads are kept as-is; "
            "other auto-proposed quads are replaced. Your hand-placed quads are "
            "never touched."
        )
        self._propose_btn.clicked.connect(self._on_propose)
        row.addWidget(self._propose_btn, stretch=1)
        self._detect_settings_btn = QPushButton("Settings…")
        self._detect_settings_btn.setFixedWidth(88)
        self._detect_settings_btn.setToolTip(
            "Open the light-detection settings window."
        )
        self._detect_settings_btn.clicked.connect(self._open_detect_settings)
        row.addWidget(self._detect_settings_btn)
        self._build_detection_dialog()
        return box

    def _open_detect_settings(self) -> None:
        self._detect_dialog.show()
        self._detect_dialog.raise_()
        self._detect_dialog.activateWindow()

    def _build_detection_dialog(self) -> None:
        """The light-detection parameters, housed in their own window so they
        don't crowd the side panel."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Light detection settings")
        outer = QVBoxLayout(dlg)

        form = QFormLayout()

        self.det_threshold = QDoubleSpinBox()
        self.det_threshold.setRange(0.0, 100.0)
        self.det_threshold.setDecimals(0)
        self.det_threshold.setSingleStep(5.0)
        self.det_threshold.setValue(_DETECT_DEFAULTS["threshold"])
        self.det_threshold.setSuffix(" %")
        self.det_threshold.setToolTip(
            "Luma key: threshold as a percentage of the scene's brightest "
            "luminance. Pixels above it form a flat white blob. Lower engulfs "
            "more of each light (down its gradient to the dim edges); higher "
            "keeps only the hottest core."
        )
        form.addRow("Brightness", self.det_threshold)

        self.det_max = QSpinBox()
        self.det_max.setRange(1, 64)
        self.det_max.setValue(_DETECT_DEFAULTS["max_lights"])
        self.det_max.setToolTip(
            "Keep at most this many proposals — the brightest win. Lights "
            "below the cap simply stay baked into the dome."
        )
        form.addRow("Max lights", self.det_max)

        self.det_blur = QDoubleSpinBox()
        self.det_blur.setRange(0.0, 10.0)
        self.det_blur.setDecimals(1)
        self.det_blur.setSingleStep(0.5)
        self.det_blur.setValue(_DETECT_DEFAULTS["blur"])
        self.det_blur.setSuffix(" °")
        self.det_blur.setToolTip(
            "Blur before blob detection. Higher = blobbier — bridges window "
            "panes, sheer curtains and beam shadows so the quad engulfs the "
            "whole light instead of fragmenting."
        )
        form.addRow("Blur", self.det_blur)

        self.det_min_size = QDoubleSpinBox()
        self.det_min_size.setRange(0.0, 45.0)
        self.det_min_size.setDecimals(1)
        self.det_min_size.setSingleStep(0.5)
        self.det_min_size.setValue(_DETECT_DEFAULTS["min_size"])
        self.det_min_size.setSuffix(" °")
        self.det_min_size.setToolTip(
            "Discard bright blobs smaller than this angular diameter "
            "(filters specular sparkle / hot pixels)."
        )
        form.addRow("Min size", self.det_min_size)

        self.det_merge = QDoubleSpinBox()
        self.det_merge.setRange(0.0, 90.0)
        self.det_merge.setDecimals(1)
        self.det_merge.setSingleStep(0.5)
        self.det_merge.setValue(_DETECT_DEFAULTS["merge"])
        self.det_merge.setSuffix(" °")
        self.det_merge.setToolTip(
            "Merge lights whose centres are within this angular distance into "
            "a single quad — use it to group clusters like a row of ceiling "
            "spots or a lit-up tree. 0 keeps every light separate."
        )
        form.addRow("Merge dist", self.det_merge)
        outer.addLayout(form)

        self.det_floor = QCheckBox("Suppress floor reflections")
        self.det_floor.setChecked(_DETECT_DEFAULTS["floor"])
        self.det_floor.setToolTip(
            "Ignore bright blobs pointing well below the horizon — usually "
            "sun / light reflections on the floor. Wall mirrors and other "
            "near-horizon reflections are kept."
        )
        outer.addWidget(self.det_floor)

        self.det_preview = QCheckBox("Show key mask")
        self.det_preview.setToolTip(
            "Overlay the luma-key mask on the panorama so you can see what the "
            "Brightness / Blur knobs select — dial Brightness until the mask "
            "covers the lights without flooding walls or clipping the light."
        )
        self.det_preview.toggled.connect(self.key_preview_changed)
        outer.addWidget(self.det_preview)
        # Live-update the preview while dragging Brightness / Blur.
        for sb in (self.det_threshold, self.det_blur):
            sb.valueChanged.connect(
                lambda _=0: self.key_preview_changed.emit(self.det_preview.isChecked())
            )

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._det_reset_btn = QPushButton("Reset")
        self._det_reset_btn.setToolTip(
            "Restore the detection parameters to their defaults."
        )
        self._det_reset_btn.clicked.connect(self._reset_detect_params)
        btn_row.addWidget(self._det_reset_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.hide)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)
        self._detect_dialog = dlg

    def _reset_detect_params(self):
        """Restore every auto-detect control to its default."""
        self.det_threshold.setValue(_DETECT_DEFAULTS["threshold"])
        self.det_max.setValue(_DETECT_DEFAULTS["max_lights"])
        self.det_blur.setValue(_DETECT_DEFAULTS["blur"])
        self.det_min_size.setValue(_DETECT_DEFAULTS["min_size"])
        self.det_merge.setValue(_DETECT_DEFAULTS["merge"])
        self.det_floor.setChecked(_DETECT_DEFAULTS["floor"])

    def detect_params(self) -> dict:
        """Current auto-detect parameters as a kwargs-style dict."""
        return {
            "threshold": self.det_threshold.value() / 100.0,
            "blur_deg": self.det_blur.value(),
            "max_quads": self.det_max.value(),
            "min_diameter_deg": self.det_min_size.value(),
            "merge_distance_deg": self.det_merge.value(),
            "suppress_floor": self.det_floor.isChecked(),
        }

    def _on_propose(self):
        self.propose_quads_requested.emit(self.detect_params())

    # ---------- quad list ----------

    def add_quad(self, q: LightQuad) -> None:
        self._suppress_item_changed = True
        fitted = bool(getattr(q, "is_rect_fitted", False))
        item = QListWidgetItem(self._row_label(q.name, fitted))
        item.setData(_ROLE_NAME, q.name)  # canonical name, used to track renames
        item.setData(_ROLE_FITTED, fitted)
        item.setFlags(
            item.flags()
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        # The row's checkbox is the lock toggle.
        state = Qt.CheckState.Checked if getattr(q, "locked", False) else Qt.CheckState.Unchecked
        item.setCheckState(state)
        item.setData(_ROLE_LOCK, state)
        item.setToolTip("Checkbox = lock. Locked quads survive 'Propose quads'.")
        self._list.addItem(item)
        self._suppress_item_changed = False

    @staticmethod
    def _row_label(name: str, fitted: bool) -> str:
        """List-row label. A leading ✓ marks quads whose corners have been
        depth-snapped to a rigid rectangle (see app._on_fit_to_rect). The ✓
        falls off automatically the next time any vertex is dragged. The
        canonical name lives in _ROLE_NAME — the ✓ is purely decoration and
        is stripped on edit (`sanitize_name` ignores non-ASCII anyway)."""
        return f"✓ {name}" if fitted else name

    def set_quad_fitted(self, name: str, fitted: bool) -> None:
        """Reflect a rect-fitted state change onto the row's display label."""
        self._suppress_item_changed = True
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_ROLE_NAME) == name:
                it.setData(_ROLE_FITTED, bool(fitted))
                it.setText(self._row_label(name, bool(fitted)))
                break
        self._suppress_item_changed = False

    def set_quad_locked(self, name: str, locked: bool) -> None:
        """Reflect a lock change that originated elsewhere (e.g. auto-lock on
        edit) onto the row's checkbox, without re-emitting lock_toggled."""
        self._suppress_item_changed = True
        state = Qt.CheckState.Checked if locked else Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_ROLE_NAME) == name:
                it.setCheckState(state)
                it.setData(_ROLE_LOCK, state)
                break
        self._suppress_item_changed = False

    def rename(self, old_name: str, new_name: str) -> None:
        """Update the row that owns `old_name` to display `new_name`. Called
        by the app after MainWindow propagates the rename to the viewer.
        Preserves the rect-fitted ✓ prefix by reading _ROLE_FITTED."""
        self._suppress_item_changed = True
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_ROLE_NAME) == old_name:
                fitted = bool(it.data(_ROLE_FITTED))
                it.setText(self._row_label(new_name, fitted))
                it.setData(_ROLE_NAME, new_name)
                break
        self._suppress_item_changed = False

    def remove_quad(self, name: str) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(_ROLE_NAME) == name:
                self._list.takeItem(i)
                return

    def clear_quads(self) -> None:
        """Drop every entry. Used when loading a new EXR."""
        self._suppress_item_changed = True
        self._list.clear()
        self._suppress_item_changed = False

    def set_selected(self, name: str | None) -> None:
        self._list.blockSignals(True)
        if not name:
            self._list.clearSelection()
        else:
            for i in range(self._list.count()):
                if self._list.item(i).data(_ROLE_NAME) == name:
                    self._list.setCurrentRow(i)
                    break
        self._list.blockSignals(False)

    def _on_selection_changed(self):
        item = self._list.currentItem()
        self.select_quad.emit(item.data(_ROLE_NAME) if item else "")

    def set_window_checkbox(self, name: str, is_window: bool) -> None:
        """Point the window checkbox at `name` (empty string → disabled)."""
        self._sel_name = name or ""
        self._window_cb.blockSignals(True)
        self._window_cb.setEnabled(bool(name))
        self._window_cb.setChecked(bool(is_window))
        self._window_cb.blockSignals(False)

    def _on_window_toggled(self, checked: bool):
        if self._sel_name:
            self.window_toggled.emit(self._sel_name, checked)

    def _on_item_changed(self, item: QListWidgetItem):
        if self._suppress_item_changed:
            return
        # itemChanged fires for both rename edits and lock checkbox toggles.
        # A change in check state means the user toggled the lock.
        cur_lock = item.checkState()
        if cur_lock != item.data(_ROLE_LOCK):
            item.setData(_ROLE_LOCK, cur_lock)
            self.lock_toggled.emit(
                item.data(_ROLE_NAME) or "", cur_lock == Qt.CheckState.Checked
            )
            return
        old_name = item.data(_ROLE_NAME) or ""
        typed = item.text()
        new_name = sanitize_name(typed)
        fitted = bool(item.data(_ROLE_FITTED))
        if not new_name or new_name == old_name:
            # Empty after sanitize, or no real change — revert visually (and
            # re-apply the ✓ prefix that sanitize would have stripped).
            self._suppress_item_changed = True
            item.setText(self._row_label(old_name, fitted))
            self._suppress_item_changed = False
            return
        # If sanitization changed what the user typed, reflect the cleaned
        # value (with the ✓ if applicable) so the row shows what's stored.
        if new_name != typed:
            self._suppress_item_changed = True
            item.setText(self._row_label(new_name, fitted))
            self._suppress_item_changed = False
        # The app calls back to rename() with the final accepted name (might
        # get a `_2` suffix on collision).
        self.rename_quad.emit(old_name, new_name)

    def _on_delete(self):
        item = self._list.currentItem()
        if item is None:
            return
        self.delete_quad.emit(item.data(_ROLE_NAME))

    def _on_add_clicked(self):
        # Only emit on transition to checked; if user clicks again, cancel.
        if self._add_btn.isChecked():
            self.add_quad_requested.emit()
        # MainWindow drives add-mode state; we sync visuals via set_add_mode_active.

    def set_add_mode_active(self, active: bool) -> None:
        """Mirror the viewer's add-mode state on the button."""
        self._add_btn.blockSignals(True)
        self._add_btn.setChecked(active)
        self._add_btn.setText(
            "Cancel add (Esc)" if active else "Add quad   (click 4 corners)"
        )
        self._add_btn.blockSignals(False)

    def output_path(self) -> str:
        return self._out_path_edit.text().strip()

    def export_state(self) -> dict:
        """Snapshot of all export-option checkboxes + dome rotation +
        output path. Used by the project-file save path."""
        return {
            "dome": self.opt_dome.isChecked(),
            "rect": self.opt_rect.isChecked(),
            "usd": self.opt_usd.isChecked(),
            "depth_exr": self.opt_depth_exr.isChecked(),
            "depth_mesh": self.opt_depth_mesh.isChecked(),
            "masks": self.opt_masks.isChecked(),
            "output_dir": self.output_path(),
            "dome_rotate_y_deg": self.opt_dome_rotate.value(),
            "geom_inflation_pct": self.opt_geom_inflation.value(),
            "open_sky": self.opt_open_sky.isChecked(),
        }

    def apply_export_state(self, state: dict) -> None:
        """Restore export-option checkboxes from a saved project."""
        self.opt_dome.setChecked(bool(state.get("dome", True)))
        self.opt_rect.setChecked(bool(state.get("rect", True)))
        self.opt_usd.setChecked(bool(state.get("usd", True)))
        self.opt_depth_exr.setChecked(bool(state.get("depth_exr", False)))
        self.opt_depth_mesh.setChecked(bool(state.get("depth_mesh", True)))
        self.opt_masks.setChecked(bool(state.get("masks", True)))
        out = state.get("output_dir", "")
        if out:
            self.force_set_output_path(out)
        rot = state.get("dome_rotate_y_deg")
        if rot is not None:
            self.opt_dome_rotate.setValue(float(rot))
        infl = state.get("geom_inflation_pct")
        if infl is not None:
            self.opt_geom_inflation.setValue(float(infl))
        self.opt_open_sky.setChecked(bool(state.get("open_sky", True)))

    def set_output_path(self, path: str) -> None:
        # Only auto-populate if the user hasn't already entered something custom.
        if not self._out_path_edit.text().strip():
            self._out_path_edit.setText(path)

    def force_set_output_path(self, path: str) -> None:
        self._out_path_edit.setText(path)

    def _on_browse(self):
        start = self._out_path_edit.text().strip() or ""
        chosen = QFileDialog.getExistingDirectory(self, "Output directory", start)
        if chosen:
            self._out_path_edit.setText(chosen)

    def _on_bake(self):
        opts = {
            "dome": self.opt_dome.isChecked(),
            "rect": self.opt_rect.isChecked(),
            "usd": self.opt_usd.isChecked(),
            "depth_exr": self.opt_depth_exr.isChecked(),
            "depth_mesh": self.opt_depth_mesh.isChecked(),
            "masks": self.opt_masks.isChecked(),
            "output_dir": self.output_path(),
            "dome_rotate_y_deg": self.opt_dome_rotate.value(),
            "geom_inflation_pct": self.opt_geom_inflation.value(),
            "open_sky": self.opt_open_sky.isChecked(),
            "preview": False,
        }
        self.bake_requested.emit(opts)

    def _on_preview(self):
        # Preview = same pipeline, no file writes anywhere.
        self.preview_requested.emit()
