"""Side panel: quad list, add/delete buttons, output path, export options."""

from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from env2lgt.ui.viewer import LightQuad


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
    add_quad_requested = Signal()
    bake_requested = Signal(dict)
    preview_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        list_box = QGroupBox("Light quads", self)
        lb = QVBoxLayout(list_box)
        self._list = QListWidget()
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
            (self.opt_depth_mesh, False),
            (self.opt_masks, True),
        ):
            cb.setChecked(default)
            eb.addWidget(cb)
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

        self._preview_btn = QPushButton("Preview (no files)")
        self._preview_btn.clicked.connect(self._on_preview)
        eb.addWidget(self._preview_btn)
        self._bake_btn = QPushButton("Bake light rig")
        self._bake_btn.clicked.connect(self._on_bake)
        eb.addWidget(self._bake_btn)
        layout.addWidget(export_box)

        layout.addStretch(1)

    def add_quad(self, q: LightQuad) -> None:
        self._suppress_item_changed = True
        item = QListWidgetItem(q.name)
        item.setData(0x0100, q.name)  # canonical name, used to track renames
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._list.addItem(item)
        self._suppress_item_changed = False

    def rename(self, old_name: str, new_name: str) -> None:
        """Update the row that owns `old_name` to display `new_name`. Called
        by the app after MainWindow propagates the rename to the viewer."""
        self._suppress_item_changed = True
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(0x0100) == old_name:
                it.setText(new_name)
                it.setData(0x0100, new_name)
                break
        self._suppress_item_changed = False

    def remove_quad(self, name: str) -> None:
        for i in range(self._list.count()):
            if self._list.item(i).data(0x0100) == name:
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
                if self._list.item(i).data(0x0100) == name:
                    self._list.setCurrentRow(i)
                    break
        self._list.blockSignals(False)

    def _on_selection_changed(self):
        item = self._list.currentItem()
        self.select_quad.emit(item.data(0x0100) if item else "")

    def _on_item_changed(self, item: QListWidgetItem):
        if self._suppress_item_changed:
            return
        old_name = item.data(0x0100) or ""
        typed = item.text()
        new_name = sanitize_name(typed)
        if not new_name or new_name == old_name:
            # Empty after sanitize, or no real change — revert visually.
            self._suppress_item_changed = True
            item.setText(old_name)
            self._suppress_item_changed = False
            return
        # If sanitization changed what the user typed, reflect the cleaned
        # value in the list immediately so they can see what's being stored.
        if new_name != typed:
            self._suppress_item_changed = True
            item.setText(new_name)
            self._suppress_item_changed = False
        # The app calls back to rename() with the final accepted name (might
        # get a `_2` suffix on collision).
        self.rename_quad.emit(old_name, new_name)

    def _on_delete(self):
        item = self._list.currentItem()
        if item is None:
            return
        self.delete_quad.emit(item.data(0x0100))

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
            "preview": False,
        }
        self.bake_requested.emit(opts)

    def _on_preview(self):
        # Preview = same pipeline, no file writes anywhere.
        self.preview_requested.emit()
