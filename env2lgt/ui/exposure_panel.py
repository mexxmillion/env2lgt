# Copyright 2024-2026 Maung Maung Hla Win <mexxmillion@gmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Exposure-mode side panel: baseline exposure offset + white balance.

These adjustments shift the HDRI itself (not just the viewport) and are baked
into the exported dome / rect textures. White balance follows the Lightroom
model — Temperature + Tint are the single source of truth; the area-sample
eyedropper and the auto-meter back-solve and set those sliders.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

# Slider scale factors.
_EV_RANGE = (-80, 80)      # tenths of a stop -> +/- 8 EV
_KELVIN_RANGE = (2000, 15000)
_TINT_RANGE = (-100, 100)  # hundredths -> +/- 1.0
_DEFAULT_KELVIN = 6500
_DEFAULT_TINT = 0


class _RegionRow(QWidget):
    """One row in the region-pair outliner. Click the row body to select it
    (the QLineEdit and the delete button still get their own clicks)."""

    select_clicked = Signal(int)

    def __init__(self, pair_index: int, parent=None):
        super().__init__(parent)
        self._pair_index = int(pair_index)
        self._selected = False
        self._apply_style()

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                "_RegionRow { background:#3a3a3a; border-radius:3px; }"
            )
        else:
            self.setStyleSheet(
                "_RegionRow { background:transparent; border-radius:3px; }"
                "_RegionRow:hover { background:#2c2c2c; }"
            )

    def set_selected(self, on: bool) -> None:
        self._selected = bool(on)
        self._apply_style()

    def mousePressEvent(self, event):  # noqa: N802
        # Don't steal clicks from child widgets (QLineEdit, buttons) — childAt
        # is non-None for those.
        if self.childAt(event.position().toPoint()) is None:
            self.select_clicked.emit(self._pair_index)
        super().mousePressEvent(event)


class ExposurePanel(QWidget):
    exposure_offset_changed = Signal(float)        # baseline EV offset
    wb_changed = Signal(float, float)              # kelvin, tint
    sample_exposure_requested = Signal()           # spot-meter an area
    sample_wb_requested = Signal()                 # WB-eyedropper an area
    auto_meter_requested = Signal()                # convolve-the-dome auto
    input_cs_changed = Signal(str)                 # source colorspace
    output_cs_changed = Signal(str)                # bake output colorspace
    pick_chart_requested = Signal()                # place a colour-checker
    clear_chart_requested = Signal()
    use_builtin_target = Signal()                  # target = built-in CC24
    load_json_target_requested = Signal()          # target = a JSON file
    use_reference_target = Signal()                # target = reference image
    load_reference_image_requested = Signal()      # load a flat reference image
    reference_cs_changed = Signal(str)             # reference-image colorspace
    reference_view_toggled = Signal(bool)          # HDRI <-> reference view
    fit_mode_changed = Signal(str)                 # exposure | wb | matrix
    solve_chart_requested = Signal()               # solve + apply correction
    save_correction_requested = Signal()           # export the correction JSON
    load_correction_requested = Signal()           # load a correction JSON
    # ---- region calibration sub-mode ----
    calibration_mode_changed = Signal(str)         # "chart" | "regions" | "auto"
    add_region_pair_requested = Signal()           # create a new pair (centred)
    region_select_requested = Signal(int)          # pair_index (-1 deselect)
    region_rename_requested = Signal(int, str)     # pair_index, new name
    region_delete_requested = Signal(int)          # pair_index
    region_fit_changed = Signal(str)               # "gain" | "gain_gamma"
    solve_regions_requested = Signal()             # solve + apply gain/gamma
    clear_regions_requested = Signal()             # wipe all pairs + correction
    auto_match_requested = Signal(str)             # fit mode: gain | gain_gamma

    def __init__(self, parent=None):
        super().__init__(parent)
        # Cap the panel width — VFX side dock should be a tight column, not a
        # half-screen sheet. Even if the user widens the dock manually, the
        # contents stay compact and left-aligned.
        self.setMaximumWidth(410)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        intro = QLabel(
            "Shift the HDRI baseline exposure + white balance. These are baked "
            "into the exported light rig."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#888;")
        layout.addWidget(intro)

        # ---- exposure offset ----
        exp_box = QGroupBox("Exposure offset", self)
        eb = QVBoxLayout(exp_box)
        row = QHBoxLayout()
        self._exp_slider = QSlider(Qt.Orientation.Horizontal)
        self._exp_slider.setRange(*_EV_RANGE)
        self._exp_slider.setValue(0)
        self._exp_slider.valueChanged.connect(self._on_exp_slider)
        row.addWidget(self._exp_slider, stretch=1)
        self._exp_label = QLabel("+0.0 EV")
        self._exp_label.setMinimumWidth(70)
        row.addWidget(self._exp_label)
        reset_exp = QPushButton("Reset")
        reset_exp.setObjectName("subtle")
        reset_exp.setFixedWidth(56)
        reset_exp.clicked.connect(lambda: self._exp_slider.setValue(0))
        row.addWidget(reset_exp)
        eb.addLayout(row)
        self._spot_btn = QPushButton("◎  Spot meter")
        self._spot_btn.setCheckable(True)
        self._spot_btn.setToolTip(
            "Spot meter — drag a rectangle on the panorama; its average is "
            "metered to 18% middle grey, setting the exposure offset "
            "(camera spot meter)."
        )
        self._spot_btn.clicked.connect(self._on_spot_clicked)
        eb.addWidget(self._spot_btn)
        layout.addWidget(exp_box)

        # ---- white balance ----
        wb_box = QGroupBox("White balance", self)
        wb = QVBoxLayout(wb_box)
        temp_row = QHBoxLayout()
        temp_row.addWidget(QLabel("Temp"))
        self._temp_slider = QSlider(Qt.Orientation.Horizontal)
        self._temp_slider.setRange(*_KELVIN_RANGE)
        self._temp_slider.setValue(_DEFAULT_KELVIN)
        self._temp_slider.valueChanged.connect(self._on_wb_slider)
        temp_row.addWidget(self._temp_slider, stretch=1)
        self._temp_label = QLabel(f"{_DEFAULT_KELVIN} K")
        self._temp_label.setMinimumWidth(64)
        temp_row.addWidget(self._temp_label)
        wb.addLayout(temp_row)

        tint_row = QHBoxLayout()
        tint_row.addWidget(QLabel("Tint"))
        self._tint_slider = QSlider(Qt.Orientation.Horizontal)
        self._tint_slider.setRange(*_TINT_RANGE)
        self._tint_slider.setValue(_DEFAULT_TINT)
        self._tint_slider.valueChanged.connect(self._on_wb_slider)
        tint_row.addWidget(self._tint_slider, stretch=1)
        self._tint_label = QLabel("+0.00")
        self._tint_label.setMinimumWidth(64)
        tint_row.addWidget(self._tint_label)
        wb.addLayout(tint_row)

        wb_reset_row = QHBoxLayout()
        self._wb_eyedrop_btn = QPushButton("◐  WB eyedropper")
        self._wb_eyedrop_btn.setCheckable(True)
        self._wb_eyedrop_btn.setToolTip(
            "WB eyedropper — drag a rectangle over something neutral (a grey "
            "wall); its average is neutralised and the Temp/Tint sliders are "
            "set to match."
        )
        self._wb_eyedrop_btn.clicked.connect(self._on_wb_eyedrop_clicked)
        wb_reset_row.addWidget(self._wb_eyedrop_btn, stretch=1)
        reset_wb = QPushButton("Reset")
        reset_wb.setObjectName("subtle")
        reset_wb.setFixedWidth(56)
        reset_wb.clicked.connect(self._reset_wb)
        wb_reset_row.addWidget(reset_wb)
        wb.addLayout(wb_reset_row)
        self._wb_readout = QLabel("scale  R 1.000  G 1.000  B 1.000")
        self._wb_readout.setStyleSheet("color:#888;")
        wb.addWidget(self._wb_readout)
        layout.addWidget(wb_box)

        # ---- auto meter ----
        auto_box = QGroupBox("Auto meter (convolve the dome)", self)
        ab = QVBoxLayout(auto_box)
        desc = QLabel(
            "Renders a cosine-weighted Lambertian gray ball lit by the whole "
            "dome, then sets exposure + WB from it."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#888;")
        ab.addWidget(desc)
        self._auto_btn = QPushButton("⚡  Auto exposure + WB")
        self._auto_btn.setToolTip(
            "Auto-meter the dome — convolve a cosine-weighted Lambertian "
            "gray ball lit by the whole dome, then set exposure + WB from it."
        )
        self._auto_btn.clicked.connect(self.auto_meter_requested)
        ab.addWidget(self._auto_btn)
        layout.addWidget(auto_box)

        layout.addWidget(self._build_colour_management_group())
        layout.addWidget(self._build_chart_group())
        layout.addStretch(1)

    # ---------- colour management ----------

    def _build_colour_management_group(self) -> QGroupBox:
        box = QGroupBox("Colour management (OCIO)", self)
        form = QFormLayout(box)
        self._input_cs_combo = QComboBox()
        self._input_cs_combo.setToolTip(
            "Colorspace of the source EXR. Converted into the ACEScg working "
            "space on load."
        )
        self._input_cs_combo.currentTextChanged.connect(self.input_cs_changed)
        form.addRow("Input", self._input_cs_combo)
        self._output_cs_combo = QComboBox()
        self._output_cs_combo.setToolTip(
            "Colorspace the baked dome / rect EXRs are written in."
        )
        self._output_cs_combo.currentTextChanged.connect(self.output_cs_changed)
        form.addRow("Output", self._output_cs_combo)
        return box

    # ---------- calibration (chart | regions) ----------

    def _build_chart_group(self) -> QGroupBox:
        box = QGroupBox("Calibration", self)
        v = QVBoxLayout(box)

        # Active correction pill — always shown so the sub-mode toggle never
        # silently hides what's actually baked into the preview.
        self._active_corr_label = QLabel("No correction.")
        self._active_corr_label.setStyleSheet(
            "QLabel { background:#2a2a2a; color:#bbb; padding:4px 8px;"
            " border-radius:3px; }"
        )
        self._active_corr_label.setWordWrap(True)
        v.addWidget(self._active_corr_label)

        # Shared reference-image controls (both sub-modes can use a ref image).
        self._ref_view_btn = QPushButton("Show reference")
        self._ref_view_btn.setCheckable(True)
        self._ref_view_btn.setEnabled(False)
        self._ref_view_btn.setToolTip(
            "Switch the viewport between the HDRI panorama and the loaded "
            "flat reference image. Charts and region rects stick to their "
            "own view."
        )
        self._ref_view_btn.toggled.connect(self.reference_view_toggled)
        v.addWidget(self._ref_view_btn)
        self._load_ref_btn = QPushButton("Load reference…")
        self._load_ref_btn.setObjectName("subtle")
        self._load_ref_btn.setToolTip(
            "Load a regular 2D photo (JPG/PNG/EXR) as the calibration "
            "reference."
        )
        self._load_ref_btn.clicked.connect(self.load_reference_image_requested)
        v.addWidget(self._load_ref_btn)

        ref_cs_row = QHBoxLayout()
        ref_cs_row.addWidget(QLabel("Image colorspace"))
        self._ref_cs_combo = QComboBox()
        self._ref_cs_combo.setToolTip(
            "Colorspace of the loaded reference image. Converted into the "
            "ACEScg working space, the same way the source EXR is. Auto-set "
            "on load (sRGB for 8-bit images); override here if it's wrong."
        )
        self._ref_cs_combo.currentTextChanged.connect(self.reference_cs_changed)
        ref_cs_row.addWidget(self._ref_cs_combo, stretch=1)
        v.addLayout(ref_cs_row)

        # Sub-mode selector.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#444;")
        v.addWidget(sep)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self._mode_chart_rb = QRadioButton("Chart")
        self._mode_chart_rb.setChecked(True)
        self._mode_regions_rb = QRadioButton("Regions")
        self._mode_auto_rb = QRadioButton("Auto")
        self._mode_chart_rb.setToolTip(
            "Solve a 3×3 matrix from a 24-patch CC chart."
        )
        self._mode_regions_rb.setToolTip(
            "Solve per-channel gain (and optional gamma) from paired sample "
            "rectangles you drop on the HDRI and the reference image."
        )
        self._mode_auto_rb.setToolTip(
            "NLE-style auto match — analyse the whole HDRI vs the whole "
            "reference and solve a per-channel correction. No ROIs needed."
        )
        grp = QButtonGroup(self)
        grp.addButton(self._mode_chart_rb)
        grp.addButton(self._mode_regions_rb)
        grp.addButton(self._mode_auto_rb)
        self._mode_chart_rb.toggled.connect(self._on_mode_radio)
        self._mode_regions_rb.toggled.connect(self._on_mode_radio)
        self._mode_auto_rb.toggled.connect(self._on_mode_radio)
        mode_row.addWidget(self._mode_chart_rb)
        mode_row.addWidget(self._mode_regions_rb)
        mode_row.addWidget(self._mode_auto_rb)
        mode_row.addStretch(1)
        v.addLayout(mode_row)

        self._calib_stack = QStackedWidget()
        self._calib_stack.addWidget(self._build_chart_page())
        self._calib_stack.addWidget(self._build_regions_page())
        self._calib_stack.addWidget(self._build_auto_page())
        v.addWidget(self._calib_stack)
        return box

    _MODE_TO_INDEX = {"chart": 0, "regions": 1, "auto": 2}

    def _on_mode_radio(self, checked: bool) -> None:
        if not checked:
            return  # only act on the radio that turned on
        if self._mode_chart_rb.isChecked():
            mode = "chart"
        elif self._mode_regions_rb.isChecked():
            mode = "regions"
        else:
            mode = "auto"
        self._calib_stack.setCurrentIndex(self._MODE_TO_INDEX[mode])
        self.calibration_mode_changed.emit(mode)

    def set_calibration_mode(self, mode: str) -> None:
        """Reflect a sub-mode without re-emitting (project restore)."""
        if mode == "regions":
            rb = self._mode_regions_rb
        elif mode == "auto":
            rb = self._mode_auto_rb
        else:
            rb = self._mode_chart_rb
            mode = "chart"
        rb.blockSignals(True)
        rb.setChecked(True)
        rb.blockSignals(False)
        self._calib_stack.setCurrentIndex(self._MODE_TO_INDEX.get(mode, 0))

    def set_active_correction_label(self, text: str) -> None:
        self._active_corr_label.setText(text)

    def _build_chart_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        desc = QLabel(
            "Place a 4-corner quad over a 24-patch chart (corner 1 = dark "
            "skin, then clockwise). The cells show the reference colours so "
            "you can line it up. Pick a target and solve a colour match."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#888;")
        v.addWidget(desc)

        pick_row = QHBoxLayout()
        self._pick_chart_btn = QPushButton("▦  Pick chart")
        self._pick_chart_btn.setCheckable(True)
        self._pick_chart_btn.setToolTip(
            "Pick a colour chart — place 4 corners over a 24-patch chart on "
            "the current view (HDRI or reference)."
        )
        self._pick_chart_btn.clicked.connect(self._on_pick_chart)
        pick_row.addWidget(self._pick_chart_btn, stretch=1)
        self._clear_chart_btn = QPushButton("Clear")
        self._clear_chart_btn.setObjectName("danger")
        self._clear_chart_btn.setFixedWidth(56)
        self._clear_chart_btn.clicked.connect(self.clear_chart_requested)
        pick_row.addWidget(self._clear_chart_btn)
        v.addLayout(pick_row)

        v.addWidget(QLabel("Target:"))
        self._target_label = QLabel("Built-in CC24")
        self._target_label.setStyleSheet("color:#6bd66b;")
        v.addWidget(self._target_label)
        tgt_row = QHBoxLayout()
        tgt_row.setSpacing(4)
        builtin_btn = QPushButton("CC24")
        builtin_btn.setObjectName("subtle")
        builtin_btn.setToolTip("Built-in CC24 reference (linear sRGB / D65).")
        builtin_btn.clicked.connect(self.use_builtin_target)
        tgt_row.addWidget(builtin_btn)
        json_btn = QPushButton("JSON…")
        json_btn.setObjectName("subtle")
        json_btn.setToolTip("Load a custom 24-swatch target JSON.")
        json_btn.clicked.connect(self.load_json_target_requested)
        tgt_row.addWidget(json_btn)
        ref_tgt_btn = QPushButton("Reference")
        ref_tgt_btn.setObjectName("subtle")
        ref_tgt_btn.setToolTip(
            "Match to the chart placed on the loaded reference image."
        )
        ref_tgt_btn.clicked.connect(self.use_reference_target)
        tgt_row.addWidget(ref_tgt_btn)
        tgt_row.addStretch(1)
        v.addLayout(tgt_row)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self._fit_combo = QComboBox()
        self._fit_combo.addItem("Exposure only", "exposure")
        self._fit_combo.addItem("White balance", "wb")
        self._fit_combo.addItem("Full 3×3 matrix", "matrix")
        self._fit_combo.setCurrentIndex(2)
        self._fit_combo.currentIndexChanged.connect(
            lambda _=0: self.fit_mode_changed.emit(self._fit_combo.currentData())
        )
        fit_row.addWidget(self._fit_combo, stretch=1)
        v.addLayout(fit_row)

        self._solve_btn = QPushButton("Solve & apply")
        self._solve_btn.setObjectName("primary")
        self._solve_btn.clicked.connect(self.solve_chart_requested)
        v.addWidget(self._solve_btn)
        self._chart_status = QLabel("No chart placed.")
        self._chart_status.setStyleSheet("color:#888;")
        self._chart_status.setWordWrap(True)
        v.addWidget(self._chart_status)

        # Save / reload the solved correction — for batch-matching a set of
        # HDRIs: solve once, save the JSON, load it on the rest.
        corr_row = QHBoxLayout()
        corr_row.setSpacing(4)
        self._save_corr_btn = QPushButton("Save…")
        self._save_corr_btn.setObjectName("subtle")
        self._save_corr_btn.setEnabled(False)
        self._save_corr_btn.setToolTip(
            "Save the solved correction to JSON so it can be reapplied to "
            "other HDRIs in the same set."
        )
        self._save_corr_btn.clicked.connect(self.save_correction_requested)
        corr_row.addWidget(self._save_corr_btn)
        self._load_corr_btn = QPushButton("Load…")
        self._load_corr_btn.setObjectName("subtle")
        self._load_corr_btn.setToolTip(
            "Load a saved correction JSON and apply it directly — no chart "
            "needed. Use for batch-matching similar HDRIs."
        )
        self._load_corr_btn.clicked.connect(self.load_correction_requested)
        corr_row.addWidget(self._load_corr_btn)
        corr_row.addStretch(1)
        v.addLayout(corr_row)
        return page

    def _build_regions_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        desc = QLabel(
            "Drop pairs of sample rectangles — one on the HDRI, one on the "
            "matching spot on the reference. Each pair contributes one log-"
            "space sample to a per-channel gain (and optional gamma) fit."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#888;")
        v.addWidget(desc)

        add_row = QHBoxLayout()
        self._add_pair_btn = QPushButton("＋  Add pair")
        self._add_pair_btn.setToolTip("Add a new region pair, centred on each view.")
        self._add_pair_btn.clicked.connect(self.add_region_pair_requested)
        add_row.addWidget(self._add_pair_btn, stretch=1)
        self._clear_regions_btn = QPushButton("Clear")
        self._clear_regions_btn.setObjectName("danger")
        self._clear_regions_btn.setFixedWidth(56)
        self._clear_regions_btn.clicked.connect(self.clear_regions_requested)
        add_row.addWidget(self._clear_regions_btn)
        v.addLayout(add_row)

        # Container for per-pair rows (rebuilt by set_region_pairs()).
        self._regions_list_widget = QWidget()
        self._regions_list_layout = QVBoxLayout(self._regions_list_widget)
        self._regions_list_layout.setContentsMargins(0, 0, 0, 0)
        self._regions_list_layout.setSpacing(4)
        v.addWidget(self._regions_list_widget)
        self._regions_empty = QLabel("No pairs yet.")
        self._regions_empty.setStyleSheet("color:#666;")
        v.addWidget(self._regions_empty)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self._region_fit_combo = QComboBox()
        self._region_fit_combo.addItem("Gain only", "gain")
        self._region_fit_combo.addItem("Gain + gamma", "gain_gamma")
        # Gain-only is the honest model for properly IDT'd linear data. Gamma
        # is a non-linearity fudge for when "linear" isn't — opt in explicitly.
        self._region_fit_combo.setCurrentIndex(0)
        self._region_fit_combo.currentIndexChanged.connect(
            lambda _=0: self.region_fit_changed.emit(self._region_fit_combo.currentData())
        )
        fit_row.addWidget(self._region_fit_combo, stretch=1)
        v.addLayout(fit_row)

        self._solve_regions_btn = QPushButton("Solve & apply")
        self._solve_regions_btn.setObjectName("primary")
        self._solve_regions_btn.clicked.connect(self.solve_regions_requested)
        v.addWidget(self._solve_regions_btn)
        self._regions_status = QLabel("No regions solved.")
        self._regions_status.setStyleSheet("color:#888;")
        self._regions_status.setWordWrap(True)
        v.addWidget(self._regions_status)
        return page

    def set_region_pairs(
        self,
        pairs: list,
        selected: int = -1,
        residuals: list | None = None,
        colors: list | None = None,
    ) -> None:
        """Rebuild the outliner. Each pair becomes a row with a colour swatch,
        an editable name, a select-hit area, and a delete button. `selected`
        is the highlighted pair index (-1 = none). `colors` matches `pairs`
        and supplies an (r,g,b) tuple per row; falls back to neutral if None.
        `residuals` (same length as `pairs`) tags each row with its per-pair
        RMSE from the last solve."""
        while self._regions_list_layout.count():
            it = self._regions_list_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not pairs:
            self._regions_empty.setVisible(True)
            return
        self._regions_empty.setVisible(False)
        for i, p in enumerate(pairs):
            row_w = _RegionRow(i, parent=self)
            row_w.select_clicked.connect(self.region_select_requested)
            row = QHBoxLayout(row_w)
            row.setContentsMargins(4, 2, 4, 2)
            row.setSpacing(6)
            # Colour swatch.
            col = QColor(*(colors[i] if colors and i < len(colors) else (150, 150, 150)))
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f"background:{col.name()}; border:1px solid #111;"
            )
            row.addWidget(swatch)
            # Name (editable).
            name_edit = QLineEdit(str(p.get("name") or f"Region {i + 1}"))
            name_edit.setFrame(False)
            name_edit.editingFinished.connect(
                lambda idx=i, le=name_edit: self.region_rename_requested.emit(idx, le.text())
            )
            row.addWidget(name_edit, stretch=1)
            # Per-pair RMSE.
            if residuals is not None and i < len(residuals) and residuals[i] is not None:
                res = QLabel(f"{residuals[i]:.3f}")
                res.setStyleSheet("color:#888;")
                res.setMinimumWidth(48)
                row.addWidget(res)
            # Delete.
            del_btn = QPushButton("✕")
            del_btn.setObjectName("danger")
            del_btn.setFixedWidth(24)
            del_btn.clicked.connect(lambda _=False, idx=i: self.region_delete_requested.emit(idx))
            row.addWidget(del_btn)
            row_w.set_selected(i == selected)
            self._regions_list_layout.addWidget(row_w)

    def region_fit(self) -> str:
        return self._region_fit_combo.currentData()

    def set_region_fit(self, mode: str) -> None:
        idx = self._region_fit_combo.findData(mode)
        if idx >= 0:
            self._region_fit_combo.blockSignals(True)
            self._region_fit_combo.setCurrentIndex(idx)
            self._region_fit_combo.blockSignals(False)

    def set_regions_status(self, text: str) -> None:
        self._regions_status.setText(text)

    def _build_auto_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(0, 0, 0, 0)
        desc = QLabel(
            "Whole-image auto match — analyse the HDRI vs the loaded "
            "reference and solve a per-channel correction. Robust to bright "
            "outliers (suns / lamps) via 99th-percentile clipping. Like an "
            "NLE \"match colour\" — no chart, no ROIs."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#888;")
        v.addWidget(desc)

        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self._auto_fit_combo = QComboBox()
        self._auto_fit_combo.addItem("Gain only", "gain")
        self._auto_fit_combo.addItem("Gain + gamma", "gain_gamma")
        self._auto_fit_combo.setCurrentIndex(0)
        self._auto_fit_combo.setToolTip(
            "Gain only: per-channel scalar match (EV + white balance).\n"
            "Gain + gamma: also fit a per-channel power curve from two "
            "percentile anchors. Use when the inputs aren't truly linear."
        )
        fit_row.addWidget(self._auto_fit_combo, stretch=1)
        v.addLayout(fit_row)

        self._auto_match_btn = QPushButton("⚡  Auto match")
        self._auto_match_btn.setObjectName("primary")
        self._auto_match_btn.clicked.connect(
            lambda: self.auto_match_requested.emit(self._auto_fit_combo.currentData())
        )
        v.addWidget(self._auto_match_btn)

        self._auto_status = QLabel("No auto match solved.")
        self._auto_status.setStyleSheet("color:#888;")
        self._auto_status.setWordWrap(True)
        v.addWidget(self._auto_status)
        return page

    def set_auto_status(self, text: str) -> None:
        self._auto_status.setText(text)

    def _on_pick_chart(self):
        if self._pick_chart_btn.isChecked():
            self.pick_chart_requested.emit()

    # ---------- slider handlers ----------

    def _on_exp_slider(self, val: int):
        ev = val / 10.0
        self._exp_label.setText(f"{ev:+.1f} EV")
        self.exposure_offset_changed.emit(ev)

    def _on_wb_slider(self, _val: int):
        kelvin = float(self._temp_slider.value())
        tint = self._tint_slider.value() / 100.0
        self._temp_label.setText(f"{int(kelvin)} K")
        self._tint_label.setText(f"{tint:+.2f}")
        self.wb_changed.emit(kelvin, tint)

    def _reset_wb(self):
        self.set_wb(_DEFAULT_KELVIN, 0.0)
        self.wb_changed.emit(float(_DEFAULT_KELVIN), 0.0)

    def _on_spot_clicked(self):
        if self._spot_btn.isChecked():
            self._wb_eyedrop_btn.setChecked(False)
            self.sample_exposure_requested.emit()

    def _on_wb_eyedrop_clicked(self):
        if self._wb_eyedrop_btn.isChecked():
            self._spot_btn.setChecked(False)
            self.sample_wb_requested.emit()

    # ---------- programmatic setters (auto-meter / project restore) ----------

    def set_exposure_offset(self, ev: float) -> None:
        self._exp_slider.blockSignals(True)
        self._exp_slider.setValue(int(round(float(ev) * 10.0)))
        self._exp_slider.blockSignals(False)
        self._exp_label.setText(f"{self._exp_slider.value() / 10.0:+.1f} EV")

    def set_wb(self, kelvin: float, tint: float) -> None:
        self._temp_slider.blockSignals(True)
        self._tint_slider.blockSignals(True)
        self._temp_slider.setValue(int(round(float(kelvin))))
        self._tint_slider.setValue(int(round(float(tint) * 100.0)))
        self._temp_slider.blockSignals(False)
        self._tint_slider.blockSignals(False)
        self._temp_label.setText(f"{self._temp_slider.value()} K")
        self._tint_label.setText(f"{self._tint_slider.value() / 100.0:+.2f}")

    def set_wb_readout(self, scale) -> None:
        self._wb_readout.setText(
            f"scale  R {scale[0]:.3f}  G {scale[1]:.3f}  B {scale[2]:.3f}"
        )

    def clear_sample_buttons(self) -> None:
        """Un-check the sample buttons (sampling finished or was cancelled)."""
        for b in (self._spot_btn, self._wb_eyedrop_btn):
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

    # ---------- colour-management / chart setters ----------

    def populate_colorspaces(
        self, names: list[str], input_cs: str, output_cs: str
    ) -> None:
        for combo, cur in (
            (self._input_cs_combo, input_cs),
            (self._output_cs_combo, output_cs),
            (self._ref_cs_combo, input_cs),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if cur in names:
                combo.setCurrentText(cur)
            combo.blockSignals(False)

    def set_reference_cs(self, name: str) -> None:
        """Reflect a colorspace pick on the combo without re-emitting — used
        when a reference image loads and its colorspace is auto-detected."""
        self._ref_cs_combo.blockSignals(True)
        if name and self._ref_cs_combo.findText(name) < 0:
            self._ref_cs_combo.addItem(name)
        self._ref_cs_combo.setCurrentText(name)
        self._ref_cs_combo.blockSignals(False)

    def reference_cs(self) -> str:
        return self._ref_cs_combo.currentText()

    def set_colour_management_enabled(self, enabled: bool) -> None:
        self._input_cs_combo.setEnabled(enabled)
        self._output_cs_combo.setEnabled(enabled)
        self._ref_cs_combo.setEnabled(enabled)

    def fit_mode(self) -> str:
        return self._fit_combo.currentData()

    def set_fit_mode(self, mode: str) -> None:
        """Reflect a fit mode on the combo without re-emitting (e.g. restored
        from a loaded correction)."""
        idx = self._fit_combo.findData(mode)
        if idx >= 0:
            self._fit_combo.blockSignals(True)
            self._fit_combo.setCurrentIndex(idx)
            self._fit_combo.blockSignals(False)

    def set_correction_available(self, available: bool) -> None:
        """Enable the 'Save correction' button once a correction is solved."""
        self._save_corr_btn.setEnabled(available)

    def set_pick_chart_active(self, active: bool) -> None:
        self._pick_chart_btn.blockSignals(True)
        self._pick_chart_btn.setChecked(active)
        self._pick_chart_btn.setText(
            "Cancel (Esc)" if active else "▦  Pick chart"
        )
        self._pick_chart_btn.blockSignals(False)

    def set_target_label(self, name: str) -> None:
        self._target_label.setText(name)

    def set_reference_loaded(self, loaded: bool) -> None:
        self._ref_view_btn.setEnabled(loaded)

    def set_reference_view(self, on: bool) -> None:
        """Reflect the active view on the toggle without re-emitting."""
        self._ref_view_btn.blockSignals(True)
        self._ref_view_btn.setChecked(on)
        self._ref_view_btn.setText(
            "Show HDRI panorama" if on else "Show reference image"
        )
        self._ref_view_btn.blockSignals(False)

    def set_chart_status(
        self, has_chart: bool, rmse: float | None = None, applied: bool = False
    ) -> None:
        if not has_chart:
            self._chart_status.setText("No chart placed.")
        elif rmse is not None:
            self._chart_status.setText(f"Correction applied — RMSE {rmse:.4f}")
        elif applied:
            self._chart_status.setText("Correction applied (restored from project).")
        else:
            self._chart_status.setText("Chart placed — pick a target and solve.")
