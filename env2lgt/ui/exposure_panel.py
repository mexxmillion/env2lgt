"""Exposure-mode side panel: baseline exposure offset + white balance.

These adjustments shift the HDRI itself (not just the viewport) and are baked
into the exported dome / rect textures. White balance follows the Lightroom
model — Temperature + Tint are the single source of truth; the area-sample
eyedropper and the auto-meter back-solve and set those sliders.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# Slider scale factors.
_EV_RANGE = (-80, 80)      # tenths of a stop -> +/- 8 EV
_KELVIN_RANGE = (2000, 15000)
_TINT_RANGE = (-100, 100)  # hundredths -> +/- 1.0
_DEFAULT_KELVIN = 6500
_DEFAULT_TINT = 0


class ExposurePanel(QWidget):
    exposure_offset_changed = Signal(float)        # baseline EV offset
    wb_changed = Signal(float, float)              # kelvin, tint
    sample_exposure_requested = Signal()           # spot-meter an area
    sample_wb_requested = Signal()                 # WB-eyedropper an area
    auto_meter_requested = Signal()                # convolve-the-dome auto

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

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
        reset_exp.setFixedWidth(56)
        reset_exp.clicked.connect(lambda: self._exp_slider.setValue(0))
        row.addWidget(reset_exp)
        eb.addLayout(row)
        self._spot_btn = QPushButton("Spot meter — sample an area")
        self._spot_btn.setCheckable(True)
        self._spot_btn.setToolTip(
            "Drag a rectangle on the panorama; its average is metered to 18% "
            "middle grey, setting the exposure offset (camera spot meter)."
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
        self._wb_eyedrop_btn = QPushButton("WB eyedropper — sample an area")
        self._wb_eyedrop_btn.setCheckable(True)
        self._wb_eyedrop_btn.setToolTip(
            "Drag a rectangle over something neutral (a grey wall); its average "
            "is neutralised and the Temp/Tint sliders are set to match."
        )
        self._wb_eyedrop_btn.clicked.connect(self._on_wb_eyedrop_clicked)
        wb_reset_row.addWidget(self._wb_eyedrop_btn, stretch=1)
        reset_wb = QPushButton("Reset")
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
        self._auto_btn = QPushButton("Auto exposure + WB")
        self._auto_btn.clicked.connect(self.auto_meter_requested)
        ab.addWidget(self._auto_btn)
        layout.addWidget(auto_box)

        layout.addStretch(1)

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
