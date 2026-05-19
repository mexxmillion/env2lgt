"""Central dark theme for env2lgt — a single QSS stylesheet + palette.

VFX pipeline tools (Nuke, Mari, Katana, Houdini) share a flat, low-contrast
dark look: a near-neutral grey surface, hairline borders, compact controls and
one restrained accent colour. Applying the theme once on the QApplication keeps
every widget consistent and leaves all per-widget logic untouched.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# ---- design tokens -------------------------------------------------------
# Greyscale ramp, darkest → lightest.
BG_WINDOW   = "#26282b"   # app background, behind docks
BG_SURFACE  = "#303338"   # panels, toolbar, menus
BG_RAISED   = "#3a3d43"   # buttons, combos at rest
BG_INPUT    = "#1f2123"   # text fields, lists, sliders' troughs
BG_HOVER    = "#44474e"   # hovered button / row
BORDER      = "#191a1c"   # hairline separators
BORDER_SOFT = "#4a4d54"   # inner control outline

TEXT        = "#d6d8dc"   # primary text
TEXT_MUTED  = "#8b8f97"   # secondary / descriptive text
TEXT_DIM    = "#62666d"   # disabled text

ACCENT      = "#4f9fe0"   # primary accent (selection, focus, primary button)
ACCENT_HOT  = "#63b0ee"   # accent hover
ACCENT_DEEP = "#2f6ea6"   # accent pressed

_STYLESHEET = f"""
* {{
    outline: 0;
}}

QWidget {{
    background-color: {BG_SURFACE};
    color: {TEXT};
    font-size: 12px;
}}

QMainWindow, QDialog {{
    background-color: {BG_WINDOW};
}}

QToolTip {{
    background-color: {BG_INPUT};
    color: {TEXT};
    border: 1px solid {BORDER_SOFT};
    padding: 4px 6px;
}}

/* ---- group boxes ---- */
QGroupBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 5px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 8px;
    padding: 0 4px;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 10px;
}}

/* ---- buttons ---- */
QPushButton {{
    background-color: {BG_RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 12px;
    min-height: 16px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {BORDER_SOFT};
}}
QPushButton:pressed {{
    background-color: {BG_INPUT};
}}
QPushButton:checked {{
    background-color: {ACCENT_DEEP};
    border-color: {ACCENT};
    color: #ffffff;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background-color: {BG_SURFACE};
    border-color: {BORDER};
}}
QPushButton#primary {{
    background-color: {ACCENT};
    border-color: {ACCENT_DEEP};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background-color: {ACCENT_HOT};
}}
QPushButton#primary:pressed {{
    background-color: {ACCENT_DEEP};
}}
QPushButton#primary:disabled {{
    background-color: {BG_RAISED};
    border-color: {BORDER};
    color: {TEXT_DIM};
}}

/* ---- toolbar ---- */
QToolBar {{
    background-color: {BG_SURFACE};
    border: 0;
    border-bottom: 1px solid {BORDER};
    padding: 4px 6px;
    spacing: 4px;
}}
QToolBar::separator {{
    background-color: {BORDER};
    width: 1px;
    margin: 4px 6px;
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
}}
QToolButton:hover {{
    background-color: {BG_HOVER};
}}
QToolButton:checked {{
    background-color: {ACCENT_DEEP};
    border-color: {ACCENT};
}}

/* ---- menu bar / menus ---- */
QMenuBar {{
    background-color: {BG_WINDOW};
    border-bottom: 1px solid {BORDER};
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
}}
QMenuBar::item:selected {{
    background-color: {BG_HOVER};
    border-radius: 4px;
}}
QMenu {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item {{
    padding: 5px 24px 5px 20px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_DEEP};
    color: #ffffff;
}}
QMenu::separator {{
    height: 1px;
    background-color: {BORDER};
    margin: 4px 6px;
}}

/* ---- inputs ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled {{
    color: {TEXT_DIM};
}}
QComboBox::drop-down {{
    border: 0;
    width: 18px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_MUTED};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DEEP};
    selection-color: #ffffff;
    outline: 0;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_RAISED};
    border: 0;
    width: 16px;
}}
QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {BG_HOVER};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {TEXT_MUTED};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {TEXT_MUTED};
}}

/* ---- sliders ---- */
QSlider::groove:horizontal {{
    height: 4px;
    background-color: {BG_INPUT};
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {TEXT};
    border: 1px solid {BORDER};
    width: 12px;
    height: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background-color: {ACCENT_HOT};
}}
QSlider::handle:horizontal:disabled {{
    background-color: {TEXT_DIM};
}}

/* ---- lists ---- */
QListWidget {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 2px;
}}
QListWidget::item {{
    padding: 4px 4px;
    border-radius: 3px;
}}
QListWidget::item:hover {{
    background-color: {BG_HOVER};
}}
QListWidget::item:selected {{
    background-color: {ACCENT_DEEP};
    color: #ffffff;
}}

/* ---- check boxes ---- */
QCheckBox {{
    spacing: 7px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER_SOFT};
    border-radius: 3px;
    background-color: {BG_INPUT};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox:disabled {{
    color: {TEXT_DIM};
}}

/* ---- dock widget ---- */
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: {BG_WINDOW};
    padding: 6px 10px;
    border-bottom: 1px solid {BORDER};
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 10px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}

/* ---- status bar ---- */
QStatusBar {{
    background-color: {BG_WINDOW};
    border-top: 1px solid {BORDER};
    color: {TEXT_MUTED};
}}
QStatusBar::item {{
    border: 0;
}}

/* ---- progress bar ---- */
QProgressBar {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT};
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ---- scroll bars ---- */
QScrollBar:vertical {{
    background-color: {BG_WINDOW};
    width: 11px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background-color: {BG_WINDOW};
    height: 11px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background-color: {BG_HOVER};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:hover {{
    background-color: {BORDER_SOFT};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
    width: 0;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}

/* ---- misc ---- */
QScrollArea {{
    border: 0;
}}
QLabel {{
    background: transparent;
}}
"""


def apply_theme(app: QApplication) -> None:
    """Apply the env2lgt dark theme to the whole application."""
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG_WINDOW))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(BG_INPUT))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(BG_SURFACE))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(BG_RAISED))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(BG_INPUT))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_DIM)
    )
    pal.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_DIM)
    )
    pal.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(TEXT_DIM)
    )
    app.setPalette(pal)

    app.setFont(QFont("Segoe UI", 9))
    app.setStyleSheet(_STYLESHEET)
