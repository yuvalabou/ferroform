"""PyQt5 graphical user interface with embedded 3D PyVista renderer.

Provides interactive parameter tuning, 3D viewport visualization, drag-and-drop
STL/preset file loading, and mesh export controls.
"""

import json
import os
import sys
import time

import numpy as np
import qdarkstyle
from PyQt5.QtCore import QEvent, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonDataModel import vtkPlane

from pyroid.core import GyroidGenerator
from pyroid.models import GyroidParams, STLConversionParams
from pyroid.tpms import get_all_tpms_kernels, get_tpms_kernel


class GeneratorWorker(QThread):
    """Worker thread for parametric and STL conversion tasks."""

    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, task_fn):
        super().__init__()
        self.task_fn = task_fn

    def run(self):
        def _report(pct: int, msg: str) -> None:
            self.progress.emit(pct)
            self.status.emit(msg)

        try:
            generator = self.task_fn(_report)
            self.progress.emit(100)
            self.finished.emit(generator)
        except Exception as e:
            self.error.emit(str(e))


class GradingPreviewBar(QWidget):
    """Lightweight 1D visual preview of Start -> End functional grading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(26)
        self.active = False
        self.start_wall = 0.8
        self.end_wall = 0.8
        self.start_cell = 5.0
        self.end_cell = 5.0
        self.start_label = "Start"
        self.end_label = "End"

    def set_state(
        self,
        active: bool,
        start_wall: float,
        end_wall: float,
        start_cell: float,
        end_cell: float,
        start_label: str = "Start",
        end_label: str = "End",
    ):
        self.active = active
        self.start_wall = max(0.05, start_wall)
        self.end_wall = max(0.05, end_wall)
        self.start_cell = max(0.5, start_cell)
        self.end_cell = max(0.5, end_cell)
        self.start_label = start_label
        self.end_label = end_label
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)

        if not self.active:
            painter.setBrush(QColor("#2c3e50"))
            painter.setPen(QColor("#4a607a"))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor("#95a5a6"))
            painter.drawText(rect, Qt.AlignCenter, "Uniform (no spatial grading)")
            return

        grad = QLinearGradient(rect.left(), 0, rect.right(), 0)
        grad.setColorAt(0.0, QColor(52, 152, 219, 130))
        grad.setColorAt(1.0, QColor(230, 126, 34, 130))
        painter.setBrush(grad)
        painter.setPen(QColor("#5d738a"))
        painter.drawRoundedRect(rect, 4, 4)

        # Draw sample lattice ticks whose spacing and thickness morph from Start -> End
        max_w = max(self.start_wall, self.end_wall, 0.1)
        x = float(rect.left() + 8)
        right = float(rect.right() - 8)
        span = max(1.0, right - x)
        while x <= right:
            u = (x - rect.left() - 8) / span
            w_u = self.start_wall + (self.end_wall - self.start_wall) * u
            c_u = self.start_cell + (self.end_cell - self.start_cell) * u
            pen_w = max(1.0, min(5.0, 1.0 + 3.5 * (w_u / max_w)))
            painter.setPen(QPen(QColor(236, 240, 241, 180), pen_w))
            painter.drawLine(int(x), rect.top() + 4, int(x), rect.bottom() - 4)
            step = max(5.0, min(22.0, 9.0 * (c_u / max(0.5, min(self.start_cell, self.end_cell)))))
            x += step

        painter.setPen(QColor("#ecf0f1"))
        painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, self.start_label)
        painter.drawText(rect.adjusted(6, 0, -6, 0), Qt.AlignVCenter | Qt.AlignRight, self.end_label)


class GyroidGeneratorGUI(QMainWindow):
    """Modern GUI for PyRoid with Parametric and STL Conversion tabs."""

    def __init__(self):
        super().__init__()
        self.generator: GyroidGenerator | None = None
        self.current_preset = "default"
        self._gen_start_time = 0.0
        self._clip_plane = vtkPlane()
        self._mesh_actors = []
        self._preview_stl_mesh = None
        self._updating_grade_ui = False

        self.init_ui()
        self.apply_preset(self.current_preset)

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("PyRoid - Radially Symmetrical Gyroid Generator & STL Volume Infill")
        self.setGeometry(100, 100, 1280, 750)
        self.setAcceptDrops(True)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # Left Panel (Tabbed Controls & Actions)
        self.create_control_panel(main_layout)

        # Right Panel (3D PyVista Visualization)
        self.create_visualization_panel(main_layout)

        # Status Bar & Progress Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self.progressBar = QProgressBar()
        self.progressBar.setMaximumWidth(200)
        self.progressBar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progressBar)

        self.create_menu_bar()
        self.apply_theme("dark")

    def create_control_panel(self, main_layout):
        """Create left side control panel with QTabWidget."""
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, 1)

        # Tab Widget
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs)

        # TAB 1: Parametric Gyroid Generator
        tab_parametric = QWidget()
        param_tab_layout = QVBoxLayout(tab_parametric)

        # Presets Section
        preset_box = QGroupBox("Presets")
        preset_layout = QVBoxLayout(preset_box)
        preset_combo_layout = QHBoxLayout()
        preset_combo_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        for p in GyroidGenerator.get_presets():
            self.preset_combo.addItem(p)
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_combo_layout.addWidget(self.preset_combo)
        preset_layout.addLayout(preset_combo_layout)

        self.save_preset_btn = QPushButton("Save Preset...")
        self.save_preset_btn.clicked.connect(self.save_current_as_preset)
        preset_layout.addWidget(self.save_preset_btn)
        param_tab_layout.addWidget(preset_box)

        # Scaffold Geometry Section
        tpms_box = QGroupBox("Scaffold Geometry")
        tpms_layout = QVBoxLayout(tpms_box)
        self.param_tpms_combo, self.param_tpms_desc = self._add_tpms_selector(
            tpms_layout, "TPMS Type:"
        )
        param_tab_layout.addWidget(tpms_box)

        # Parameters Section
        params_box = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_box)
        self.params_widgets = {}
        param_descriptions = GyroidGenerator.get_param_descriptions()
        param_labels = {
            "res": "Resolution",
            "a": "X-axis Length",
            "b": "Y-axis Length",
            "c": "Z-axis Length",
            "r1": "Inner Radius",
            "r2": "Outer Radius",
            "phi_scale": "Angular Scaling",
            "wall_thickness": "Wall Thickness",
            "cell_radius": "Cell Radius",
            "cell_height": "Cell Height",
            "bias": "Cavity A/B Bias (-1..1)",
        }
        for k, label in param_labels.items():
            row = QHBoxLayout()
            tip = param_descriptions.get(k, "")
            lbl = QLabel(f"{label}:")
            lbl.setToolTip(tip)
            row.addWidget(lbl)
            line_edit = QLineEdit()
            line_edit.setToolTip(tip)
            row.addWidget(line_edit)
            self.params_widgets[k] = line_edit
            params_layout.addLayout(row)
        param_tab_layout.addWidget(params_box)

        self.generate_btn = QPushButton("Generate Gyroid")
        self.generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.generate_btn.clicked.connect(self.on_generate_clicked)
        param_tab_layout.addWidget(self.generate_btn)

        self.tabs.addTab(tab_parametric, "Parametric Gyroid")

        # TAB 2: STL Volume Conversion
        tab_stl = QWidget()
        stl_tab_layout = QVBoxLayout(tab_stl)

        stl_file_box = QGroupBox("Input Mesh (STL)")
        stl_file_layout = QVBoxLayout(stl_file_box)
        self.stl_path_edit = QLineEdit()
        self.stl_path_edit.setPlaceholderText("Select STL file...")
        self.stl_path_edit.setToolTip("Path to the closed/manifold 3D STL model to infill")
        stl_browse_btn = QPushButton("Browse...")
        stl_browse_btn.clicked.connect(self.on_browse_stl)

        file_row = QHBoxLayout()
        file_row.addWidget(self.stl_path_edit)
        file_row.addWidget(stl_browse_btn)
        stl_file_layout.addLayout(file_row)
        stl_tab_layout.addWidget(stl_file_box)

        stl_params_box = QGroupBox("Conversion Settings")
        stl_params_layout = QVBoxLayout(stl_params_box)

        # TPMS Geometry
        self.stl_tpms_combo, self.stl_tpms_desc = self._add_tpms_selector(
            stl_params_layout,
            "TPMS Scaffold:",
            tooltip="Select TPMS lattice topology (Gyroid, Diamond, Primitive, I-WP, Fischer-Koch S, Neovius)",
        )

        # Resolution
        res_row = QHBoxLayout()
        res_lbl = QLabel("Voxel Resolution:")
        res_tip = "Voxel grid resolution along the longest STL axis (higher = finer detail, slower)"
        res_lbl.setToolTip(res_tip)
        res_row.addWidget(res_lbl)
        self.stl_res_edit = QLineEdit("80")
        self.stl_res_edit.setToolTip(res_tip)
        res_row.addWidget(self.stl_res_edit)
        stl_params_layout.addLayout(res_row)

        # Target Infill %
        infill_row = QHBoxLayout()
        self.stl_infill_label = QLabel("Target Infill (%):")
        infill_tip = "Target solid volume percentage (1 to 99); automatically determines unit cell size from Wall Thickness"
        self.stl_infill_label.setToolTip(infill_tip)
        infill_row.addWidget(self.stl_infill_label)
        self.stl_infill_edit = QLineEdit("30")
        self.stl_infill_edit.setToolTip(infill_tip)
        infill_row.addWidget(self.stl_infill_edit)
        stl_params_layout.addLayout(infill_row)

        # Wall Thickness
        wall_row = QHBoxLayout()
        wall_lbl = QLabel("Wall Thickness (mm):")
        wall_tip = "Physical thickness of the solid wall separating Cavity A and Cavity B (in mm)"
        wall_lbl.setToolTip(wall_tip)
        wall_row.addWidget(wall_lbl)
        self.stl_wall_edit = QLineEdit("0.8")
        self.stl_wall_edit.setToolTip(wall_tip)
        wall_row.addWidget(self.stl_wall_edit)
        stl_params_layout.addLayout(wall_row)

        # Optional Cell Size Override
        cell_row = QHBoxLayout()
        cell_tip = "Override automatic cell sizing with a fixed unit cell period in mm"
        self.stl_cell_check = QCheckBox("Custom Cell Size (mm):")
        self.stl_cell_check.setChecked(False)
        self.stl_cell_check.setToolTip(cell_tip)
        self.stl_cell_check.toggled.connect(self.on_cell_override_toggled)
        cell_row.addWidget(self.stl_cell_check)
        self.stl_cell_edit = QLineEdit("5.0")
        self.stl_cell_edit.setToolTip(cell_tip)
        self.stl_cell_edit.setEnabled(False)
        cell_row.addWidget(self.stl_cell_edit)
        stl_params_layout.addLayout(cell_row)

        # Asymmetric Cavity Bias
        bias_row = QHBoxLayout()
        self.stl_bias_label = QLabel("Cavity A/B Bias (-1..1):")
        bias_tip = "Shift dividing wall: >0 widens Cavity A (Red), <0 widens Cavity B (Blue), 0 = equal 50/50 split"
        self.stl_bias_label.setToolTip(bias_tip)
        bias_row.addWidget(self.stl_bias_label)
        self.stl_bias_edit = QLineEdit("0.0")
        self.stl_bias_edit.setToolTip(bias_tip)
        bias_row.addWidget(self.stl_bias_edit)
        stl_params_layout.addLayout(bias_row)

        # Skeletal Mode Checkbox
        self.stl_skeletal_check = QCheckBox("Skeletal lattice (struts with single drain void)")
        self.stl_skeletal_check.setChecked(False)
        self.stl_skeletal_check.setToolTip(
            "Generate network-based struts with a single continuous void channel, ideal for resin draining."
        )
        self.stl_skeletal_check.toggled.connect(self.on_skeletal_toggled)
        stl_params_layout.addWidget(self.stl_skeletal_check)

        # Flat Capping
        self.stl_cap_check = QCheckBox("Planar slice & cap surfaces for 3D printing")
        self.stl_cap_check.setChecked(False)
        self.stl_cap_check.setToolTip(
            "Trim and cap top and bottom Z extremes for a flat build-plate interface"
        )
        stl_params_layout.addWidget(self.stl_cap_check)

        # Preserve Shell
        self.stl_shell_check = QCheckBox("Preserve original STL outer shell")
        self.stl_shell_check.setChecked(False)
        self.stl_shell_check.setToolTip(
            "Keep the outer skin of the input STL solid and fill only the interior"
        )
        self.stl_shell_check.toggled.connect(self.on_shell_check_toggled)
        stl_params_layout.addWidget(self.stl_shell_check)

        shell_thick_row = QHBoxLayout()
        shell_tip = "Thickness of the preserved outer STL skin in mm"
        self.stl_shell_label = QLabel("Shell Thickness (mm):")
        self.stl_shell_label.setToolTip(shell_tip)
        self.stl_shell_edit = QLineEdit("1.0")
        self.stl_shell_edit.setToolTip(shell_tip)
        self.stl_shell_label.setEnabled(False)
        self.stl_shell_edit.setEnabled(False)
        shell_thick_row.addWidget(self.stl_shell_label)
        shell_thick_row.addWidget(self.stl_shell_edit)
        stl_params_layout.addLayout(shell_thick_row)

        stl_tab_layout.addWidget(stl_params_box)

        # Functional Grading Group
        grade_box = QGroupBox("Functional Grading (Optional)")
        grade_layout = QVBoxLayout(grade_box)

        preset_row = QHBoxLayout()
        preset_lbl = QLabel("Quick Preset:")
        preset_lbl.setToolTip("1-click grading recipes that auto-fill direction and end values")
        preset_row.addWidget(preset_lbl)
        self.stl_grade_preset_combo = QComboBox()
        self.stl_grade_preset_combo.addItem("None (Uniform)", "none")
        self.stl_grade_preset_combo.addItem("Strong Shell → Light Core", "strong_shell_light_core")
        self.stl_grade_preset_combo.addItem("Light Shell → Strong Core", "light_shell_strong_core")
        self.stl_grade_preset_combo.addItem("Heavy Bottom → Light Top (Z)", "heavy_bottom_light_top")
        self.stl_grade_preset_combo.addItem("Stiff Center → Soft Edge (XY)", "stiff_center_soft_edge")
        self.stl_grade_preset_combo.addItem("Custom...", "custom")
        self.stl_grade_preset_combo.currentIndexChanged.connect(self.on_grade_preset_changed)
        preset_row.addWidget(self.stl_grade_preset_combo)
        grade_layout.addLayout(preset_row)

        grade_mode_row = QHBoxLayout()
        grade_mode_lbl = QLabel("Direction:")
        grade_mode_lbl.setToolTip("Where Start (0%) and End (100%) are located in 3D space")
        grade_mode_row.addWidget(grade_mode_lbl)
        self.stl_grade_combo = QComboBox()
        self.stl_grade_combo.addItem("None (Uniform)", "none")
        self.stl_grade_combo.addItem("Outside Surface → Inside Core (EDT)", "shell_to_core")
        self.stl_grade_combo.addItem("Left → Right (X)", "axial_x")
        self.stl_grade_combo.addItem("Front → Back (Y)", "axial_y")
        self.stl_grade_combo.addItem("Bottom → Top (Z)", "axial_z")
        self.stl_grade_combo.addItem("Center → Outer Edge (XY)", "radial")
        self.stl_grade_combo.currentIndexChanged.connect(self.on_grade_mode_changed)
        grade_mode_row.addWidget(self.stl_grade_combo)
        grade_layout.addLayout(grade_mode_row)

        def _make_grade_row(
            label_text: str,
            start_text: str,
            placeholder: str,
            tooltip: str,
            with_start: bool = True,
        ):
            row = QHBoxLayout()
            lbl = QLabel(label_text)
            lbl.setToolTip(tooltip)
            lbl.setEnabled(False)
            row.addWidget(lbl)
            start_lbl = None
            if with_start:
                start_lbl = QLabel(start_text)
                start_lbl.setStyleSheet("color: #95a5a6; font-size: 11px;")
                start_lbl.setEnabled(False)
                row.addWidget(start_lbl)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setToolTip(tooltip)
            edit.setEnabled(False)
            row.addWidget(edit)
            grade_layout.addLayout(row)
            return lbl, start_lbl, edit

        self.stl_grade_infill_label, self.stl_grade_infill_start_lbl, self.stl_grade_infill_edit = (
            _make_grade_row(
                "End Infill (%):",
                "30% →",
                "Same as Start",
                "Optional shortcut: type target End infill % to auto-compute End Wall Thickness",
            )
        )
        self.stl_grade_wall_label, self.stl_grade_wall_start_lbl, self.stl_grade_wall_edit = (
            _make_grade_row(
                "Wall (mm):",
                "0.80 →",
                "Same as Start",
                "Target wall thickness (mm) at grading End",
            )
        )
        self.stl_grade_cell_label, self.stl_grade_cell_start_lbl, self.stl_grade_cell_edit = (
            _make_grade_row(
                "Cell Size (mm):",
                "5.00 →",
                "Same as Start",
                "Target TPMS unit cell size (mm) at grading End",
            )
        )
        self.stl_grade_bias_label, self.stl_grade_bias_start_lbl, self.stl_grade_bias_edit = (
            _make_grade_row(
                "Bias (-1.4..1.4):",
                "0.00 →",
                "Same as Start",
                "Target cavity A/B bias at grading End",
            )
        )
        self.stl_grade_depth_label, _, self.stl_grade_depth_edit = _make_grade_row(
            "Transition Depth (mm):",
            "",
            "Auto (to center)",
            "Distance from outer surface to core for Surface → Core grading (blank = auto)",
            with_start=False,
        )

        self.stl_grade_preview_bar = GradingPreviewBar()
        grade_layout.addWidget(self.stl_grade_preview_bar)

        self.stl_grade_summary_label = QLabel(
            "Uniform lattice — same wall & cell size everywhere."
        )
        self.stl_grade_summary_label.setWordWrap(True)
        self.stl_grade_summary_label.setStyleSheet("color: #95a5a6; font-size: 11px;")
        grade_layout.addWidget(self.stl_grade_summary_label)

        # Wire live feedback on every relevant parameter change
        self.stl_grade_infill_edit.textEdited.connect(self.on_grade_infill_edited)
        for edit in (
            self.stl_grade_wall_edit,
            self.stl_grade_cell_edit,
            self.stl_grade_bias_edit,
            self.stl_grade_depth_edit,
        ):
            edit.textEdited.connect(self._on_grade_manual_edit)
            edit.textChanged.connect(self._update_grading_feedback)
        self.stl_grade_depth_edit.textChanged.connect(self._refresh_stl_preview)

        for start_edit in (
            self.stl_infill_edit,
            self.stl_wall_edit,
            self.stl_cell_edit,
            self.stl_bias_edit,
        ):
            start_edit.textChanged.connect(self._update_grading_feedback)
        self.stl_tpms_combo.currentIndexChanged.connect(self._update_grading_feedback)

        stl_tab_layout.addWidget(grade_box)

        self.convert_stl_btn = QPushButton("Convert STL Volume to Gyroid")
        self.convert_stl_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.convert_stl_btn.clicked.connect(self.on_convert_stl_clicked)
        stl_tab_layout.addWidget(self.convert_stl_btn)
        stl_tab_layout.addStretch()

        self.tabs.addTab(tab_stl, "STL to Gyroid Volume")

        # Shared Export Actions Section
        actions_box = QGroupBox("Export Options")
        actions_layout = QHBoxLayout(actions_box)

        self.save_stl_btn = QPushButton("Export STL")
        self.save_stl_btn.clicked.connect(lambda: self.save_mesh("stl"))
        self.save_obj_btn = QPushButton("Export OBJ")
        self.save_obj_btn.clicked.connect(lambda: self.save_mesh("obj"))

        actions_layout.addWidget(self.save_stl_btn)
        actions_layout.addWidget(self.save_obj_btn)
        left_layout.addWidget(actions_box)

        # View Options Section
        view_box = QGroupBox("View Options")
        view_layout = QVBoxLayout(view_box)
        self.show_wall_check = QCheckBox("Show Solid Wall")
        self.show_wall_check.setChecked(True)
        self.show_wall_check.setToolTip(
            "Display the solid gyroid wall (renders translucent when fluid cavities are shown)"
        )
        self.show_wall_check.toggled.connect(self.on_view_mode_changed)
        view_layout.addWidget(self.show_wall_check)

        color_row = QHBoxLayout()
        color_lbl = QLabel("Color By:")
        color_lbl.setToolTip("Choose whether solid wall colors show Z height or local infill density (%)")
        color_row.addWidget(color_lbl)
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItem("Height (Z)", "z")
        self.color_mode_combo.addItem("Infill Density (%)", "density")
        self.color_mode_combo.currentIndexChanged.connect(self.on_view_mode_changed)
        color_row.addWidget(self.color_mode_combo)
        view_layout.addLayout(color_row)

        cav_row = QHBoxLayout()
        self.show_cav_a_check = QCheckBox("Cavity A (Red)")
        self.show_cav_a_check.setToolTip("Display Fluid Channel A volume (G > t) in red")
        self.show_cav_a_check.toggled.connect(self.on_view_mode_changed)
        self.show_cav_b_check = QCheckBox("Cavity B (Blue)")
        self.show_cav_b_check.setToolTip("Display Fluid Channel B volume (G < -t) in blue")
        self.show_cav_b_check.toggled.connect(self.on_view_mode_changed)
        cav_row.addWidget(self.show_cav_a_check)
        cav_row.addWidget(self.show_cav_b_check)
        view_layout.addLayout(cav_row)

        slice_row = QHBoxLayout()
        self.slice_enable_check = QCheckBox("Slice Plane:")
        self.slice_enable_check.setToolTip(
            "Clip mesh along an interactive X/Y/Z plane to inspect internal lattice"
        )
        self.slice_enable_check.toggled.connect(self.on_slice_toggled)
        self.slice_axis_combo = QComboBox()
        self.slice_axis_combo.addItems(["X", "Y", "Z"])
        self.slice_axis_combo.setEnabled(False)
        self.slice_axis_combo.currentIndexChanged.connect(self.on_slice_changed)
        self.slice_reverse_check = QCheckBox("Reverse")
        self.slice_reverse_check.setToolTip("Flip which side of the slice plane is clipped")
        self.slice_reverse_check.setEnabled(False)
        self.slice_reverse_check.toggled.connect(self.on_slice_changed)
        self.slice_pos_label = QLabel("50%")
        slice_row.addWidget(self.slice_enable_check)
        slice_row.addWidget(self.slice_axis_combo)
        slice_row.addWidget(self.slice_reverse_check)
        slice_row.addStretch()
        slice_row.addWidget(self.slice_pos_label)
        view_layout.addLayout(slice_row)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.setEnabled(False)
        self.slice_slider.setToolTip("Adjust slice plane position across mesh bounds")
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        view_layout.addWidget(self.slice_slider)

        left_layout.addWidget(view_box)
        self.tabs.currentChanged.connect(self.on_skeletal_toggled)

        # Stats section
        self.stats_box = QGroupBox("Mesh Statistics")
        stats_layout = QVBoxLayout(self.stats_box)
        self.stats_labels = {
            "n_points": QLabel("Points: -"),
            "n_cells": QLabel("Cells: -"),
            "volume": QLabel("Volume: -"),
            "surface_area": QLabel("Surface Area: -"),
            "porosity": QLabel("Porosity: -"),
            "sa_to_vol": QLabel("Surface/Volume: -"),
            "watertight": QLabel("Watertight: -"),
        }
        for label in self.stats_labels.values():
            stats_layout.addWidget(label)
        self.stats_box.setVisible(False)
        left_layout.addWidget(self.stats_box)

    def create_visualization_panel(self, main_layout):
        """Create 3D visualization panel with PyVistaQt."""
        self.plotter = QtInteractor(self)
        self.plotter.installEventFilter(self)
        main_layout.addWidget(self.plotter, 2)

    def create_menu_bar(self):
        """Create top menu bar."""
        menubar = self.menuBar()
        assert menubar is not None
        file_menu = menubar.addMenu("&File")
        assert file_menu is not None

        new_action = QAction("&New Project", self)
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)

        open_stl_action = QAction("Open &STL...", self)
        open_stl_action.triggered.connect(self.on_browse_stl)
        file_menu.addAction(open_stl_action)

        export_stl = QAction("Export as &STL", self)
        export_stl.triggered.connect(lambda: self.save_mesh("stl"))
        file_menu.addAction(export_stl)

        file_menu.addSeparator()
        exit_action = QAction("E&xit", self)
        exit_action.triggered.connect(lambda: self.close())
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu("&View")
        assert view_menu is not None
        reset_camera_action = QAction("Reset &Camera", self)
        reset_camera_action.triggered.connect(self.plotter.reset_camera)
        view_menu.addAction(reset_camera_action)

    def _add_tpms_selector(
        self, parent_layout: QVBoxLayout, label_text: str, tooltip: str = ""
    ) -> tuple[QComboBox, QLabel]:
        """Build a TPMS combo box + description label pair wired to update on selection."""
        row = QHBoxLayout()
        lbl = QLabel(label_text)
        combo = QComboBox()
        for k in get_all_tpms_kernels():
            combo.addItem(k.display_name, k.tpms_type.value)
        if tooltip:
            lbl.setToolTip(tooltip)
            combo.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(combo)
        parent_layout.addLayout(row)

        desc = QLabel(get_tpms_kernel("gyroid").description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #888888; font-size: 11px;")
        parent_layout.addWidget(desc)
        combo.currentIndexChanged.connect(
            lambda: desc.setText(get_tpms_kernel(combo.currentData()).description)
        )
        return combo, desc

    def on_preset_changed(self, preset_name):
        self.apply_preset(preset_name)
        self.current_preset = preset_name

    def apply_preset(self, preset_name):
        presets = GyroidGenerator.get_presets()
        if preset_name in presets:
            data = presets[preset_name]
            for k, v in data.items():
                if k in self.params_widgets:
                    self.params_widgets[k].setText(str(v))
            if "tpms_type" in data and hasattr(self, "param_tpms_combo"):
                idx = self.param_tpms_combo.findData(data["tpms_type"])
                if idx >= 0:
                    self.param_tpms_combo.setCurrentIndex(idx)

    def save_current_as_preset(self):
        name, ok = QFileDialog.getSaveFileName(self, "Save Preset", "", "Preset Files (*.json)")
        if ok and name:
            if not name.endswith(".json"):
                name += ".json"
            try:
                params = self.get_params_from_ui()
                with open(name, "w") as f:
                    json.dump(params.to_dict(), f, indent=4)
                QMessageBox.information(self, "Success", f"Preset saved as {name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save preset: {e!s}")

    def get_params_from_ui(self) -> GyroidParams:
        raw_dict = {}
        for k, widget in self.params_widgets.items():
            val = widget.text()
            raw_dict[k] = int(val) if k == "res" else float(val or 0.0)
        if hasattr(self, "param_tpms_combo"):
            raw_dict["tpms_type"] = self.param_tpms_combo.currentData()
        return GyroidParams.from_dict(raw_dict)

    def eventFilter(self, watched, event):
        """Intercept drag and drop events on child widgets (e.g. 3D plotter viewport)."""
        if watched == getattr(self, "plotter", None):
            if event.type() == QEvent.Type.DragEnter:
                self.dragEnterEvent(event)
                return True
            elif event.type() == QEvent.Type.DragMove:
                self.dragMoveEvent(event)
                return True
            elif event.type() == QEvent.Type.Drop:
                self.dropEvent(event)
                return True
        return super().eventFilter(watched, event)

    def _accept_supported_drag(self, event) -> None:
        """Accept drag event if idle and mime URLs contain a .stl or .json file."""
        if not self.progressBar.isVisible() and event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path and path.lower().endswith((".stl", ".json")):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragEnterEvent(self, event):
        """Accept drag event if it contains supported file types (.stl or .json)."""
        self._accept_supported_drag(event)

    def dragMoveEvent(self, event):
        """Accept drag movement over valid drop areas."""
        self._accept_supported_drag(event)

    def dropEvent(self, event):
        """Handle dropped files (.stl or .json)."""
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        handled = False
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if not file_path or not os.path.isfile(file_path):
                continue

            _, ext = os.path.splitext(file_path.lower())
            match ext:
                case ".stl":
                    self.load_stl_file(file_path)
                    handled = True
                    break
                case ".json":
                    self.load_preset_file(file_path)
                    handled = True
                    break
                case _:
                    continue

        if handled:
            event.acceptProposedAction()
        else:
            event.ignore()

    def load_stl_file(self, file_path: str):
        """Load an STL file, switch to STL conversion tab, and preview in 3D."""
        if not file_path or not os.path.isfile(file_path):
            QMessageBox.warning(self, "File Not Found", f"STL file does not exist:\n{file_path}")
            return

        self.stl_path_edit.setText(file_path)
        self.tabs.setCurrentIndex(1)
        self.status_bar.showMessage(f"Loaded STL: {os.path.basename(file_path)}")

        try:
            import pyvista as pv

            mesh = pv.read(file_path)
            if mesh is not None and getattr(mesh, "n_points", 0) > 0:
                self._preview_stl_mesh = mesh
                self.generator = None
                self.stats_box.setVisible(False)
                self._refresh_stl_preview(reset_camera=True)
                self._update_grading_feedback()
        except Exception:
            pass

    def load_preset_file(self, file_path: str):
        """Load parameter preset from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                preset_data = json.load(f)
            if not isinstance(preset_data, dict):
                raise ValueError("Preset file must contain a JSON object.")
            for k, v in preset_data.items():
                if k in self.params_widgets:
                    self.params_widgets[k].setText(str(v))
            if "tpms_type" in preset_data and hasattr(self, "param_tpms_combo"):
                idx = self.param_tpms_combo.findData(str(preset_data["tpms_type"]))
                if idx >= 0:
                    self.param_tpms_combo.setCurrentIndex(idx)
            self.tabs.setCurrentIndex(0)
            self.status_bar.showMessage(f"Loaded preset: {os.path.basename(file_path)}")
        except Exception as e:
            QMessageBox.warning(self, "Invalid Preset", f"Failed to load preset: {e!s}")

    def on_browse_stl(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Input STL File", "", "STL Files (*.stl)"
        )
        if filename:
            self.load_stl_file(filename)

    def on_generate_clicked(self):
        try:
            params = self.get_params_from_ui()
            valid, err = params.validate()
            if not valid:
                QMessageBox.warning(self, "Validation Error", err)
                return

            self._gen_start_time = time.perf_counter()
            self.generate_btn.setEnabled(False)
            self.status_bar.showMessage("Starting parametric generation...")
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)

            def _gen_task(cb):
                gen = GyroidGenerator(params)
                gen.generate(progress_callback=cb)
                return gen

            self.worker = GeneratorWorker(_gen_task)
            self.worker.finished.connect(self.on_generation_finished)
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(self.progressBar.setValue)
            self.worker.status.connect(self.status_bar.showMessage)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.generate_btn.setEnabled(True)

    def on_convert_stl_clicked(self):
        stl_path = self.stl_path_edit.text().strip()
        if not stl_path or not os.path.exists(stl_path):
            QMessageBox.warning(self, "Warning", "Please select a valid existing STL file.")
            return

        try:
            res = int(self.stl_res_edit.text())
            wall_thick = float(self.stl_wall_edit.text())
            is_skeletal = self.stl_skeletal_check.isChecked()
            tpms_type = (
                self.stl_tpms_combo.currentData() if hasattr(self, "stl_tpms_combo") else "gyroid"
            )
            if self.stl_cell_check.isChecked():
                cell_size = float(self.stl_cell_edit.text())
                infill_pct = None
            else:
                infill_pct = float(self.stl_infill_edit.text())
                cell_size = STLConversionParams.estimate_cell_size(
                    infill_pct, wall_thick, is_skeletal, tpms_type=tpms_type
                )
            bias = float(self.stl_bias_edit.text() or 0.0)
            flat_cap = self.stl_cap_check.isChecked()
            preserve_shell = self.stl_shell_check.isChecked()
            shell_thick = float(self.stl_shell_edit.text()) if preserve_shell else 1.0

            def _opt_float(edit: QLineEdit) -> float | None:
                txt = edit.text().strip()
                return float(txt) if txt else None

            grade_mode = self.stl_grade_combo.currentData() or "none"
            grade_wall_end = _opt_float(self.stl_grade_wall_edit) if grade_mode != "none" else None
            grade_bias_end = _opt_float(self.stl_grade_bias_edit) if grade_mode != "none" else None
            grade_cell_end = _opt_float(self.stl_grade_cell_edit) if grade_mode != "none" else None
            grade_depth = (
                _opt_float(self.stl_grade_depth_edit) if grade_mode == "shell_to_core" else None
            )

            stl_params = STLConversionParams(
                stl_path=stl_path,
                tpms_type=tpms_type,
                resolution=res,
                cell_size=cell_size,
                wall_thickness=wall_thick,
                bias=bias,
                flat_cap=flat_cap,
                preserve_shell=preserve_shell,
                shell_thickness=shell_thick,
                is_skeletal=is_skeletal,
                infill_percent=infill_pct,
                grade_mode=grade_mode,
                grade_wall_end=grade_wall_end,
                grade_bias_end=grade_bias_end,
                grade_cell_end=grade_cell_end,
                grade_depth=grade_depth,
            )

            self._gen_start_time = time.perf_counter()
            self.convert_stl_btn.setEnabled(False)
            self.status_bar.showMessage("Starting STL lattice conversion...")
            self.progressBar.setVisible(True)
            self.progressBar.setValue(0)

            def _stl_task(cb):
                gen = GyroidGenerator()
                gen.convert_from_stl(stl_params, progress_callback=cb)
                return gen

            self.worker = GeneratorWorker(_stl_task)
            self.worker.finished.connect(self.on_generation_finished)
            self.worker.error.connect(self.on_generation_error)
            self.worker.progress.connect(self.progressBar.setValue)
            self.worker.status.connect(self.status_bar.showMessage)
            self.worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid parameters: {e!s}")

    def on_cell_override_toggled(self, checked: bool):
        self.stl_cell_edit.setEnabled(checked)
        self.stl_infill_label.setEnabled(not checked)
        self.stl_infill_edit.setEnabled(not checked)
        self._update_grading_feedback()

    def on_shell_check_toggled(self, checked: bool):
        self.stl_shell_label.setEnabled(checked)
        self.stl_shell_edit.setEnabled(checked)

    def _safe_float(self, text: str, default: float) -> float:
        try:
            return float(text.strip())
        except (ValueError, AttributeError):
            return default

    def _current_stl_start_values(self) -> tuple[float, float, float, float, bool, str]:
        """Return (start_wall, start_cell, start_bias, start_infill, is_skeletal, tpms_type)."""
        wall = max(0.05, self._safe_float(self.stl_wall_edit.text(), 0.8))
        is_skeletal = self.stl_skeletal_check.isChecked()
        tpms_type = (
            self.stl_tpms_combo.currentData() if hasattr(self, "stl_tpms_combo") else "gyroid"
        )
        if self.stl_cell_check.isChecked():
            cell = max(0.1, self._safe_float(self.stl_cell_edit.text(), 5.0))
            infill = STLConversionParams.estimate_infill(
                cell, wall, is_skeletal, tpms_type=tpms_type
            )
        else:
            infill = min(99.0, max(1.0, self._safe_float(self.stl_infill_edit.text(), 30.0)))
            cell = STLConversionParams.estimate_cell_size(
                infill, wall, is_skeletal, tpms_type=tpms_type
            )
        bias = 0.0 if is_skeletal else self._safe_float(self.stl_bias_edit.text(), 0.0)
        return wall, cell, bias, infill, is_skeletal, tpms_type

    def on_grade_preset_changed(self, _index: int = 0):
        if self._updating_grade_ui:
            return
        preset = self.stl_grade_preset_combo.currentData() or "none"
        if preset == "custom":
            return

        start_wall, _cell, _bias, _infill, _skel, _tpms = self._current_stl_start_values()
        self._updating_grade_ui = True
        try:
            if preset == "none":
                idx = self.stl_grade_combo.findData("none")
                if idx >= 0:
                    self.stl_grade_combo.setCurrentIndex(idx)
                for edit in (
                    self.stl_grade_infill_edit,
                    self.stl_grade_wall_edit,
                    self.stl_grade_cell_edit,
                    self.stl_grade_bias_edit,
                    self.stl_grade_depth_edit,
                ):
                    edit.clear()
            else:
                mode_map = {
                    "strong_shell_light_core": ("shell_to_core", 0.5),
                    "light_shell_strong_core": ("shell_to_core", 1.8),
                    "heavy_bottom_light_top": ("axial_z", 0.5),
                    "stiff_center_soft_edge": ("radial", 0.5),
                }
                mode, wall_mult = mode_map[preset]
                idx = self.stl_grade_combo.findData(mode)
                if idx >= 0:
                    self.stl_grade_combo.setCurrentIndex(idx)
                self.stl_grade_wall_edit.setText(f"{start_wall * wall_mult:.2f}")
                self.stl_grade_cell_edit.clear()
                self.stl_grade_bias_edit.clear()
                self.stl_grade_infill_edit.clear()
        finally:
            self._updating_grade_ui = False

        self.on_grade_mode_changed()

    def _on_grade_manual_edit(self):
        if self._updating_grade_ui:
            return
        self._updating_grade_ui = True
        try:
            idx = self.stl_grade_preset_combo.findData("custom")
            if idx >= 0:
                self.stl_grade_preset_combo.setCurrentIndex(idx)
        finally:
            self._updating_grade_ui = False

    def on_grade_infill_edited(self, text: str):
        if self._updating_grade_ui:
            return
        self._on_grade_manual_edit()
        txt = text.strip()
        if not txt:
            return
        try:
            end_infill = float(txt)
        except ValueError:
            return
        if not (1.0 <= end_infill <= 99.0):
            return
        _sw, start_cell, _sb, _si, is_skeletal, tpms_type = self._current_stl_start_values()
        end_cell_txt = self.stl_grade_cell_edit.text().strip()
        target_cell = self._safe_float(end_cell_txt, start_cell) if end_cell_txt else start_cell
        if is_skeletal:
            ratio = max(0.05, 0.2 + (end_infill - 50.0) / 95.5)
            end_wall = target_cell * ratio
        else:
            factor = get_tpms_kernel(tpms_type).surface_factor
            end_wall = (end_infill * target_cell) / max(factor, 1e-6)
        self._updating_grade_ui = True
        try:
            self.stl_grade_wall_edit.setText(f"{max(0.05, end_wall):.2f}")
        finally:
            self._updating_grade_ui = False
        self._update_grading_feedback()

    def on_grade_mode_changed(self, _index: int = 0):
        mode = self.stl_grade_combo.currentData() or "none"
        active = mode != "none"
        skeletal = self.stl_skeletal_check.isChecked()

        if not self._updating_grade_ui and mode == "none":
            self._updating_grade_ui = True
            try:
                idx = self.stl_grade_preset_combo.findData("none")
                if idx >= 0:
                    self.stl_grade_preset_combo.setCurrentIndex(idx)
            finally:
                self._updating_grade_ui = False

        for widgets in (
            (self.stl_grade_infill_label, self.stl_grade_infill_start_lbl, self.stl_grade_infill_edit),
            (self.stl_grade_wall_label, self.stl_grade_wall_start_lbl, self.stl_grade_wall_edit),
            (self.stl_grade_cell_label, self.stl_grade_cell_start_lbl, self.stl_grade_cell_edit),
        ):
            for w in widgets:
                if w is not None:
                    w.setEnabled(active)

        for w in (self.stl_grade_bias_label, self.stl_grade_bias_start_lbl, self.stl_grade_bias_edit):
            if w is not None:
                w.setEnabled(active and not skeletal)

        is_edt = mode == "shell_to_core"
        self.stl_grade_depth_label.setEnabled(is_edt)
        self.stl_grade_depth_edit.setEnabled(is_edt)

        self._update_grading_feedback()
        self._refresh_stl_preview(reset_camera=False)

    def _update_grading_feedback(self):
        start_wall, start_cell, start_bias, start_infill, is_skeletal, tpms_type = (
            self._current_stl_start_values()
        )
        self.stl_grade_infill_start_lbl.setText(f"{start_infill:.0f}% →")
        self.stl_grade_wall_start_lbl.setText(f"{start_wall:.2f} →")
        self.stl_grade_cell_start_lbl.setText(f"{start_cell:.2f} →")
        self.stl_grade_bias_start_lbl.setText(f"{start_bias:.2f} →")

        mode = self.stl_grade_combo.currentData() or "none"
        if mode == "none":
            self.stl_grade_preview_bar.set_state(
                False, start_wall, start_wall, start_cell, start_cell
            )
            self.stl_grade_summary_label.setStyleSheet("color: #95a5a6; font-size: 11px;")
            self.stl_grade_summary_label.setText(
                f"Uniform lattice — ~{start_infill:.0f}% infill ({start_wall:.2f} mm wall, {start_cell:.2f} mm cell) everywhere."
            )
            return

        def _opt(edit: QLineEdit) -> float | None:
            txt = edit.text().strip()
            if not txt:
                return None
            try:
                return float(txt)
            except ValueError:
                return None

        end_wall_opt = _opt(self.stl_grade_wall_edit)
        end_cell_opt = _opt(self.stl_grade_cell_edit)
        end_bias_opt = None if is_skeletal else _opt(self.stl_grade_bias_edit)
        depth_opt = _opt(self.stl_grade_depth_edit) if mode == "shell_to_core" else None

        end_wall = end_wall_opt if end_wall_opt is not None else start_wall
        end_cell = end_cell_opt if end_cell_opt is not None else start_cell
        end_infill = STLConversionParams.estimate_infill(
            end_cell, end_wall, is_skeletal, tpms_type=tpms_type
        )

        endpoints = {
            "shell_to_core": ("Outer Surface", "Core"),
            "axial_x": ("Left (X min)", "Right (X max)"),
            "axial_y": ("Front (Y min)", "Back (Y max)"),
            "axial_z": ("Bottom (Z min)", "Top (Z max)"),
            "radial": ("Center (XY)", "Outer Edge"),
        }
        start_name, end_name = endpoints.get(mode, ("Start", "End"))
        has_change = any(v is not None for v in (end_wall_opt, end_cell_opt, end_bias_opt))

        self.stl_grade_preview_bar.set_state(
            has_change, start_wall, end_wall, start_cell, end_cell, start_name, end_name
        )

        if not has_change:
            self.stl_grade_summary_label.setStyleSheet("color: #f39c12; font-size: 11px;")
            self.stl_grade_summary_label.setText(
                "Pick a Quick Preset or enter at least one End value to activate grading."
            )
            return

        depth_note = ""
        if mode == "shell_to_core":
            if depth_opt is not None and depth_opt > 0:
                depth_note = f" over {depth_opt:.1f} mm depth"
            elif self._preview_stl_mesh is not None:
                b = self._preview_stl_mesh.bounds
                est_depth = 0.5 * min(b[1] - b[0], b[3] - b[2], b[5] - b[4])
                depth_note = f" (auto depth ~{est_depth:.1f} mm)"
            else:
                depth_note = " (auto depth)"

        self.stl_grade_summary_label.setStyleSheet("color: #d0d7de; font-size: 11px;")
        self.stl_grade_summary_label.setText(
            f"{start_name}: ~{start_infill:.0f}% ({start_wall:.2f} mm wall, {start_cell:.2f} mm cell)  →  "
            f"{end_name}{depth_note}: ~{end_infill:.0f}% ({end_wall:.2f} mm wall, {end_cell:.2f} mm cell)"
        )

    def _refresh_stl_preview(self, reset_camera: bool = False):
        """Render the loaded input STL colored by the active grading profile (instant preview)."""
        if self._preview_stl_mesh is None or self.generator is not None:
            return
        mesh = self._preview_stl_mesh
        mode = self.stl_grade_combo.currentData() or "none"

        self.plotter.clear()
        if mode == "none":
            self.plotter.add_mesh(mesh, color="#5dade2", show_edges=True)
        else:
            pts = np.asarray(mesh.points, dtype=np.float64)
            b = mesh.bounds
            ex = max(b[1] - b[0], 1e-6)
            ey = max(b[3] - b[2], 1e-6)
            ez = max(b[5] - b[4], 1e-6)

            if mode == "axial_x":
                u = np.clip((pts[:, 0] - b[0]) / ex, 0.0, 1.0)
            elif mode == "axial_y":
                u = np.clip((pts[:, 1] - b[2]) / ey, 0.0, 1.0)
            elif mode == "axial_z":
                u = np.clip((pts[:, 2] - b[4]) / ez, 0.0, 1.0)
            elif mode == "radial":
                cx = 0.5 * (b[0] + b[1])
                cy = 0.5 * (b[2] + b[3])
                max_r = max(0.5 * max(ex, ey), 1e-6)
                u = np.clip(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / max_r, 0.0, 1.0)
            else:  # shell_to_core: outer surface is u=0 (Start)
                u = np.zeros(len(pts), dtype=np.float64)

            self.plotter.add_mesh(
                mesh,
                scalars=u,
                cmap="coolwarm",
                clim=[0.0, 1.0],
                show_edges=True,
                show_scalar_bar=True,
                scalar_bar_args={"title": "Grading: Start (0) → End (1)"},
            )

        self.plotter.add_bounding_box()
        self.plotter.show_axes()
        if reset_camera:
            self.plotter.reset_camera()
        else:
            self.plotter.render()

    def on_skeletal_toggled(self, _checked: bool = False):
        is_skeletal_stl = self.tabs.currentIndex() == 1 and self.stl_skeletal_check.isChecked()
        skeletal = self.stl_skeletal_check.isChecked()
        self.stl_bias_label.setEnabled(not skeletal)
        self.stl_bias_edit.setEnabled(not skeletal)
        grade_active = (self.stl_grade_combo.currentData() or "none") != "none"
        for w in (self.stl_grade_bias_label, self.stl_grade_bias_start_lbl, self.stl_grade_bias_edit):
            if w is not None:
                w.setEnabled(grade_active and not skeletal)
        self.show_cav_a_check.setEnabled(not is_skeletal_stl)
        self.show_cav_b_check.setEnabled(not is_skeletal_stl)
        if is_skeletal_stl:
            self.show_cav_a_check.setChecked(False)
            self.show_cav_b_check.setChecked(False)
            self.show_wall_check.setChecked(True)
        self._update_grading_feedback()

    def on_view_mode_changed(self):
        if self.generator and self.generator.mesh:
            self.render_current_mesh(reset_camera=False, refresh_stats=False)

    def on_slice_toggled(self, checked: bool):
        self.slice_axis_combo.setEnabled(checked)
        self.slice_reverse_check.setEnabled(checked)
        self.slice_slider.setEnabled(checked)
        self._update_slice_label()
        self._apply_clip_plane()
        if self.generator and self.generator.mesh:
            self.plotter.render()

    def on_slice_changed(self):
        self._update_slice_label()
        self._apply_clip_plane()
        if self.generator and self.generator.mesh:
            self.plotter.render()

    def _get_slice_plane(self) -> tuple[tuple[float, float, float], list[float], float] | None:
        if self.generator is None or self.generator.mesh is None or self.generator.mesh.n_points == 0:
            return None
        axis_idx = self.slice_axis_combo.currentIndex()
        sign = -1.0 if self.slice_reverse_check.isChecked() else 1.0
        normals = ((sign, 0.0, 0.0), (0.0, sign, 0.0), (0.0, 0.0, sign))
        bounds = self.generator.mesh.bounds
        b_min, b_max = bounds[2 * axis_idx], bounds[2 * axis_idx + 1]
        t = self.slice_slider.value() / 100.0
        coord = b_min + t * (b_max - b_min)
        origin = list(self.generator.mesh.center)
        origin[axis_idx] = coord
        return normals[axis_idx], origin, coord

    def _update_slice_label(self):
        pct = self.slice_slider.value()
        plane = self._get_slice_plane()
        if plane is not None:
            _, _, coord = plane
            self.slice_pos_label.setText(f"{pct}% ({coord:.2f})")
        else:
            self.slice_pos_label.setText(f"{pct}%")

    def _apply_clip_plane(self):
        plane = self._get_slice_plane()
        enabled = self.slice_enable_check.isChecked() and plane is not None
        if enabled:
            normal, origin, _ = plane
            self._clip_plane.SetNormal(*normal)
            self._clip_plane.SetOrigin(*origin)

        for actor in self._mesh_actors:
            mapper = getattr(actor, "mapper", None)
            if mapper is None:
                continue
            mapper.RemoveAllClippingPlanes()
            if enabled:
                mapper.AddClippingPlane(self._clip_plane)

    def _compute_density_scalars(self, wall) -> np.ndarray:
        """Compute per-vertex estimated infill density (%) for the generated mesh."""
        pts = np.asarray(wall.points, dtype=np.float64)
        n_pts = len(pts)
        stl_params = getattr(self.generator, "last_stl_params", None) if self.generator else None

        if stl_params is None:
            # Parametric gyroid: uniform solid fraction from mesh stats if available
            stats = self.generator.get_mesh_stats() if self.generator else {}
            porosity = stats.get("porosity_pct")
            val = max(0.0, min(100.0, 100.0 - float(porosity))) if porosity is not None else 50.0
            return np.full(n_pts, val, dtype=np.float64)

        mode = stl_params.grade_mode
        if mode == "none":
            val = STLConversionParams.estimate_infill(
                stl_params.cell_size,
                stl_params.wall_thickness,
                stl_params.is_skeletal,
                tpms_type=stl_params.tpms_type,
            )
            return np.full(n_pts, val, dtype=np.float64)

        bounds = getattr(self.generator, "_input_bounds", None)
        if bounds is not None:
            min_b, max_b = bounds[0], bounds[1]
        else:
            b = wall.bounds
            min_b = np.array([b[0], b[2], b[4]], dtype=np.float64)
            max_b = np.array([b[1], b[3], b[5]], dtype=np.float64)
        extent = np.maximum(max_b - min_b, 1e-6)

        if mode == "axial_x":
            u = np.clip((pts[:, 0] - min_b[0]) / extent[0], 0.0, 1.0)
        elif mode == "axial_y":
            u = np.clip((pts[:, 1] - min_b[1]) / extent[1], 0.0, 1.0)
        elif mode == "axial_z":
            u = np.clip((pts[:, 2] - min_b[2]) / extent[2], 0.0, 1.0)
        elif mode == "radial":
            cx = min_b[0] + 0.5 * extent[0]
            cy = min_b[1] + 0.5 * extent[1]
            max_r = max(0.5 * max(extent[0], extent[1]), 1e-6)
            u = np.clip(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) / max_r, 0.0, 1.0)
        elif mode == "shell_to_core":
            surf_pts = getattr(self.generator, "_input_surface_points", None)
            if surf_pts is not None and len(surf_pts) > 0:
                from scipy.spatial import cKDTree

                dists, _ = cKDTree(surf_pts).query(pts, workers=-1)
                depth = (
                    stl_params.grade_depth
                    if (stl_params.grade_depth is not None and stl_params.grade_depth > 0)
                    else max(float(dists.max()), 1e-6)
                )
                u = np.clip(dists / depth, 0.0, 1.0)
            else:
                u = np.zeros(n_pts, dtype=np.float64)
        else:
            u = np.zeros(n_pts, dtype=np.float64)

        w_start = stl_params.wall_thickness
        w_end = stl_params.grade_wall_end if stl_params.grade_wall_end is not None else w_start
        c_start = stl_params.cell_size
        c_end = stl_params.grade_cell_end if stl_params.grade_cell_end is not None else c_start

        w_u = w_start + (w_end - w_start) * u
        c_u = np.maximum(c_start + (c_end - c_start) * u, 1e-6)
        ratio = w_u / c_u

        if stl_params.is_skeletal:
            density = np.clip(50.0 + 95.5 * (ratio - 0.2), 5.0, 95.0)
        else:
            factor = get_tpms_kernel(stl_params.tpms_type).surface_factor
            density = np.clip(factor * ratio, 5.0, 95.0)
        return density

    def render_current_mesh(self, reset_camera: bool = True, refresh_stats: bool = True):
        self._mesh_actors.clear()
        self.plotter.clear()
        if self.generator is None or self.generator.mesh is None:
            return

        if self.generator.mesh.n_points == 0:
            QMessageBox.warning(
                self,
                "Empty Mesh",
                "Generated mesh contains 0 points. Check cell size or wall thickness.",
            )
            return

        self._update_slice_label()
        show_cavs = self.show_cav_a_check.isChecked() or self.show_cav_b_check.isChecked()

        if self.show_wall_check.isChecked():
            wall = self.generator.mesh
            if show_cavs:
                actor = self.plotter.add_mesh(wall, color="#cccccc", opacity=0.25)
            else:
                color_mode = (
                    self.color_mode_combo.currentData()
                    if hasattr(self, "color_mode_combo")
                    else "z"
                )
                if color_mode == "density":
                    scalars = self._compute_density_scalars(wall)
                    actor = self.plotter.add_mesh(
                        wall,
                        scalars=scalars,
                        cmap="coolwarm",
                        show_scalar_bar=True,
                        scalar_bar_args={"title": "Infill Density (%)"},
                    )
                else:
                    actor = self.plotter.add_mesh(
                        wall, scalars=wall.points[:, -1], show_scalar_bar=False
                    )
            self._mesh_actors.append(actor)

        if self.show_cav_a_check.isChecked() and self.generator.cavity_a is not None:
            cav_a = self.generator.cavity_a
            if cav_a.n_points > 0:
                self._mesh_actors.append(
                    self.plotter.add_mesh(cav_a, color="#e74c3c", opacity=0.85)
                )

        if self.show_cav_b_check.isChecked() and self.generator.cavity_b is not None:
            cav_b = self.generator.cavity_b
            if cav_b.n_points > 0:
                self._mesh_actors.append(
                    self.plotter.add_mesh(cav_b, color="#3498db", opacity=0.85)
                )

        self._apply_clip_plane()
        self.plotter.add_bounding_box()
        self.plotter.show_axes()
        if reset_camera:
            self.plotter.reset_camera()
        if refresh_stats:
            self.update_stats()
        self.stats_box.setVisible(True)

    def on_generation_finished(self, generator: GyroidGenerator):
        self.generator = generator
        elapsed = time.perf_counter() - self._gen_start_time
        n_cells = generator.mesh.n_cells if generator.mesh is not None else 0
        self.status_bar.showMessage(f"Ready: {n_cells:,} faces generated in {elapsed:.2f}s")
        self.generate_btn.setEnabled(True)
        self.convert_stl_btn.setEnabled(True)
        self.progressBar.setVisible(False)
        stl_params = getattr(generator, "last_stl_params", None)
        if (
            stl_params is not None
            and stl_params.grade_mode != "none"
            and any(
                v is not None
                for v in (
                    stl_params.grade_wall_end,
                    stl_params.grade_cell_end,
                    stl_params.grade_bias_end,
                )
            )
            and hasattr(self, "color_mode_combo")
        ):
            idx = self.color_mode_combo.findData("density")
            if idx >= 0:
                self.color_mode_combo.blockSignals(True)
                self.color_mode_combo.setCurrentIndex(idx)
                self.color_mode_combo.blockSignals(False)
        self.render_current_mesh()

    def on_generation_error(self, error_msg):
        QMessageBox.critical(self, "Generation Failed", error_msg)
        self.generate_btn.setEnabled(True)
        self.convert_stl_btn.setEnabled(True)
        self.progressBar.setVisible(False)
        self.status_bar.showMessage("Operation failed")

    def update_stats(self):
        if self.generator and self.generator.mesh:
            stats = self.generator.get_mesh_stats()
            self.stats_labels["n_points"].setText(f"Points: {stats['n_points']:,}")
            self.stats_labels["n_cells"].setText(f"Cells: {stats['n_cells']:,}")
            self.stats_labels["volume"].setText(f"Volume: {stats['volume']:.2f} cubic units")
            self.stats_labels["surface_area"].setText(
                f"Surface Area: {stats['surface_area']:.2f} square units"
            )
            porosity = stats.get("porosity_pct")
            porosity_str = f"{porosity:.1f}%" if porosity is not None else "-"
            self.stats_labels["porosity"].setText(f"Porosity: {porosity_str}")
            self.stats_labels["sa_to_vol"].setText(
                f"Surface/Volume: {stats['sa_to_vol_ratio']:.3f} unit\u207b\u00b9"
            )
            if stats["is_watertight"]:
                self.stats_labels["watertight"].setText("Watertight: Yes (0 open edges)")
            else:
                self.stats_labels["watertight"].setText(
                    f"Watertight: No ({stats['n_open_edges']:,} open edges)"
                )

    def save_mesh(self, file_type):
        if not self.generator or not self.generator.mesh:
            QMessageBox.warning(self, "Warning", "Please generate a mesh first")
            return

        ext_filter = "STL Files (*.stl)" if file_type == "stl" else "OBJ Files (*.obj)"
        filename, _ = QFileDialog.getSaveFileName(
            self, f"Save as {file_type.upper()}", "", ext_filter
        )
        if filename:
            if not filename.lower().endswith(f".{file_type}"):
                filename += f".{file_type}"
            match file_type:
                case "stl":
                    is_stl_tab = self.tabs.currentIndex() == 1
                    flat_cap = self.stl_cap_check.isChecked() if is_stl_tab else False
                    self.generator.save_stl(filename, flat_cap=flat_cap)
                case "obj":
                    self.generator.save_obj(filename)
            self.status_bar.showMessage(f"Saved to {filename}")

    def new_project(self):
        self.generator = None
        self._mesh_actors.clear()
        self.stats_box.setVisible(False)
        self.apply_preset("default")
        self.stl_shell_check.setChecked(False)
        self.show_wall_check.setChecked(True)
        self.show_cav_a_check.setChecked(False)
        self.show_cav_b_check.setChecked(False)
        self.slice_enable_check.setChecked(False)
        self.slice_axis_combo.setCurrentIndex(0)
        self.slice_reverse_check.setChecked(False)
        self.slice_slider.setValue(50)
        self._update_slice_label()
        self.plotter.clear()
        self.status_bar.showMessage("New project started")

    def apply_theme(self, theme_name="dark"):
        if theme_name == "dark":
            self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        else:
            self.setStyleSheet("")


def main():
    app = QApplication(sys.argv)
    window = GyroidGeneratorGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
