"""Tests for PyRoid PyQt5 graphical user interface and interaction."""

import json
import os

import pytest
import trimesh
from PyQt5.QtCore import QMimeData, QPoint, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication

from pyroid.core import GyroidGenerator
from pyroid.gui import GyroidGeneratorGUI
from pyroid.models import GyroidParams


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def gui_window(qapp):
    win = GyroidGeneratorGUI()
    yield win
    win.close()


def test_drag_and_drop_stl(gui_window, tmp_path):
    stl_path = str(tmp_path / "test_cube.stl")
    cube = trimesh.creation.box(extents=[10, 10, 10])
    cube.export(stl_path)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(stl_path)])

    # Test dragEnterEvent accepts STL
    enter_event = QDragEnterEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dragEnterEvent(enter_event)
    assert enter_event.isAccepted()

    # Test dropEvent processes STL
    drop_event = QDropEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dropEvent(drop_event)
    assert drop_event.isAccepted()
    assert os.path.samefile(gui_window.stl_path_edit.text(), stl_path)
    assert gui_window.tabs.currentIndex() == 0  # Switched to STL Conversion tab


def test_drag_and_drop_plotter_event_filter(gui_window, tmp_path):
    stl_path = str(tmp_path / "test_plotter_cube.stl")
    cube = trimesh.creation.box(extents=[10, 10, 10])
    cube.export(stl_path)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(stl_path)])

    enter_event = QDragEnterEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    drop_event = QDropEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)

    # Deliver directly to plotter child widget
    QApplication.sendEvent(gui_window.plotter, enter_event)
    QApplication.sendEvent(gui_window.plotter, drop_event)

    assert drop_event.isAccepted()
    assert os.path.samefile(gui_window.stl_path_edit.text(), stl_path)
    assert gui_window.tabs.currentIndex() == 0


def test_drag_and_drop_preset_json(gui_window, tmp_path):
    preset_path = str(tmp_path / "custom_preset.json")
    preset_content = {"res": 55, "a": 18.5, "wall_thickness": 0.4}
    with open(preset_path, "w", encoding="utf-8") as f:
        json.dump(preset_content, f)

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(preset_path)])

    enter_event = QDragEnterEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dragEnterEvent(enter_event)
    assert enter_event.isAccepted()

    drop_event = QDropEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dropEvent(drop_event)
    assert drop_event.isAccepted()

    assert gui_window.tabs.currentIndex() == 1  # Switched to Parametric tab
    assert gui_window.params_widgets["res"].text() == "55"
    assert gui_window.params_widgets["a"].text() == "18.5"
    assert gui_window.params_widgets["wall_thickness"].text() == "0.4"


def test_drag_and_drop_reject_unsupported(gui_window, tmp_path):
    txt_path = str(tmp_path / "readme.txt")
    with open(txt_path, "w") as f:
        f.write("unsupported file")

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(txt_path)])

    enter_event = QDragEnterEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dragEnterEvent(enter_event)
    assert not enter_event.isAccepted()

    drop_event = QDropEvent(QPoint(0, 0), Qt.CopyAction, mime, Qt.NoButton, Qt.NoModifier)
    gui_window.dropEvent(drop_event)
    assert not drop_event.isAccepted()


def test_slice_inspector_and_manufacturability_stats(gui_window):
    generator = GyroidGenerator(GyroidParams(res=20))
    generator.generate()
    gui_window.on_generation_finished(generator)

    assert not gui_window.stats_box.isHidden()
    assert "Porosity:" in gui_window.stats_labels["porosity"].text()
    assert "%" in gui_window.stats_labels["porosity"].text()
    assert "Surface/Volume:" in gui_window.stats_labels["sa_to_vol"].text()
    assert "Watertight:" in gui_window.stats_labels["watertight"].text()

    # Slice controls start disabled, then enable when toggled
    assert not gui_window.slice_enable_check.isChecked()
    assert not gui_window.slice_axis_combo.isEnabled()
    assert not gui_window.slice_reverse_check.isEnabled()
    assert not gui_window.slice_slider.isEnabled()

    gui_window.slice_enable_check.setChecked(True)
    assert gui_window.slice_axis_combo.isEnabled()
    assert gui_window.slice_reverse_check.isEnabled()
    assert gui_window.slice_slider.isEnabled()

    # Test switching axes (X, Y, Z), reversing plane, and moving the slider
    expected_normals = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for axis_idx, expected_normal in enumerate(expected_normals):
        gui_window.slice_reverse_check.setChecked(False)
        gui_window.slice_axis_combo.setCurrentIndex(axis_idx)
        gui_window.slice_slider.setValue(25)
        plane = gui_window._get_slice_plane()
        assert plane is not None
        assert plane[0] == expected_normal
        assert "25%" in gui_window.slice_pos_label.text()
        assert gui_window._clip_plane.GetNormal() == expected_normal
        assert len(gui_window._mesh_actors) > 0
        assert gui_window._mesh_actors[0].mapper.GetNumberOfClippingPlanes() == 1

        gui_window.slice_reverse_check.setChecked(True)
        flipped_normal = tuple(-c if c != 0.0 else 0.0 for c in expected_normal)
        assert gui_window._clip_plane.GetNormal() == flipped_normal

    gui_window.slice_enable_check.setChecked(False)
    assert gui_window._mesh_actors[0].mapper.GetNumberOfClippingPlanes() == 0


def test_functional_grading_ui_controls(gui_window):
    # Starts in Uniform (none) mode with grading inputs disabled
    assert gui_window.stl_grade_combo.currentData() == "none"
    assert not gui_window.stl_grade_wall_edit.isEnabled()
    assert not gui_window.stl_grade_bias_edit.isEnabled()
    assert not gui_window.stl_grade_cell_edit.isEnabled()
    assert not gui_window.stl_grade_depth_edit.isEnabled()

    # Switching to Shell-to-Core enables all grading inputs including transition depth
    idx_edt = gui_window.stl_grade_combo.findData("shell_to_core")
    gui_window.stl_grade_combo.setCurrentIndex(idx_edt)
    assert gui_window.stl_grade_wall_edit.isEnabled()
    assert gui_window.stl_grade_bias_edit.isEnabled()
    assert gui_window.stl_grade_cell_edit.isEnabled()
    assert gui_window.stl_grade_depth_edit.isEnabled()

    # Switching to Axial Z keeps wall/bias/cell enabled, disables transition depth
    idx_z = gui_window.stl_grade_combo.findData("axial_z")
    gui_window.stl_grade_combo.setCurrentIndex(idx_z)
    assert gui_window.stl_grade_wall_edit.isEnabled()
    assert not gui_window.stl_grade_depth_edit.isEnabled()

    # Quick preset auto-populates direction, end wall, and live summary
    idx_preset = gui_window.stl_grade_preset_combo.findData("strong_shell_light_core")
    gui_window.stl_grade_preset_combo.setCurrentIndex(idx_preset)
    assert gui_window.stl_grade_combo.currentData() == "shell_to_core"
    assert float(gui_window.stl_grade_wall_edit.text()) < float(gui_window.stl_wall_edit.text())
    assert gui_window.stl_grade_preview_bar.active
    assert "Outer Surface" in gui_window.stl_grade_summary_label.text()

    # Typing End Infill % auto-computes End Wall Thickness
    gui_window.stl_grade_infill_edit.setText("15")
    gui_window.on_grade_infill_edited("15")
    assert float(gui_window.stl_grade_wall_edit.text()) > 0.0

    # Skeletal mode disables End Bias
    gui_window.stl_skeletal_check.setChecked(True)
    assert not gui_window.stl_grade_bias_edit.isEnabled()


def test_color_by_infill_density(gui_window, tmp_path):
    import trimesh

    from pyroid.models import STLConversionParams

    stl_path = str(tmp_path / "box.stl")
    trimesh.creation.box(extents=[10.0, 10.0, 10.0]).export(stl_path)

    gen = GyroidGenerator()
    params = STLConversionParams(
        stl_path=stl_path,
        resolution=20,
        cell_size=4.0,
        wall_thickness=1.0,
        grade_mode="axial_z",
        grade_wall_end=0.4,
    )
    gen.convert_from_stl(params, include_cavities=False)
    gui_window.on_generation_finished(gen)

    assert gui_window.color_mode_combo.currentData() == "density"
    scalars = gui_window._compute_density_scalars(gen.mesh)
    assert len(scalars) == gen.mesh.n_points
    assert scalars.max() > scalars.min()
