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
    assert gui_window.tabs.currentIndex() == 1  # Switched to STL Conversion tab


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
    assert gui_window.tabs.currentIndex() == 1


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

    assert gui_window.tabs.currentIndex() == 0  # Switched to Parametric tab
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
    assert not gui_window.slice_slider.isEnabled()

    gui_window.slice_enable_check.setChecked(True)
    assert gui_window.slice_axis_combo.isEnabled()
    assert gui_window.slice_slider.isEnabled()

    # Test switching axes (X, Y, Z) and moving the slider (GPU clipping plane, no CPU re-clip)
    expected_normals = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    for axis_idx, expected_normal in enumerate(expected_normals):
        gui_window.slice_axis_combo.setCurrentIndex(axis_idx)
        gui_window.slice_slider.setValue(25)
        plane = gui_window._get_slice_plane()
        assert plane is not None
        assert plane[0] == expected_normal
        assert "25%" in gui_window.slice_pos_label.text()
        assert gui_window._clip_plane.GetNormal() == expected_normal
        assert len(gui_window._mesh_actors) > 0
        assert gui_window._mesh_actors[0].mapper.GetNumberOfClippingPlanes() == 1

    gui_window.slice_enable_check.setChecked(False)
    assert gui_window._mesh_actors[0].mapper.GetNumberOfClippingPlanes() == 0
