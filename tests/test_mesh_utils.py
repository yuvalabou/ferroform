"""Unit tests for mesh processing utilities and STL infill conversion."""

import numpy as np
import pyroid_rs
import pyvista as pv
import trimesh

from pyroid import core
from pyroid.core import (
    GyroidGenerator,
    convert_stl_to_gyroid,
    export_stl_binary,
    load_trimesh,
    prepare_for_printing,
)
from pyroid.models import GyroidParams, STLConversionParams


def test_stl_gyroid_conversion(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    assert len(tri_mesh.vertices) > 0

    params = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=3.0, wall_thickness=0.5
    )
    gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
    assert isinstance(gyroid_mesh, pv.PolyData)
    assert gyroid_mesh.n_points > 0

    tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
    assert isinstance(tri_printable, trimesh.Trimesh)


def test_stl_gyroid_conversion_with_preserved_shell(tmp_path):
    cube = trimesh.creation.box(extents=[12, 12, 12])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    params = STLConversionParams(
        stl_path=tmp_path_stl,
        resolution=30,
        cell_size=3.0,
        wall_thickness=0.5,
        preserve_shell=True,
        shell_thickness=1.0,
    )

    gyroid_mesh = convert_stl_to_gyroid(tri_mesh, params)
    assert isinstance(gyroid_mesh, pv.PolyData)
    assert gyroid_mesh.n_points > 0

    tri_printable = prepare_for_printing(gyroid_mesh, flat_cap=False)
    assert isinstance(tri_printable, trimesh.Trimesh)
    assert tri_printable.is_watertight
    assert 0 < tri_printable.volume < cube.volume

    np.testing.assert_allclose(tri_printable.bounds, cube.bounds, atol=1e-3)
    diffs = np.min(
        np.linalg.norm(tri_printable.vertices[:, None, :] - cube.vertices[None, :, :], axis=2),
        axis=0,
    )
    assert (diffs < 1e-4).all()


def test_stl_gyroid_calibrated_wall_thickness(tmp_path):
    cube = trimesh.creation.box(extents=[15, 15, 15])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    params_thin = STLConversionParams(
        stl_path=tmp_path_stl, resolution=40, cell_size=5.0, wall_thickness=0.4
    )
    mesh_thin = convert_stl_to_gyroid(tri_mesh, params_thin)

    params_thick = STLConversionParams(
        stl_path=tmp_path_stl, resolution=40, cell_size=5.0, wall_thickness=1.0
    )
    mesh_thick = convert_stl_to_gyroid(tri_mesh, params_thick)

    assert mesh_thin.n_points > 0
    assert mesh_thick.n_points > 0
    assert mesh_thick.volume > mesh_thin.volume


def test_stl_gyroid_progress_callback(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    params = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8
    )
    updates = []

    convert_stl_to_gyroid(
        tri_mesh, params, progress_callback=lambda pct, msg: updates.append((pct, msg))
    )

    assert len(updates) >= 4
    percentages = [u[0] for u in updates]
    assert percentages == sorted(percentages)
    assert all(0 <= p <= 100 for p in percentages)
    assert any("containment" in u[1].lower() for u in updates)


def test_stl_gyroid_skeletal_conversion(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    params_sheet = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=False
    )
    params_skeletal = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.8, is_skeletal=True
    )

    mesh_sheet = convert_stl_to_gyroid(tri_mesh, params_sheet)
    mesh_skeletal = convert_stl_to_gyroid(tri_mesh, params_skeletal)

    assert mesh_skeletal.n_points > 0
    assert mesh_skeletal.n_cells > 0
    assert mesh_skeletal.n_cells < mesh_sheet.n_cells

    params_skeletal_shell = STLConversionParams(
        stl_path=tmp_path_stl,
        resolution=30,
        cell_size=5.0,
        wall_thickness=0.8,
        preserve_shell=True,
        shell_thickness=0.8,
        is_skeletal=True,
    )
    mesh_shell = convert_stl_to_gyroid(tri_mesh, params_skeletal_shell)
    assert mesh_shell.n_points > 0
    assert mesh_shell.n_cells > 0


def test_stl_gyroid_asymmetric_cavities(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)

    tri_mesh = load_trimesh(tmp_path_stl)
    params_pos = STLConversionParams(
        stl_path=tmp_path_stl, resolution=30, cell_size=5.0, wall_thickness=0.6, bias=0.4
    )
    mesh_pos = convert_stl_to_gyroid(tri_mesh, params_pos)
    assert hasattr(mesh_pos, "cavity_a") and hasattr(mesh_pos, "cavity_b")
    assert mesh_pos.cavity_a.n_cells > mesh_pos.cavity_b.n_cells


def test_stl_gyroid_target_infill_percent(tmp_path):
    cube = trimesh.creation.box(extents=[12, 12, 12])
    tmp_path_stl = str(tmp_path / "cube.stl")
    cube.export(tmp_path_stl)
    tri_mesh = load_trimesh(tmp_path_stl)

    cell_25 = STLConversionParams.estimate_cell_size(25.0, 0.8)
    cell_60 = STLConversionParams.estimate_cell_size(60.0, 0.8)
    assert cell_25 > cell_60
    assert abs(STLConversionParams.estimate_infill(cell_25, 0.8) - 25.0) < 0.5

    mesh_low = convert_stl_to_gyroid(
        tri_mesh,
        STLConversionParams(
            stl_path=tmp_path_stl,
            resolution=35,
            cell_size=cell_25,
            wall_thickness=0.8,
            infill_percent=25.0,
        ),
    )
    mesh_high = convert_stl_to_gyroid(
        tri_mesh,
        STLConversionParams(
            stl_path=tmp_path_stl,
            resolution=35,
            cell_size=cell_60,
            wall_thickness=0.8,
            infill_percent=60.0,
        ),
    )
    assert mesh_high.volume > mesh_low.volume


def test_export_stl_binary_buildstar_compatibility(tmp_path):
    cube = trimesh.creation.box(extents=[10, 10, 10])
    in_stl = str(tmp_path / "in_cube.stl")
    out_stl = str(tmp_path / "out_buildstar.stl")
    cube.export(in_stl)

    tri_mesh = load_trimesh(in_stl)
    params = STLConversionParams(
        stl_path=in_stl, resolution=30, cell_size=4.0, wall_thickness=0.6
    )
    mesh = convert_stl_to_gyroid(tri_mesh, params)
    export_stl_binary(mesh, out_stl)

    # 1. Inspect raw binary STL structure
    with open(out_stl, "rb") as f:
        header = f.read(80)
        face_count_bytes = f.read(4)
        body = f.read()

    assert len(header) == 80
    assert not header.lower().startswith(b"solid")
    assert header.strip().startswith(b"PyRoid")
    face_count = int.from_bytes(face_count_bytes, "little")
    assert len(body) == face_count * 50

    # 2. Inspect mesh topology and normals
    loaded = trimesh.load(out_stl, force="mesh")
    assert loaded.is_watertight
    assert loaded.is_winding_consistent
    assert (loaded.area_faces < 1e-12).sum() == 0
    norms = np.linalg.norm(loaded.face_normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    assert not np.isnan(loaded.face_normals).any()


def test_parametric_export_buildstar_compatibility(tmp_path):
    out_stl = str(tmp_path / "out_parametric.stl")
    generator = GyroidGenerator(GyroidParams(res=25))
    generator.generate()
    generator.save_stl(out_stl)

    with open(out_stl, "rb") as f:
        header = f.read(80)
        face_count_bytes = f.read(4)
        body = f.read()

    assert len(header) == 80
    assert not header.lower().startswith(b"solid")
    assert header.strip().startswith(b"PyRoid")
    face_count = int.from_bytes(face_count_bytes, "little")
    assert len(body) == face_count * 50

    loaded = trimesh.load(out_stl, force="mesh")
    assert loaded.is_winding_consistent
    assert (loaded.area_faces < 1e-12).sum() == 0
    norms = np.linalg.norm(loaded.face_normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-3)
    assert not np.isnan(loaded.face_normals).any()


def test_direct_rust_stl_writer_fast_path(tmp_path, monkeypatch):
    # Direct pyroid_rs.write_stl_binary test (with "solid" header guard & inverted winding)
    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    tris = np.array([[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]], dtype=np.uint32)
    direct_stl = str(tmp_path / "direct_rust.stl")
    pyroid_rs.write_stl_binary(pts, tris, direct_stl, "solid test")

    with open(direct_stl, "rb") as f:
        hdr = f.read(80)
    assert not hdr.lower().startswith(b"solid")
    assert hdr.startswith(b"PyRoid solid test")

    loaded_tet = trimesh.load(direct_stl, force="mesh")
    assert loaded_tet.is_watertight
    assert loaded_tet.volume > 0

    # Verify export_stl_binary bypasses prepare_for_printing when flat_cap=False & preserve_shell=False
    cube = trimesh.creation.box(extents=[10, 10, 10])
    in_stl = str(tmp_path / "in_cube_fast.stl")
    out_stl = str(tmp_path / "out_fast.stl")
    cube.export(in_stl)

    tri_mesh = load_trimesh(in_stl)
    params = STLConversionParams(
        stl_path=in_stl, resolution=25, cell_size=4.0, wall_thickness=0.6
    )
    mesh = convert_stl_to_gyroid(tri_mesh, params)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("prepare_for_printing should be bypassed on Rust fast path")

    monkeypatch.setattr(core, "prepare_for_printing", _fail_if_called)
    export_stl_binary(mesh, out_stl, flat_cap=False)

    loaded = trimesh.load(out_stl, force="mesh")
    assert loaded.is_watertight
    assert loaded.is_winding_consistent
    assert (loaded.area_faces < 1e-12).sum() == 0


def test_stl_gyroid_flat_cap_and_shell_native_rust(tmp_path, monkeypatch):
    cube = trimesh.creation.box(extents=[12, 12, 12])
    in_stl = str(tmp_path / "in_cap.stl")
    out_stl = str(tmp_path / "out_cap.stl")
    cube.export(in_stl)

    tri_mesh = load_trimesh(in_stl)
    params = STLConversionParams(
        stl_path=in_stl,
        resolution=30,
        cell_size=4.0,
        wall_thickness=0.6,
        preserve_shell=True,
        shell_thickness=1.0,
        flat_cap=True,
    )
    mesh = convert_stl_to_gyroid(tri_mesh, params)
    assert getattr(mesh, "_rust_manifold", False)
    assert getattr(mesh, "_rust_flat_capped", False)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("prepare_for_printing should be bypassed on Rust flat_cap fast path")

    monkeypatch.setattr(core, "prepare_for_printing", _fail_if_called)
    export_stl_binary(mesh, out_stl, flat_cap=True)

    loaded = trimesh.load(out_stl, force="mesh")
    assert loaded.is_watertight
    assert loaded.is_winding_consistent
    assert 0 < loaded.volume < cube.volume
    # 1% margin on 12mm height -> z in [-5.88, +5.88]
    np.testing.assert_allclose(loaded.bounds[:, 2], [-5.88, 5.88], atol=1e-2)

    # Also test prepare_for_printing(flat_cap=True) directly via Rust trim_flat_cap
    monkeypatch.undo()
    capped_cube = prepare_for_printing(cube, flat_cap=True)
    assert capped_cube.is_watertight
    np.testing.assert_allclose(capped_cube.bounds[:, 2], [-5.88, 5.88], atol=1e-3)


def test_export_3mf_multibody(tmp_path):
    import xml.etree.ElementTree as ET
    import zipfile

    out_3mf = str(tmp_path / "multibody.3mf")
    generator = GyroidGenerator(GyroidParams(res=20))
    generator.generate(include_cavities=True)
    generator.save_3mf(out_3mf, include_cavities=True)

    with zipfile.ZipFile(out_3mf, "r") as zf:
        names = set(zf.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model"} <= names
        model_xml = zf.read("3D/3dmodel.model")

    root = ET.fromstring(model_xml)
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert root.attrib.get("unit") == "millimeter"
    objects = root.findall("m:resources/m:object", ns)
    obj_names = {obj.attrib["name"] for obj in objects}
    assert "lattice" in obj_names
    assert "cavity_a" in obj_names
    assert "cavity_b" in obj_names

    build_items = root.findall("m:build/m:item", ns)
    assert len(build_items) == len(objects)

    scene = trimesh.load(out_3mf)
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == len(objects)
