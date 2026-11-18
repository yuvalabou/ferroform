"""Core mathematical engine and mesh generation for PyRoid.

Contains GyroidGenerator for computing parametric gyroid surfaces in cylindrical
coordinates and coordinating STL mesh infill conversions.
"""

import math
import os
from collections.abc import Callable
from dataclasses import asdict, fields
from typing import Any

import numpy as np
import pyroid_rs
import pyvista as pv
import trimesh

from pyroid.models import GyroidParams, STLConversionParams


def polydata_from_buffers(
    points: np.ndarray, cells: np.ndarray, *, rust_manifold: bool = False
) -> pv.PolyData:
    """Construct a PyVista PolyData mesh from zero-copy Rust vertex and VTK cell buffers."""
    if len(points) == 0 or len(cells) == 0:
        mesh = pv.PolyData()
    else:
        mesh = pv.PolyData(points, cells)
    if rust_manifold:
        pv.set_new_attribute(mesh, "_rust_manifold", True)
    return mesh


def load_trimesh(filepath: str) -> trimesh.Trimesh:
    """Load an STL file as a Trimesh object."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")
    loaded = trimesh.load(filepath, force="mesh")
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh object from {filepath}, got {type(loaded).__name__}")
    return loaded


def convert_stl_to_gyroid(
    stl_mesh: trimesh.Trimesh,
    params: STLConversionParams,
    progress_callback: Callable[[int, str], None] | None = None,
    include_cavities: bool = True,
) -> pv.PolyData:
    """
    Convert an input STL 3D mesh volume into a gyroid lattice structure.

    Args:
        stl_mesh: Input Trimesh object (should be a manifold/closed surface)
        params: STLConversionParams configuration

    Returns:
        PyVista PolyData object representing the generated gyroid mesh within the STL boundary.
    """
    params_dict = asdict(params)
    params_dict["include_cavities"] = include_cavities

    clean_stl = stl_mesh
    if params.preserve_shell and not stl_mesh.is_watertight:
        clean_stl = stl_mesh.copy()
        clean_stl.fill_holes()

    _is_shell_void, mesh_buf, cav_a_buf, cav_b_buf = pyroid_rs.convert_stl_to_tpms(
        np.asarray(clean_stl.vertices, dtype=np.float64),
        np.asarray(clean_stl.faces, dtype=np.uint32),
        params_dict,
        progress_callback,
    )

    def _attach_cavities(out_mesh: pv.PolyData) -> pv.PolyData:
        if cav_a_buf is not None:
            pv.set_new_attribute(out_mesh, "cavity_a", polydata_from_buffers(*cav_a_buf))
        if cav_b_buf is not None:
            pv.set_new_attribute(out_mesh, "cavity_b", polydata_from_buffers(*cav_b_buf))
        return out_mesh

    mesh = polydata_from_buffers(*mesh_buf, rust_manifold=True)
    if params.flat_cap:
        pv.set_new_attribute(mesh, "_rust_flat_capped", True)
    return _attach_cavities(mesh)


def prepare_for_printing(
    mesh: pv.PolyData | trimesh.Trimesh, flat_cap: bool = False
) -> trimesh.Trimesh:
    """
    Ensure generated mesh is manifold/watertight with consistent winding for 3D printing.
    Optimized for industrial SLS/SLM slicers (such as Farsoon BuildStar) and Netfabb.
    """
    if isinstance(mesh, pv.PolyData):
        tri_pv = mesh.triangulate() if mesh.faces.size > 0 else mesh
        faces = tri_pv.faces.reshape((-1, 4))[:, 1:]
        tri = trimesh.Trimesh(vertices=tri_pv.points, faces=faces, process=False)
    else:
        tri = mesh.copy()

    # Clean initial duplicate/degenerate geometry
    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()

    if flat_cap and len(tri.vertices) > 0:
        z_min, z_max = tri.bounds[:, 2]
        margin = (z_max - z_min) * 0.01
        if margin > 0.001:
            pts, cells = pyroid_rs.trim_flat_cap(
                np.asarray(tri.vertices, dtype=np.float32),
                np.asarray(tri.faces, dtype=np.uint32),
            )
            if len(pts) > 0 and len(cells) > 0:
                faces = cells.reshape((-1, 4))[:, 1:]
                tri = trimesh.Trimesh(vertices=pts, faces=faces, process=False)

    # Ensure watertightness and repair broken faces
    if not tri.is_watertight:
        tri.fill_holes()

    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()

        # Unify counter-clockwise winding order and outward normals
        trimesh.repair.fix_winding(tri)
        trimesh.repair.fix_normals(tri)
        trimesh.repair.fix_inversion(tri)

        if tri.volume < 0:
            tri.faces = tri.faces[:, ::-1]
            trimesh.repair.fix_normals(tri)

    return tri


def export_stl_binary(
    mesh: pv.PolyData | trimesh.Trimesh,
    filepath: str,
    header_text: str = "PyRoid 3D Model - Farsoon BuildStar and Netfabb Compatible",
    flat_cap: bool = False,
) -> None:
    """
    Export mesh to a strictly standard-compliant binary STL file.

    Guarantees compatibility with industrial Laser Powder Bed Fusion (SLS/SLM)
    slicers (such as Farsoon BuildStar, EOS RP-Tools, Materialise Magics) and CAD tools (Netfabb):
      - 80-byte ASCII header padded with spaces (never starts with 'solid' or null bytes).
      - Little-endian 32-bit unsigned facet count.
      - 50 bytes per facet with valid, normalized non-zero unit normals.
      - Watertight topology, consistent counter-clockwise face winding, no degenerate facets.
    """
    if (
        isinstance(mesh, pv.PolyData)
        and getattr(mesh, "_rust_manifold", False)
        and (not flat_cap or getattr(mesh, "_rust_flat_capped", False))
        and (mesh.n_cells == 0 or mesh.faces.size == mesh.n_cells * 4)
    ):
        if mesh.n_cells == 0:
            pts = np.empty((0, 3), dtype=np.float32)
            tris = np.empty((0, 3), dtype=np.uint32)
        else:
            pts = np.ascontiguousarray(mesh.points, dtype=np.float32)
            tris = np.ascontiguousarray(mesh.faces.reshape((-1, 4))[:, 1:], dtype=np.uint32)
        pyroid_rs.write_stl_binary(pts, tris, filepath, header_text)
        return

    tri = prepare_for_printing(mesh, flat_cap=flat_cap)

    # Compute explicit unit normal vectors per face to prevent any zero/NaN vectors in CAM slicers
    v = tri.vertices[tri.faces]
    e1 = v[:, 1] - v[:, 0]
    e2 = v[:, 2] - v[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    zero_mask = (norms < 1e-12).squeeze()
    normals[zero_mask] = [0.0, 0.0, 1.0]
    norms[zero_mask] = 1.0
    normals = (normals / norms).astype("<f4")

    n_faces = len(tri.faces)

    # 80-byte header: valid ASCII, padded with spaces, never starts with 'solid'
    header_clean = (
        header_text.replace("\r", " ").replace("\n", " ").encode("ascii", errors="replace")
    )
    if header_clean.lower().startswith(b"solid"):
        header_clean = b"PyRoid " + header_clean
    header = header_clean[:80].ljust(80, b" ")

    dtype = np.dtype([
        ("normals", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attr", "<u2"),
    ])
    data = np.empty(n_faces, dtype=dtype)
    data["normals"] = normals
    data["vertices"] = v.astype("<f4")
    data["attr"] = 0

    with open(filepath, "wb") as f:
        f.write(header)
        f.write(np.uint32(n_faces).tobytes())
        f.write(data.tobytes())


def render_offscreen(
    mesh: pv.PolyData, output_image_path: str, window_size: tuple[int, int] = (800, 600)
):
    """Render a thumbnail image of the mesh headlessly."""
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    plotter.add_bounding_box()  # type: ignore[call-arg]
    plotter.show_axes()  # type: ignore[call-arg]
    plotter.reset_camera()  # type: ignore[call-arg]
    plotter.screenshot(output_image_path, window_size=window_size)
    plotter.close()


class GyroidGenerator:
    """Core class for generating parametric gyroid structures and converting STL meshes."""

    def __init__(self, params: GyroidParams | dict[str, Any] | None = None):
        """
        Initialize the gyroid generator with parameters.

        Args:
            params: GyroidParams instance or dictionary of parameter overrides.
        """
        if isinstance(params, GyroidParams):
            self.params = params
        elif isinstance(params, dict):
            self.params = GyroidParams.from_dict(params)
        else:
            self.params = GyroidParams()

        self.mesh: pv.PolyData | None = None
        self.cavity_a: pv.PolyData | None = None
        self.cavity_b: pv.PolyData | None = None
        self._reference_volume: float | None = None
        self.last_stl_params: STLConversionParams | None = None
        self._input_bounds: np.ndarray | None = None
        self._input_surface_points: np.ndarray | None = None

    def validate_params(self) -> tuple[bool, str]:
        """Validate parameters using the GyroidParams data model."""
        return self.params.validate()

    def generate(
        self,
        progress_callback: Callable[[int, str], None] | None = None,
        include_cavities: bool = True,
    ) -> pv.PolyData:
        """
        Generate the radially symmetrical gyroid/TPMS mesh using current parameters.

        Returns:
            PyVista PolyData mesh object.

        Raises:
            ValueError: If parameters are invalid.
        """
        is_valid, error = self.validate_params()
        if not is_valid:
            raise ValueError(error)

        params_dict = self.params.to_dict()
        params_dict["include_cavities"] = include_cavities
        mesh_buf, cav_a_buf, cav_b_buf = pyroid_rs.generate_parametric_tpms(
            params_dict, progress_callback
        )
        self.mesh = polydata_from_buffers(*mesh_buf, rust_manifold=True)
        self.cavity_a = polydata_from_buffers(*cav_a_buf) if include_cavities else None
        self.cavity_b = polydata_from_buffers(*cav_b_buf) if include_cavities else None
        if self.params.shape.strip().lower() in ("box", "cube", "cartesian"):
            self._reference_volume = self.params.a * self.params.b * self.params.c
        else:
            r_inner = max(0.0, self.params.r1 - self.params.wall_thickness)
            r_outer = self.params.r1
            self._reference_volume = (
                math.pi * max(0.0, r_outer**2 - r_inner**2) * self.params.c
            )
        self.last_stl_params = None
        self._input_bounds = None
        self._input_surface_points = None

        return self.mesh

    def convert_from_stl(
        self,
        stl_params: STLConversionParams,
        progress_callback: Callable[[int, str], None] | None = None,
        include_cavities: bool = True,
    ) -> pv.PolyData:
        """
        Import an STL file and convert its 3D volume into a gyroid lattice.

        Args:
            stl_params: STLConversionParams configuration object.
            progress_callback: Optional callback receiving (percent, status_message).

        Returns:
            PyVista PolyData mesh of the gyroidized STL volume.
        """
        is_valid, err = stl_params.validate()
        if not is_valid:
            raise ValueError(err)

        if progress_callback:
            progress_callback(5, "Loading STL geometry...")

        tri_mesh = load_trimesh(stl_params.stl_path)
        self._reference_volume = float(abs(tri_mesh.volume))
        self.last_stl_params = stl_params
        self._input_bounds = tri_mesh.bounds.copy()
        verts = np.asarray(tri_mesh.vertices)
        if len(verts) > 20000:
            step = max(1, len(verts) // 20000)
            verts = verts[::step]
        self._input_surface_points = verts.copy()
        self.mesh = convert_stl_to_gyroid(
            tri_mesh,
            stl_params,
            progress_callback=progress_callback,
            include_cavities=include_cavities,
        )
        self.cavity_a = getattr(self.mesh, "cavity_a", None)
        self.cavity_b = getattr(self.mesh, "cavity_b", None)
        return self.mesh

    def save_stl(self, filename: str, flat_cap: bool = False) -> None:
        """Save the generated mesh as a 3D-printable binary STL file."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        export_stl_binary(
            self.mesh,
            filename,
            header_text="PyRoid 3D Model - Farsoon BuildStar and Netfabb Compatible",
            flat_cap=flat_cap,
        )

    def save_obj(self, filename: str) -> None:
        """Save the generated mesh as an OBJ file."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")
        self.mesh.save(filename)

    def save_screenshot(self, filename: str) -> None:
        """Save a rendered PNG thumbnail of the mesh."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")
        render_offscreen(self.mesh, filename)

    def get_mesh_stats(self) -> dict[str, Any]:
        """Return quantitative mesh and manufacturability metrics."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        volume = float(self.mesh.volume)
        surface_area = float(self.mesh.area)
        sa_to_vol_ratio = (surface_area / volume) if volume > 0.0 else 0.0
        n_open_edges = int(self.mesh.n_open_edges)
        is_watertight = n_open_edges == 0 and self.mesh.n_cells > 0

        porosity_pct: float | None = None
        if self._reference_volume is not None and self._reference_volume > 0.0:
            raw_porosity = (1.0 - (volume / self._reference_volume)) * 100.0
            porosity_pct = max(0.0, min(100.0, raw_porosity))

        return {
            "n_points": self.mesh.n_points,
            "n_cells": self.mesh.n_cells,
            "volume": volume,
            "surface_area": surface_area,
            "porosity_pct": porosity_pct,
            "sa_to_vol_ratio": sa_to_vol_ratio,
            "n_open_edges": n_open_edges,
            "is_watertight": is_watertight,
        }

    @classmethod
    def get_param_descriptions(cls) -> dict[str, str]:
        """Return parameter descriptions derived from model metadata."""
        return {f.name: str(f.metadata.get("help", "")) for f in fields(GyroidParams)}

    @classmethod
    def get_presets(cls) -> dict[str, dict[str, Any]]:
        """Return predefined parameter presets."""
        return {
            "default": GyroidParams().to_dict(),
            "fine_detail": {
                "res": 120,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 8,
                "wall_thickness": 11.5,
                "cell_radius": 1.5,
                "cell_height": 2.5,
                "bias": 0.0,
            },
            "thick_walls": {
                "res": 80,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 8,
                "wall_thickness": 15.0,
                "cell_radius": 2,
                "cell_height": 3,
                "bias": 0.0,
            },
            "dense_pattern": {
                "res": 100,
                "a": 24,
                "b": 24,
                "c": 10,
                "r1": 12,
                "r2": 0,
                "phi_scale": 12,
                "wall_thickness": 11.5,
                "cell_radius": 1.2,
                "cell_height": 1.8,
                "bias": 0.0,
            },
        }
