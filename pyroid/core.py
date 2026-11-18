"""Core mathematical engine and mesh generation for PyRoid.

Contains GyroidGenerator for computing parametric gyroid surfaces in cylindrical
coordinates and coordinating STL mesh infill conversions.
"""

import math
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import pyroid_rs
import pyvista as pv

from pyroid.mesh_utils import (
    convert_stl_to_gyroid,
    export_stl_binary,
    load_trimesh,
    polydata_from_buffers,
    render_offscreen,
)
from pyroid.models import GyroidParams, STLConversionParams


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
        self._input_bounds = None
        self._input_surface_points = None

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
        verts = tri_mesh.vertices
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
