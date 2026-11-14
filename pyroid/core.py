"""Core mathematical engine and mesh generation for PyRoid.

Contains GyroidGenerator for computing parametric gyroid surfaces in cylindrical
coordinates and coordinating STL mesh infill conversions.
"""

from collections.abc import Callable
from dataclasses import fields
from typing import Any

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

        import pyroid_rs

        params_dict = self.params.to_dict()
        params_dict["include_cavities"] = include_cavities
        mesh_buf, cav_a_buf, cav_b_buf = pyroid_rs.generate_parametric_tpms(
            params_dict, progress_callback
        )
        self.mesh = polydata_from_buffers(*mesh_buf)
        self.cavity_a = polydata_from_buffers(*cav_a_buf) if include_cavities else None
        self.cavity_b = polydata_from_buffers(*cav_b_buf) if include_cavities else None

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
        """Return quantitative mesh metrics."""
        if self.mesh is None:
            raise ValueError("No mesh has been generated yet")

        return {
            "n_points": self.mesh.n_points,
            "n_cells": self.mesh.n_cells,
            "volume": self.mesh.volume,
            "surface_area": self.mesh.area,
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
