"""Mesh processing and volumetric utility functions.

Provides utilities for STL loading, volumetric voxel containment testing,
gyroid lattice infill generation, watertight manifold preparation, and offscreen rendering.
"""

import os
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pyvista as pv
import trimesh

from pyroid.models import STLConversionParams


def polydata_from_buffers(points: np.ndarray, cells: np.ndarray) -> pv.PolyData:
    """Construct a PyVista PolyData mesh from zero-copy Rust vertex and VTK cell buffers."""
    if len(points) == 0 or len(cells) == 0:
        return pv.PolyData()
    return pv.PolyData(points, cells)


def load_stl(filepath: str) -> tuple[trimesh.Trimesh, pv.PolyData]:
    """Load an STL file into both Trimesh and PyVista objects."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"STL file not found: {filepath}")

    tri_mesh = trimesh.load(filepath, force="mesh")
    pv_mesh = pv.read(filepath)
    return tri_mesh, pv_mesh


def convert_stl_to_gyroid(
    stl_mesh: trimesh.Trimesh,
    params: STLConversionParams,
    progress_callback: Callable[[int, str], None] | None = None,
) -> pv.PolyData:
    """
    Convert an input STL 3D mesh volume into a gyroid lattice structure.

    Args:
        stl_mesh: Input Trimesh object (should be a manifold/closed surface)
        params: STLConversionParams configuration

    Returns:
        PyVista PolyData object representing the generated gyroid mesh within the STL boundary.
    """
    import pyroid_rs

    is_shell_void, mesh_buf, cav_a_buf, cav_b_buf = pyroid_rs.convert_stl_to_tpms(
        np.asarray(stl_mesh.vertices, dtype=np.float64),
        np.asarray(stl_mesh.faces, dtype=np.uint32),
        asdict(params),
        progress_callback,
    )

    def _attach_cavities(out_mesh: pv.PolyData) -> pv.PolyData:
        if cav_a_buf is not None:
            pv.set_new_attribute(out_mesh, "cavity_a", polydata_from_buffers(*cav_a_buf))
        if cav_b_buf is not None:
            pv.set_new_attribute(out_mesh, "cavity_b", polydata_from_buffers(*cav_b_buf))
        return out_mesh

    if is_shell_void:
        void_pts, void_cells = mesh_buf
        if len(void_pts) == 0 or len(void_cells) == 0:
            if progress_callback:
                progress_callback(95, "Finalizing mesh...")
            return _attach_cavities(pv.wrap(stl_mesh))

        void_faces = void_cells.reshape((-1, 4))[:, 1:].astype(np.uint32)

        # Prepare clean STL mesh for boolean subtraction
        clean_stl = stl_mesh.copy()
        if not clean_stl.is_watertight:
            clean_stl.fill_holes()
        if clean_stl.volume < 0:
            clean_stl.faces = clean_stl.faces[:, ::-1]

        # Robust boolean subtraction: outer STL minus internal gyroid void.
        # Keeps original CAD surface triangles intact while skipping redundant Python validation.
        import manifold3d

        if progress_callback:
            progress_callback(85, "Performing manifold boolean subtraction...")

        m_outer = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.asarray(clean_stl.vertices, dtype=np.float32),
                tri_verts=np.asarray(clean_stl.faces, dtype=np.uint32),
            )
        )
        m_void = manifold3d.Manifold(
            manifold3d.Mesh(
                vert_properties=np.asarray(void_pts, dtype=np.float32),
                tri_verts=void_faces,
            )
        )
        diff = m_outer - m_void
        res_mesh = diff.to_mesh()
        tri_result = trimesh.Trimesh(
            vertices=res_mesh.vert_properties, faces=res_mesh.tri_verts, process=False
        )
        if tri_result.volume < 0:
            tri_result.faces = tri_result.faces[:, ::-1]

        if progress_callback:
            progress_callback(95, "Wrapping final mesh...")

        return _attach_cavities(pv.wrap(tri_result))

    mesh = polydata_from_buffers(*mesh_buf)
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
            clipped = False
            try:
                import manifold3d

                m = manifold3d.Manifold(
                    manifold3d.Mesh(
                        vert_properties=np.asarray(tri.vertices, dtype=np.float32),
                        tri_verts=np.asarray(tri.faces, dtype=np.uint32),
                    )
                )
                m = m.trim_by_plane((0, 0, -1), -(z_max - margin)).trim_by_plane(
                    (0, 0, 1), z_min + margin
                )
                res = m.to_mesh()
                tri = trimesh.Trimesh(
                    vertices=res.vert_properties, faces=res.tri_verts, process=False
                )
                clipped = True
            except Exception:
                clipped = False

            if not clipped:
                try:
                    tri = trimesh.intersections.slice_mesh_plane(
                        tri,
                        plane_normal=[0, 0, -1],
                        plane_origin=[0, 0, z_max - margin],
                        cap=True,
                    )
                    tri = trimesh.intersections.slice_mesh_plane(
                        tri,
                        plane_normal=[0, 0, 1],
                        plane_origin=[0, 0, z_min + margin],
                        cap=True,
                    )
                except Exception:
                    pass

    # Ensure watertightness and repair broken faces
    if not tri.is_watertight:
        tri.fill_holes()

    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()

        # If manifold3d is available, validate 2-manifold orientation
        try:
            import manifold3d

            m_check = manifold3d.Manifold(
                manifold3d.Mesh(
                    vert_properties=np.asarray(tri.vertices, dtype=np.float32),
                    tri_verts=np.asarray(tri.faces, dtype=np.uint32),
                )
            )
            if m_check.status().name == "NoError" and m_check.volume() > 0:
                m_mesh = m_check.to_mesh()
                tri = trimesh.Trimesh(
                    vertices=m_mesh.vert_properties, faces=m_mesh.tri_verts, process=False
                )
        except Exception:
            pass

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
    tri = prepare_for_printing(mesh, flat_cap=flat_cap)

    # Clean any residual degenerate or unreferenced elements
    if len(tri.faces) > 0:
        tri.update_faces(tri.nondegenerate_faces())
        tri.update_faces(tri.unique_faces())
        tri.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(tri)
        trimesh.repair.fix_normals(tri)

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

    vertices = tri.vertices[tri.faces].astype("<f4")
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
    data = np.zeros(n_faces, dtype=dtype)
    data["normals"] = normals
    data["vertices"] = vertices
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
    plotter.add_bounding_box()
    plotter.show_axes()
    plotter.reset_camera()
    plotter.screenshot(output_image_path, window_size=window_size)
    plotter.close()
