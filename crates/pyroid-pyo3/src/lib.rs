//! PyO3 Python bindings exposing `pyroid_core` as the `pyroid_rs` extension module.

use std::path::Path;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyroid_core::manifold::{write_binary_stl_slices, DEFAULT_STL_HEADER};
use pyroid_core::marching::MeshData;
use pyroid_core::tpms::TpmsType;
use pyroid_core::{convert_stl, generate_parametric, ParametricParams, StlParams};

type MeshArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<i64>>);

fn dict_get<'py, T: FromPyObject<'py>>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<T> {
    let val = dict
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("Missing required parameter: {key}")))?;
    val.extract()
}

fn dict_get_opt<'py, T: FromPyObject<'py>>(
    dict: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match dict.get_item(key)? {
        Some(val) if !val.is_none() => Ok(Some(val.extract()?)),
        _ => Ok(None),
    }
}

fn mesh_to_py<'py>(py: Python<'py>, mesh: MeshData) -> PyResult<MeshArrays<'py>> {
    let n_pts = mesh.n_points();
    let py_points = mesh
        .points_flat()
        .into_pyarray_bound(py)
        .reshape([n_pts, 3])?;
    let py_cells = mesh.vtk_cells().into_pyarray_bound(py);
    Ok((py_points, py_cells))
}

fn parse_parametric_params(params: &Bound<'_, PyDict>) -> PyResult<ParametricParams> {
    let tpms_str: String = dict_get(params, "tpms_type")?;
    let tpms_type = TpmsType::from_str(&tpms_str).map_err(PyValueError::new_err)?;
    Ok(ParametricParams {
        tpms_type,
        res: dict_get(params, "res")?,
        a: dict_get(params, "a")?,
        b: dict_get(params, "b")?,
        c: dict_get(params, "c")?,
        r1: dict_get(params, "r1")?,
        r2: dict_get(params, "r2")?,
        phi_scale: dict_get(params, "phi_scale")?,
        wall_thickness: dict_get(params, "wall_thickness")?,
        cell_radius: dict_get(params, "cell_radius")?,
        cell_height: dict_get(params, "cell_height")?,
        bias: dict_get(params, "bias")?,
        include_cavities: dict_get_opt(params, "include_cavities")?.unwrap_or(true),
    })
}

fn parse_stl_params(params: &Bound<'_, PyDict>) -> PyResult<StlParams> {
    let tpms_str: String = dict_get(params, "tpms_type")?;
    let tpms_type = TpmsType::from_str(&tpms_str).map_err(PyValueError::new_err)?;
    Ok(StlParams {
        tpms_type,
        resolution: dict_get(params, "resolution")?,
        cell_size: dict_get(params, "cell_size")?,
        wall_thickness: dict_get(params, "wall_thickness")?,
        bias: dict_get(params, "bias")?,
        flat_cap: dict_get(params, "flat_cap")?,
        preserve_shell: dict_get(params, "preserve_shell")?,
        shell_thickness: dict_get(params, "shell_thickness")?,
        is_skeletal: dict_get(params, "is_skeletal")?,
        infill_percent: dict_get_opt(params, "infill_percent")?,
        include_cavities: dict_get_opt(params, "include_cavities")?.unwrap_or(true),
    })
}

/// Generate a radially symmetrical parametric TPMS structure and dual cavities.
#[pyfunction]
#[pyo3(signature = (params, progress_callback=None))]
fn generate_parametric_tpms<'py>(
    py: Python<'py>,
    params: &Bound<'py, PyDict>,
    progress_callback: Option<PyObject>,
) -> PyResult<(MeshArrays<'py>, MeshArrays<'py>, MeshArrays<'py>)> {
    let rust_params = parse_parametric_params(params)?;
    let out = py
        .allow_threads(|| {
            generate_parametric(&rust_params, |pct, msg| {
                if let Some(ref cb) = progress_callback {
                    Python::with_gil(|py| {
                        let _ = cb.call1(py, (pct, msg));
                    });
                }
            })
        })
        .map_err(PyValueError::new_err)?;

    let mesh_py = mesh_to_py(py, out.mesh)?;
    let cav_a_py = mesh_to_py(py, out.cavity_a.unwrap_or_default())?;
    let cav_b_py = mesh_to_py(py, out.cavity_b.unwrap_or_default())?;
    Ok((mesh_py, cav_a_py, cav_b_py))
}

/// Convert an input STL triangle mesh volume into a TPMS lattice (or shell internal void).
#[pyfunction]
#[pyo3(signature = (vertices, faces, params, progress_callback=None))]
fn convert_stl_to_tpms<'py>(
    py: Python<'py>,
    vertices: PyReadonlyArray2<'py, f64>,
    faces: PyReadonlyArray2<'py, u32>,
    params: &Bound<'py, PyDict>,
    progress_callback: Option<PyObject>,
) -> PyResult<(
    bool,
    MeshArrays<'py>,
    Option<MeshArrays<'py>>,
    Option<MeshArrays<'py>>,
)> {
    let rust_params = parse_stl_params(params)?;
    let v_arr = vertices.as_array();
    let f_arr = faces.as_array();
    if v_arr.ncols() != 3 || f_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "vertices and faces must have shape (N, 3)",
        ));
    }

    let verts: Vec<[f64; 3]> = v_arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    let tris: Vec<[u32; 3]> = f_arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();

    let out = py
        .allow_threads(|| {
            convert_stl(&verts, &tris, &rust_params, |pct, msg| {
                if let Some(ref cb) = progress_callback {
                    Python::with_gil(|py| {
                        let _ = cb.call1(py, (pct, msg));
                    });
                }
            })
        })
        .map_err(PyValueError::new_err)?;

    Ok((
        out.is_shell_void,
        mesh_to_py(py, out.mesh)?,
        out.cavity_a.map(|m| mesh_to_py(py, m)).transpose()?,
        out.cavity_b.map(|m| mesh_to_py(py, m)).transpose()?,
    ))
}

/// Write triangle mesh buffers directly to a standard-compliant binary STL file.
#[pyfunction]
#[pyo3(signature = (points, faces, filepath, header_text=None))]
fn write_stl_binary<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    faces: PyReadonlyArray2<'py, u32>,
    filepath: &str,
    header_text: Option<&str>,
) -> PyResult<()> {
    let p_arr = points.as_array();
    let f_arr = faces.as_array();
    if p_arr.ncols() != 3 || f_arr.ncols() != 3 {
        return Err(PyValueError::new_err(
            "points and faces must have shape (N, 3)",
        ));
    }

    let pts: Vec<[f32; 3]> = if let Some(s) = p_arr.as_slice() {
        s.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        p_arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()
    };

    let tris: Vec<[u32; 3]> = if let Some(s) = f_arr.as_slice() {
        s.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
    } else {
        f_arr.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect()
    };

    let header = header_text.unwrap_or(DEFAULT_STL_HEADER).to_string();
    let path = Path::new(filepath).to_path_buf();

    py.allow_threads(|| write_binary_stl_slices(&pts, &tris, &path, &header))
        .map_err(PyIOError::new_err)
}

#[pymodule]
fn pyroid_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_parametric_tpms, m)?)?;
    m.add_function(wrap_pyfunction!(convert_stl_to_tpms, m)?)?;
    m.add_function(wrap_pyfunction!(write_stl_binary, m)?)?;
    Ok(())
}
