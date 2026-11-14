use pyo3::prelude::*;

#[pyfunction]
fn core_version() -> &'static str {
    pyroid_core::version()
}

#[pymodule]
fn pyroid_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(core_version, m)?)?;
    Ok(())
}
