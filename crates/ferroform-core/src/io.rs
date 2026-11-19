//! Pure-Rust mesh I/O for STL (binary & ASCII), Wavefront OBJ, and 3MF archives.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::marching::MeshData;
use crate::Trimesh;

const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

fn quantize_vertex(v: [f64; 3]) -> [i64; 3] {
    [
        (v[0] * 1_000_000.0).round() as i64,
        (v[1] * 1_000_000.0).round() as i64,
        (v[2] * 1_000_000.0).round() as i64,
    ]
}

fn push_welded_triangle(
    vertices: &mut Vec<[f64; 3]>,
    faces: &mut Vec<[u32; 3]>,
    weld_map: &mut HashMap<[i64; 3], u32>,
    tri: [[f64; 3]; 3],
) {
    let mut indices = [0u32; 3];
    for (k, v) in tri.into_iter().enumerate() {
        let key = quantize_vertex(v);
        let idx = *weld_map.entry(key).or_insert_with(|| {
            let next = vertices.len() as u32;
            vertices.push(v);
            next
        });
        indices[k] = idx;
    }
    if indices[0] != indices[1] && indices[1] != indices[2] && indices[2] != indices[0] {
        faces.push(indices);
    }
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn is_binary_stl(data: &[u8]) -> bool {
    if data.len() < 84 {
        return false;
    }
    let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected_len = 84usize.saturating_add(tri_count.saturating_mul(50));
    if data.len() == expected_len {
        return true;
    }
    // If it doesn't start with "solid", treat as binary if length >= 84
    let header_trim = &data[..80.min(data.len())];
    !header_trim.starts_with(b"solid")
}

/// Read a binary or ASCII STL file into a welded `Trimesh`.
pub fn read_stl(path: &Path) -> Result<Trimesh, String> {
    let data = fs::read(path)
        .map_err(|e| format!("Failed to read STL file '{}': {e}", path.display()))?;
    if data.is_empty() {
        return Err(format!("STL file '{}' is empty", path.display()));
    }

    if is_binary_stl(&data) {
        parse_binary_stl(&data)
    } else {
        let text = std::str::from_utf8(&data)
            .map_err(|e| format!("Invalid UTF-8 in ASCII STL '{}': {e}", path.display()))?;
        parse_ascii_stl(text)
    }
}

fn parse_binary_stl(data: &[u8]) -> Result<Trimesh, String> {
    if data.len() < 84 {
        return Err("Binary STL is truncated (< 84 bytes)".to_string());
    }
    let n_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let available = (data.len() - 84) / 50;
    let count = n_tris.min(available);

    let mut vertices = Vec::with_capacity(count / 2 + 16);
    let mut faces = Vec::with_capacity(count);
    let mut weld_map = HashMap::with_capacity(count / 2 + 16);

    for i in 0..count {
        let base = 84 + i * 50;
        // Skip 12-byte normal; read 3 vertices (36 bytes)
        let mut tri = [[0.0f64; 3]; 3];
        for v_idx in 0..3 {
            let off = base + 12 + v_idx * 12;
            tri[v_idx] = [
                read_f32_le(&data[off..off + 4]) as f64,
                read_f32_le(&data[off + 4..off + 8]) as f64,
                read_f32_le(&data[off + 8..off + 12]) as f64,
            ];
        }
        push_welded_triangle(&mut vertices, &mut faces, &mut weld_map, tri);
    }

    if faces.is_empty() {
        return Err("STL file contains no valid triangles".to_string());
    }

    Ok(Trimesh { vertices, faces })
}

fn parse_ascii_stl(text: &str) -> Result<Trimesh, String> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut weld_map = HashMap::new();
    let mut current_tri = Vec::with_capacity(3);

    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("vertex") {
            let mut coords = rest.split_whitespace().filter_map(|s| s.parse::<f64>().ok());
            if let (Some(x), Some(y), Some(z)) = (coords.next(), coords.next(), coords.next()) {
                current_tri.push([x, y, z]);
                if current_tri.len() == 3 {
                    push_welded_triangle(
                        &mut vertices,
                        &mut faces,
                        &mut weld_map,
                        [current_tri[0], current_tri[1], current_tri[2]],
                    );
                    current_tri.clear();
                }
            }
        } else if trimmed.starts_with("endloop") || trimmed.starts_with("endfacet") {
            current_tri.clear();
        }
    }

    if faces.is_empty() {
        return Err("ASCII STL file contains no valid triangles".to_string());
    }

    Ok(Trimesh { vertices, faces })
}

/// Write a `MeshData` to a Wavefront `.obj` file.
pub fn write_obj(mesh: &MeshData, path: &Path) -> Result<(), String> {
    let file = File::create(path)
        .map_err(|e| format!("Failed to create OBJ file '{}': {e}", path.display()))?;
    let mut w = BufWriter::with_capacity(256 * 1024, file);

    writeln!(w, "# FerroForm OBJ Export")
        .map_err(|e| format!("OBJ write error: {e}"))?;
    for p in &mesh.points {
        writeln!(w, "v {:.6} {:.6} {:.6}", p[0], p[1], p[2])
            .map_err(|e| format!("OBJ write error: {e}"))?;
    }
    for f in &mesh.faces {
        writeln!(w, "f {} {} {}", f[0] + 1, f[1] + 1, f[2] + 1)
            .map_err(|e| format!("OBJ write error: {e}"))?;
    }
    w.flush().map_err(|e| format!("OBJ flush error: {e}"))?;
    Ok(())
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}

/// Write one or more named meshes into a compressed multi-body `.3mf` archive.
pub fn write_3mf(parts: &[(&str, &MeshData)], path: &Path, unit: &str) -> Result<(), String> {
    let non_empty: Vec<(usize, &str, &MeshData)> = parts
        .iter()
        .enumerate()
        .filter(|(_, (_, m))| !m.points.is_empty() && !m.faces.is_empty())
        .map(|(i, (name, m))| (i + 1, *name, *m))
        .collect();

    if non_empty.is_empty() {
        return Err("No non-empty meshes to export".to_string());
    }

    let mut model_xml = String::with_capacity(64 * 1024);
    model_xml.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <model unit=\"{}\" xml:lang=\"en-US\" xmlns=\"http://schemas.microsoft.com/3dmanufacturing/core/2015/02\">\n\
         \x20 <resources>\n",
        xml_escape(unit)
    ));

    use std::fmt::Write as _;
    for (obj_id, name, mesh) in &non_empty {
        let _ = writeln!(
            model_xml,
            "    <object id=\"{obj_id}\" name=\"{}\" type=\"model\">\n      <mesh>\n        <vertices>",
            xml_escape(name)
        );
        for p in &mesh.points {
            let _ = writeln!(
                model_xml,
                "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>",
                p[0], p[1], p[2]
            );
        }
        model_xml.push_str("        </vertices>\n        <triangles>\n");
        for f in &mesh.faces {
            let _ = writeln!(
                model_xml,
                "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>",
                f[0], f[1], f[2]
            );
        }
        model_xml.push_str("        </triangles>\n      </mesh>\n    </object>\n");
    }

    model_xml.push_str("  </resources>\n  <build>\n");
    for (obj_id, _, _) in &non_empty {
        let _ = writeln!(model_xml, "    <item objectid=\"{obj_id}\"/>");
    }
    model_xml.push_str("  </build>\n</model>\n");

    let file = File::create(path)
        .map_err(|e| format!("Failed to create 3MF file '{}': {e}", path.display()))?;
    let mut zip = ZipWriter::new(file);
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);

    zip.start_file("[Content_Types].xml", opts)
        .map_err(|e| format!("3MF zip error: {e}"))?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
        .map_err(|e| format!("3MF zip write error: {e}"))?;

    zip.start_file("_rels/.rels", opts)
        .map_err(|e| format!("3MF zip error: {e}"))?;
    zip.write_all(RELS_XML.as_bytes())
        .map_err(|e| format!("3MF zip write error: {e}"))?;

    zip.start_file("3D/3dmodel.model", opts)
        .map_err(|e| format!("3MF zip error: {e}"))?;
    zip.write_all(model_xml.as_bytes())
        .map_err(|e| format!("3MF zip write error: {e}"))?;

    zip.finish().map_err(|e| format!("3MF finalize error: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::write_binary_stl;

    fn sample_tetrahedron() -> MeshData {
        MeshData {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            faces: vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
        }
    }

    #[test]
    fn binary_stl_round_trip() {
        let mesh = sample_tetrahedron();
        let tmp = std::env::temp_dir().join("ferroform_test_roundtrip.stl");
        write_binary_stl(&mesh, &tmp, "FerroForm test").unwrap();
        let loaded = read_stl(&tmp).unwrap();
        let _ = fs::remove_file(&tmp);
        assert_eq!(loaded.vertices.len(), 4);
        assert_eq!(loaded.faces.len(), 4);
    }

    #[test]
    fn ascii_stl_parse() {
        let ascii = "\
solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
";
        let loaded = parse_ascii_stl(ascii).unwrap();
        assert_eq!(loaded.vertices.len(), 3);
        assert_eq!(loaded.faces.len(), 1);
    }

    #[test]
    fn obj_and_3mf_export() {
        let mesh = sample_tetrahedron();
        let obj_path = std::env::temp_dir().join("ferroform_test_export.obj");
        let mf_path = std::env::temp_dir().join("ferroform_test_export.3mf");
        write_obj(&mesh, &obj_path).unwrap();
        write_3mf(&[("lattice", &mesh)], &mf_path, "millimeter").unwrap();

        let obj_text = fs::read_to_string(&obj_path).unwrap();
        assert!(obj_text.contains("v 0.000000 0.000000 0.000000"));
        assert!(obj_text.contains("f "));

        let mf_bytes = fs::read(&mf_path).unwrap();
        assert!(mf_bytes.starts_with(b"PK"));

        let _ = fs::remove_file(&obj_path);
        let _ = fs::remove_file(&mf_path);
    }
}
