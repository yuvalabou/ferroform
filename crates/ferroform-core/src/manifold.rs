//! Manifold winding, signed volume, and direct binary STL export utilities.

use std::fs::File;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

use crate::marching::MeshData;

/// Default binary STL header matching Farsoon BuildStar and Netfabb requirements.
pub const DEFAULT_STL_HEADER: &str = "FerroForm 3D Model - Farsoon BuildStar and Netfabb Compatible";

/// Compute the signed volume of indexed triangle slices via the divergence theorem.
pub fn signed_volume_slices(points: &[[f32; 3]], faces: &[[u32; 3]]) -> f64 {
    let mut sum = 0.0f64;
    for f in faces {
        let p0 = points[f[0] as usize];
        let p1 = points[f[1] as usize];
        let p2 = points[f[2] as usize];

        let (x0, y0, z0) = (p0[0] as f64, p0[1] as f64, p0[2] as f64);
        let (x1, y1, z1) = (p1[0] as f64, p1[1] as f64, p1[2] as f64);
        let (x2, y2, z2) = (p2[0] as f64, p2[1] as f64, p2[2] as f64);

        // det([p0, p1, p2])
        let cx = y1 * z2 - z1 * y2;
        let cy = z1 * x2 - x1 * z2;
        let cz = x1 * y2 - y1 * x2;
        sum += x0 * cx + y0 * cy + z0 * cz;
    }
    sum / 6.0
}

/// Compute the signed volume of an indexed triangle mesh via divergence theorem.
pub fn signed_volume(mesh: &MeshData) -> f64 {
    signed_volume_slices(&mesh.points, &mesh.faces)
}

/// Ensure positive outward signed volume by reversing triangle winding if negative.
pub fn ensure_positive_volume(mesh: &mut MeshData) {
    if !mesh.faces.is_empty() && signed_volume(mesh) < 0.0 {
        for f in &mut mesh.faces {
            f.swap(1, 2);
        }
    }
}

fn format_stl_header(header_text: &str) -> [u8; 80] {
    let mut cleaned: Vec<u8> = header_text
        .chars()
        .map(|c| match c {
            '\r' | '\n' => b' ',
            c if c.is_ascii() => c as u8,
            _ => b'?',
        })
        .collect();

    if cleaned.len() >= 5 && cleaned[..5].eq_ignore_ascii_case(b"solid") {
        let mut prefixed = Vec::with_capacity(10 + cleaned.len());
        prefixed.extend_from_slice(b"FerroForm ");
        prefixed.extend_from_slice(&cleaned);
        cleaned = prefixed;
    }

    let mut header = [b' '; 80];
    let copy_len = cleaned.len().min(80);
    header[..copy_len].copy_from_slice(&cleaned[..copy_len]);
    header
}

#[inline]
fn write_vec3(dst: &mut [u8], v: [f32; 3]) {
    dst[0..4].copy_from_slice(&v[0].to_le_bytes());
    dst[4..8].copy_from_slice(&v[1].to_le_bytes());
    dst[8..12].copy_from_slice(&v[2].to_le_bytes());
}

/// Write indexed triangle slices directly to a standard-compliant binary STL file.
///
/// Enforces an 80-byte space-padded ASCII header (never starting with `solid`),
/// outward CCW winding via signed volume, explicit unit normals, and degenerate
/// triangle filtering for Farsoon BuildStar / Netfabb compatibility.
pub fn write_binary_stl_slices(
    points: &[[f32; 3]],
    faces: &[[u32; 3]],
    path: &Path,
    header_text: &str,
) -> Result<(), String> {
    let n_pts = points.len();
    for (fi, f) in faces.iter().enumerate() {
        if f[0] as usize >= n_pts || f[1] as usize >= n_pts || f[2] as usize >= n_pts {
            return Err(format!(
                "Face {fi} references out-of-bounds vertex index (n_points={n_pts})"
            ));
        }
    }

    let flip_winding = !faces.is_empty() && signed_volume_slices(points, faces) < 0.0;

    let file = File::create(path)
        .map_err(|e| format!("Failed to create STL file '{}': {e}", path.display()))?;
    let mut writer = BufWriter::with_capacity(256 * 1024, file);

    let header = format_stl_header(header_text);
    writer
        .write_all(&header)
        .map_err(|e| format!("Failed to write STL header: {e}"))?;
    writer
        .write_all(&0u32.to_le_bytes())
        .map_err(|e| format!("Failed to write STL facet count: {e}"))?;

    let mut valid_faces: u32 = 0;
    let mut record = [0u8; 50];

    for f in faces {
        let p0 = points[f[0] as usize];
        let (p1, p2) = if flip_winding {
            (points[f[2] as usize], points[f[1] as usize])
        } else {
            (points[f[1] as usize], points[f[2] as usize])
        };

        let e1x = (p1[0] as f64) - (p0[0] as f64);
        let e1y = (p1[1] as f64) - (p0[1] as f64);
        let e1z = (p1[2] as f64) - (p0[2] as f64);
        let e2x = (p2[0] as f64) - (p0[0] as f64);
        let e2y = (p2[1] as f64) - (p0[1] as f64);
        let e2z = (p2[2] as f64) - (p0[2] as f64);

        let nx = e1y * e2z - e1z * e2y;
        let ny = e1z * e2x - e1x * e2z;
        let nz = e1x * e2y - e1y * e2x;
        let norm = (nx * nx + ny * ny + nz * nz).sqrt();

        // Skip degenerate / zero-area triangles (area = 0.5 * norm < 1e-12)
        if !norm.is_finite() || norm < 2e-12 {
            continue;
        }

        if valid_faces == u32::MAX {
            return Err("Mesh exceeds maximum binary STL facet count (u32::MAX)".to_string());
        }
        valid_faces += 1;

        let inv_norm = 1.0 / norm;
        let n = [
            (nx * inv_norm) as f32,
            (ny * inv_norm) as f32,
            (nz * inv_norm) as f32,
        ];

        write_vec3(&mut record[0..12], n);
        write_vec3(&mut record[12..24], p0);
        write_vec3(&mut record[24..36], p1);
        write_vec3(&mut record[36..48], p2);
        record[48] = 0;
        record[49] = 0;

        writer
            .write_all(&record)
            .map_err(|e| format!("Failed to write STL facet: {e}"))?;
    }

    writer
        .flush()
        .map_err(|e| format!("Failed to flush STL writer: {e}"))?;

    let mut file = writer
        .into_inner()
        .map_err(|e| format!("Failed to finalize STL writer: {e}"))?;
    file.seek(SeekFrom::Start(80))
        .map_err(|e| format!("Failed to seek to STL facet count: {e}"))?;
    file.write_all(&valid_faces.to_le_bytes())
        .map_err(|e| format!("Failed to write final STL facet count: {e}"))?;

    Ok(())
}

/// Write a `MeshData` directly to a standard-compliant binary STL file.
pub fn write_binary_stl(mesh: &MeshData, path: &Path, header_text: &str) -> Result<(), String> {
    write_binary_stl_slices(&mesh.points, &mesh.faces, path, header_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensure_positive_volume_flips_inverted_mesh() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Inverted tetrahedron winding
        let faces = vec![[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]];
        let mut mesh = MeshData { points, faces };
        assert!(signed_volume(&mesh) < 0.0);
        ensure_positive_volume(&mut mesh);
        assert!(signed_volume(&mesh) > 0.0);
        assert!((signed_volume(&mesh) - (1.0 / 6.0)).abs() < 1e-6);
    }

    #[test]
    fn test_write_binary_stl_header_winding_and_degenerate_filter() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        // Inverted tetrahedron + 1 degenerate triangle
        let faces = vec![
            [0, 1, 2],
            [0, 3, 1],
            [1, 3, 2],
            [2, 3, 0],
            [0, 1, 1],
        ];
        let mesh = MeshData { points, faces };
        let tmp_path = std::env::temp_dir().join("ferroform_test_direct_writer.stl");

        write_binary_stl(&mesh, &tmp_path, "solid\nexported").unwrap();
        let bytes = std::fs::read(&tmp_path).unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(&bytes[..24], b"FerroForm solid exported");
        assert_eq!(&bytes[24..80], &[b' '; 56]);

        let n_faces = u32::from_le_bytes(bytes[80..84].try_into().unwrap());
        assert_eq!(n_faces, 4);
        assert_eq!(bytes.len(), 84 + 4 * 50);

        // Verify unit normals for all 4 facets
        for i in 0..4 {
            let off = 84 + i * 50;
            let nx = f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let ny = f32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let nz = f32::from_le_bytes(bytes[off + 8..off + 12].try_into().unwrap());
            let norm = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_write_binary_stl_empty_mesh() {
        let mesh = MeshData::default();
        let tmp_path = std::env::temp_dir().join("ferroform_test_empty_writer.stl");
        write_binary_stl(&mesh, &tmp_path, DEFAULT_STL_HEADER).unwrap();
        let bytes = std::fs::read(&tmp_path).unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(bytes.len(), 84);
        assert_eq!(u32::from_le_bytes(bytes[80..84].try_into().unwrap()), 0);
    }
}
