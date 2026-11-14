//! Manifold winding and signed volume utilities.

use crate::marching::MeshData;

/// Compute the signed volume of an indexed triangle mesh via divergence theorem.
pub fn signed_volume(mesh: &MeshData) -> f64 {
    let mut sum = 0.0f64;
    for f in &mesh.faces {
        let p0 = mesh.points[f[0] as usize];
        let p1 = mesh.points[f[1] as usize];
        let p2 = mesh.points[f[2] as usize];

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

/// Ensure positive outward signed volume by reversing triangle winding if negative.
pub fn ensure_positive_volume(mesh: &mut MeshData) {
    if !mesh.faces.is_empty() && signed_volume(mesh) < 0.0 {
        for f in &mut mesh.faces {
            f.swap(1, 2);
        }
    }
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
}
