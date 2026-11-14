//! Parallel AABB BVH point-in-mesh containment and quantile estimation.

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

const MAX_LEAF_SIZE: usize = 8;

/// Triangle face stored with precomputed YZ bounds for +X ray intersection.
#[derive(Clone, Copy, Debug)]
struct TriBBox {
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    cy: f64,
    cz: f64,
}

impl TriBBox {
    #[inline]
    fn from_vertices(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3]) -> Self {
        Self {
            v0,
            v1,
            v2,
            y_min: v0[1].min(v1[1]).min(v2[1]),
            y_max: v0[1].max(v1[1]).max(v2[1]),
            z_min: v0[2].min(v1[2]).min(v2[2]),
            z_max: v0[2].max(v1[2]).max(v2[2]),
            cy: (v0[1] + v1[1] + v2[1]) / 3.0,
            cz: (v0[2] + v1[2] + v2[2]) / 3.0,
        }
    }
}

/// BVH node storing 2D YZ bounding box.
#[derive(Clone, Copy, Debug)]
struct BvhNode {
    y_min: f64,
    y_max: f64,
    z_min: f64,
    z_max: f64,
    /// If `count > 0`, leaf covering `tris[first..first + count]`.
    /// If `count == 0`, internal node with children at `first` and `first + 1`.
    first: usize,
    count: usize,
}

/// Acceleration structure for fast +X ray-casting against triangle meshes.
pub struct Bvh {
    tris: Vec<TriBBox>,
    nodes: Vec<BvhNode>,
}

impl Bvh {
    /// Build a BVH over indexed triangle mesh vertices and faces.
    pub fn build(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Self {
        let mut tris: Vec<TriBBox> = faces
            .iter()
            .filter_map(|f| {
                let a = vertices.get(f[0] as usize)?;
                let b = vertices.get(f[1] as usize)?;
                let c = vertices.get(f[2] as usize)?;
                Some(TriBBox::from_vertices(*a, *b, *c))
            })
            .collect();

        if tris.is_empty() {
            return Self {
                tris,
                nodes: Vec::new(),
            };
        }

        let mut nodes = Vec::with_capacity(tris.len() * 2);
        nodes.push(BvhNode {
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            first: 0,
            count: tris.len(),
        });
        Self::subdivide(0, 0, tris.len(), &mut tris, &mut nodes);
        Self { tris, nodes }
    }

    fn subdivide(
        node_idx: usize,
        start: usize,
        end: usize,
        tris: &mut [TriBBox],
        nodes: &mut Vec<BvhNode>,
    ) {
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        let mut cy_min = f64::INFINITY;
        let mut cy_max = f64::NEG_INFINITY;
        let mut cz_min = f64::INFINITY;
        let mut cz_max = f64::NEG_INFINITY;

        for t in &tris[start..end] {
            y_min = y_min.min(t.y_min);
            y_max = y_max.max(t.y_max);
            z_min = z_min.min(t.z_min);
            z_max = z_max.max(t.z_max);
            cy_min = cy_min.min(t.cy);
            cy_max = cy_max.max(t.cy);
            cz_min = cz_min.min(t.cz);
            cz_max = cz_max.max(t.cz);
        }

        nodes[node_idx].y_min = y_min;
        nodes[node_idx].y_max = y_max;
        nodes[node_idx].z_min = z_min;
        nodes[node_idx].z_max = z_max;

        let count = end - start;
        if count <= MAX_LEAF_SIZE {
            nodes[node_idx].first = start;
            nodes[node_idx].count = count;
            return;
        }

        let span_y = cy_max - cy_min;
        let span_z = cz_max - cz_min;
        if span_y <= 0.0 && span_z <= 0.0 {
            nodes[node_idx].first = start;
            nodes[node_idx].count = count;
            return;
        }

        let mid = start + count / 2;
        let slice = &mut tris[start..end];
        let local_mid = count / 2;
        if span_y >= span_z {
            slice.select_nth_unstable_by(local_mid, |a, b| a.cy.total_cmp(&b.cy));
        } else {
            slice.select_nth_unstable_by(local_mid, |a, b| a.cz.total_cmp(&b.cz));
        }

        let left_idx = nodes.len();
        nodes.push(BvhNode {
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            first: 0,
            count: 0,
        });
        nodes.push(BvhNode {
            y_min: 0.0,
            y_max: 0.0,
            z_min: 0.0,
            z_max: 0.0,
            first: 0,
            count: 0,
        });

        nodes[node_idx].first = left_idx;
        nodes[node_idx].count = 0;

        Self::subdivide(left_idx, start, mid, tris, nodes);
        Self::subdivide(left_idx + 1, mid, end, tris, nodes);
    }

    /// Collect all `+X` ray intersection X coordinates at `(ry, rz)`.
    #[inline]
    fn intersect_x_ray(&self, ry: f64, rz: f64, stack: &mut Vec<usize>, hits: &mut Vec<f64>) {
        hits.clear();
        if self.nodes.is_empty() {
            return;
        }
        stack.clear();
        stack.push(0);

        while let Some(idx) = stack.pop() {
            let node = &self.nodes[idx];
            if ry < node.y_min || ry > node.y_max || rz < node.z_min || rz > node.z_max {
                continue;
            }
            if node.count > 0 {
                for t in &self.tris[node.first..node.first + node.count] {
                    if ry < t.y_min || ry > t.y_max || rz < t.z_min || rz > t.z_max {
                        continue;
                    }
                    let e1y = t.v1[1] - t.v0[1];
                    let e1z = t.v1[2] - t.v0[2];
                    let e2y = t.v2[1] - t.v0[1];
                    let e2z = t.v2[2] - t.v0[2];
                    let det = e1y * e2z - e1z * e2y;
                    if det.abs() < 1e-18 {
                        continue;
                    }
                    let inv_det = 1.0 / det;
                    let py = ry - t.v0[1];
                    let pz = rz - t.v0[2];
                    let u = (py * e2z - pz * e2y) * inv_det;
                    if u < 0.0 || u > 1.0 {
                        continue;
                    }
                    let v = (e1y * pz - e1z * py) * inv_det;
                    if v < 0.0 || u + v > 1.0 {
                        continue;
                    }
                    let x_hit = t.v0[0] + u * (t.v1[0] - t.v0[0]) + v * (t.v2[0] - t.v0[0]);
                    hits.push(x_hit);
                }
            } else {
                stack.push(node.first);
                stack.push(node.first + 1);
            }
        }
    }
}

/// Evaluate 3D grid point containment inside a closed triangle mesh in C-order `(nx, ny, nz)`.
pub fn compute_containment_grid(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
) -> Vec<bool> {
    let nx = xs.len();
    let ny = ys.len();
    let nz = zs.len();
    let total = nx * ny * nz;
    if total == 0 || vertices.is_empty() || faces.is_empty() {
        return vec![false; total];
    }

    let bvh = Bvh::build(vertices, faces);
    let inside: Vec<AtomicBool> = (0..total).map(|_| AtomicBool::new(false)).collect();

    // Scale perturbation gently so tiny and huge meshes both avoid edge/vertex hits.
    let y_span = (ys[ny - 1] - ys[0]).abs();
    let z_span = (zs[nz - 1] - zs[0]).abs();
    let scale = y_span.max(z_span).max(1.0);
    let eps_y = 1.37e-9 * scale;
    let eps_z = 2.71e-9 * scale;

    (0..ny * nz).into_par_iter().for_each_init(
        || (Vec::with_capacity(64), Vec::with_capacity(16)),
        |(stack, hits), jk| {
            let j = jk / nz;
            let k = jk % nz;
            let ry = ys[j] + eps_y;
            let rz = zs[k] + eps_z;

            bvh.intersect_x_ray(ry, rz, stack, hits);
            if hits.is_empty() {
                return;
            }
            hits.sort_unstable_by(|a, b| a.total_cmp(b));

            let mut h_idx = 0usize;
            let n_hits = hits.len();
            for (i, &xi) in xs.iter().enumerate() {
                while h_idx < n_hits && hits[h_idx] <= xi {
                    h_idx += 1;
                }
                if (n_hits - h_idx) % 2 == 1 {
                    let flat = (i * ny + j) * nz + k;
                    inside[flat].store(true, Ordering::Relaxed);
                }
            }
        },
    );

    inside.into_iter().map(AtomicBool::into_inner).collect()
}

/// Compute linear-interpolated `q`-quantile (matching `np.quantile`) over masked values.
pub fn compute_quantile(values: &[f64], mask: &[bool], q: f64) -> f64 {
    compute_quantile_mapped(values, mask, q, |v| v)
}

/// Compute linear-interpolated `q`-quantile over masked `|v|` values without allocating an intermediate `abs` buffer.
pub fn compute_abs_quantile(values: &[f64], mask: &[bool], q: f64) -> f64 {
    compute_quantile_mapped(values, mask, q, f64::abs)
}

fn compute_quantile_mapped(
    values: &[f64],
    mask: &[bool],
    q: f64,
    map: impl Fn(f64) -> f64,
) -> f64 {
    let mut masked: Vec<f64> = values
        .iter()
        .zip(mask.iter())
        .filter_map(|(&v, &m)| {
            if m && !v.is_nan() {
                Some(map(v))
            } else {
                None
            }
        })
        .collect();

    let n = masked.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return masked[0];
    }

    let q_clamped = q.clamp(0.0, 1.0);
    let pos = q_clamped * ((n - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);

    masked.select_nth_unstable_by(lo, f64::total_cmp);
    let lo_val = masked[lo];
    if hi == lo || lo + 1 >= n {
        return lo_val;
    }
    let hi_val = masked[lo + 1..]
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    lo_val * (1.0 - frac) + hi_val * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube_mesh() -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        let vertices = vec![
            [-5.0, -5.0, -5.0],
            [5.0, -5.0, -5.0],
            [5.0, 5.0, -5.0],
            [-5.0, 5.0, -5.0],
            [-5.0, -5.0, 5.0],
            [5.0, -5.0, 5.0],
            [5.0, 5.0, 5.0],
            [-5.0, 5.0, 5.0],
        ];
        let faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ];
        (vertices, faces)
    }

    #[test]
    fn test_cube_containment_grid() {
        let (verts, faces) = unit_cube_mesh();
        let xs: Vec<f64> = (-7..=7).map(|v| v as f64).collect();
        let ys = xs.clone();
        let zs = xs.clone();
        let nx = xs.len();
        let ny = ys.len();
        let nz = zs.len();

        let mask = compute_containment_grid(&verts, &faces, &xs, &ys, &zs);
        assert_eq!(mask.len(), nx * ny * nz);

        for (i, &x) in xs.iter().enumerate() {
            for (j, &y) in ys.iter().enumerate() {
                for (k, &z) in zs.iter().enumerate() {
                    let idx = (i * ny + j) * nz + k;
                    if x.abs() < 4.9 && y.abs() < 4.9 && z.abs() < 4.9 {
                        assert!(mask[idx], "Expected inside at ({x}, {y}, {z})");
                    } else if x.abs() > 5.1 || y.abs() > 5.1 || z.abs() > 5.1 {
                        assert!(!mask[idx], "Expected outside at ({x}, {y}, {z})");
                    }
                }
            }
        }
    }

    #[test]
    fn test_compute_quantile_linear() {
        let vals = vec![10.0, 1.0, 5.0, 2.0, 99.0, 3.0, 4.0];
        let mask = vec![false, true, true, true, false, true, true];
        // Masked sorted values: [1.0, 2.0, 3.0, 4.0, 5.0]
        assert!((compute_quantile(&vals, &mask, 0.0) - 1.0).abs() < 1e-12);
        assert!((compute_quantile(&vals, &mask, 0.5) - 3.0).abs() < 1e-12);
        assert!((compute_quantile(&vals, &mask, 0.25) - 2.0).abs() < 1e-12);
        assert!((compute_quantile(&vals, &mask, 0.375) - 2.5).abs() < 1e-12);
        assert!((compute_quantile(&vals, &mask, 1.0) - 5.0).abs() < 1e-12);
    }
}
