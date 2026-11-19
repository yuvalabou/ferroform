//! Multithreaded 3D Euclidean Distance Transform (Felzenszwalb & Huttenlocher).

use rayon::prelude::*;

/// Exact 1D lower envelope squared distance transform with spacing `spacing`.
fn edt_1d(f: &[f64], out: &mut [f64], spacing: f64, v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    if n == 0 {
        return;
    }

    let w = spacing * spacing;
    let mut len = 0usize;
    z[0] = f64::NEG_INFINITY;

    for q in 0..n {
        if !f[q].is_finite() {
            continue;
        }
        let fq = f[q];
        let qf = q as f64;
        let mut s_inter = f64::NEG_INFINITY;

        while len > 0 {
            let r = v[len - 1];
            let rf = r as f64;
            s_inter = ((fq - f[r]) / w + (qf * qf - rf * rf)) / (2.0 * (qf - rf));
            if s_inter <= z[len - 1] {
                len -= 1;
            } else {
                break;
            }
        }

        v[len] = q;
        z[len] = if len == 0 { f64::NEG_INFINITY } else { s_inter };
        z[len + 1] = f64::INFINITY;
        len += 1;
    }

    if len == 0 {
        out.fill(f64::INFINITY);
        return;
    }

    let mut k = 0usize;
    for (p, slot) in out.iter_mut().enumerate() {
        let pf = p as f64;
        while z[k + 1] < pf {
            k += 1;
        }
        let q = v[k];
        let diff = pf - (q as f64);
        *slot = f[q] + w * diff * diff;
    }
}

/// Compute the 3D Euclidean Distance Transform to the nearest `false` voxel in `inside`.
///
/// Layout is C-order `(nx, ny, nz)` matching `numpy.mgrid` (`idx = (i * ny + j) * nz + k`).
pub fn edt_3d(
    inside: &[bool],
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Vec<f64> {
    let total = nx * ny * nz;
    if total == 0 {
        return Vec::new();
    }
    debug_assert_eq!(inside.len(), total);

    let mut dist: Vec<f64> = inside
        .par_iter()
        .map(|&is_in| if is_in { f64::INFINITY } else { 0.0 })
        .collect();

    // Pass 1: along Z (contiguous slices of length nz).
    dist.par_chunks_mut(nz).for_each_init(
        || (vec![0.0; nz], vec![0usize; nz], vec![0.0; nz + 1]),
        |(f_buf, v, z), slice| {
            f_buf.copy_from_slice(slice);
            edt_1d(f_buf, slice, dz, v, z);
        },
    );

    // Pass 2: along Y (parallel over i planes of size ny * nz).
    dist.par_chunks_mut(ny * nz).for_each_init(
        || (vec![0.0; ny], vec![0.0; ny], vec![0usize; ny], vec![0.0; ny + 1]),
        |(f_buf, out_buf, v, z), plane| {
            for k in 0..nz {
                for j in 0..ny {
                    f_buf[j] = plane[j * nz + k];
                }
                edt_1d(f_buf, out_buf, dy, v, z);
                for j in 0..ny {
                    plane[j * nz + k] = out_buf[j];
                }
            }
        },
    );

    // Pass 3: along X — transpose (i, j, k) -> (j, k, i) so X is contiguous for L1 cache.
    let mut x_buf = vec![0.0; total];
    x_buf.par_chunks_mut(nz * nx).enumerate().for_each(|(j, j_slab)| {
        for k in 0..nz {
            let col_off = k * nx;
            for i in 0..nx {
                j_slab[col_off + i] = dist[(i * ny + j) * nz + k];
            }
        }
    });

    x_buf.par_chunks_mut(nx).for_each_init(
        || (vec![0.0; nx], vec![0usize; nx], vec![0.0; nx + 1]),
        |(f_buf, v, z), slice| {
            f_buf.copy_from_slice(slice);
            edt_1d(f_buf, slice, dx, v, z);
            for val in slice.iter_mut() {
                *val = val.sqrt();
            }
        },
    );

    // Transpose back (j, k, i) -> (i, j, k).
    dist.par_chunks_mut(ny * nz).enumerate().for_each(|(i, plane)| {
        for j in 0..ny {
            let j_off = j * nz;
            for k in 0..nz {
                plane[j_off + k] = x_buf[(j * nz + k) * nx + i];
            }
        }
    });

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edt_3d_box_analytical() {
        let (nx, ny, nz) = (9, 11, 13);
        let (dx, dy, dz) = (0.5, 1.0, 2.0);
        let mut inside = vec![false; nx * ny * nz];

        // Interior box: i in 2..=6, j in 2..=8, k in 2..=10
        for i in 2..=6 {
            for j in 2..=8 {
                for k in 2..=10 {
                    inside[(i * ny + j) * nz + k] = true;
                }
            }
        }

        let edt = edt_3d(&inside, nx, ny, nz, dx, dy, dz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = (i * ny + j) * nz + k;
                    if !inside[idx] {
                        assert_eq!(edt[idx], 0.0);
                    } else {
                        // Nearest background voxel is 1 step outside the box on some face.
                        let d_x = ((i - 1).min(7 - i) as f64) * dx;
                        let d_y = ((j - 1).min(9 - j) as f64) * dy;
                        let d_z = ((k - 1).min(11 - k) as f64) * dz;
                        let expected = d_x.min(d_y).min(d_z);
                        assert!(
                            (edt[idx] - expected).abs() < 1e-12,
                            "mismatch at ({i},{j},{k}): got {} expected {}",
                            edt[idx],
                            expected
                        );
                    }
                }
            }
        }
    }
}
