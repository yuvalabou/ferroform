//! 3D Voronoi cellular lattice generation with Bridson's Poisson-disk sampling.
//!
//! Provides deterministic, $O(N)$ 3D Poisson-disk seed generation with a maximal
//! gap-filling sweep, an $O(1)$ spatial-hash nearest-neighbor evaluator for
//! closed-cell foam ($d_2 - d_1$) and open-cell trabecular struts ($d_3 - d_1$),
//! slenderness enforcement ($L_{\text{strut}} \le 8\,r_{\text{strut}}$), and an
//! optional separable 3D voxel scalar-field smoothing pass for blended junctions.

use rayon::prelude::*;

/// Maximum allowed strut slenderness ratio ($L_{\text{strut}} / r_{\text{strut}}$).
pub const MAX_VORONOI_SLENDERNESS: f64 = 8.0;

/// Conservative upper bound on $L_{\max} / r_{\min}$ for maximal 3D Poisson-disk Voronoi edges.
pub const POISSON_MAX_EDGE_RATIO: f64 = 1.35;

/// Deterministic 64-bit SplitMix64 PRNG (zero external dependencies).
#[derive(Clone, Debug)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    #[inline]
    fn gen_range_usize(&mut self, upper: usize) -> usize {
        ((self.next_u64() as u128 * upper as u128) >> 64) as usize
    }

    /// Sample a random point uniformly in the spherical annulus `[r_min, 2 * r_min]`.
    fn next_annulus_3d(&mut self, r_min: f64) -> [f64; 3] {
        loop {
            let x = 2.0 * self.next_f64() - 1.0;
            let y = 2.0 * self.next_f64() - 1.0;
            let z = 2.0 * self.next_f64() - 1.0;
            let r2 = x * x + y * y + z * z;
            if r2 > 1e-12 && r2 <= 1.0 {
                let inv_len = 1.0 / r2.sqrt();
                let rad = r_min * (1.0 + self.next_f64() * 7.0).cbrt();
                return [x * inv_len * rad, y * inv_len * rad, z * inv_len * rad];
            }
        }
    }
}

/// Compute the effective Poisson-disk `min_dist` ($r_{\min}$) respecting the
/// strut slenderness constraint $L_{\text{strut}} \le 8 \, r_{\text{strut}}$ when
/// `is_skeletal` is enabled.
#[inline]
pub fn effective_min_dist(cell_size: f64, wall_thickness: f64, is_skeletal: bool) -> f64 {
    let base = cell_size.max(1e-3);
    if !is_skeletal {
        return base;
    }
    let r_strut = 0.5 * wall_thickness.max(1e-3);
    let max_r_min = (MAX_VORONOI_SLENDERNESS * r_strut) / POISSON_MAX_EDGE_RATIO;
    base.min(max_r_min.max(1e-3))
}

/// 3D Poisson-disk Voronoi field backed by a flat spatial hash grid for $O(1)$
/// nearest-seed queries.
#[derive(Clone, Debug)]
pub struct VoronoiField {
    pub seeds: Vec<[f64; 3]>,
    pub min_dist: f64,
    grid: Vec<u32>,
    origin: [f64; 3],
    cell_size: f64,
    inv_cell: f64,
    dims: [usize; 3],
}

impl VoronoiField {
    /// Generate a deterministic 3D Poisson-disk seed cloud over `[min_b, max_b]`
    /// with minimum separation `min_dist` and a maximal gap-filling pass.
    pub fn new(min_b: [f64; 3], max_b: [f64; 3], min_dist: f64, seed: u64) -> Self {
        let min_dist = min_dist.max(1e-4);
        let cell_size = min_dist / 3.0_f64.sqrt();
        let inv_cell = 1.0 / cell_size;

        let nx = (((max_b[0] - min_b[0]).max(cell_size) * inv_cell).ceil() as usize).max(1);
        let ny = (((max_b[1] - min_b[1]).max(cell_size) * inv_cell).ceil() as usize).max(1);
        let nz = (((max_b[2] - min_b[2]).max(cell_size) * inv_cell).ceil() as usize).max(1);
        let dims = [nx, ny, nz];

        let mut field = Self {
            seeds: Vec::new(),
            min_dist,
            grid: vec![u32::MAX; nx * ny * nz],
            origin: min_b,
            cell_size,
            inv_cell,
            dims,
        };

        field.sample_bridson(max_b, seed);
        field.fill_gaps();
        field
    }

    #[inline]
    fn grid_index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (ix * self.dims[1] + iy) * self.dims[2] + iz
    }

    #[inline]
    fn coord_to_cell(&self, p: [f64; 3]) -> Option<[usize; 3]> {
        let fx = (p[0] - self.origin[0]) * self.inv_cell;
        let fy = (p[1] - self.origin[1]) * self.inv_cell;
        let fz = (p[2] - self.origin[2]) * self.inv_cell;
        if fx < 0.0 || fy < 0.0 || fz < 0.0 {
            return None;
        }
        let ix = fx as usize;
        let iy = fy as usize;
        let iz = fz as usize;
        if ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2] {
            Some([ix, iy, iz])
        } else {
            None
        }
    }

    fn is_valid_candidate(&self, p: [f64; 3], cell: [usize; 3]) -> bool {
        let min_d2 = self.min_dist * self.min_dist;
        let x0 = cell[0].saturating_sub(2);
        let x1 = (cell[0] + 2).min(self.dims[0] - 1);
        let y0 = cell[1].saturating_sub(2);
        let y1 = (cell[1] + 2).min(self.dims[1] - 1);
        let z0 = cell[2].saturating_sub(2);
        let z1 = (cell[2] + 2).min(self.dims[2] - 1);

        for ix in x0..=x1 {
            for iy in y0..=y1 {
                let row = (ix * self.dims[1] + iy) * self.dims[2];
                for iz in z0..=z1 {
                    let s_idx = self.grid[row + iz];
                    if s_idx != u32::MAX {
                        let s = self.seeds[s_idx as usize];
                        let dx = p[0] - s[0];
                        let dy = p[1] - s[1];
                        let dz = p[2] - s[2];
                        if dx * dx + dy * dy + dz * dz < min_d2 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn insert_seed(&mut self, p: [f64; 3], cell: [usize; 3]) -> u32 {
        let id = self.seeds.len() as u32;
        self.seeds.push(p);
        let gidx = self.grid_index(cell[0], cell[1], cell[2]);
        self.grid[gidx] = id;
        id
    }

    fn sample_bridson(&mut self, max_b: [f64; 3], seed: u64) {
        const K_ATTEMPTS: usize = 30;
        let mut rng = SplitMix64::new(seed);

        let first = [
            self.origin[0] + rng.next_f64() * (max_b[0] - self.origin[0]).max(self.cell_size),
            self.origin[1] + rng.next_f64() * (max_b[1] - self.origin[1]).max(self.cell_size),
            self.origin[2] + rng.next_f64() * (max_b[2] - self.origin[2]).max(self.cell_size),
        ];
        if let Some(c) = self.coord_to_cell(first) {
            let id = self.insert_seed(first, c);
            let mut active = vec![id];

            while !active.is_empty() {
                let a_pos = rng.gen_range_usize(active.len());
                let base = self.seeds[active[a_pos] as usize];
                let mut found = false;

                for _ in 0..K_ATTEMPTS {
                    let off = rng.next_annulus_3d(self.min_dist);
                    let cand = [base[0] + off[0], base[1] + off[1], base[2] + off[2]];
                    if cand[0] < self.origin[0]
                        || cand[1] < self.origin[1]
                        || cand[2] < self.origin[2]
                        || cand[0] > max_b[0]
                        || cand[1] > max_b[1]
                        || cand[2] > max_b[2]
                    {
                        continue;
                    }
                    if let Some(cell) = self.coord_to_cell(cand) {
                        let gidx = self.grid_index(cell[0], cell[1], cell[2]);
                        if self.grid[gidx] == u32::MAX && self.is_valid_candidate(cand, cell) {
                            let new_id = self.insert_seed(cand, cell);
                            active.push(new_id);
                            found = true;
                            break;
                        }
                    }
                }

                if !found {
                    active.swap_remove(a_pos);
                }
            }
        }
    }

    /// Near-maximal gap-filling pass: inserts seeds at empty cell centers whose
    /// distance to every existing seed is at least `min_dist`, bounding covering radius.
    fn fill_gaps(&mut self) {
        for ix in 0..self.dims[0] {
            let cx = self.origin[0] + (ix as f64 + 0.5) * self.cell_size;
            for iy in 0..self.dims[1] {
                let cy = self.origin[1] + (iy as f64 + 0.5) * self.cell_size;
                for iz in 0..self.dims[2] {
                    let gidx = self.grid_index(ix, iy, iz);
                    if self.grid[gidx] != u32::MAX {
                        continue;
                    }
                    let p = [cx, cy, self.origin[2] + (iz as f64 + 0.5) * self.cell_size];
                    if self.is_valid_candidate(p, [ix, iy, iz]) {
                        self.insert_seed(p, [ix, iy, iz]);
                    }
                }
            }
        }
    }

    /// Return the three smallest Euclidean distances `(d1, d2, d3)` from `(x, y, z)`
    /// to the nearest Voronoi seeds in $O(1)$ amortized time.
    #[inline]
    pub fn nearest_three_distances(&self, x: f64, y: f64, z: f64) -> (f64, f64, f64) {
        let mut d1_sq = f64::INFINITY;
        let mut d2_sq = f64::INFINITY;
        let mut d3_sq = f64::INFINITY;

        let ix = (((x - self.origin[0]) * self.inv_cell).floor() as isize)
            .clamp(0, (self.dims[0] - 1) as isize) as usize;
        let iy = (((y - self.origin[1]) * self.inv_cell).floor() as isize)
            .clamp(0, (self.dims[1] - 1) as isize) as usize;
        let iz = (((z - self.origin[2]) * self.inv_cell).floor() as isize)
            .clamp(0, (self.dims[2] - 1) as isize) as usize;

        const R: usize = 4;
        let x0 = ix.saturating_sub(R);
        let x1 = (ix + R).min(self.dims[0] - 1);
        let y0 = iy.saturating_sub(R);
        let y1 = (iy + R).min(self.dims[1] - 1);
        let z0 = iz.saturating_sub(R);
        let z1 = (iz + R).min(self.dims[2] - 1);

        for gx in x0..=x1 {
            for gy in y0..=y1 {
                let row = (gx * self.dims[1] + gy) * self.dims[2];
                for gz in z0..=z1 {
                    let s_idx = self.grid[row + gz];
                    if s_idx == u32::MAX {
                        continue;
                    }
                    let s = self.seeds[s_idx as usize];
                    let dx = x - s[0];
                    let dy = y - s[1];
                    let dz = z - s[2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 < d1_sq {
                        d3_sq = d2_sq;
                        d2_sq = d1_sq;
                        d1_sq = d2;
                    } else if d2 < d2_sq {
                        d3_sq = d2_sq;
                        d2_sq = d2;
                    } else if d2 < d3_sq {
                        d3_sq = d2;
                    }
                }
            }
        }

        // Fallback if query is far outside the padded grid or seed count < 3.
        if !d3_sq.is_finite() {
            d1_sq = f64::INFINITY;
            d2_sq = f64::INFINITY;
            d3_sq = f64::INFINITY;
            for s in &self.seeds {
                let dx = x - s[0];
                let dy = y - s[1];
                let dz = z - s[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < d1_sq {
                    d3_sq = d2_sq;
                    d2_sq = d1_sq;
                    d1_sq = d2;
                } else if d2 < d2_sq {
                    d3_sq = d2_sq;
                    d2_sq = d2;
                } else if d2 < d3_sq {
                    d3_sq = d2;
                }
            }
        }

        (d1_sq.sqrt(), d2_sq.sqrt(), d3_sq.sqrt())
    }

    /// Closed-cell Voronoi foam scalar field ($d_2 - d_1 \ge 0$).
    /// Zero at cell boundary faces; `d2 - d1 <= wall_thickness` gives wall thickness `wall_thickness`.
    #[inline]
    pub fn eval_foam(&self, x: f64, y: f64, z: f64) -> f64 {
        let (d1, d2, _) = self.nearest_three_distances(x, y, z);
        if d2.is_finite() && d1.is_finite() {
            d2 - d1
        } else {
            0.0
        }
    }

    /// Open-cell trabecular Voronoi strut scalar field ($d_3 - d_1 \ge 0$).
    /// Zero along Voronoi edges; `d3 - d1 <= sqrt(3) * r_strut` gives strut radius `r_strut`.
    #[inline]
    pub fn eval_strut(&self, x: f64, y: f64, z: f64) -> f64 {
        let (d1, _, d3) = self.nearest_three_distances(x, y, z);
        if d3.is_finite() && d1.is_finite() {
            d3 - d1
        } else {
            0.0
        }
    }
}

/// Apply `passes` iterations of a separable 3D `[0.25, 0.5, 0.25]` smoothing
/// kernel to a scalar field of dimensions `(nx, ny, nz)` in parallel.
pub fn smooth_scalar_field(field: &mut Vec<f64>, nx: usize, ny: usize, nz: usize, passes: usize) {
    if passes == 0 || nx < 2 || ny < 2 || nz < 2 || field.len() != nx * ny * nz {
        return;
    }
    let mut scratch = vec![0.0; field.len()];
    let stride_x = ny * nz;

    for _ in 0..passes {
        // 1. Smooth along Z
        scratch
            .par_chunks_mut(stride_x)
            .enumerate()
            .for_each(|(ix, out_slab)| {
                let in_slab = &field[ix * stride_x..(ix + 1) * stride_x];
                for iy in 0..ny {
                    let row = iy * nz;
                    for iz in 0..nz {
                        let zm = iz.saturating_sub(1);
                        let zp = (iz + 1).min(nz - 1);
                        out_slab[row + iz] = 0.25 * in_slab[row + zm]
                            + 0.5 * in_slab[row + iz]
                            + 0.25 * in_slab[row + zp];
                    }
                }
            });

        // 2. Smooth along Y
        field
            .par_chunks_mut(stride_x)
            .enumerate()
            .for_each(|(ix, out_slab)| {
                let in_slab = &scratch[ix * stride_x..(ix + 1) * stride_x];
                for iy in 0..ny {
                    let ym = iy.saturating_sub(1) * nz;
                    let y0 = iy * nz;
                    let yp = (iy + 1).min(ny - 1) * nz;
                    for iz in 0..nz {
                        out_slab[y0 + iz] =
                            0.25 * in_slab[ym + iz] + 0.5 * in_slab[y0 + iz] + 0.25 * in_slab[yp + iz];
                    }
                }
            });

        // 3. Smooth along X
        scratch
            .par_chunks_mut(stride_x)
            .enumerate()
            .for_each(|(ix, out_slab)| {
                let xm = ix.saturating_sub(1) * stride_x;
                let x0 = ix * stride_x;
                let xp = (ix + 1).min(nx - 1) * stride_x;
                for idx in 0..stride_x {
                    out_slab[idx] =
                        0.25 * field[xm + idx] + 0.5 * field[x0 + idx] + 0.25 * field[xp + idx];
                }
            });

        std::mem::swap(field, &mut scratch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poisson_min_distance_and_determinism() {
        let a = VoronoiField::new([0.0, 0.0, 0.0], [20.0, 20.0, 20.0], 4.0, 42);
        let b = VoronoiField::new([0.0, 0.0, 0.0], [20.0, 20.0, 20.0], 4.0, 42);
        let c = VoronoiField::new([0.0, 0.0, 0.0], [20.0, 20.0, 20.0], 4.0, 99);

        assert!(a.seeds.len() >= 20);
        assert_eq!(a.seeds, b.seeds);
        assert_ne!(a.seeds, c.seeds);

        let min_d2 = a.min_dist * a.min_dist - 1e-9;
        for i in 0..a.seeds.len() {
            for j in (i + 1)..a.seeds.len() {
                let dx = a.seeds[i][0] - a.seeds[j][0];
                let dy = a.seeds[i][1] - a.seeds[j][1];
                let dz = a.seeds[i][2] - a.seeds[j][2];
                assert!(dx * dx + dy * dy + dz * dz >= min_d2);
            }
        }
    }

    #[test]
    fn test_spatial_hash_matches_bruteforce() {
        let vf = VoronoiField::new([-5.0, -5.0, -5.0], [15.0, 15.0, 15.0], 3.5, 123);
        let mut rng = SplitMix64::new(777);

        for _ in 0..100 {
            let q = [
                rng.next_f64() * 12.0,
                rng.next_f64() * 12.0,
                rng.next_f64() * 12.0,
            ];
            let (d1, d2, d3) = vf.nearest_three_distances(q[0], q[1], q[2]);

            let mut dists: Vec<f64> = vf
                .seeds
                .iter()
                .map(|s| {
                    let dx = q[0] - s[0];
                    let dy = q[1] - s[1];
                    let dz = q[2] - s[2];
                    (dx * dx + dy * dy + dz * dz).sqrt()
                })
                .collect();
            dists.sort_by(|x, y| x.partial_cmp(y).unwrap());

            assert!((d1 - dists[0]).abs() < 1e-12);
            assert!((d2 - dists[1]).abs() < 1e-12);
            assert!((d3 - dists[2]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_slenderness_clamp() {
        // wall_thickness = 1.0 => r_strut = 0.5 => max r_min = 8 * 0.5 / 1.35 ~= 2.963
        let eff = effective_min_dist(5.0, 1.0, true);
        assert!(eff * POISSON_MAX_EDGE_RATIO <= MAX_VORONOI_SLENDERNESS * 0.5 + 1e-9);
        assert_eq!(effective_min_dist(5.0, 1.0, false), 5.0);
    }

    #[test]
    fn test_smooth_scalar_field_preserves_constant() {
        let mut field = vec![3.25; 8 * 8 * 8];
        smooth_scalar_field(&mut field, 8, 8, 8, 3);
        for v in field {
            assert!((v - 3.25).abs() < 1e-12);
        }
    }
}
