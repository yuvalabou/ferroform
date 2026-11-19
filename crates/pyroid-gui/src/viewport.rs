//! Interactive 3D wgpu viewport embedded in egui via an offscreen render target.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use eframe::egui;
use eframe::egui_wgpu::{self, wgpu};
use eframe::wgpu::util::DeviceExt;
use glam::{Mat4, Vec3};
use pyroid_core::marching::MeshData;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    clip_plane: [f32; 4],
    clip_params: [f32; 4],
}

/// Orbit camera state.
#[derive(Clone, Debug)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 45.0,
            yaw: 0.78,
            pitch: 0.52,
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        self.target + Vec3::new(cp * cy, cp * sy, sp) * self.distance
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let eye = self.eye();
        let view = Mat4::look_at_rh(eye, self.target, Vec3::Z);
        let near = (self.distance * 0.005).max(0.01);
        let far = (self.distance * 20.0).max(500.0);
        let proj = Mat4::perspective_rh(40.0f32.to_radians(), aspect.max(0.1), near, far);
        proj * view
    }

    pub fn fit_bounds(&mut self, min_b: [f32; 3], max_b: [f32; 3]) {
        let min_v = Vec3::from_array(min_b);
        let max_v = Vec3::from_array(max_b);
        self.target = 0.5 * (min_v + max_v);
        let diag = (max_v - min_v).length().max(1.0);
        self.distance = diag * 1.45;
        self.yaw = 0.78;
        self.pitch = 0.52;
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.54, 1.54);
    }

    pub fn pan(&mut self, dx: f32, dy: f32) {
        let forward = (self.target - self.eye()).normalize_or_zero();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward).normalize_or_zero();
        let scale = self.distance * 0.0015;
        self.target -= right * (dx * scale);
        self.target += up * (dy * scale);
    }

    pub fn zoom(&mut self, scroll_delta: f32) {
        let factor = (1.0 - scroll_delta * 0.0015).clamp(0.5, 2.0);
        self.distance = (self.distance * factor).clamp(0.5, 5000.0);
    }
}

/// Coolwarm colormap matching PyVista/matplotlib's `coolwarm` LUT.
pub fn coolwarm_color(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    // 5-stop coolwarm approximation
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [0.230, 0.299, 0.754]),
        (0.25, [0.482, 0.620, 0.973]),
        (0.50, [0.865, 0.865, 0.865]),
        (0.75, [0.949, 0.588, 0.451]),
        (1.00, [0.706, 0.016, 0.150]),
    ];
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let u = (t - t0) / (t1 - t0);
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
                1.0,
            ];
        }
    }
    [0.706, 0.016, 0.150, 1.0]
}

/// Viridis-like colormap for Z height.
pub fn viridis_color(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    const STOPS: [(f32, [f32; 3]); 5] = [
        (0.00, [0.267, 0.005, 0.329]),
        (0.25, [0.230, 0.322, 0.545]),
        (0.50, [0.129, 0.569, 0.549]),
        (0.75, [0.369, 0.788, 0.384]),
        (1.00, [0.992, 0.906, 0.145]),
    ];
    for w in STOPS.windows(2) {
        let (t0, c0) = w[0];
        let (t1, c1) = w[1];
        if t <= t1 {
            let u = (t - t0) / (t1 - t0);
            return [
                c0[0] + (c1[0] - c0[0]) * u,
                c0[1] + (c1[1] - c0[1]) * u,
                c0[2] + (c1[2] - c0[2]) * u,
                1.0,
            ];
        }
    }
    [0.992, 0.906, 0.145, 1.0]
}

const MSAA_SAMPLES: u32 = 4;

/// Compute crease-aware normals for normal meshes, and memory-efficient indexed normals for huge meshes.
pub fn build_mesh_vertices(
    mesh: &MeshData,
    colors: &[[f32; 4]],
) -> (Vec<MeshVertex>, Vec<u32>) {
    let n_pts = mesh.points.len();
    let n_faces = mesh.faces.len();

    // For dense meshes (> 250k faces), use indexed smooth normals (~6x less GPU memory)
    if n_faces > 250_000 {
        let mut normals = vec![[0.0f32; 3]; n_pts];
        for f in &mesh.faces {
            let i0 = f[0] as usize;
            let i1 = f[1] as usize;
            let i2 = f[2] as usize;
            let p0 = Vec3::from_array(mesh.points[i0]);
            let p1 = Vec3::from_array(mesh.points[i1]);
            let p2 = Vec3::from_array(mesh.points[i2]);
            let n = (p1 - p0).cross(p2 - p0);
            for idx in [i0, i1, i2] {
                normals[idx][0] += n.x;
                normals[idx][1] += n.y;
                normals[idx][2] += n.z;
            }
        }
        let vertices: Vec<MeshVertex> = (0..n_pts)
            .map(|i| {
                let n = Vec3::from_array(normals[i]).normalize_or_zero();
                MeshVertex {
                    position: mesh.points[i],
                    normal: if n.length_squared() < 1e-12 {
                        [0.0, 0.0, 1.0]
                    } else {
                        n.to_array()
                    },
                    color: colors.get(i).copied().unwrap_or([0.75, 0.78, 0.82, 1.0]),
                }
            })
            .collect();
        let indices: Vec<u32> = mesh.faces.iter().flat_map(|f| *f).collect();
        return (vertices, indices);
    }

    let mut face_normals = Vec::with_capacity(n_faces);
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); n_pts];

    for (fi, f) in mesh.faces.iter().enumerate() {
        let i0 = f[0] as usize;
        let i1 = f[1] as usize;
        let i2 = f[2] as usize;
        let p0 = Vec3::from_array(mesh.points[i0]);
        let p1 = Vec3::from_array(mesh.points[i1]);
        let p2 = Vec3::from_array(mesh.points[i2]);
        let cross = (p1 - p0).cross(p2 - p0);
        let n = cross.normalize_or_zero();
        face_normals.push(if n.length_squared() < 1e-12 { Vec3::Z } else { n });
        vertex_faces[i0].push(fi);
        vertex_faces[i1].push(fi);
        vertex_faces[i2].push(fi);
    }

    // cos(75 deg) ~= 0.258: smooth across curved Marching Cubes walls, crisp on 90 deg caps
    const CREASE_COS: f32 = 0.258;
    let mut vertices = Vec::with_capacity(n_faces * 3);
    let mut indices = Vec::with_capacity(n_faces * 3);

    for (fi, f) in mesh.faces.iter().enumerate() {
        let fn_cur = face_normals[fi];
        for &vi_u32 in f {
            let vi = vi_u32 as usize;
            let mut accum = Vec3::ZERO;
            for &adj_fi in &vertex_faces[vi] {
                let adj_n = face_normals[adj_fi];
                if fn_cur.dot(adj_n) >= CREASE_COS {
                    accum += adj_n;
                }
            }
            let n = accum.normalize_or_zero();
            let normal = if n.length_squared() < 1e-12 {
                fn_cur.to_array()
            } else {
                n.to_array()
            };
            let out_idx = vertices.len() as u32;
            vertices.push(MeshVertex {
                position: mesh.points[vi],
                normal,
                color: colors.get(vi).copied().unwrap_or([0.75, 0.78, 0.82, 1.0]),
            });
            indices.push(out_idx);
        }
    }

    (vertices, indices)
}

struct GpuMesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    n_indices: u32,
}

struct GpuLines {
    vbuf: wgpu::Buffer,
    n_vertices: u32,
}

pub struct GpuViewport {
    device: wgpu::Device,
    queue: wgpu::Queue,
    egui_renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,

    opaque_pipeline: wgpu::RenderPipeline,
    translucent_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,

    color_tex: Option<wgpu::Texture>,
    color_view: Option<wgpu::TextureView>,
    msaa_color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    pub tex_id: Option<egui::TextureId>,
    tex_size: [u32; 2],

    wall_gpu: Option<GpuMesh>,
    cav_a_gpu: Option<GpuMesh>,
    cav_b_gpu: Option<GpuMesh>,
    lines_gpu: Option<GpuLines>,

    pub camera: OrbitCamera,
    pub dirty: bool,
}

impl GpuViewport {
    pub fn new(wgpu_state: &egui_wgpu::RenderState) -> Self {
        let device = wgpu_state.device.clone();
        let queue = wgpu_state.queue.clone();
        let egui_renderer = wgpu_state.renderer.clone();

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyroid_mesh_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pyroid_uniforms"),
            contents: bytemuck::bytes_of(&Uniforms {
                view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                camera_pos: [0.0, 0.0, 50.0, 1.0],
                clip_plane: [1.0, 0.0, 0.0, 0.0],
                clip_params: [0.0, 1.0, 0.0, 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("pyroid_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pyroid_bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pyroid_pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let mesh_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let make_mesh_pipeline = |label: &str, depth_write: bool, blend: Option<wgpu::BlendState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[mesh_vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: depth_write,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: MSAA_SAMPLES,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
        };

        let opaque_pipeline = make_mesh_pipeline("pyroid_opaque", true, None);
        let translucent_pipeline = make_mesh_pipeline(
            "pyroid_translucent",
            false,
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );

        let line_vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let line_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pyroid_lines"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_line"),
                buffers: &[line_vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_line"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: MSAA_SAMPLES,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        });

        Self {
            device,
            queue,
            egui_renderer,
            opaque_pipeline,
            translucent_pipeline,
            line_pipeline,
            uniform_buf,
            bind_group,
            color_tex: None,
            color_view: None,
            msaa_color_view: None,
            depth_view: None,
            tex_id: None,
            tex_size: [0, 0],
            wall_gpu: None,
            cav_a_gpu: None,
            cav_b_gpu: None,
            lines_gpu: None,
            camera: OrbitCamera::default(),
            dirty: true,
        }
    }

    fn upload_mesh(&self, vertices: &[MeshVertex], indices: &[u32]) -> Option<GpuMesh> {
        if vertices.is_empty() || indices.is_empty() {
            return None;
        }
        let v_bytes: &[u8] = bytemuck::cast_slice(vertices);
        let i_bytes: &[u8] = bytemuck::cast_slice(indices);
        let max_buf = self.device.limits().max_buffer_size;
        if (v_bytes.len() as u64) > max_buf || (i_bytes.len() as u64) > max_buf {
            return None;
        }
        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_vbuf"),
            contents: v_bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ibuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh_ibuf"),
            contents: i_bytes,
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(GpuMesh {
            vbuf,
            ibuf,
            n_indices: indices.len() as u32,
        })
    }

    pub fn set_wall(&mut self, vertices: &[MeshVertex], indices: &[u32]) {
        self.wall_gpu = self.upload_mesh(vertices, indices);
        self.dirty = true;
    }

    pub fn set_cavity_a(&mut self, vertices: &[MeshVertex], indices: &[u32]) {
        self.cav_a_gpu = self.upload_mesh(vertices, indices);
        self.dirty = true;
    }

    pub fn set_cavity_b(&mut self, vertices: &[MeshVertex], indices: &[u32]) {
        self.cav_b_gpu = self.upload_mesh(vertices, indices);
        self.dirty = true;
    }

    pub fn set_bounds_lines(&mut self, min_b: [f32; 3], max_b: [f32; 3]) {
        let [x0, y0, z0] = min_b;
        let [x1, y1, z1] = max_b;
        let c = [0.45, 0.50, 0.58, 0.7];
        let mut verts = vec![
            // Bottom rectangle
            LineVertex { position: [x0, y0, z0], color: c },
            LineVertex { position: [x1, y0, z0], color: c },
            LineVertex { position: [x1, y0, z0], color: c },
            LineVertex { position: [x1, y1, z0], color: c },
            LineVertex { position: [x1, y1, z0], color: c },
            LineVertex { position: [x0, y1, z0], color: c },
            LineVertex { position: [x0, y1, z0], color: c },
            LineVertex { position: [x0, y0, z0], color: c },
            // Top rectangle
            LineVertex { position: [x0, y0, z1], color: c },
            LineVertex { position: [x1, y0, z1], color: c },
            LineVertex { position: [x1, y0, z1], color: c },
            LineVertex { position: [x1, y1, z1], color: c },
            LineVertex { position: [x1, y1, z1], color: c },
            LineVertex { position: [x0, y1, z1], color: c },
            LineVertex { position: [x0, y1, z1], color: c },
            LineVertex { position: [x0, y0, z1], color: c },
            // Verticals
            LineVertex { position: [x0, y0, z0], color: c },
            LineVertex { position: [x0, y0, z1], color: c },
            LineVertex { position: [x1, y0, z0], color: c },
            LineVertex { position: [x1, y0, z1], color: c },
            LineVertex { position: [x1, y1, z0], color: c },
            LineVertex { position: [x1, y1, z1], color: c },
            LineVertex { position: [x0, y1, z0], color: c },
            LineVertex { position: [x0, y1, z1], color: c },
        ];

        // Origin RGB axes scaled to 25% of bounding diagonal
        let len = (Vec3::from_array(max_b) - Vec3::from_array(min_b))
            .length()
            .max(1.0)
            * 0.25;
        let o = [x0, y0, z0];
        verts.extend_from_slice(&[
            LineVertex { position: o, color: [0.95, 0.25, 0.25, 1.0] },
            LineVertex { position: [o[0] + len, o[1], o[2]], color: [0.95, 0.25, 0.25, 1.0] },
            LineVertex { position: o, color: [0.25, 0.90, 0.35, 1.0] },
            LineVertex { position: [o[0], o[1] + len, o[2]], color: [0.25, 0.90, 0.35, 1.0] },
            LineVertex { position: o, color: [0.30, 0.55, 0.98, 1.0] },
            LineVertex { position: [o[0], o[1], o[2] + len], color: [0.30, 0.55, 0.98, 1.0] },
        ]);

        let vbuf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lines_vbuf"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.lines_gpu = Some(GpuLines {
            vbuf,
            n_vertices: verts.len() as u32,
        });
        self.dirty = true;
    }

    pub fn clear(&mut self) {
        self.wall_gpu = None;
        self.cav_a_gpu = None;
        self.cav_b_gpu = None;
        self.lines_gpu = None;
        self.dirty = true;
    }

    fn ensure_targets(&mut self, width: u32, height: u32) {
        let w = width.max(16);
        let h = height.max(16);
        if self.tex_size == [w, h] && self.color_tex.is_some() {
            return;
        }
        self.tex_size = [w, h];

        let color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&Default::default());

        let msaa_color_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_msaa_color"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_color_view = msaa_color_tex.create_view(&Default::default());

        let depth_tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport_depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: MSAA_SAMPLES,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());

        let mut renderer = self.egui_renderer.write();
        if let Some(id) = self.tex_id {
            renderer.update_egui_texture_from_wgpu_texture(
                &self.device,
                &color_view,
                wgpu::FilterMode::Linear,
                id,
            );
        } else {
            let id = renderer.register_native_texture(
                &self.device,
                &color_view,
                wgpu::FilterMode::Linear,
            );
            self.tex_id = Some(id);
        }

        self.color_tex = Some(color_tex);
        self.color_view = Some(color_view);
        self.msaa_color_view = Some(msaa_color_view);
        self.depth_view = Some(depth_view);
        self.dirty = true;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_if_needed(
        &mut self,
        width: u32,
        height: u32,
        dark_mode: bool,
        show_wall: bool,
        show_cav_a: bool,
        show_cav_b: bool,
        clip_plane: Option<([f32; 3], [f32; 3])>,
    ) {
        self.ensure_targets(width, height);
        if !self.dirty {
            return;
        }
        self.dirty = false;

        let Some(ref color_view) = self.color_view else {
            return;
        };
        let Some(ref msaa_view) = self.msaa_color_view else {
            return;
        };
        let Some(ref depth_view) = self.depth_view else {
            return;
        };

        let aspect = self.tex_size[0] as f32 / (self.tex_size[1].max(1) as f32);
        let view_proj = self.camera.view_proj(aspect).to_cols_array_2d();
        let eye = self.camera.eye();
        let (clip_vec, clip_enabled) = if let Some((n, o)) = clip_plane {
            let d = -(n[0] * o[0] + n[1] * o[1] + n[2] * o[2]);
            ([n[0], n[1], n[2], d], 1.0f32)
        } else {
            ([1.0, 0.0, 0.0, 0.0], 0.0f32)
        };

        let clear_color = if dark_mode {
            wgpu::Color {
                r: 0.035,
                g: 0.042,
                b: 0.055,
                a: 1.0,
            }
        } else {
            wgpu::Color {
                r: 0.90,
                g: 0.92,
                b: 0.95,
                a: 1.0,
            }
        };

        let show_cavs = show_cav_a || show_cav_b;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport_encoder"),
            });

        // Pass 1: Opaque meshes (Cavities + opaque Wall + Lines)
        self.queue.write_buffer(
            &self.uniform_buf,
            0,
            bytemuck::bytes_of(&Uniforms {
                view_proj,
                camera_pos: [eye.x, eye.y, eye.z, 1.0],
                clip_plane: clip_vec,
                clip_params: [clip_enabled, 1.0, 0.0, 0.0],
            }),
        );

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport_opaque_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: msaa_view,
                    resolve_target: Some(color_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_pipeline(&self.opaque_pipeline);

            if show_cav_a {
                if let Some(ref mesh) = self.cav_a_gpu {
                    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.n_indices, 0, 0..1);
                }
            }
            if show_cav_b {
                if let Some(ref mesh) = self.cav_b_gpu {
                    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.n_indices, 0, 0..1);
                }
            }
            if show_wall && !show_cavs {
                if let Some(ref mesh) = self.wall_gpu {
                    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..mesh.n_indices, 0, 0..1);
                }
            }

            if let Some(ref lines) = self.lines_gpu {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, lines.vbuf.slice(..));
                pass.draw(0..lines.n_vertices, 0..1);
            }
        }

        // Pass 2: Translucent Wall when cavities are visible
        if show_wall && show_cavs {
            if let Some(ref mesh) = self.wall_gpu {
                self.queue.write_buffer(
                    &self.uniform_buf,
                    0,
                    bytemuck::bytes_of(&Uniforms {
                        view_proj,
                        camera_pos: [eye.x, eye.y, eye.z, 1.0],
                        clip_plane: clip_vec,
                        clip_params: [clip_enabled, 0.25, 0.0, 0.0],
                    }),
                );

                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("viewport_translucent_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: msaa_view,
                        resolve_target: Some(color_view),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_pipeline(&self.translucent_pipeline);
                pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
                pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.n_indices, 0, 0..1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
    }
}
