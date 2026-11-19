struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    clip_plane: vec4<f32>, // xyz = normal, w = -dot(normal, origin)
    clip_params: vec4<f32>, // x = 1.0 if enabled else 0.0, y = opacity
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.world_pos = in.position;
    out.world_normal = in.normal;
    out.color = in.color;
    return out;
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

@fragment
fn fs_main(in: VertexOut, @builtin(front_facing) front_facing: bool) -> @location(0) vec4<f32> {
    if (u.clip_params.x > 0.5) {
        let d = dot(in.world_pos, u.clip_plane.xyz) + u.clip_plane.w;
        if (d < 0.0) {
            discard;
        }
    }

    var n = normalize(in.world_normal);
    if (!front_facing) {
        n = -n;
    }

    let view_dir = normalize(u.camera_pos.xyz - in.world_pos);
    let key_dir = normalize(vec3<f32>(0.45, 0.55, 0.70));
    let fill_dir = normalize(vec3<f32>(-0.65, -0.30, 0.35));
    let back_dir = normalize(vec3<f32>(0.10, -0.80, -0.45));

    // Wrap diffuse for smooth TPMS channel readability
    let ndl_key = max(dot(n, key_dir), 0.0);
    let ndl_fill = max(dot(n, fill_dir), 0.0) * 0.35;
    let ndl_back = max(dot(n, back_dir), 0.0) * 0.18;

    let half_vec = normalize(key_dir + view_dir);
    let spec = pow(max(dot(n, half_vec), 0.0), 48.0) * 0.28;

    // Subtle Fresnel rim highlight to separate overlapping gyroid walls
    let ndv = max(dot(n, view_dir), 0.0);
    let rim = pow(1.0 - ndv, 3.0) * 0.14;

    // Hemisphere ambient (sky vs ground)
    let hemi = 0.5 + 0.5 * n.z;
    let ambient = mix(0.14, 0.26, hemi);

    let base_lin = srgb_to_linear(in.color.rgb);
    let lit = base_lin * (ambient + ndl_key * 0.78 + ndl_fill + ndl_back)
        + vec3<f32>(spec + rim);

    let alpha = in.color.a * u.clip_params.y;
    return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), alpha);
}

struct LineIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct LineOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_line(in: LineIn) -> LineOut {
    var out: LineOut;
    out.clip_pos = u.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    return vec4<f32>(srgb_to_linear(in.color.rgb), in.color.a);
}
