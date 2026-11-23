import { withViews } from "./webgpu-noise.js";

// Tile adjacency and accumulation order are shared with the CUDA v5 rasterizer.
const source = `
struct Parameters {
    count: u32, width: u32, tile: u32, mode: u32,
    dispatchWidth: u32, columns: u32, padding: vec2u,
}
@group(0) @binding(0) var<storage, read> patches: array<f32>;
@group(0) @binding(1) var<storage, read> offsets: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> fieldA: array<f32>;
@group(0) @binding(4) var<storage, read> fieldB: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> settings: Parameters;

fn eased(lo: f32, hi: f32, value: f32) -> f32 {
    let t = clamp((value - lo) / (hi - lo), 0, 1);
    return (t * t) * (3 - 2 * t);
}
fn oneMinusExp(x: f32) -> f32 {
    if (x < 0.001) { return x * (1 - x * (0.5 - x / 6)); }
    return 1 - exp(-x);
}
fn angle(v: f32, u: f32) -> f32 {
    if (u == 0 && v == 0) { return 0; }
    return atan2(v, u);
}
fn sixth(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x2;
}
fn dust(base: u32, dx: f32, dy: f32, detail: f32) -> f32 {
    var u = dx;
    var v = (dy + patches[base + 6] * u * u) / patches[base + 7];
    let strength = patches[base + 12];
    if (strength > 0) {
        let prominence = patches[base + 13];
        let tail = patches[base + 14];
        u = u / (1.29 + 0.60 * prominence) - 0.21 * tail;
        v += tail * patches[base + 15] * u * u;
        let theta = angle(v, u);
        let r2 = u * u + v * v;
        let outline = 0.97 + 0.18 * cos(2 * theta + patches[base + 8])
            + (0.11 * sin(3 * theta + patches[base + 9]) + 0.06 * cos(7 * theta + patches[base + 10]));
        let core = max(0, 1 - r2 / (1.13 * 1.13));
        let preferred = sixth(max(0, cos(theta - patches[base + 16])));
        let edge = eased(0, 0.57 - 0.42 * preferred, 1 - sqrt(r2) / outline);
        let shaped = edge * (0.49 + 0.51 * exp(-1.65 * r2));
        let shape = (0.13 + 0.66 * prominence) * ((1 - prominence) * core * core * core + prominence * shaped);
        return strength * shape * detail;
    }
    let r2 = u * u + v * v;
    if (r2 >= 1) { return 0; }
    let theta = angle(v, u);
    let core = 1 - r2;
    let contour = 0.62 + 0.14 * cos(2 * theta + patches[base + 8])
        + (0.07 * sin(3 * theta + patches[base + 9]) + 0.035 * cos(7 * theta + patches[base + 10]));
    let delta = (sqrt(r2) - contour) / 0.11;
    let arc = exp(-(delta * delta)) * eased(-0.7, 0.8, cos(theta + patches[base + 11])) * (core * core);
    return strength * (core * core * core * detail + 0.68 * arc * (detail * detail));
}

@compute @workgroup_size(128)
fn raster(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * settings.dispatchWidth;
    if (i >= settings.count) { return; }
    let ix = i % settings.width;
    let iy = i / settings.width;
    let bin = (iy / settings.tile) * settings.columns + ix / settings.tile;
    let x = f32(ix);
    let y = f32(iy);
    var total = 0.0;
    var edges = vec3f(0);
    for (var j = offsets[bin]; j < offsets[bin + 1]; j++) {
        let p = 19 * indices[j];
        if (abs(x - patches[p + 17]) > patches[p + 3]
            || abs(y - patches[p + 18]) > patches[p + 3]) { continue; }
        let dx = (x - patches[p]) / patches[p + 2];
        let dy = (y - patches[p + 1]) / patches[p + 2];
        var u = patches[p + 4] * dx + patches[p + 5] * dy;
        var v = patches[p + 4] * dy - patches[p + 5] * dx;
        if (settings.mode == 0) {
            total += dust(p, u, v, 0.23 + 0.77 * fieldA[i]);
        } else if (settings.mode == 1) {
            v = ((v - 0.12) + patches[p + 6] * u * u) / 0.56;
            let r2 = u * u + v * v;
            let side = fieldA[i] * fieldB[i];
            if (r2 >= 1 || v <= -0.50 || side == 0) { continue; }
            let radius = sqrt(r2);
            let preference = max(0, -v / max(radius, 1e-12));
            let body = eased(0, 0.57 - 0.42 * sixth(preference), 1 - radius)
                * exp(-1.75 * r2) * eased(-0.50, 0.38, v);
            total += (0.56 * patches[p + 7]) * body * side;
        } else {
            u /= patches[p + 6];
            v *= patches[p + 6];
            let u2 = u*u;
            let v2 = v*v;
            var r2 = u2 + v2;
            let base = 1 + r2;
            let skew = u * (u2 - 2.3 * v2) / (base * sqrt(base));
            r2 *= 1 + patches[p + 7] * skew;
            let tau = 6.283185307179586;
            let core = 0.85 * 0.85 + patches[p + 12];
            let inner = 2.2 * 2.2 + patches[p + 12];
            let outer = 5.2 * 5.2 + patches[p + 12];
            let psf = 0.951 * exp(-r2 / (2 * core)) / (tau * core)
                + 0.042 * exp(-r2 / (2 * inner)) / (tau * inner)
                + 0.007 * exp(-r2 / (2 * outer)) / (tau * outer);
            let received = patches[p + 8] * psf;
            total += received;
            let wing = oneMinusExp(0.31 * (r2 * r2));
            edges += (received * wing) * vec3f(patches[p + 9], patches[p + 10], patches[p + 11]);
        }
    }
    if (settings.mode == 2) {
        let brightness = oneMinusExp(total);
        let response = select(brightness / total, 0, total == 0);
        let rgb = brightness * vec3f(224, 229, 238) + response * edges;
        output[3 * i] = rgb.r;
        output[3 * i + 1] = rgb.g;
        output[3 * i + 2] = rgb.b;
    } else {
        output[i] = tanh(total);
    }
}
`;

export async function createRaster(device) {
  const module = device.createShaderModule({ code: source, label: "v5 tile rasterization" });
  const report = await module.getCompilationInfo();
  const errors = report.messages.filter(message => message.type === "error");
  if (errors.length) throw new Error(errors.map(message => `${message.lineNum}: ${message.message}`).join("\n"));
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module, entryPoint: "raster" },
  });
  const input = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const uploads = [];
  let output, readback;
  let capacity = 0;
  const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const workgroupLimit = device.limits.maxComputeWorkgroupsPerDimension;

  function upload(data, index) {
    const bytes = Math.max(4, data.byteLength);
    if (!uploads[index] || uploads[index].size < bytes) {
      uploads[index]?.destroy();
      uploads[index] = device.createBuffer({ size: bytes, usage: input });
    }
    if (data.byteLength) device.queue.writeBuffer(uploads[index], 0, data);
  }

  return async (patches, offsets, indices, a, b, width, height, tile, mode) => {
    const count = width * height;
    const bytes = count * (mode === 2 ? 12 : 4);
    if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error("WebGPU raster buffer exceeds device limits");
    if (capacity < bytes) {
      output?.destroy();
      readback?.destroy();
      output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      readback = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      capacity = bytes;
    }
    withViews([patches, offsets, indices, a, b], ["f32", "u32", "u32", "f32", "f32"],
      (...inputs) => inputs.forEach(upload));
    const groupsX = Math.min(Math.ceil(count / 128), workgroupLimit);
    const dimensions = new Uint32Array([count, width, tile, mode, groupsX * 128, Math.ceil(width / tile), 0, 0]);
    device.queue.writeBuffer(params, 0, dimensions);
    const binding = [...uploads, output, params];
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: binding.map((buffer, index) => ({ binding: index, resource: { buffer } })),
    });
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, Math.ceil(count / (groupsX * 128)));
    pass.end();
    commands.copyBufferToBuffer(output, 0, readback, 0, bytes);
    device.queue.submit([commands.finish()]);
    await readback.mapAsync(GPUMapMode.READ, 0, bytes);
    try {
      return new Float32Array(readback.getMappedRange(0, bytes).slice(0));
    } finally {
      readback.unmap();
    }
  };
}
