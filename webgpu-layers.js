import { withViews } from "./webgpu-noise.js";

// Fused geometry, density and compositing from desktop/wallpaper_compute.py.
const source = `
struct Parameters {
    count: u32, stage: u32, dispatchWidth: u32, padding: u32,
    values: array<vec4f, 5>,
}
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;

fn value(i: u32) -> f32 { return p.values[i / 4][i % 4]; }
fn readPlane(j: u32, i: u32) -> f32 { return input[j * p.count + i]; }
fn writePlane(j: u32, i: u32, x: f32) { output[j * p.count + i] = x; }
fn ease(lo: f32, hi: f32, x: f32) -> f32 {
    let t = clamp((x - lo) / (hi - lo), 0, 1);
    return (t*t) * (3 - 2*t);
}
fn norm(x: f32, y: f32) -> f32 { return sqrt(x*x + y*y); }

fn geometry(i: u32) {
    let slab = readPlane(2, i);
    let folds = readPlane(3, i);
    let broad = readPlane(4, i);
    let stars = readPlane(5, i);
    let cavityStrength = value(10);
    let x = (readPlane(0,i) - value(0)) + 0.021 * slab;
    let y = (readPlane(1,i) - value(1)) + 0.034 * slab - value(4) * x;
    let u = x / value(2);
    let v = y / value(3);
    let envelope = exp(-0.62 * pow(norm(u, v), 2.15));
    let radius = norm((u - value(5)) / value(7), ((v - value(6)) + 0.24 * folds) / value(8));
    let cavity = exp(-0.60 * (radius * radius));
    let wall = (radius - 1.25) / 0.56;
    let compression = exp(-0.5 * (wall * wall))
        * ((0.72 + 0.28 * tanh(2.3 * folds + slab)) * (0.56 + 0.44 * cavityStrength));
    let damping = 0.075 + 0.065 * cavity * cavityStrength - 0.066 * compression + 0.074 * (1 - stars);
    let cx = (u + 0.36) + 0.40 * slab;
    let cy = v - 0.28;
    let core = exp(-0.95 * (cx * cx + cy * cy));
    let rim = ease(-0.42, 0.48, 0.44 * u - 0.34 * v + 0.68 * slab);
    let diffuse = (value(9) * envelope) * exp(0.49 * broad + 0.37 * slab) * (1 - cavityStrength * cavity);
    let definition = ease(0.11, 0.77, diffuse) * (0.54 + 0.46 * stars);
    let bx = u + 0.28;
    let by = v - 0.35;
    let exposure = exp(-0.50 * (bx * bx + by * by));
    writePlane(0,i,envelope); writePlane(1,i,cavity); writePlane(2,i,compression); writePlane(3,i,damping); writePlane(4,i,core);
    writePlane(5,i,rim); writePlane(6,i,diffuse); writePlane(7,i,definition); writePlane(8,i,exposure);
    writePlane(9,i,definition * ease(0.25, 0.81, exposure));
}

fn density(i: u32) {
    let definition = readPlane(1,i);
    let turbulence = readPlane(2,i);
    let details = readPlane(3,i);
    let gas = readPlane(0,i) * exp((0.30 + 0.79 * definition) * turbulence - (0.13 + 0.75 * definition) * details);
    var opacity = 1 - exp(-gas);
    if (gas < 0.001) { opacity = gas * (1 - gas * (0.5 - gas / 6)); }
    let dust = readPlane(4,i) * (0.105 + 0.43 * value(0) * readPlane(5,i))
        * exp((0.25 * readPlane(6,i) + 0.37 * readPlane(7,i)) + (0.39 + 0.30 * readPlane(8,i)) * turbulence + 1.35 * details);
    let at = 10*p.count + 3*i;
    let transmission = (input[at] + input[at+1] + input[at+2]) / 3;
    writePlane(0,i,opacity); writePlane(1,i,dust); writePlane(2,i,opacity * readPlane(9,i) * transmission * exp(-dust));
}

fn composite(i: u32) {
    let core = readPlane(0,i);
    let rim = readPlane(1,i);
    let exposure = readPlane(3,i);
    let relief = readPlane(4,i);
    let opacity = readPlane(5,i);
    let dust = readPlane(6,i);
    let illumination = 0.74 + 0.43 * exposure + 0.64 * relief;
    let tone = ease(0, 1, 0.25 + 0.39 * exposure + 0.20 * readPlane(2,i) + 0.34 * relief);
    let rimWeight = (1 - core) * rim;
    let f = pow(max(0, (1 - core) - rimWeight), 2.3);
    let c = pow(core, 2.3);
    let r = pow(rimWeight, 2.3);
    var color = vec3f(0);
    for (var k = 0u; k < 3; k++) {
        let base = (f * value(1+k) + c * value(4+k)) + r * value(7+k);
        let shift = (f * value(10+k) + c * value(13+k)) + r * value(16+k);
        color[k] = (base + tone * shift) / ((f + c) + r);
    }
    let largest = max(color.r,max(color.g,color.b));
    let weight = clamp(0.48 * value(0) * readPlane(7,i) + 1.65 * max(0,readPlane(8,i)), 0, 0.96);
    let base = vec3f(0.8, 1.0, 1.19);
    for (var k = 0u; k < 3; k++) {
        let j = 3*i + k;
        let transmission = input[12*p.count + j];
        let selective = 0.81 + 0.82 * (1 - color[k] / largest);
        let loss = (1 - weight) * base[k] + weight * selective;
        output[j] = input[9*p.count+j] + transmission * opacity * color[k] * illumination;
        output[3*p.count+j] = transmission * (1 - opacity) * exp(-dust * loss);
    }
    writePlane(6,i,readPlane(15,i)+dust);
}

@compute @workgroup_size(128)
fn layer(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * p.dispatchWidth;
    if (i >= p.count) { return; }
    switch p.stage {
        case 0: { geometry(i); }
        case 1: { density(i); }
        default: { composite(i); }
    }
}
`;

export async function createLayers(device) {
  const module = device.createShaderModule({ code: source, label: "v5 cloud layers" });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  if (errors.length) throw new Error(errors.map(message => `${message.lineNum}: ${message.message}`).join("\n"));
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "layer" },
  });
  const params = device.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  let input;
  let output;
  let staging;
  const make = (prior, size, usage) => {
    if (prior?.size >= size) return prior;
    prior?.destroy();
    return device.createBuffer({ size, usage });
  };
  return async (stage, count, outPlanes, settingsProxy, ...fields) => {
    const outBytes = count * outPlanes * 4;
    const limit = device.limits.maxStorageBufferBindingSize;
    const groupsX = Math.min(Math.ceil(count / 128), device.limits.maxComputeWorkgroupsPerDimension);
    withViews([settingsProxy, ...fields], Array(fields.length + 1).fill("f32"), (settings, ...values) => {
      const inBytes = values.reduce((sum, data) => sum + data.byteLength, 0);
      if (Math.max(inBytes, outBytes) > limit) throw new Error("WebGPU cloud layer exceeds device buffer limits");
      input = make(input, inBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      output = make(output, outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      staging = make(staging, outBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      let cursor = 0;
      for (const data of values) {
        device.queue.writeBuffer(input, cursor, data);
        cursor += data.byteLength;
      }
      const uniform = new ArrayBuffer(96);
      new Uint32Array(uniform, 0, 4).set([count, stage, groupsX * 128, 0]);
      new Float32Array(uniform).set(settings, 4);
      device.queue.writeBuffer(params, 0, uniform);
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [input, output, params].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, Math.ceil(count / (groupsX * 128)));
    pass.end();
    commands.copyBufferToBuffer(output, 0, staging, 0, outBytes);
    device.queue.submit([commands.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, outBytes);
    try {
      return new Float32Array(staging.getMappedRange(0, outBytes).slice(0));
    } finally {
      staging.unmap();
    }
  };
}
