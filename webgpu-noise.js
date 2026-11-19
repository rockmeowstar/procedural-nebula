// The octave recurrences, weights and gradient lookup match desktop/wallpaper_compute.py.
const source = `
struct Parameters {
    count: u32, octaves: u32, cascade: u32, dispatchWidth: u32,
    outerScale: f32, step: f32, padding: vec2f,
    weights: array<vec4f, 2>, frequencies: array<vec4f, 2>,
}
@group(0) @binding(0) var<storage, read> xs: array<f32>;
@group(0) @binding(1) var<storage, read> ys: array<f32>;
@group(0) @binding(2) var<storage, read> gradients: array<vec2f>;
@group(0) @binding(3) var<storage, read> damping: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<uniform> p: Parameters;

fn corner(x: i32, y: i32, dx: f32, dy: f32) -> f32 {
    let g = gradients[u32((y & 127) * 128 + (x & 127))];
    return g.x * dx + g.y * dy;
}

fn noise(x: f32, y: f32) -> f32 {
    let ix = i32(floor(x));
    let iy = i32(floor(y));
    let dx = x - f32(ix);
    let dy = y - f32(iy);
    let u = (dx * dx * dx) * (dx * (dx * 6 - 15) + 10);
    let v = (dy * dy * dy) * (dy * (dy * 6 - 15) + 10);
    let top = (1 - u) * corner(ix, iy, dx, dy)
        + u * corner(ix + 1, iy, dx - 1, dy);
    let bottom = (1 - u) * corner(ix, iy + 1, dx, dy - 1)
        + u * corner(ix + 1, iy + 1, dx - 1, dy - 1);
    return (1 - v) * top + v * bottom;
}

@compute @workgroup_size(128)
fn render(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * p.dispatchWidth;
    if (i >= p.count) { return; }
    var x = xs[i];
    var y = ys[i];
    var total = 0.0;
    if (p.cascade != 0) {
        x /= p.outerScale;
        y /= p.outerScale;
    }
    for (var n = 0u; n < p.octaves; n++) {
        let weight = p.weights[n / 4][n % 4];
        let sample = noise(x, y) * weight;
        if (p.cascade != 0) {
            let decay = p.frequencies[n / 4][n % 4] * damping[i];
            total += sample * exp(-(decay * decay));
            let nextX = p.step * (0.87 * x - 0.49 * y) + 17;
            y = p.step * (0.49 * x + 0.87 * y) + 29;
            x = nextX;
        } else {
            total += sample;
            let nextX = 1.76 * x - 1.02 * y + 19;
            y = 1.02 * x + 1.76 * y + 31;
            x = nextX;
        }
    }
    output[i] = total;
}
`;

export function withViews(proxies, types, work) {
  const views = [];
  try {
    proxies.forEach((proxy, i) => views.push(proxy.getBuffer(types[i])));
    return work(...views.map(view => view.data));
  } finally {
    views.forEach(view => view.release());
  }
}

export async function createNoiseDevice() {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return null;
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024),
    maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024),
  } });
  const module = device.createShaderModule({ code: source });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "render" },
  });
  const make = (size, usage) => device.createBuffer({ size, usage });
  const input = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const gradients = make(128 * 128 * 8, input);
  const params = make(96, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  // Keep the numpy-to-WebGPU uploads bounded and share staging buffers across octaves.
  let size = 0;
  let buffers;
  let bindGroup;

  async function noise(x, y, g, damping, octaves, decay, outerScale, step) {
    let count;
    let groupsX;
    withViews([x, y, g, damping], ["f32", "f32", "f32", "f32"], (xs, ys, grid, localDamping) => {
      count = xs.length;
      const bytes = count * 4;
      if (bytes > device.limits.maxStorageBufferBindingSize) throw new Error("WebGPU noise buffer exceeds device limits");
      if (size !== bytes) {
        buffers?.forEach(buffer => buffer.destroy());
        size = bytes;
        buffers = [
          make(size, input), make(size, input), make(size, input),
          make(size, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
          make(size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        ];
        const binding = [buffers[0], buffers[1], gradients, buffers[2], buffers[3], params];
        bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: binding.map((buffer, index) => ({ binding: index, resource: { buffer } })),
        });
      }
      const cascade = step !== 0;
      groupsX = Math.min(Math.ceil(count / 128), device.limits.maxComputeWorkgroupsPerDimension);
      const settings = new ArrayBuffer(96);
      new Uint32Array(settings, 0, 4).set([count, octaves, Number(cascade), groupsX * 128]);
      const floats = new Float32Array(settings);
      floats.set([outerScale, step], 4);
      let frequency = 1;
      for (let i = 0; i < octaves; i++) {
        floats[8 + i] = cascade ? frequency ** (-1 / 3) : decay ** i;
        floats[16 + i] = frequency;
        frequency *= step;
      }
      device.queue.writeBuffer(params, 0, settings);
      device.queue.writeBuffer(gradients, 0, grid);
      device.queue.writeBuffer(buffers[0], 0, xs);
      device.queue.writeBuffer(buffers[1], 0, ys);
      if (cascade) device.queue.writeBuffer(buffers[2], 0, localDamping);
    });
    const commands = device.createCommandEncoder();
    const pass = commands.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, Math.ceil(count / (groupsX * 128)));
    pass.end();
    commands.copyBufferToBuffer(buffers[3], 0, buffers[4], 0, size);
    device.queue.submit([commands.finish()]);
    await buffers[4].mapAsync(GPUMapMode.READ);
    try {
      return new Float32Array(buffers[4].getMappedRange().slice(0));
    } finally {
      buffers[4].unmap();
    }
  }
  return { noise, device };
}

// The discrete star catalogue selection uses double precision, as the CUDA v5 sparse projection does.
export function projectStars(px, py, grid, width, height, cosine, sine, warpX, warpY, phase, scale) {
  return withViews([px, py, grid], ["i32", "i32", "f64"], (xs, ys, gradients) => {
    function corner(x, y, dx, dy) {
      const i = 2 * ((y & 127) * 128 + (x & 127));
      return gradients[i] * dx + gradients[i + 1] * dy;
    }
    function noise(x, y) {
      const ix = Math.floor(x), iy = Math.floor(y);
      const dx = x - ix, dy = y - iy;
      const u = (dx * dx * dx) * (dx * (dx * 6 - 15) + 10);
      const v = (dy * dy * dy) * (dy * (dy * 6 - 15) + 10);
      const top = (1 - u) * corner(ix, iy, dx, dy) + u * corner(ix + 1, iy, dx - 1, dy);
      const bottom = (1 - u) * corner(ix, iy + 1, dx, dy - 1) + u * corner(ix + 1, iy + 1, dx - 1, dy - 1);
      return (1 - v) * top + v * bottom;
    }
    function fractal(x, y, weights) {
      let total = 0;
      for (let i = 0; i < weights.length; i++) {
        total += noise(x, y) * weights[i];
        const nextX = 1.76 * x - 1.02 * y + 19;
        y = 1.02 * x + 1.76 * y + 31;
        x = nextX;
      }
      return total;
    }
    const warp = Array.from({ length: 5 }, (_, i) => 0.53 ** i);
    const mist = Array.from({ length: 4 }, (_, i) => 0.41 ** i);
    const count = xs.length;
    const projected = new Float64Array(2 * count);
    // Multiple candidates can sample the same pixel, especially on a phone.
    const cache = width * height <= 2 * count ? new Float64Array(2 * width * height) : null;
    cache?.fill(NaN);
    for (let i = 0; i < count; i++) {
      const pixel = 2 * (ys[i] * width + xs[i]);
      if (cache && !Number.isNaN(cache[pixel])) {
        projected[i] = cache[pixel];
        projected[count + i] = cache[pixel + 1];
        continue;
      }
      const x = ((xs[i] - width / 2) / width) * scale;
      const y = ((ys[i] - height / 2) / width) * scale;
      let u = x * cosine - y * sine, v = x * sine + y * cosine;
      const dx = fractal(2.4 * u + warpX, 2.4 * v + warpY, warp);
      const dy = fractal(2.4 * u + warpY, 2.4 * v + warpX + 17, warp);
      u += 0.33 * dx;
      v += 0.14 * dy;
      projected[i] = u;
      projected[count + i] = v - 0.037 * Math.sin(6 * u + phase)
        + 0.028 * fractal(4 * u + 52, 6 * v + 10, mist);
      if (cache) {
        cache[pixel] = projected[i];
        cache[pixel + 1] = projected[count + i];
      }
    }
    return projected;
  });
}
