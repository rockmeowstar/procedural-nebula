import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs";
import { createNoiseDevice, projectStars } from "./webgpu-noise.js";
import { createRaster } from "./webgpu-raster.js";
import { createLayers } from "./webgpu-layers.js";

const engineFiles = ["generate.py", "milky_way.py", "wallpaper_compute.py"];

async function loadEngine(forceCPU) {
  const sources = Promise.all(engineFiles.map(async filename => {
    const response = await fetch(new URL(`./desktop/${filename}`, import.meta.url), { cache: "no-cache" });
    if (!response.ok) throw new Error(`Could not load ${filename}: ${response.status}`);
    return response.text();
  }));
  const runtime = loadPyodide().then(async pyodide => {
    await pyodide.loadPackage(["numpy", "pillow"]);
    return pyodide;
  });
  const gpu = !forceCPU && typeof WebAssembly.Suspending === "function"
    ? createNoiseDevice().then(async accelerator => {
      if (accelerator) {
        [accelerator.raster, accelerator.layers] = await Promise.all([
          createRaster(accelerator.device), createLayers(accelerator.device),
        ]);
      }
      return accelerator;
    }).catch(error => {
      console.warn("WebGPU initialization failed:", error);
      return null;
    }) : null;
  const [pyodide, files, accelerator] = await Promise.all([runtime, sources, gpu]);
  engineFiles.forEach((filename, index) => pyodide.FS.writeFile(filename, files[index]));
  pyodide.globals.set("gpu_noise", accelerator?.noise ?? null);
  pyodide.globals.set("gpu_raster", accelerator?.raster ?? null);
  pyodide.globals.set("gpu_layers", accelerator?.layers ?? null);
  pyodide.globals.set("gpu_buffer_limit", accelerator?.device.limits.maxStorageBufferBindingSize ?? 0);
  pyodide.globals.set("project_stars", projectStars);
  pyodide.runPython(`
import numpy as np
from generate import DEFAULT_WALLPAPER_VERSION, wallpaper_image
from pyodide.ffi import run_sync
from wallpaper_compute import star_tile_graph

class WebGPU:
    name = "webgpu"
    optimized = True
    fast = False
    empty = np.empty(0, dtype=np.float32)

    @staticmethod
    def array(value):
        return np.ascontiguousarray(value, dtype=np.float32)

    def _noise(self, x, y, gradients, damping, octaves, decay, outer_scale, step):
        axes = [np.ascontiguousarray(field, dtype=np.float32) for field in (x, y, gradients, damping)]
        result = run_sync(gpu_noise(*axes, octaves, decay, outer_scale, step))
        return np.frombuffer(result.to_py(), dtype=np.float32).reshape(x.shape).astype(x.dtype)

    def fractal(self, x, y, gradients, octaves=5, decay=0.53):
        return self._noise(x, y, gradients, self.empty, octaves, decay, 0.0, 0.0)

    def turbulence(self, x, y, gradients, outer_scale, damping, step):
        return self._noise(x, y, gradients, damping, 7, 0.0, outer_scale, step)

    def star_projection(self, x, y, gradients, width, height, geometry):
        result = project_stars(x.astype(np.int32), y.astype(np.int32), gradients, width, height, *geometry)
        return np.frombuffer(result.to_py(), dtype=np.float64).reshape(2, x.size)

    def splat(self, shape, patches, mode, field_a=None, field_b=None):
        output_shape = (*shape, 3) if mode == 2 else shape
        if len(patches) == 0:
            return np.zeros(output_shape, dtype=np.float32)
        height, width = shape
        tile = 16 if mode == 2 else 32
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        offsets, indices = star_tile_graph(patches, width, height, tile)
        a, b = (self.empty if field is None else np.ascontiguousarray(field, dtype=np.float32)
                for field in (field_a, field_b))
        result = run_sync(gpu_raster(patches, offsets.astype(np.uint32), indices.astype(np.uint32),
                                     a, b, width, height, tile, mode))
        return np.frombuffer(result.to_py(), dtype=np.float32).reshape(output_shape)

    def _layer(self, stage, shape, planes, settings, fields):
        count = shape[0] * shape[1]
        args = [self.array(field).reshape(count, -1) for field in fields]
        settings = self.array(settings)
        stride = max(planes, sum(field.shape[1] for field in args))
        tile = gpu_buffer_limit // (4 * stride)

        def run(start, end):
            result = run_sync(gpu_layers(stage, end - start, planes, settings,
                                         *(field[start:end] for field in args)))
            return np.frombuffer(result.to_py(), dtype=np.float32).reshape(planes, -1)

        if count <= tile:
            return run(0, count).reshape(planes, *shape)

        output = np.empty((planes, count), dtype=np.float32)
        for start in range(0, count, tile):
            end = min(count, start + tile)
            chunk = run(start, end)
            if stage == 2:
                output[:3].reshape(-1, 3)[start:end] = chunk[:3].reshape(-1, 3)
                output[3:6].reshape(-1, 3)[start:end] = chunk[3:6].reshape(-1, 3)
                output[6, start:end] = chunk[6]
            else:
                output[:, start:end] = chunk
        return output.reshape(planes, *shape)

    def layer_geometry(self, wx, wy, slab, folds, broad, stars, *settings):
        return self._layer(0, wx.shape, 10, settings, (wx, wy, slab, folds, broad, stars))

    def layer_density(self, diffuse, definition, turbulent, details, envelope, cavity, dust, slab,
                      compression, cavity_strength, broad_light, transmission):
        return self._layer(1, diffuse.shape, 3, (cavity_strength,),
                           (diffuse, definition, turbulent, details, envelope, cavity, dust, slab,
                            compression, broad_light, transmission))

    def layer_composite(self, core, rim, definition, exposure, relief, opacity, dust, cavity, details,
                        cavity_strength, bands, light, transmission, dust_column):
        frame = self._layer(2, core.shape, 7, (cavity_strength, *bands.ravel()),
                            (core, rim, definition, exposure, relief, opacity, dust, cavity, details,
                             light, transmission, dust_column))
        light[...] = frame[:3].reshape(light.shape)
        transmission[...] = frame[3:6].reshape(transmission.shape)
        dust_column[...] = frame[6]

def render_frame(seed, width, height, use_gpu):
    compute = WebGPU() if use_gpu else None
    image = wallpaper_image(seed, DEFAULT_WALLPAPER_VERSION, False, compute,
                            uniform_palette=True, size=(width, height))
    return np.asarray(image.convert("RGBA"))
`);
  return { renderFrame: pyodide.globals.get("render_frame"), accelerator };
}

let engine;
self.onmessage = async ({ data: { seed, width, height, forceCPU } }) => {
  try {
    engine ??= loadEngine(forceCPU);
    const { renderFrame, accelerator } = await engine;
    self.postMessage({ type: "rendering" });
    const started = performance.now();
    let backend = accelerator && !forceCPU ? "webgpu" : "cpu";
    let rgba;
    if (backend === "webgpu") {
      try {
        rgba = await renderFrame.callPromising(seed, width, height, true);
      } catch (error) {
        console.warn("WebGPU rendering failed:", error);
        backend = "cpu";
      }
    }
    rgba ??= renderFrame(seed, width, height, false);
    try {
      const view = rgba.getBuffer("u8");
      try {
        const pixels = view.data.slice().buffer;
        self.postMessage({ type: "frame", width, height, pixels, backend, elapsed: performance.now() - started }, [pixels]);
      } finally {
        view.release();
      }
    } finally {
      rgba.destroy();
    }
  } catch (error) {
    self.postMessage({ type: "error", message: String(error) });
  }
};
