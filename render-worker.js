import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.29.4/full/pyodide.mjs";

const engineFiles = ["generate.py", "milky_way.py", "wallpaper_compute.py"];

async function loadEngine() {
  const sources = Promise.all(engineFiles.map(async filename => {
    const response = await fetch(new URL(`./desktop/${filename}`, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${filename}: ${response.status}`);
    return response.text();
  }));
  const runtime = loadPyodide().then(async pyodide => {
    await pyodide.loadPackage(["numpy", "pillow"]);
    return pyodide;
  });
  const [pyodide, files] = await Promise.all([runtime, sources]);
  engineFiles.forEach((filename, index) => pyodide.FS.writeFile(filename, files[index]));
  pyodide.runPython(`
import numpy as np
from generate import DEFAULT_WALLPAPER_VERSION, wallpaper_image

def render_frame(seed, width, height):
    image = wallpaper_image(seed, DEFAULT_WALLPAPER_VERSION, False, None,
                            uniform_palette=True, size=(width, height))
    return np.asarray(image.convert("RGBA"))
`);
  return pyodide.globals.get("render_frame");
}

let engine;
self.onmessage = async ({ data: { seed, width, height } }) => {
  try {
    engine ??= loadEngine();
    const renderFrame = await engine;
    self.postMessage({ type: "rendering" });
    const started = performance.now();
    const rgba = renderFrame(seed, width, height);
    try {
      const view = rgba.getBuffer("u8");
      try {
        const pixels = view.data.slice().buffer;
        self.postMessage({ type: "frame", width, height, pixels, elapsed: performance.now() - started }, [pixels]);
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
