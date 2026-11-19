import { fullscreen, clouds, display, stars, starLight } from "./shaders.js";

const querySeed = new URLSearchParams(location.search).get("seed");
const seed = /^[0-9a-f]{1,8}$/i.test(querySeed ?? "")
  ? Number.parseInt(querySeed, 16)
  : crypto.getRandomValues(new Uint32Array(1))[0];

function random(initial) {
  let state = initial;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let x = Math.imul(state ^ (state >>> 15), state | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

const rng = random(seed);
const scene = {
  seed: rng(),
  scale: 0.8 + 0.4 * rng(),
  starFactor: 0.7 + 0.3 * rng(),
  angle: (rng() - 0.5) * 0.76,
  hue: rng(),
  layers: Array.from({ length: 4 }, (_, i) => [
    (i - 1.5) * 0.235 + (rng() - 0.5) * 0.12,
    (rng() - 0.5) * 0.18,
    0.15 + rng() * 0.13,
    0.06 + rng() * 0.09,
  ]),
};
scene.layers.push([0, 0, 0.48, 0.15]);

function starVertices(width, height) {
  const next = random(seed ^ 0x89c30b2f);
  const cos = Math.cos(scene.angle), sin = Math.sin(scene.angle);
  const max = Math.max(width, height) * scene.scale;
  const count = Math.round(19000 * scene.starFactor * Math.min(1, width * height / (2560 * 1440)));
  const data = new Float32Array(count * 5);
  for (let i = 0; i < count;) {
    const x = 2 * next() - 1, y = 2 * next() - 1;
    const sx = x * width / (2 * max), sy = y * height / (2 * max);
    const along = sx * cos - sy * sin;
    const orbit = sx * sin + sy * cos - 0.047 * Math.sin(6 * along + scene.seed * 5);
    const density = 0.20 + 0.77 * Math.exp(-Math.abs(orbit) / 0.083);
    if (next() > density) continue;
    const distance = Math.cbrt(0.001 + next());
    // Many faint stars and a sparse bright tail, with a distance-dependent dust column.
    const flux = Math.min(5, 0.045 * Math.pow(Math.max(next(), 0.0001), -0.68) / (0.25 + distance ** 2));
    const behind = 1 - Math.exp(-3.2 * distance ** 2);
    data.set([x, y, flux, behind, next() * 2 * Math.PI], 5 * i++);
  }
  return data;
}

function program(gl, vertex, fragment) {
  const build = (kind, source) => {
    const shader = gl.createShader(kind);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      throw new Error(gl.getShaderInfoLog(shader));
    }
    return shader;
  };
  const vert = build(gl.VERTEX_SHADER, vertex);
  const frag = build(gl.FRAGMENT_SHADER, fragment);
  const result = gl.createProgram();
  gl.attachShader(result, vert);
  gl.attachShader(result, frag);
  gl.linkProgram(result);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(result, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(result));
  return result;
}

function gpuRenderer(canvas, gl) {
  const gas = program(gl, fullscreen, clouds);
  const show = program(gl, fullscreen, display);
  const starlight = program(gl, stars, starLight);
  const screen = gl.createVertexArray(), points = gl.createVertexArray();
  const sceneTexture = gl.createTexture(), framebuffer = gl.createFramebuffer();
  const buffer = gl.createBuffer();
  gl.bindVertexArray(points);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 20, 16);
  gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.useProgram(gas);
  gl.uniform1f(gl.getUniformLocation(gas, "seed"), scene.seed);
  gl.uniform1f(gl.getUniformLocation(gas, "sceneScale"), scene.scale);
  gl.uniform1f(gl.getUniformLocation(gas, "angle"), scene.angle);
  gl.uniform1f(gl.getUniformLocation(gas, "hue"), scene.hue);
  gl.uniform4fv(gl.getUniformLocation(gas, "layers[0]"), scene.layers.flat());
  gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

  return (width, height) => {
    gl.viewport(0, 0, width, height);
    gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTexture, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error("WebGL framebuffer is incomplete");
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(screen);
    gl.useProgram(gas);
    gl.uniform2f(gl.getUniformLocation(gas, "resolution"), width, height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, sceneTexture);
    gl.useProgram(show);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const vertices = starVertices(width, height);
    gl.bindVertexArray(points);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.useProgram(starlight);
    gl.uniform1f(gl.getUniformLocation(starlight, "sizeScale"),
      scene.scale * Math.max(0.65, Math.min(1.65, Math.max(width, height) / 1150)));
    gl.enable(gl.BLEND);
    gl.drawArrays(gl.POINTS, 0, vertices.length / 5);
    gl.disable(gl.BLEND);
    canvas.dataset.renderer = "webgl2";
  };
}

function cpuRenderer(canvas) {
  const context = canvas.getContext("2d", { alpha: false });
  return (width, height) => {
    context.fillStyle = "#050711";
    context.fillRect(0, 0, width, height);
    context.save();
    context.translate(width / 2, height / 2);
    context.scale(1, -1);
    context.rotate(-scene.angle);
    const space = Math.max(width, height) * scene.scale;
    const next = random(seed ^ 0x271556bc);
    for (const [x, y, rx, ry] of scene.layers) {
      for (let i = 0; i < 26; i++) {
        const offset = (next() - 0.5) * 1.8;
        context.save();
        context.translate((x + offset * rx) * space, (y + (next() - 0.5) * ry) * space);
        context.rotate((next() - 0.5) * 0.6);
        context.scale(rx * space * (0.35 + next() * 0.55), ry * space * (0.35 + next() * 0.8));
        const gradient = context.createRadialGradient(0, 0, 0, 0, 0, 1);
        const h = scene.hue * 360 + 30 * offset;
        gradient.addColorStop(0, `hsl(${h} 58% 45% / 0.125)`);
        gradient.addColorStop(1, `hsl(${h} 60% 26% / 0)`);
        context.fillStyle = gradient;
        context.fillRect(-1, -1, 2, 2);
        context.restore();
      }
    }
    context.restore();
    const points = starVertices(width, height);
    for (let i = 0; i < points.length; i += 5) {
      const x = (points[i] + 1) * width / 2;
      const y = (1 - points[i + 1]) * height / 2;
      const radius = scene.scale * (points[i + 2] > 1.4 ? 1.4 : 0.65);
      context.fillStyle = `rgba(227,234,255,${Math.min(1, 0.91 * points[i + 2])})`;
      context.beginPath();
      context.arc(x, y, radius, 0, 2 * Math.PI);
      context.fill();
    }
    canvas.dataset.renderer = "canvas2d";
  };
}

let canvas = document.querySelector("canvas");
let render;
function useFallback() {
  const replacement = canvas.cloneNode(false);
  canvas.replaceWith(replacement);
  canvas = replacement;
  render = cpuRenderer(canvas);
}

try {
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, powerPreference: "high-performance" });
  render = gl ? gpuRenderer(canvas, gl) : cpuRenderer(canvas);
  canvas.addEventListener("webglcontextlost", () => {
    useFallback();
    redraw();
  });
} catch (error) {
  console.warn("WebGL unavailable:", error);
  useFallback();
}

function redraw() {
  const { clientWidth: cssWidth, clientHeight: cssHeight } = canvas;
  if (!cssWidth || !cssHeight) return;
  const ratio = Math.min(devicePixelRatio, Math.sqrt(2400000 / (cssWidth * cssHeight)));
  const width = Math.max(1, Math.round(cssWidth * ratio));
  const height = Math.max(1, Math.round(cssHeight * ratio));
  canvas.width = width;
  canvas.height = height;
  try {
    render(width, height);
  } catch (error) {
    if (canvas.dataset.renderer === "canvas2d") throw error;
    console.warn("WebGL rendering failed:", error);
    useFallback();
    canvas.width = width;
    canvas.height = height;
    render(width, height);
  }
}

let frame;
function scheduleRedraw() {
  cancelAnimationFrame(frame);
  frame = requestAnimationFrame(redraw);
}
window.addEventListener("resize", scheduleRedraw);
scheduleRedraw();
