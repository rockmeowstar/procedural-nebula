const canvas = document.querySelector("canvas");
const status = document.querySelector("#status");
const context = canvas.getContext("2d", { alpha: false });
const params = new URLSearchParams(location.search);
const candidate = params.get("seed");
const forceCPU = params.get("backend") === "cpu";
const seed = /^[0-9a-f]{40}$/i.test(candidate ?? "")
  ? candidate.toLowerCase()
  : Array.from(crypto.getRandomValues(new Uint8Array(20)), byte =>
      byte.toString(16).padStart(2, "0")).join("");

canvas.dataset.seed = seed;
canvas.dataset.state = "loading";
const worker = new Worker(new URL("./render-worker.js?v=webgpu-1", import.meta.url), { type: "module" });
let pending = false;
let hasFrame = false;
let sizeSent;
let resizeTimer;

function viewportSize() {
  return [Math.max(2, canvas.clientWidth), Math.max(2, canvas.clientHeight)];
}

function render() {
  if (pending || canvas.dataset.state === "error") return;
  const [width, height] = viewportSize();
  if (hasFrame && canvas.width === width && canvas.height === height) {
    canvas.dataset.state = "ready";
    return;
  }
  if (sizeSent?.[0] === width && sizeSent[1] === height) return;
  sizeSent = [width, height];
  pending = true;
  worker.postMessage({ seed, width, height, forceCPU });
}

function showError(message) {
  console.error(message);
  canvas.dataset.state = "error";
  status.hidden = false;
  status.textContent = "Не удалось сгенерировать туманность. Перезагрузите страницу.";
  worker.terminate();
}

worker.addEventListener("error", event => showError(event.message));
worker.addEventListener("message", ({ data }) => {
  if (data.type === "error") {
    showError(data.message);
  } else if (data.type === "rendering") {
    canvas.dataset.state = "rendering";
    status.textContent = "Генерируется туманность…";
  } else if (data.type === "frame") {
    pending = false;
    const [width, height] = viewportSize();
    if (hasFrame && (data.width !== width || data.height !== height)) {
      render();
      return;
    }
    canvas.width = data.width;
    canvas.height = data.height;
    const pixels = new ImageData(new Uint8ClampedArray(data.pixels), data.width, data.height);
    context.putImageData(pixels, 0, 0);
    hasFrame = true;
    canvas.dataset.state = "ready";
    canvas.dataset.backend = data.backend;
    canvas.dataset.renderMs = data.elapsed.toFixed(0);
    status.hidden = true;
    render();
  }
});

window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(render, 250);
});
render();
