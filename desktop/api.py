"""Serve CUDA-generated nebulae as PNG or JPEG over HTTP."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
import re
from secrets import token_hex
from threading import BoundedSemaphore, Lock
from urllib.parse import parse_qs, urlsplit

from generate import DEFAULT_WALLPAPER_VERSION, wallpaper_backend, wallpaper_image


MAX_SIDE = 8192
MAX_PIXELS = 3840 * 2160
PREVIEW_SIZE = (1920, 1080)
IMAGE_TYPES = {"png": "image/png", "jpeg": "image/jpeg"}


def render_parameters(query):
    params = parse_qs(query, keep_blank_values=True, max_num_fields=4)
    if params.keys() - {"width", "height", "seed", "format"} or any(len(values) != 1 for values in params.values()):
        raise ValueError("Expected width, height, optional seed and format, each at most once")
    image_format = params.get("format", ["png"])[0]
    if image_format not in IMAGE_TYPES:
        raise ValueError("format must be png or jpeg")
    raw_width = params.get("width", [""])[0]
    raw_height = params.get("height", [""])[0]
    if not all(re.fullmatch(r"[0-9]{1,4}", value) for value in (raw_width, raw_height)):
        raise ValueError("width and height must be integers in pixels")
    size = int(raw_width), int(raw_height)
    if not all(2 <= side <= MAX_SIDE for side in size) or size[0] * size[1] > MAX_PIXELS:
        raise ValueError(f"Each side must be 2–{MAX_SIDE} pixels, with at most {MAX_PIXELS} pixels total")
    seed = params.get("seed", [None])[0]
    if seed is None:
        return size, None, image_format
    if not re.fullmatch(r"[0-9a-fA-F]{40}", seed):
        raise ValueError("seed must be a 40-character hexadecimal string")
    return size, seed.lower(), image_format


def render_image(compute, size, image_format, seed=None):
    seed = token_hex(20) if seed is None else seed
    image = wallpaper_image(seed, DEFAULT_WALLPAPER_VERSION, False, compute,
                            uniform_palette=True, size=size)
    buffer = BytesIO()
    if image_format == "jpeg":
        image.convert("RGB").save(buffer, format="JPEG", quality=95, subsampling=0)
    else:
        image.save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue(), seed


class NebulaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = 15
    disable_nagle_algorithm = True

    def send_payload(self, status, content_type, payload, *, seed=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        if seed is not None:
            self.send_header("X-Nebula-Seed", seed)
        try:
            self.end_headers()
            self.wfile.write(payload)
        except ConnectionError:
            self.close_connection = True

    def send_json(self, status, obj):
        self.send_payload(status, "application/json", json.dumps(obj).encode("utf-8"))

    def image_future(self, size, seed, image_format):
        server = self.server
        if size != PREVIEW_SIZE or seed is not None:
            return server.worker.submit(render_image, server.compute, size, image_format, seed)
        with server.preview_lock:
            frame = server.previews[image_format]
            server.previews[image_format] = server.worker.submit(render_image, server.compute, PREVIEW_SIZE, image_format)
        return frame

    def do_GET(self):
        try:
            target = urlsplit(self.path)
        except ValueError:
            self.send_json(400, {"error": "Invalid request URL"})
            return
        if target.path == "/healthz":
            self.send_json(200, {"backend": "cuda", "version": DEFAULT_WALLPAPER_VERSION})
            return
        if target.path == "/":
            self.send_payload(200, "text/html; charset=utf-8", self.server.page)
            return
        if target.path != "/api/nebula":
            self.send_json(404, {"error": "Use GET /api/nebula?width=2560&height=1440"})
            return
        try:
            size, seed, image_format = render_parameters(target.query)
        except ValueError as error:
            self.send_json(400, {"error": str(error)})
            return

        if not self.server.render_slots.acquire(blocking=False):
            self.send_json(503, {"error": "Renderer is busy; retry shortly"})
            return
        try:
            payload, seed = self.image_future(size, seed, image_format).result()
        except Exception as error:
            self.log_error("Rendering failed: %s", error)
            self.send_json(500, {"error": "CUDA rendering failed"})
            return
        finally:
            self.server.render_slots.release()
        self.send_payload(200, IMAGE_TYPES[image_format], payload, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="IPv4 address to listen on (default: loopback)")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    try:
        # The same worker initializes, renders with, and closes the CUDA backend.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="nebula-cuda") as worker:
            resources = ExitStack()
            try:
                backend = wallpaper_backend(DEFAULT_WALLPAPER_VERSION, "cuda")
                compute = worker.submit(resources.enter_context, backend).result()
                previews = {
                    image_format: worker.submit(render_image, compute, PREVIEW_SIZE, image_format)
                    for image_format in IMAGE_TYPES
                }
                for preview in previews.values():
                    preview.result()
                with ThreadingHTTPServer((args.host, args.port), NebulaHandler) as server:
                    server.page = Path(__file__).with_name("lan.html").read_bytes()
                    server.worker = worker
                    server.compute = compute
                    server.previews = previews
                    server.preview_lock = Lock()
                    server.render_slots = BoundedSemaphore(4)
                    print(f"CUDA API listening on http://{args.host}:{server.server_port}", flush=True)
                    server.serve_forever()
            finally:
                worker.submit(resources.close).result()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
