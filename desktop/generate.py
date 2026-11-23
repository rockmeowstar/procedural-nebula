"""Render individual static PNGs and looping WebP banners for GitHub releases."""

import argparse
from colorsys import hsv_to_rgb, rgb_to_hsv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from secrets import token_hex
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from milky_way import REFERENCE_STAR_COUNT, sample_visible_stars, star_density_map, star_field


DEFAULT_REPOS = (
    "rockmeowstar/KSP_koe=KerbalOverEngineering",
    "rockmeowstar/KSP_ShineFix=ShineFix",
    "rockmeowstar/KSP_Stockalike_Hairstyles=Stockalike Hairstyles",
)
WIDTH, HEIGHT = 624, 80
WALLPAPER_SIZE = (2560, 1440)
DEFAULT_WALLPAPER_VERSION = 5
FORUM_SIZE = (280, 42)
FOG_PAD_X, FOG_PAD_Y = 176, 44
FRAME_COUNT, FRAME_MS = 96, 65
DEFAULT_DRIFT, MAX_DRIFT = 42, 48
PALETTES = (
    ("060b1a", "112a47", "176c74", "81cfbc", "e9e2b7"),
    ("10081b", "342249", "8f405a", "e29d76", "ffe4ba"),
    ("070d21", "1c2e58", "494cac", "a7a5f3", "f5d0be"),
    ("080f18", "1d353a", "4b7368", "bfb571", "f3e4b7"),
    ("0d0919", "332746", "834360", "ce82ad", "f6e4de"),
    ("07101c", "17314a", "386c9a", "8ac1e1", "ebe6cc"),
)
MOD_PALETTES = {
    "rockmeowstar/ksp_koe": ("090b25", "202c58", "474a9c", "926db3", "cbc9f1"),
    "rockmeowstar/ksp_shinefix": ("071526", "142d50", "28699b", "4ba6ad", "b7e0d8"),
    "rockmeowstar/ksp_stockalike_hairstyles": ("081715", "173a32", "427451", "89a76a", "d5ca92"),
}
FONT_CANDIDATES = (
    ("SpaceGrotesk-Bold.ttf", "bahnschrift.ttf", "DejaVuSans-Bold.ttf"),
    ("IBMPlexMono-Regular.ttf", "consola.ttf", "DejaVuSansMono.ttf"),
)


@dataclass(frozen=True)
class Release:
    repo: str
    title: str
    stars: int
    tag: str
    sha: str
    published: datetime

    @property
    def filename(self):
        return self.repo.lower().replace("/", "--") + ".png"


def github(path):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "static-release-banner",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(f"https://api.github.com/repos/{path}", headers=headers),
                     timeout=25) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as error:
        raise RuntimeError(f"GitHub API {path}: {error}") from error


def fetch_release(spec, *, tag=None):
    repo, _, title = spec.partition("=")
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"Expected owner/repo[=Title], got {spec!r}")
    repo_data = github(repo)
    endpoint = "latest" if tag is None else f"tags/{quote(tag, safe='')}"
    release = github(f"{repo}/releases/{endpoint}")
    if release.get("draft") or not release.get("published_at"):
        raise ValueError(f"{repo} {tag} is not a published release")
    tag = release["tag_name"]
    # Resolve the exact Git ref. The release's target_commitish may be a moving branch.
    target = github(f"{repo}/git/ref/tags/{quote(tag, safe='/')}")["object"]
    while target["type"] == "tag":
        target = github(f"{repo}/git/tags/{target['sha']}")["object"]
    sha = target["sha"]
    if target["type"] != "commit" or not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError(f"{repo} {tag} does not resolve to a commit SHA")
    return Release(repo, title or repo_data["name"], repo_data["stargazers_count"],
                   tag, sha, datetime.fromisoformat(release["published_at"].replace("Z", "+00:00")))


def smoothstep(low, high, value, xp=np):
    t = xp.clip((value - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def noise(x, y, gradients):
    ix, iy = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    dx, dy = x - ix, y - iy
    u = dx**3 * (dx * (dx * 6 - 15) + 10)
    v = dy**3 * (dy * (dy * 6 - 15) + 10)

    def corner(cx, cy):
        gx, gy = gradients[(iy + cy) % 128, (ix + cx) % 128].transpose(2, 0, 1)
        return gx * (dx - cx) + gy * (dy - cy)

    top = (1 - u) * corner(0, 0) + u * corner(1, 0)
    bottom = (1 - u) * corner(0, 1) + u * corner(1, 1)
    return (1 - v) * top + v * bottom


def fractal(x, y, gradients, octaves=5, decay=0.53):
    total = np.zeros_like(x)
    for octave in range(octaves):
        total += noise(x, y, gradients) * decay**octave
        x, y = 1.76 * x - 1.02 * y + 19, 1.02 * x + 1.76 * y + 31
    return total


def palette_rgb(palette):
    return np.array([tuple(bytes.fromhex(color)) for color in palette], dtype=np.uint8)


def wallpaper_colors(key):
    rng = np.random.default_rng(int.from_bytes(key[8:16], "big"))
    # Analogous narrowband families: blue-violet-magenta, cyan-blue-violet,
    # and burgundy-coral-amber. The continuum remains close to neutral.
    continuum = np.array(((5, 7, 11), (15, 19, 24), (31, 34, 41), (57, 61, 69), (92, 96, 105)))
    centers = (0.82, 0.83, 0.59, 0.025)
    hue = centers[rng.integers(len(centers))] + rng.uniform(-0.025, 0.025)
    span = rng.uniform(0.16, 0.25)
    hues = (hue + span * np.array((-0.54, 0.02, 0.54)) + rng.uniform(-0.012, 0.012, 3)) % 1
    bands = [hsv_to_rgb(hue, rng.uniform(0.40, 0.68), rng.uniform(0.75, 0.94)) for hue in hues]
    rng.shuffle(bands)
    return np.vstack((continuum, 255 * np.array(bands)))


def band_light_and_shadow(bands):
    shadows, highlights = [], []
    for band in bands:
        hue, saturation, value = rgb_to_hsv(*(band / 255))
        # Move cyan toward blue and yellow toward orange in shadow.
        shift = np.cos(2 * np.pi * (hue - 0.59))
        shadows.append(hsv_to_rgb((hue + 0.037 * shift) % 1, min(1, 1.12 * saturation), 0.94 * value))
        highlights.append(hsv_to_rgb((hue - 0.030 * shift) % 1, 0.86 * saturation, min(1, 1.05 * value)))
    return 255 * np.array(shadows), 255 * np.array(highlights)


IRREGULARITY_SIZES = (
    1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
)


def prime_placements(probability, compression, outer_scale, occupied, rng, *, minimum=1, count=1450,
                     fields=(), compute=None, sampler=None, scene_scale=1.0):
    height, width = compression.shape
    pixel_scale = width / WALLPAPER_SIZE[0] * scene_scale
    largest = outer_scale * WALLPAPER_SIZE[0]
    sizes = [size for size in IRREGULARITY_SIZES if minimum <= size <= largest]
    x = rng.integers(width, size=count)
    y = rng.integers(height, size=x.size)
    samples = (sampler(x, y) if sampler is not None else
               [field[y, x] for field in (probability, compression, *fields)])
    if compute is not None and compute.name == "cuda":
        packed = compute.cp.stack(samples).get()
        samples = [row.astype(field.dtype, copy=False) for row, field in zip(packed, samples, strict=True)]
    chance, pressure, *extra = samples
    kept = np.flatnonzero(rng.random(x.size) < chance)
    indexed = not isinstance(occupied, dict)
    for i in kept:
        cx, cy = x[i], y[i]
        if indexed:
            cx, cy = int(cx), int(cy)
            eligible = [size for size in sizes if occupied.available(size, cx, cy)]
        else:
            eligible = [
                size for size in sizes
                if all((cx - px)**2 + (cy - py)**2 >= (3 * max(0.72, size * pixel_scale))**2
                       for px, py in occupied.get(size, ()))
            ]
        if not eligible:
            continue
        compressed = pressure[i]
        typical_size = 3 + largest * (0.28 - 0.20 * compressed)
        candidates = np.array(eligible)
        weights = np.exp(-candidates / typical_size) / candidates**0.52
        size = int(rng.choice(candidates, p=weights / weights.sum()))
        if indexed:
            occupied.add(size, cx, cy)
        else:
            occupied.setdefault(size, []).append((cx, cy))
        yield cx, cy, max(0.72, size * pixel_scale), *(field[i] for field in extra)


def feature_patch(shape, cx, cy, radius, extent=1.4):
    height, width = shape
    reach = int(np.ceil(extent * radius))
    left, right = max(0, cx - reach), min(width, cx + reach + 1)
    top, bottom = max(0, cy - reach), min(height, cy + reach + 1)
    dy, dx = np.ogrid[top:bottom, left:right]
    return np.s_[top:bottom, left:right], (dx - cx) / radius, (dy - cy) / radius


def feathered_sides(u, v, radius, sharp_angle):
    angle = np.arctan2(v, u)
    preference = np.maximum(0, np.cos(angle - sharp_angle))**6
    feather = 0.57 - 0.42 * preference
    return smoothstep(0, feather, 1 - radius)


def dust_fold(u, v, outline, sharp_side, prominence):
    distance = np.hypot(u, v)
    diffuse = np.maximum(0, 1 - distance**2 / 1.13**2)**3
    shaped = feathered_sides(u, v, distance / outline, sharp_side)
    shaped *= 0.49 + 0.51 * np.exp(-1.65 * distance**2)
    return (0.13 + 0.66 * prominence) * ((1 - prominence) * diffuse + prominence * shaped)


def irregularity_field(envelope, compression, star_density, outer_scale, occupied, rng, *, hue_gradient=None,
                       fine=False, compute=None, bands=None, scene_scale=1.0):
    """Splat irregular features without repeating a scale in its neighborhood."""
    field = np.zeros_like(envelope) if compute is None else None
    patches = []
    frame_shape = envelope.shape
    if bands is None:
        probability = envelope**1.7 * (0.18 + 0.82 * compression)
        sampler = None
    else:
        probability = None

        def sampler(x, y):
            return compute.sample_dust_sites(envelope, compression, bands, x, y)
    count = round((2000 if fine else 1450) / scene_scale**2)
    fields = () if hue_gradient is None else tuple(hue_gradient)
    for cx, cy, radius, *gradient in prime_placements(probability, compression, outer_scale, occupied, rng,
                                                      count=count, fields=fields, compute=compute, sampler=sampler,
                                                      scene_scale=scene_scale):
        extent = 2.8 if hue_gradient is not None or bands is not None else 1.4
        angle = rng.uniform(0, 2 * np.pi)
        if gradient:
            dy_hue, dx_hue = gradient
            normal = np.arctan2(dy_hue, dx_hue)
            direction = np.hypot(dy_hue, dx_hue) * frame_shape[1] * scene_scale
            angle = normal - np.pi / 2 + 0.14 * np.sin(angle) + 0.75 * (1 - smoothstep(0.1, 2, direction)) * np.sin(angle)
        bend, squeeze = rng.uniform(-0.45, 0.45), rng.uniform(0.47, 1)
        phase_2, phase_3, phase_7, arc_phase = rng.uniform(0, 2 * np.pi, 4)
        amplitude = rng.uniform(-0.55, 0.90)
        prominence = tail = curl = sharp_side = 0
        if gradient and amplitude > 0:
            prominence = smoothstep(11, 42, radius * WALLPAPER_SIZE[0] / frame_shape[1])
            sides = np.arange(4) * np.pi / 2
            preference = np.exp(1.19 * np.cos(sides - (normal - angle)))
            sharp_side = rng.choice(sides, p=preference / preference.sum())
            tail = rng.choice((-1, 1))
            curl = rng.uniform(0.24, 0.60)
        if compute is not None:
            patches.append((cx, cy, radius, np.ceil(extent * radius), np.cos(angle), np.sin(angle),
                            bend, squeeze, phase_2, phase_3, phase_7, arc_phase, amplitude,
                            prominence, tail, curl, sharp_side, cx, cy))
            continue
        region, dx, dy = feature_patch(frame_shape, cx, cy, radius, extent)
        u = np.cos(angle) * dx + np.sin(angle) * dy
        v = np.cos(angle) * dy - np.sin(angle) * dx
        v = (v + bend * u**2) / squeeze
        radius_sq = u**2 + v**2
        shape = np.maximum(0, 1 - radius_sq)**3
        theta = np.arctan2(v, u)
        contour = 0.62 + 0.14 * np.cos(2 * theta + phase_2)
        contour += 0.07 * np.sin(3 * theta + phase_3) + 0.035 * np.cos(7 * theta + phase_7)
        arc_gate = smoothstep(-0.7, 0.8, np.cos(theta + arc_phase))
        arc = np.exp(-((np.sqrt(radius_sq) - contour) / 0.11)**2) * arc_gate
        arc *= np.maximum(0, 1 - radius_sq)**2
        detail = 0.23 + 0.77 * star_density[region]
        if tail:
            # A fold trails to one side along the corridor, with a curved spine.
            u = u / (1.29 + 0.60 * prominence) - 0.21 * tail
            v = v + tail * curl * u**2
            theta = np.arctan2(v, u)
            outline = 0.97 + 0.18 * np.cos(2 * theta + phase_2)
            outline += 0.11 * np.sin(3 * theta + phase_3) + 0.06 * np.cos(7 * theta + phase_7)
            shape = dust_fold(u, v, outline, sharp_side, prominence)
            arc = 0
        field[region] += amplitude * (shape * detail + 0.68 * arc * detail**2)
    if compute is not None:
        return compute.splat(frame_shape, patches, 0, star_density)
    return np.tanh(field)


def corridor_relief(core, rim, lit, compression, outer_scale, occupied, rng):
    """Place broad, tangent-aligned relief on illuminated transitions between bands."""
    core_y, core_x = np.gradient(core)
    rim_y, rim_x = np.gradient(rim)
    core_edge = np.exp(-0.5 * ((core - 0.51) / 0.17)**2)
    rim_edge = (1 - core) * np.exp(-0.5 * ((rim - 0.50) / 0.19)**2)
    prefer_core = core_edge * np.hypot(core_x, core_y) >= rim_edge * np.hypot(rim_x, rim_y)
    nx = np.where(prefer_core, core_x, rim_x)
    ny = np.where(prefer_core, core_y, rim_y)
    boundary_strength = np.hypot(nx, ny) * core.shape[1]
    front = lit * np.where(prefer_core, core_edge, rim_edge)
    front *= smoothstep(0.6, 3.5, boundary_strength)

    relief = np.zeros_like(core)
    for cx, cy, radius in prime_placements(front**1.7, compression, outer_scale, occupied, rng,
                                           minimum=23, count=480):
        region, dx, dy = feature_patch(relief.shape, cx, cy, radius)
        tangent = np.arctan2(ny[cy, cx], nx[cy, cx]) - np.pi / 2
        u = np.cos(tangent) * dx + np.sin(tangent) * dy
        v = np.cos(tangent) * dy - np.sin(tangent) * dx
        v += rng.uniform(-0.27, 0.27) * u**2
        shape = np.maximum(0, 1 - u**2 - (v / 0.48)**2)**2
        ridge = np.exp(-((v - 0.15) / 0.25)**2) - 0.55 * np.exp(-((v + 0.19) / 0.29)**2)
        relief[region] += front[cy, cx] * shape * ridge
    return np.tanh(relief) * front


def attached_relief(core, rim, visible, lit, compression, outer_scale, occupied, rng, *, compute=None,
                    scene_scale=1.0):
    """Attach each light accent to a visible, illuminated shoulder of the cloud."""
    height, width = visible.shape
    shape = (max(2, int(width / (26 * scene_scale))), max(2, int(height / (26 * scene_scale))))
    cuda = compute if compute is not None and compute.name == "cuda" else None
    xp = np if cuda is None else cuda.cp
    if cuda is None:
        smooth = Image.fromarray(visible.astype(np.float32)).resize(shape, Image.Resampling.BILINEAR)
        visible = np.asarray(smooth.resize((width, height), Image.Resampling.BILINEAR))
    else:
        visible = cuda.bilinear(cuda.bilinear(visible, shape), (width, height))
    band = core - 0.58 * rim * (1 - core)
    if cuda is not None and cuda.fast:
        dy, dx, front, light_side, transition = cuda.relief_front(
            visible, band, core, rim, lit, width, height, width * scene_scale
        )
    else:
        dy, dx = xp.gradient(visible)
        band_y, band_x = xp.gradient(band)
        bright = smoothstep(0.14, 0.39, visible, xp)
        direction = xp.hypot(dx, dy)
        alignment = xp.abs(dx * band_x + dy * band_y) / xp.maximum(direction * xp.hypot(band_x, band_y), 1e-10)
        transition = xp.maximum(4 * core * (1 - core), 4 * rim * (1 - rim) * (1 - core))
        front = bright * lit * transition * (0.45 + 0.55 * alignment)
        front *= smoothstep(0.18, 1.45, direction * (width * scene_scale), xp)
        light_side = smoothstep(0.39, 0.78, bright, xp)
    relief = np.zeros_like(core) if compute is None else None
    patches = []
    for cx, cy, radius, normal_y, normal_x, power in prime_placements(
        front**1.24, compression, outer_scale, occupied, rng,
        minimum=23, count=round(720 / scene_scale**2), fields=(dy, dx, front), compute=compute,
        scene_scale=scene_scale
    ):
        tangent = np.arctan2(normal_y, normal_x) - np.pi / 2
        bend = rng.uniform(-0.27, 0.27)
        if compute is not None:
            patches.append((cx, cy, radius, np.ceil(1.4 * radius), np.cos(tangent), np.sin(tangent), bend,
                            power, *([0] * 9), cx, cy))
            continue
        region, dx_patch, dy_patch = feature_patch(relief.shape, cx, cy, radius)
        u = np.cos(tangent) * dx_patch + np.sin(tangent) * dy_patch
        v = np.cos(tangent) * dy_patch - np.sin(tangent) * dx_patch
        v = (v - 0.12 + bend * u**2) / 0.56
        radius_sq = u**2 + v**2
        body = feathered_sides(u, v, np.sqrt(radius_sq), -np.pi / 2)
        body *= np.exp(-1.75 * radius_sq) * smoothstep(-0.50, 0.38, v)
        relief[region] += 0.56 * power * body * (light_side[region] * transition[region])
    if compute is not None:
        # Preserve the float32 smoothstep roundoff before multiplying by transition.
        return compute.splat(core.shape, patches, 1, light_side, transition)
    return np.tanh(relief)


def nebula_turbulence(x, y, gradients, outer_scale, damping_ratio, frequency_step):
    """An illustrative cascade with an injection scale and local damping."""
    u, v = x / outer_scale, y / outer_scale
    variation = np.zeros_like(x)
    frequency = 1.0
    for _ in range(7):
        resolved = np.exp(-(frequency * damping_ratio)**2)
        variation += noise(u, v, gradients) * frequency**(-1 / 3) * resolved
        u, v = frequency_step * (0.87 * u - 0.49 * v) + 17, frequency_step * (0.49 * u + 0.87 * v) + 29
        frequency *= frequency_step
    return variation


def dust_transmittance(depth, color, shadow):
    chroma = color / color.max(axis=-1, keepdims=True)
    selective_loss = 0.81 + 0.82 * (1 - chroma)
    weight = np.clip(shadow, 0, 0.96)[..., None]
    loss = (1 - weight) * (0.80, 1.0, 1.19) + weight * selective_loss
    return np.exp(-depth[..., None] * loss)


def hue_shift(rgb, turns, xp=np):
    high = rgb.max(axis=-1)
    span = high - rgb.min(axis=-1)
    red, green, blue = np.moveaxis(rgb, -1, 0)
    safe = xp.maximum(span, 1e-8)
    sector = xp.where(high == red, (green - blue) / safe,
                      xp.where(high == green, 2 + (blue - red) / safe, 4 + (red - green) / safe))
    hue = sector + 6 * turns
    shifted = []
    for offset in (5, 3, 1):
        phase = (offset + hue) % 6
        shifted.append(high - span * xp.clip(xp.minimum(phase, 4 - phase), 0, 1))
    return xp.stack(shifted, axis=-1)


def accent_opacity(u, v, sides, xp=np):
    """Fade along two selected directions, with continuous falloff on every edge."""
    distance = xp.hypot(u, v)
    angle = xp.arctan2(v, u)
    selected = xp.maximum(0, xp.cos(angle - sides[0]))**4
    selected += xp.maximum(0, xp.cos(angle - sides[1]))**4
    extent = 1 + 0.58 * selected
    return smoothstep(0, 0.88 + 0.68 * selected, extent - distance, xp)


def color_accents(light, orbit, rng, *, compute=None, scene_scale=1.0):
    """Blend a few analogous hues into the most uniform parts of the gas."""
    height, width = light.shape[:2]
    xp = np if compute is None else compute.cp
    fast = compute is not None and compute.fast
    y = rng.integers(height, size=520)
    x = rng.integers(width, size=y.size)
    diagonal = max(2, round(width * 0.023 * scene_scale))
    offsets = ((-diagonal, 0), (diagonal, 0), (0, -diagonal), (0, diagonal))
    if fast:
        ox, oy = np.array(((0, 0), *offsets)).T[..., None]
        color, *neighbors = light[np.clip(y + oy, 0, height - 1),
                                  np.clip(x + ox, 0, width - 1)].get()
    else:
        color = light[y, x]
        if compute is not None:
            color = color.get()
        neighbors = [light[np.clip(y + oy, 0, height - 1), np.clip(x + ox, 0, width - 1)] for ox, oy in offsets]
        if compute is not None:
            neighbors = [neighbor.get() for neighbor in neighbors]
    luma = color @ (0.2126, 0.7152, 0.0722)
    variation = np.zeros_like(luma)
    chroma = color / np.maximum(color.max(axis=1, keepdims=True), 1)
    for neighbor in neighbors:
        neighbor /= np.maximum(neighbor.max(axis=1, keepdims=True), 1)
        variation += np.mean((neighbor - chroma)**2, axis=1)
    weight = smoothstep(14, 34, luma) * (1 - smoothstep(68, 115, luma)) * np.exp(-110 * variation)
    if not weight.any():
        return light, xp.zeros((height, width), dtype=bool)
    accents = xp.zeros((height, width), dtype=bool)
    covered = 0
    luminance_weights = xp.asarray((0.2126, 0.7152, 0.0722))
    output = light.copy()
    for i in rng.choice(x.size, size=min(6, np.count_nonzero(weight)), replace=False, p=weight / weight.sum()):
        cx, cy = x[i], y[i]
        radius = max(2, rng.uniform(0.019, 0.027) * width * scene_scale)
        region, u, v = feature_patch((height, width), cx, cy, radius, extent=2.4)
        u, v = xp.asarray(u), xp.asarray(v)
        step = max(1, round(radius * 0.25))
        if fast:
            right, left, bottom, top = orbit[
                [cy, cy, min(height - 1, cy + step), max(0, cy - step)],
                [min(width - 1, cx + step), max(0, cx - step), cx, cx],
            ].get()
            gx, gy = right - left, bottom - top
        else:
            gx = orbit[cy, min(width - 1, cx + step)] - orbit[cy, max(0, cx - step)]
            gy = orbit[min(height - 1, cy + step), cx] - orbit[max(0, cy - step), cx]
        angle = np.arctan2(float(gy), float(gx)) + np.pi / 2 + rng.uniform(-0.32, 0.32)
        u, v = np.cos(angle) * u + np.sin(angle) * v, np.cos(angle) * v - np.sin(angle) * u
        v = (v + rng.uniform(-0.32, 0.32) * u**2) / rng.uniform(0.40, 0.56)
        sides = rng.choice(4, size=2, replace=False) * np.pi / 2
        opacity = accent_opacity(u, v, sides, xp) * rng.uniform(0.62, 0.82)
        new = (opacity > 0) & ~accents[region]
        area = int(new.sum())
        if covered + area > int(0.028 * height * width):
            continue
        covered += area
        accents[region] |= new
        patch = output[region]
        other = hue_shift(patch, rng.choice((-1, 1)) * rng.uniform(0.043, 0.075), xp)
        local_luma = patch @ luminance_weights
        other *= (local_luma / xp.maximum(other @ luminance_weights, 1e-8))[..., None]
        # Compress toward equal-luminance gray only if the target leaves the RGB gamut.
        gamut = xp.minimum(1, (255 - local_luma) / xp.maximum(other.max(axis=-1) - local_luma, 1e-8))
        other = local_luma[..., None] + gamut[..., None] * (other - local_luma[..., None])
        patch += opacity[..., None] * (other - patch)
    return output, accents


def nebula_emission(rgb, bands, sky_x, sky_y, orbit, dust, gradients, rng, star_density, version=1, fine=False,
                    compute=None, scene_scale=1.0, resolution_scale=None, frame_extent=(1.0, 1.0)):
    """Project overlapping clouds, foreground dust and cavities along the line of sight."""
    fractal_field = fractal if compute is None else compute.fractal
    turbulence_field = nebula_turbulence if compute is None else compute.turbulence
    broad = fractal_field(7 * sky_x + 21, 9 * sky_y + 41, gradients, 4, 0.50)
    folds = fractal_field(24 * sky_x - 38, 30 * sky_y + 14, gradients, 4, 0.50)
    warped_x = sky_x + 0.042 * broad + 0.012 * folds
    warped_y = orbit + 0.062 * broad + 0.026 * folds
    compositor = compute if compute is not None and compute.name == "cuda" else None
    rasterizer = compute if compute is not None and compute.name in ("cuda", "webgpu") else None
    layers = rasterizer if rasterizer is not None and rasterizer.optimized else None
    xp = np if compositor is None else compositor.cp
    fused = layers is not None
    transmission = xp.ones(rgb.shape, dtype=rgb.dtype)
    light = xp.zeros_like(transmission)
    dust_column = xp.zeros_like(orbit, dtype=np.float32 if fused else orbit.dtype)
    shadow_bands, light_bands = band_light_and_shadow(bands)

    # Populate the available area with more volumes while preserving each volume's optical scale.
    long_extent, short_extent = frame_extent
    layer_count = max(4, min(12, round(4 * long_extent * short_extent)))
    centers = np.linspace(-0.36 * long_extent, 0.36 * long_extent, layer_count)
    centers += rng.uniform(-0.085, 0.085, layer_count)
    order = rng.permutation(layer_count)
    occupied = {}
    if compute is not None:
        from wallpaper_compute import ScaleNeighbors

        pixel_scale = orbit.shape[1] / WALLPAPER_SIZE[0] if resolution_scale is None else resolution_scale
        occupied = ScaleNeighbors(pixel_scale * scene_scale,
                                  cache=(fused and layers.fast) or compute.name == "webgpu")
    detail_rng = np.random.Generator(rng.bit_generator.jumped())
    relief_rng = np.random.Generator(rng.bit_generator.jumped(2))
    for depth, index in enumerate((*order, layer_count)):
        slab = fractal_field(10 * sky_x + 17 * depth, 13 * sky_y - 23 * depth, gradients, 3, 0.51)
        if index == layer_count:
            center_x, center_y, span_x, span_y = 0.0, 0.0, 0.47, 0.15
            strength = 0.22
        else:
            center_x = centers[index]
            center_y = rng.uniform(-0.095, 0.095) * max(1.0, short_extent)
            span_x, span_y = rng.uniform(0.13, 0.25), rng.uniform(0.065, 0.143)
            strength = rng.uniform(0.63, 0.95)
        tilt = rng.uniform(-0.48, 0.48)
        offset_x, width_x = rng.uniform(-0.42, 0.42), rng.uniform(0.28, 0.54)
        offset_y, width_y = rng.uniform(-0.32, 0.32), rng.uniform(0.36, 0.69)
        cavity_strength = 0.77 if index in (0, 2) else 0.26
        if fused:
            (envelope, cavity, compression, damping, core_weight, rim_zone,
             diffuse_gas, definition, broad_light, lit) = layers.layer_geometry(
                warped_x, warped_y, slab, folds, broad, star_density, center_x, center_y,
                span_x, span_y, tilt, offset_x, offset_y, width_x, width_y, strength, cavity_strength
            )
        else:
            local_x = warped_x - center_x + 0.021 * slab
            local_y = warped_y - center_y + 0.034 * slab - tilt * local_x
            u, v = local_x / span_x, local_y / span_y
            envelope = xp.exp(-0.62 * xp.hypot(u, v)**2.15)
            cavity_x = (u - offset_x) / width_x
            cavity_y = (v - offset_y + 0.24 * folds) / width_y
            cavity_radius = xp.hypot(cavity_x, cavity_y)
            cavity = xp.exp(-0.60 * cavity_radius**2)
            # Compressed walls retain small structures, while hot interiors damp them.
            compression = xp.exp(-0.5 * ((cavity_radius - 1.25) / 0.56)**2)
            compression *= (0.72 + 0.28 * xp.tanh(2.3 * folds + slab)) * (0.56 + 0.44 * cavity_strength)
            damping = 0.075 + 0.065 * cavity * cavity_strength - 0.066 * compression + 0.074 * (1 - star_density)
            core_weight = xp.exp(-0.95 * ((u + 0.36 + 0.40 * slab)**2 + (v - 0.28)**2))
            rim_zone = smoothstep(-0.42, 0.48, 0.44 * u - 0.34 * v + 0.68 * slab, xp)
            diffuse_gas = strength * envelope * xp.exp(0.49 * broad + 0.37 * slab) * (1 - cavity_strength * cavity)
            definition = smoothstep(0.11, 0.77, diffuse_gas, xp) * (0.54 + 0.46 * star_density)
            broad_light = xp.exp(-0.50 * ((u + 0.28)**2 + (v - 0.35)**2))
            lit = definition * smoothstep(0.25, 0.81, broad_light, xp)
        outer_scale = 1.35 * np.sqrt(span_x * span_y * width_x * width_y)
        turbulent = turbulence_field(sky_x + 3.1 * depth, sky_y - 2.7 * depth,
                                     gradients, outer_scale, damping, 1.88 + 0.04 * depth)
        dust_bands = (core_weight, rim_zone) if version == 2 and compositor is not None and compositor.fast else None
        hue_gradient = (xp.gradient(core_weight - 0.58 * rim_zone * (1 - core_weight))
                        if version == 2 and dust_bands is None else None)
        details = irregularity_field(envelope, compression, star_density, outer_scale, occupied, detail_rng,
                                     hue_gradient=hue_gradient, fine=fine, compute=rasterizer, bands=dust_bands,
                                     scene_scale=scene_scale)
        if fused:
            opacity, dust_depth, visible = layers.layer_density(
                diffuse_gas, definition, turbulent, details, envelope, cavity, dust, slab,
                compression, cavity_strength, broad_light, transmission
            )
            opacity = opacity[..., None]
        else:
            gas = diffuse_gas * xp.exp((0.30 + 0.79 * definition) * turbulent - (0.13 + 0.75 * definition) * details)
            opacity = -xp.expm1(-gas)[..., None]
            dust_depth = envelope * (0.105 + 0.43 * cavity_strength * cavity) * xp.exp(
                0.25 * dust + 0.37 * slab + (0.39 + 0.30 * compression) * turbulent + 1.35 * details
            )
        if version == 2:
            if not fused:
                visible = opacity[..., 0] * broad_light * transmission.mean(axis=2) * xp.exp(-dust_depth)
            relief = attached_relief(core_weight, rim_zone, visible, lit, compression, outer_scale,
                                     occupied, relief_rng, compute=rasterizer, scene_scale=scene_scale)
        else:
            relief = corridor_relief(core_weight, rim_zone, lit, compression, outer_scale, occupied, relief_rng)
        indices = detail_rng.permutation(3)
        fill, core, rim = shadow_bands[indices]
        fill_shift, core_shift, rim_shift = (light_bands - shadow_bands)[indices]
        if fused:
            layers.layer_composite(
                core_weight, rim_zone, definition, broad_light, relief, opacity[..., 0], dust_depth,
                cavity, details, cavity_strength, layers.array((fill, core, rim, fill_shift, core_shift, rim_shift)),
                light, transmission, dust_column
            )
            continue
        illumination = 0.74 + 0.43 * broad_light + 0.64 * relief
        tone = smoothstep(0.0, 1.0, 0.25 + 0.39 * broad_light + 0.20 * definition + 0.34 * relief, xp)
        rim_weight = (1 - core_weight) * rim_zone
        fill_power = (1 - core_weight - rim_weight)**2.3
        core_power, rim_power = core_weight**2.3, rim_weight**2.3
        dust_column += dust_depth
        shadow = 0.48 * cavity_strength * cavity + 1.65 * xp.maximum(details, 0)
        if compositor is not None:
            compositor.composite(light, transmission, (fill_power, core_power, rim_power),
                                 (fill, core, rim, fill_shift, core_shift, rim_shift),
                                 (tone, opacity[..., 0], illumination, dust_depth, shadow))
        else:
            color = (fill_power[..., None] * fill + core_power[..., None] * core + rim_power[..., None] * rim)
            color_shift = (fill_power[..., None] * fill_shift + core_power[..., None] * core_shift
                           + rim_power[..., None] * rim_shift)
            color = (color + tone[..., None] * color_shift) / (fill_power + core_power + rim_power)[..., None]
            light += transmission * opacity * color * illumination[..., None]
            transmission *= (1 - opacity) * dust_transmittance(dust_depth, color, shadow)
    if version == 2:
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, 2)
        if compositor is not None and compositor.fast:
            phase_x, phase_y = float(phase_x), float(phase_y)
        drift = 0.010 * (0.65 * xp.sin(8 * sky_x + 3 * sky_y + phase_x)
                         + 0.35 * xp.sin(3 * sky_x - 9 * sky_y + phase_y))
        shifted = compositor.hue_shift(light, drift) if compositor is not None and compositor.fast else hue_shift(light, drift, xp)
        light, _ = color_accents(shifted, orbit, rng, compute=compositor, scene_scale=scene_scale)
    return light + rgb * transmission, dust_column, light


def add_starlight(fixed, twinkle, region, light, rng):
    fixed[region] += light
    frequency = rng.integers(0, len(twinkle))
    phase = rng.uniform(0, 2 * np.pi)
    amplitude = rng.uniform(0.18, 0.43)
    twinkle[frequency, 0][region] += light * (amplitude * np.cos(phase))
    twinkle[frequency, 1][region] += light * (amplitude * np.sin(phase))


def texture(sha, colors=None, *, size=(WIDTH, HEIGHT), animated=True, wallpaper_version=2, fine=False, compute=None,
            uniform_palette=False, target_hue=None, scene_scale=1.0, star_count=REFERENCE_STAR_COUNT,
            adapt_resolution=False):
    fractal_field = fractal if compute is None else compute.fractal
    device = compute if compute is not None and compute.name == "cuda" else None
    xp = np if device is None else device.cp
    dtype = np.float64 if device is None else device.dtype
    fast = device is not None and device.fast
    key = hashlib.sha256(bytes.fromhex(sha)).digest()
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    if animated:
        palette = palette_rgb(colors or PALETTES[key[4] % len(PALETTES)])
    else:
        palette, bands = np.split(wallpaper_colors(key), [5])
    phase = rng.uniform(0, 2 * np.pi, (128, 128))
    gradients = np.stack((np.cos(phase), np.sin(phase)), axis=-1)
    width, height = size
    reference_width, reference_height = WALLPAPER_SIZE
    resolution_scale, camera_pixel_scale = None, None
    frame_extent, camera_scale = (1.0, 1.0), 1.0
    portrait = adapt_resolution and height > width
    if adapt_resolution:
        resolution_scale = np.sqrt(width * height / (reference_width * reference_height))
        long_side, short_side = max(width, height), min(width, height)
        long_ratio = long_side / reference_width
        short_ratio = short_side / reference_height
        # Use one pixel scale on both axes, with a bounded field of view for extreme formats.
        camera_pixel_scale = max(min(long_ratio, short_ratio), long_ratio / 3)
        frame_extent = (long_ratio / camera_pixel_scale, short_ratio / camera_pixel_scale)
        camera_scale = width / (reference_width * camera_pixel_scale)
    optical_scale = scene_scale if resolution_scale is None else scene_scale * resolution_scale
    pad_x, pad_y = (FOG_PAD_X, FOG_PAD_Y) if animated else (0, 0)
    y, x = xp.mgrid[-pad_y:height + pad_y, -pad_x:width + pad_x].astype(dtype)
    sx, sy = (x - width / 2) / width, (y - height / 2) / width
    if portrait:
        sx, sy = sy, -sx
    projection_scale = camera_scale / scene_scale
    if projection_scale != 1.0:
        sx *= projection_scale
        sy *= projection_scale
    angle = rng.uniform(-0.38, 0.38)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    warp_x, warp_y = rng.uniform(0, 90, 2).tolist()
    phase = rng.uniform(0, 6)
    geometry = (cosine, sine, warp_x, warp_y, phase, projection_scale, float(portrait))
    if fast:
        u, orbit, mist, dust = device.projection((height, width), gradients, geometry)

        def star_projection(x, y):
            return device.projection((height, width), gradients, geometry, (x, y))
    else:
        u = sx * cosine - sy * sine
        v = sx * sine + sy * cosine
        wx = fractal_field(2.4 * u + warp_x, 2.4 * v + warp_y, gradients)
        wy = fractal_field(2.4 * u + warp_y, 2.4 * v + warp_x + 17, gradients)
        u = u + 0.33 * wx
        v = v + 0.14 * wy
        mist = fractal_field(4 * u + 52, 6 * v + 10, gradients, 4, 0.41)
        dust = fractal_field(9 * u + 16, 13 * v - 38, gradients, 3, 0.38)
        center = 0.037 * xp.sin(6 * u + phase)
        orbit = v - center + 0.028 * mist
        if compute is not None and compute.name == "webgpu":
            def star_projection(x, y):
                return compute.star_projection(x, y, gradients, width, height, geometry)
        else:
            star_projection = None
    offset = rng.uniform(-0.25, 0.26)
    if fast:
        rgb = xp.empty((*u.shape, 3), dtype=dtype)
        core, gloss = xp.empty_like(u), xp.empty_like(u)
        device.cloud_base(u, orbit, mist, dust, sx, sy, offset, device.array(palette), core, gloss, rgb)
    else:
        halo = xp.exp(-((u - offset) / 0.44)**2 - (orbit / 0.093)**2)
        arm = xp.exp(-((orbit + 0.028 * xp.sin(10 * u)) / 0.045)**2)
        core = xp.exp(-((u - offset) / 0.17)**2 - ((orbit - 0.008) / 0.047)**2)
        obscuration = smoothstep(0.04, 0.37, dust, xp) * arm
        density = xp.clip(0.035 + halo * (0.13 + 0.15 * mist)
                          + arm * (0.07 + 0.07 * mist) + 0.19 * core
                          - 0.08 * obscuration, 0, 1)
        stops = (0, 0.12, 0.26, 0.42, 0.62) if colors or not animated else (0, 0.15, 0.39, 0.67, 1)
        rgb = xp.stack([xp.interp(density, xp.asarray(stops), xp.asarray(channel)) for channel in palette.T], axis=-1)
        gloss = xp.exp(-((0.52 * sx + sy + 0.12) / 0.15)**2)
    if not animated:
        scene_rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
        stars = sample_visible_stars(u, orbit, rng, xp=xp, projection=star_projection,
                                    evaluate=device.star_candidates if fast else None, star_count=star_count)
        grain = device.grain_worker.submit(rng.normal, 0, 0.42, (height, width, 1)) if fast else None
        density_map = star_density_map(stars, orbit.shape, compute=device, scene_scale=optical_scale)
        cloud_x, cloud_y, cloud_orbit, cloud_dust = (field.astype(dtype, copy=False) for field in (sx, sy, orbit, dust))
        rgb, dust_column, cloud_light = nebula_emission(rgb, bands, cloud_x, cloud_y, cloud_orbit, cloud_dust,
                                                        gradients, scene_rng, density_map, wallpaper_version,
                                                        fine, compute, scene_scale, camera_pixel_scale, frame_extent)
    # The cloud's light moves with the fog, while the print reflection stays fixed.
    rgb += (11 * core)[..., None] * xp.asarray((0.91, 0.94, 1), dtype=dtype)
    if not animated and uniform_palette:
        from wallpaper_compute import median_cloud_hue

        current_hue = median_cloud_hue(rgb, xp)
        if current_hue is not None:
            if target_hue is None:
                target_hue = np.random.default_rng(int.from_bytes(key[24:], "big")).random()
            turns = (target_hue - current_hue + 0.5) % 1 - 0.5
            if fast:
                rgb = device.hue_shift(rgb, turns)
                cloud_light = device.hue_shift(cloud_light, turns)
            else:
                rgb = hue_shift(rgb, turns, xp)
                cloud_light = hue_shift(cloud_light, turns, xp)

    crop = np.s_[pad_y:pad_y + height, pad_x:pad_x + width]
    fog = 0.81 * rgb
    fixed = (4 * gloss[crop])[..., None] * xp.asarray((0.91, 0.94, 1), dtype=dtype)
    if not animated:
        optics_rng = np.random.default_rng(int.from_bytes(key[16:24], "big"))
        rasterizer = compute if compute is not None and compute.name in ("cuda", "webgpu") else None
        # Approximate the extra area covered by an output pixel below the reference resolution.
        pixel_variance = max(0.0, 1.0 - resolution_scale**2) / 12 if resolution_scale is not None else 0.0
        fixed = 0.81 * fixed + star_field(stars, dust_column, cloud_light, optics_rng, compute=rasterizer,
                                         scene_scale=optical_scale, pixel_variance=pixel_variance)
        fixed += xp.asarray(grain.result() if grain is not None else rng.normal(0, 0.42, (height, width, 1)),
                            dtype=dtype)
        return fog, fixed, None
    arm, density = arm[crop], density[crop]
    twinkle = np.zeros((3, 2, height, width, 3))
    star_rng = np.random.default_rng(int.from_bytes(key[16:], "big"))

    # Fine stars and a handful of larger stars with soft light spilling into the print.
    star_count = round(440 * width * height / (WIDTH * HEIGHT))
    for _ in range(star_count):
        px, py = rng.integers((4, 4), (width - 4, height - 4))
        power = min(1.9, rng.exponential(0.43)) * (0.75 + 0.55 * arm[py, px])
        extent = 4 if power > 0.95 else 2
        ky, kx = np.mgrid[-extent:extent + 1, -extent:extent + 1]
        radius = (kx - rng.uniform(-.48, .48))**2 + (ky - rng.uniform(-.48, .48))**2
        glow = 100 * np.exp(-radius / 0.45) + 8 * np.exp(-radius / 4.1)
        region = np.s_[py-extent:py+extent+1, px-extent:px+extent+1]
        light = power * glow[..., None] * (rng.uniform(0.83, 1), 0.94, 1.0)
        add_starlight(fixed, twinkle, region, light, star_rng)
    extent = 11
    ky, kx = np.mgrid[-extent:extent + 1, -extent:extent + 1]
    spark = (120 * np.exp(-(kx**2 + ky**2) / 1.45)
             + 26 * np.exp(-kx**2 / 0.37 - ky**2 / 24)
             + 18 * np.exp(-ky**2 / 0.37 - kx**2 / 16)
             + 12 * np.exp(-(kx**2 + ky**2) / 23))
    spark_count = max(1, round(3 * width / WIDTH))
    candidates = rng.integers((extent, extent), (width - extent, height - extent), size=(max(17, 4 * spark_count), 2))
    for px, py in sorted(candidates, key=lambda point: density[point[1], point[0]])[:spark_count]:
        region = np.s_[py-extent:py+extent+1, px-extent:px+extent+1]
        add_starlight(fixed, twinkle, region, spark[..., None] * (0.57, 0.66, 0.78), star_rng)

    fixed *= 0.81
    grain = rng.normal(0, 0.42, (height, width, 1))
    x, y = x[crop], y[crop]
    edge_distance = np.minimum(np.minimum(x, width - 1 - x), np.minimum(y, height - 1 - y))
    edge_light = 7 * np.exp(-edge_distance / 3.5)
    fixed += edge_light[..., None] + grain
    return fog, fixed, 0.81 * twinkle


def sample_fog(fog, offset, drift):
    sx = FOG_PAD_X + 3.6 * drift * offset
    sy = FOG_PAD_Y + 0.84 * drift * offset
    ix, iy = int(sx), int(sy)
    dx, dy = sx - ix, sy - iy
    top = fog[iy:iy + HEIGHT, ix:ix + WIDTH] * (1 - dx) + fog[iy:iy + HEIGHT, ix + 1:ix + WIDTH + 1] * dx
    bottom = fog[iy + 1:iy + HEIGHT + 1, ix:ix + WIDTH] * (1 - dx) + fog[iy + 1:iy + HEIGHT + 1, ix + 1:ix + WIDTH + 1] * dx
    return top * (1 - dy) + bottom * dy


def background(fog, fixed, twinkle, turns, drift, size=(WIDTH, HEIGHT)):
    time = turns % 1
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    # Both cloud layers travel up and left. Blend them at staggered times across the frame.
    transition = 0.14 * (x / WIDTH + y / HEIGHT - 1)
    blend = smoothstep(0.22, 0.78, time + transition)[..., None]
    rgb = sample_fog(fog, time, drift) * (1 - blend) + sample_fog(fog, time - 1, drift) * blend + fixed
    for frequency, (sine, cosine) in enumerate(twinkle, start=1):
        angle = 2 * np.pi * frequency * time
        rgb += np.sin(angle) * sine + (np.cos(angle) - 1) * cosine
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    return image if image.size == size else image.resize(size, Image.Resampling.LANCZOS)


def font_paths(font_dir):
    roots = (font_dir, Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
             Path("/usr/share/fonts/truetype/dejavu"))
    paths = []
    for candidates in FONT_CANDIDATES:
        match = next((root / candidate for root in roots if root
                      for candidate in candidates if (root / candidate).is_file()), None)
        if match is None:
            raise ValueError("Install fonts-dejavu-core or pass --font-dir with the title and mono fonts.")
        paths.append(match)
    return paths


def fitted_font(draw, text, path, size, width):
    while size >= 6:
        font = ImageFont.truetype(str(path), size)
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        if right - left <= width:
            return font
        size -= 1
    raise ValueError(f"Text exceeds panel width: {text!r}")


def contrasting_ink(backgrounds):
    srgb = np.arange(256) / 255
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055)**2.4)
    weights = (0.2126, 0.7152, 0.0722)
    # All panels have the same area, so each contributes equally to the mean.
    luminance = np.mean([linear[np.asarray(image)].mean(axis=(0, 1)) @ weights
                         for image in backgrounds])
    white_contrast = 1.05 / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    return "white" if white_contrast >= black_contrast else "black"


def draw_text(image, position, text, font, anchor, ink, outline):
    # Blend one pixel of the panel's median color to give a subpixel-weight outline.
    ImageDraw.Draw(image).text(position, text, font=font, anchor=anchor,
                              fill=ink, stroke_width=1, stroke_fill=(*outline, 165))


def draw_star(image, position, size, ink, outline):
    scale = 4
    star = Image.new("RGBA", (size * scale, size * scale))
    draw = ImageDraw.Draw(star)
    angles = np.arange(10) * np.pi / 5 - np.pi / 2
    radius = size / 2 - 1
    radii = np.tile((radius, radius * 0.44), 5)
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    for edge, fill in ((0.7, (*outline, 165)), (0, ink)):
        points = scale * (size / 2 + directions * (radii + edge)[:, None])
        draw.polygon([tuple(point) for point in points], fill=fill)
    star = star.resize((size, size), Image.Resampling.LANCZOS)
    image.alpha_composite(star, position)


def lettering(release, fonts, reference, ink):
    median = tuple(np.median(np.asarray(reference), axis=(0, 1)).astype(int))
    image = Image.new("RGBA", reference.size)
    draw = ImageDraw.Draw(image)
    width = reference.width
    if reference.size == FORUM_SIZE:
        padding, first_baseline, second_baseline = 7, 19, 34
        heading_size, meta_size, star_gap, title_gap = 15, 11, 3, 5
    else:
        padding, first_baseline, second_baseline = 16, 36, 63
        heading_size, meta_size, star_gap, title_gap = 30, 16, 7, 14
    title = " ".join(release.title.split()) or release.repo.split("/")[1]
    stars = str(release.stars)
    for size in range(heading_size, 5, -1):
        heading = ImageFont.truetype(str(fonts[0]), size)
        star_size = 2 - heading.getbbox("H", anchor="ls")[1]
        count_x = padding + star_size + star_gap
        left_end = draw.textbbox((count_x, 0), stars, font=heading, anchor="ls")[2]
        right_start = draw.textbbox((width - padding, 0), release.tag, font=heading, anchor="rs")[0]
        title_left, _, title_right, _ = draw.textbbox((width // 2, 0), title, font=heading, anchor="ms")
        if left_end + title_gap <= title_left and title_right + title_gap <= right_start:
            break
    else:
        raise ValueError(f"Title and version exceed panel width: {title!r}, {release.tag!r}")
    draw_star(image, (padding, first_baseline - star_size + 1), star_size, ink, median)
    draw_text(image, (count_x, first_baseline), stars, heading, "ls", ink, median)
    draw_text(image, (width // 2, first_baseline), title, heading, "ms", ink, median)
    draw_text(image, (width - padding, first_baseline), release.tag, heading, "rs", ink, median)
    published = release.published.astimezone(timezone.utc)
    details = f"{published:%Y-%m-%d %H:%M:%S UTC}"
    meta = fitted_font(draw, details, fonts[1], meta_size, width - 2 * padding)
    draw_text(image, (padding, second_baseline), details, meta, "ls", ink, median)
    return image


def save_panel(output, release, layers, reference, overlay, drift):
    alpha = overlay.getchannel("A")
    reference.paste(overlay, mask=alpha)
    frames = [reference]
    for index in range(1, FRAME_COUNT):
        frame = background(*layers, index / FRAME_COUNT, drift, reference.size)
        frame.paste(overlay, mask=alpha)
        frames.append(frame)
    path = output / release.filename
    reference.save(path)
    reference.save(path.with_suffix(".webp"), save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, lossless=True)


def render_releases(releases, fonts, preset, output, drift):
    forum = preset == "forum"
    size = FORUM_SIZE if forum else (WIDTH, HEIGHT)
    layers = [texture(release.sha, MOD_PALETTES.get(release.repo.lower())) for release in releases]
    references = [background(*layer, 0, drift, size) for layer in layers]
    ink = contrasting_ink(references)
    output.mkdir(parents=True, exist_ok=True)
    for release, layer, reference in zip(releases, layers, references, strict=True):
        overlay = lettering(release, fonts, reference, ink)
        save_panel(output, release, layer, reference, overlay, drift)
    print(f"{output.resolve()} ({size[0]} x {size[1]} per panel)")


@contextmanager
def wallpaper_backend(version, backend):
    with ExitStack() as resources:
        compute = None
        if version >= 3:
            from wallpaper_compute import ThreadedNoise, cpu_pool, cuda_noise

            if backend != "cpu":
                try:
                    compute = cuda_noise(optimized=version >= 4, fast=version == 5)
                except (ImportError, OSError, RuntimeError) as error:
                    if backend == "cuda":
                        raise RuntimeError(f"CUDA initialization failed: {error}") from error
                    print(f"CUDA unavailable, using row-parallel CPU: {error}")
            if compute is None:
                compute = ThreadedNoise(resources.enter_context(cpu_pool()), fractal, nebula_turbulence)
            elif version == 5:
                compute.grain_worker = resources.enter_context(ThreadPoolExecutor(max_workers=1))
            print(f"Wallpaper backend: {compute.name}", flush=True)
        yield compute


def wallpaper_variation(seed):
    key = hashlib.sha256(b"wallpaper-scale-and-stars" + bytes.fromhex(seed)).digest()
    rng = np.random.default_rng(int.from_bytes(key, "big"))
    return rng.uniform(0.8, 1.2), round(REFERENCE_STAR_COUNT * rng.uniform(0.7, 1.0))


def wallpaper_image(seed, version, fine, compute, uniform_palette=False, target_hue=None, *, size=WALLPAPER_SIZE):
    scene_scale, star_count = wallpaper_variation(seed) if version == 5 else (1.0, REFERENCE_STAR_COUNT)
    fog, stars, _ = texture(seed, size=size, animated=False, wallpaper_version=min(version, 2),
                            fine=fine, compute=compute, uniform_palette=uniform_palette, target_hue=target_hue,
                            scene_scale=scene_scale, star_count=star_count, adapt_resolution=version == 5)
    if compute is not None and compute.name == "cuda":
        pixels = compute.cp.clip(fog + stars, 0, 255).astype(np.uint8).get()
    else:
        pixels = np.clip(fog + stars, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def save_wallpapers(args):
    version = args.wallpaper_version
    batch_size = args.wallpaper_count or 1
    seed = args.seed or token_hex(20)
    batch = args.wallpaper_count is not None
    uniform_palette = (args.wallpaper_palette or ("uniform" if batch else "reference")) == "uniform"
    targets = [None] * batch_size
    if batch and uniform_palette:
        hue_rng = np.random.default_rng(int(seed, 16))
        # Use each sector once, with a random rotation of the whole batch.
        positions = hue_rng.permutation(batch_size) + hue_rng.uniform(-0.15, 0.15, batch_size)
        targets = (positions / batch_size + hue_rng.random()) % 1
    if batch:
        args.wallpaper.mkdir(parents=True, exist_ok=True)
    def write(image, path, frame_seed):
        image.save(path, format="PNG")
        print(f"{path.resolve()} (seed {frame_seed})", flush=True)

    with wallpaper_backend(version, args.wallpaper_backend) as compute, ExitStack() as resources:
        writer = resources.enter_context(ThreadPoolExecutor(max_workers=1)) if batch_size > 1 else None
        pending = None
        for index in range(batch_size):
            frame_seed = seed
            path = args.wallpaper
            if batch:
                frame_seed = hashlib.sha1(f"{seed.lower()}:{index}".encode("ascii")).hexdigest()
                path = args.wallpaper / f"{index + 1:04d}-{frame_seed}.png"
            image = wallpaper_image(frame_seed, version, args.wallpaper_detail == "fine", compute,
                                    uniform_palette, target_hue=targets[index])
            if writer is None:
                write(image, path, frame_seed)
            else:
                if pending is not None:
                    pending.result()
                pending = writer.submit(write, image, path, frame_seed)
        if pending is not None:
            pending.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repos", nargs="*", metavar="owner/repo[=Title]", help="Defaults to the three KSP mods.")
    parser.add_argument("--output", type=Path, help="Output directory. With both, it contains images and images-forum.")
    parser.add_argument("--wallpaper", type=Path, metavar="PATH",
                        help="Render a text-free 2560 x 1440 PNG, or a directory of PNGs with --wallpaper-count.")
    parser.add_argument("--seed", help="Reuse a 40-digit hexadecimal wallpaper seed.")
    parser.add_argument("--wallpaper-version", type=int, choices=(1, 2, 3, 4, 5),
                        help=f"Wallpaper algorithm version (default: {DEFAULT_WALLPAPER_VERSION}).")
    parser.add_argument("--wallpaper-backend", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wallpaper-palette", choices=("reference", "uniform"),
                        help="Defaults to uniform in a batch and reference for a single image.")
    parser.add_argument("--wallpaper-count", type=int, help="Generate N wallpapers into the given directory.")
    parser.add_argument("--wallpaper-detail", choices=("normal", "fine"), default="normal")
    parser.add_argument("--preset", choices=("github", "forum", "both"), default="github")
    parser.add_argument("--font-dir", type=Path, help="Directory with SpaceGrotesk-Bold.ttf and IBMPlexMono-Regular.ttf.")
    parser.add_argument("--drift", type=float, default=DEFAULT_DRIFT,
                        help=f"Fog motion strength in pixels (0 to {MAX_DRIFT}, default: {DEFAULT_DRIFT}).")
    args = parser.parse_args()
    if args.wallpaper is None and (args.wallpaper_version is not None or args.wallpaper_detail != "normal"):
        parser.error("Wallpaper version and detail require --wallpaper")
    if args.wallpaper is not None and args.wallpaper_version is None:
        args.wallpaper_version = DEFAULT_WALLPAPER_VERSION
    if args.wallpaper_version == 1 and args.wallpaper_detail == "fine":
        parser.error("Fine detail requires wallpaper version 2 or later")
    accelerated = args.wallpaper is not None and args.wallpaper_version >= 3
    if args.wallpaper_backend != "auto" and not accelerated:
        parser.error("--wallpaper-backend requires --wallpaper and version 3 or later")
    if args.wallpaper_palette == "uniform" and not accelerated:
        parser.error("--wallpaper-palette uniform requires --wallpaper and version 3 or later")
    if args.wallpaper_count is not None and (not accelerated
                                             or args.wallpaper_count < 1):
        parser.error("--wallpaper-count requires version 3 or later, a destination directory and a positive count")
    if args.seed is not None and (args.wallpaper is None or not re.fullmatch(r"[a-fA-F0-9]{40}", args.seed)):
        parser.error("--seed requires --wallpaper and 40 hexadecimal digits")
    if args.wallpaper is not None:
        try:
            save_wallpapers(args)
        except (OSError, RuntimeError) as error:
            parser.exit(1, f"{error}\n")
        return
    if not 0 <= args.drift <= MAX_DRIFT:
        parser.error(f"--drift must be between 0 and {MAX_DRIFT}")
    specs = args.repos or DEFAULT_REPOS
    names = [spec.partition("=")[0].lower() for spec in specs]
    if len(set(names)) != len(names):
        parser.error("Each repository can appear once")
    try:
        fonts = font_paths(args.font_dir)
        with ThreadPoolExecutor(max_workers=min(6, len(specs))) as pool:
            releases = list(pool.map(fetch_release, specs))
        if len({release.filename for release in releases}) != len(releases):
            raise ValueError("Repository names produce identical PNG filenames")
        for release in releases:
            print(release.repo, release.stars, release.tag, release.published.isoformat(), release.sha)
        presets = ("github", "forum") if args.preset == "both" else (args.preset,)
        for preset in presets:
            folder = "images-forum" if preset == "forum" else "images"
            if args.preset == "both":
                output = (args.output or Path(__file__).parent) / folder
            else:
                output = args.output or Path(__file__).with_name(folder)
            render_releases(releases, fonts, preset, output, args.drift)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
