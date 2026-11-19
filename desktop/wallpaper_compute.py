"""CUDA wallpaper kernels and a row-parallel NumPy fallback."""

from concurrent.futures import ThreadPoolExecutor
from math import ceil
import os
import re

import numpy as np


class ScaleNeighbors:
    """Index accepted features in cells one exclusion distance wide, per scale."""

    def __init__(self, pixel_scale, cache=False):
        self.pixel_scale = pixel_scale
        self.cache = cache
        self.grids = {}
        self.bounds = {}

    def limits(self, size):
        if size not in self.bounds:
            distance = 3 * max(0.72, size * self.pixel_scale)
            self.bounds[size] = ceil(distance), distance**2
        return self.bounds[size]

    def available(self, size, x, y):
        grid = self.grids.get(size)
        if grid is None:
            return True
        if self.cache:
            side, threshold = self.limits(size)
            return all((x - px)**2 + (y - py)**2 >= threshold for px, py in grid.get((x // side, y // side), ()))
        distance = 3 * max(0.72, size * self.pixel_scale)
        side = ceil(distance)
        col, row = x // side, y // side
        return all((x - px)**2 + (y - py)**2 >= distance**2
                   for cy in range(row - 1, row + 2)
                   for cx in range(col - 1, col + 2)
                   for px, py in grid.get((cx, cy), ()))

    def add(self, size, x, y):
        if self.cache:
            side, _ = self.limits(size)
            col, row = x // side, y // side
            grid = self.grids.setdefault(size, {})
            for cy in range(row - 1, row + 2):
                for cx in range(col - 1, col + 2):
                    grid.setdefault((cx, cy), []).append((x, y))
        else:
            side = ceil(3 * max(0.72, size * self.pixel_scale))
            self.grids.setdefault(size, {}).setdefault((x // side, y // side), []).append((x, y))


CUDA_NOISE = r"""
__device__ double corner(const double* gradients, int x, int y, double dx, double dy) {
    int i = 2 * ((y & 127) * 128 + (x & 127));
    return gradients[i] * dx + gradients[i + 1] * dy;
}

__device__ double noise(double x, double y, const double* gradients) {
    int ix = (int)floor(x), iy = (int)floor(y);
    double dx = x - ix, dy = y - iy;
    double u = (dx * dx * dx) * (dx * (dx * 6 - 15) + 10);
    double v = (dy * dy * dy) * (dy * (dy * 6 - 15) + 10);
    double top = (1 - u) * corner(gradients, ix, iy, dx, dy)
                 + u * corner(gradients, ix + 1, iy, dx - 1, dy);
    double bottom = (1 - u) * corner(gradients, ix, iy + 1, dx, dy - 1)
                    + u * corner(gradients, ix + 1, iy + 1, dx - 1, dy - 1);
    return (1 - v) * top + v * bottom;
}

extern "C" __global__
void fractal(const double* xs, const double* ys, const double* gradients,
             const double* weights, double* out, int size, int octaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double x = xs[i], y = ys[i], total = 0;
    for (int octave = 0; octave < octaves; ++octave) {
        total += noise(x, y, gradients) * weights[octave];
        double next_x = 1.76 * x - 1.02 * y + 19;
        y = 1.02 * x + 1.76 * y + 31;
        x = next_x;
    }
    out[i] = total;
}

extern "C" __global__
void turbulence(const double* xs, const double* ys, const double* gradients,
                const double* damping, const double* frequencies,
                const double* weights, double* out, int size,
                double outer_scale, double step) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double u = xs[i] / outer_scale, v = ys[i] / outer_scale, total = 0;
    for (int octave = 0; octave < 7; ++octave) {
        double dx = frequencies[octave] * damping[i];
        total += (noise(u, v, gradients) * weights[octave]) * exp(-(dx * dx));
        double next_u = step * (0.87 * u - 0.49 * v) + 17;
        v = step * (0.49 * u + 0.87 * v) + 29;
        u = next_u;
    }
    out[i] = total;
}

extern "C" __global__
void composite(const double* weights, const double* bands, const double* fields,
               double* light, double* transmission, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double fill = weights[i], core = weights[size + i], rim = weights[2 * size + i];
    double tone = fields[i], opacity = fields[size + i], illumination = fields[2 * size + i];
    double depth = fields[3 * size + i], shadow = fields[4 * size + i];
    double colors[3];
    for (int c = 0; c < 3; ++c) {
        double color = (fill * bands[c] + core * bands[3 + c]) + rim * bands[6 + c];
        double shift = (fill * bands[9 + c] + core * bands[12 + c]) + rim * bands[15 + c];
        colors[c] = (color + tone * shift) / ((fill + core) + rim);
    }
    double largest = fmax(colors[0], fmax(colors[1], colors[2]));
    double weight = fmin(fmax(shadow, 0.0), 0.96);
    for (int c = 0; c < 3; ++c) {
        int at = 3 * i + c;
        light[at] += ((transmission[at] * opacity) * colors[c]) * illumination;
        double base = c == 0 ? 0.80 : (c == 1 ? 1.0 : 1.19);
        double selective = 0.81 + 0.82 * (1 - colors[c] / largest);
        double loss = (1 - weight) * base + weight * selective;
        transmission[at] *= (1 - opacity) * exp(-depth * loss);
    }
}
"""


CUDA_RASTER = r"""
__device__ double smooth(double low, double high, double value) {
    double t = fmin(fmax((value - low) / (high - low), 0.0), 1.0);
    return (t * t) * (3 - 2 * t);
}

__device__ double feather(double u, double v, double radius, double side) {
    double preferred = pow(fmax(0.0, cos(atan2(v, u) - side)), 6.0);
    return smooth(0, 0.57 - 0.42 * preferred, 1 - radius);
}

__device__ double dust_patch(const double* p, double u, double v, double detail) {
    v = (v + p[6] * u * u) / p[7];
    if (p[12] > 0) {
        double prominence = p[13];
        u = u / (1.29 + 0.60 * prominence) - 0.21 * p[14];
        v += p[14] * p[15] * u * u;
        double theta = atan2(v, u), distance = hypot(u, v);
        double outline = 0.97 + 0.18 * cos(2 * theta + p[8]);
        outline += 0.11 * sin(3 * theta + p[9]) + 0.06 * cos(7 * theta + p[10]);
        double diffuse = pow(fmax(0.0, 1 - distance * distance / (1.13 * 1.13)), 3.0);
        double shaped = feather(u, v, distance / outline, p[16]);
        shaped *= 0.49 + 0.51 * exp(-1.65 * distance * distance);
        double shape = (0.13 + 0.66 * prominence) * ((1 - prominence) * diffuse + prominence * shaped);
        return p[12] * (shape * detail);
    }
    double radius_sq = u * u + v * v, theta = atan2(v, u);
    double base = fmax(0.0, 1 - radius_sq);
    double contour = 0.62 + 0.14 * cos(2 * theta + p[8]);
    contour += 0.07 * sin(3 * theta + p[9]) + 0.035 * cos(7 * theta + p[10]);
    double delta = (sqrt(radius_sq) - contour) / 0.11;
    double arc = (exp(-(delta * delta)) * smooth(-0.7, 0.8, cos(theta + p[11]))) * (base * base);
    return p[12] * ((base * base * base) * detail + (0.68 * arc) * (detail * detail));
}

__device__ void splat_pixel(const double* patches, const int* offsets, const int* indices,
           const double* field_a, const double* field_b, double* out,
           int i, int x, int y, int bin, int stride, int mode) {
    double total = 0, edges[3] = {0, 0, 0};
    for (int cursor = offsets[bin]; cursor < offsets[bin + 1]; ++cursor) {
        const double* p = patches + stride * indices[cursor];
        double dx = x - p[0], dy = y - p[1];
        if (fabs(x - p[17]) > p[3] || fabs(y - p[18]) > p[3]) continue;
        dx /= p[2];
        dy /= p[2];
        double u = p[4] * dx + p[5] * dy, v = p[4] * dy - p[5] * dx;
        if (mode == 0) {
            total += dust_patch(p, u, v, 0.23 + 0.77 * field_a[i]);
        } else if (mode == 1) {
            v = ((v - 0.12) + p[6] * u * u) / 0.56;
            double radius_sq = u * u + v * v;
#ifdef WALLPAPER_FAST
            if (radius_sq >= 1 || v <= -0.50 || field_a[i] == 0 || field_b[i] == 0) continue;
            double radius = sqrt(radius_sq);
            double preference = radius > 0 ? fmax(0.0, -v / radius) : 0;
            double p2 = preference * preference;
            double body = smooth(0, 0.57 - 0.42 * ((p2 * p2) * p2), 1 - radius);
#else
            double body = feather(u, v, sqrt(radius_sq), -1.5707963267948966);
#endif
            body *= exp(-1.75 * radius_sq) * smooth(-0.50, 0.38, v);
            double side = field_a[i] * field_b[i];
            total += ((0.56 * p[7]) * body) * side;
        } else {
            u /= p[6];
            v *= p[6];
            double u_sq = u * u, v_sq = v * v;
            double radius_sq = u_sq + v_sq;
#ifdef WALLPAPER_FAST
            double base = 1 + radius_sq;
            double skew = u * (u_sq - 2.3 * v_sq) / (base * sqrt(base));
#else
            double skew = u * (u_sq - 2.3 * v_sq) / pow(1 + radius_sq, 1.5);
#endif
            radius_sq *= 1 + p[7] * skew;
            const double pi = 3.141592653589793;
            double psf = 0.951 * exp(-radius_sq / (2 * 0.85 * 0.85)) / (2 * pi * (0.85 * 0.85));
            psf += 0.042 * exp(-radius_sq / (2 * 2.2 * 2.2)) / (2 * pi * (2.2 * 2.2));
            psf += 0.007 * exp(-radius_sq / (2 * 5.2 * 5.2)) / (2 * pi * (5.2 * 5.2));
            double received = p[8] * psf;
            total += received;
            double wing = -expm1(-0.31 * (radius_sq * radius_sq));
            for (int c = 0; c < 3; ++c) edges[c] += (received * wing) * p[9 + c];
        }
    }
    if (mode == 2) {
        double brightness = -expm1(-total);
        double response = total > 0 ? brightness / total : 0;
        const double neutral[3] = {224, 229, 238};
        for (int c = 0; c < 3; ++c) out[3 * i + c] = brightness * neutral[c] + response * edges[c];
    } else {
        out[i] = tanh(total);
    }
}

extern "C" __global__
void splat(const double* patches, const int* offsets, const int* indices,
           const double* field_a, const double* field_b, double* out,
           int width, int height, int tile, int stride, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int x = i % width, y = i / width;
    int bin = (y / tile) * ((width + tile - 1) / tile) + x / tile;
    splat_pixel(patches, offsets, indices, field_a, field_b, out, i, x, y, bin, stride, mode);
}

extern "C" __global__
void splat_active(const double* patches, const int* offsets, const int* indices,
                  const double* field_a, const double* field_b, double* out,
                  int width, int height, int tile, int stride, int mode,
                  const int* active, int columns) {
    const int bin = active[blockIdx.x];
    const int left = (bin % columns) * tile, top = (bin / columns) * tile;
    const int x = left + threadIdx.x % tile;
    if (x >= width) return;
    for (int local_y = threadIdx.x / tile; local_y < tile; local_y += blockDim.x / tile) {
        const int y = top + local_y;
        if (y >= height) break;
        splat_pixel(patches, offsets, indices, field_a, field_b, out, y * width + x, x, y, bin, stride, mode);
    }
}

extern "C" __global__
void bilinear_f32(const float* input, float* output, const int* bounds,
                  const double* weights, int width, int height, int source_width,
                  int stride, int vertical) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * height) return;
    int x = i % width, y = i / width, at = vertical ? y : x;
    int start = bounds[2 * at], count = bounds[2 * at + 1];
    int base = vertical ? start * source_width + x : y * source_width + start;
    int step = vertical ? source_width : 1;
    double sum = 0;
    for (int tap = 0; tap < count; ++tap)
        sum += input[base + step * tap] * weights[at * stride + tap];
    output[i] = (float)sum;
}
"""


CUDA_PROJECTION = r"""
__device__ double octave_field(double x, double y, const double* gradients,
                               const double* weights, int octaves) {
    double total = 0;
    for (int octave = 0; octave < octaves; ++octave) {
        total += noise(x, y, gradients) * weights[octave];
        double next_x = 1.76 * x - 1.02 * y + 19;
        y = 1.02 * x + 1.76 * y + 31;
        x = next_x;
    }
    return total;
}

extern "C" __global__
void projection(const int* px, const int* py, const double* gradients, const double* weights,
                const double* geometry, double* output, int count, int width, int height, int sparse) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    double x = sparse ? px[i] : i % width, y = sparse ? py[i] : i / width;
    double sx = ((x - width / 2.0) / width) * geometry[5];
    double sy = ((y - height / 2.0) / width) * geometry[5];
    double u = sx * geometry[0] - sy * geometry[1], v = sx * geometry[1] + sy * geometry[0];
    double wx = octave_field(2.4 * u + geometry[2], 2.4 * v + geometry[3], gradients, weights, 5);
    double wy = octave_field(2.4 * u + geometry[3], (2.4 * v + geometry[2]) + 17, gradients, weights, 5);
    u += 0.33 * wx;
    v += 0.14 * wy;
    double mist = octave_field(4 * u + 52, 6 * v + 10, gradients, weights + 5, 4);
    double center = 0.037 * sin(6 * u + geometry[4]);
    output[i] = u;
    output[count + i] = (v - center) + 0.028 * mist;
    if (!sparse) {
        output[2 * count + i] = mist;
        output[3 * count + i] = octave_field(9 * u + 16, 13 * v - 38, gradients, weights + 9, 3);
    }
}
"""


def fp32_cuda(source):
    """Specialize the shared CUDA formulas, including constants and math overloads."""
    source = re.sub(r"\bdouble\b", "float", source)
    source = re.sub(r"\b(floor|fabs|hypot|atan2|sin|cos|sqrt|exp|expm1|pow|tanh|fmin|fmax)\(",
                    r"\1f(", source)
    return re.sub(r"(?<![\w.])(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?(?![\w.])", r"\g<0>f", source)



def star_tile_graph(patches, width, height, tile):
    """Build ordered tile adjacency in bulk from each star's bounding box."""
    cols, rows = ceil(width / tile), ceil(height / tile)
    reach = patches[:, 3].astype(np.int32)
    centers = patches[:, 17:19].astype(np.int32)
    lower = np.maximum(0, centers - reach[:, None]) // tile
    upper = np.minimum((width - 1, height - 1), centers + reach[:, None]) // tile
    sizes = upper - lower + 1
    dy, dx = np.mgrid[:sizes[:, 1].max(), :sizes[:, 0].max()]
    valid = (dx < sizes[:, 0, None, None]) & (dy < sizes[:, 1, None, None])
    tile_ids = ((lower[:, 1, None, None] + dy) * cols + lower[:, 0, None, None] + dx)[valid]
    stars = np.broadcast_to(np.arange(len(patches), dtype=np.int32)[:, None, None], valid.shape)[valid]
    offsets = np.empty(cols * rows + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(np.bincount(tile_ids, minlength=cols * rows), dtype=np.int32, out=offsets[1:])
    return offsets, stars[np.argsort(tile_ids, kind="stable")]


class CudaNoise:
    name = "cuda"

    def __init__(self, cupy, optimized=False, fast=False):
        self.cp = cupy
        self.optimized = optimized
        self.fast = fast
        self.dtype = np.float32 if fast else np.float64
        source = CUDA_NOISE + CUDA_RASTER
        if fast:
            # Pillow's filter weights and accumulation stay double precision.
            raster, resize = source.split('extern "C" __global__\nvoid bilinear_f32', maxsplit=1)
            source = ("#define WALLPAPER_FAST\n" + fp32_cuda(raster + CUDA_PROJECTION)
                      + 'extern "C" __global__\nvoid bilinear_f32' + resize)
            reference = cupy.RawModule(code=CUDA_NOISE + CUDA_PROJECTION, options=("--fmad=false",))
            self.sparse_projection_kernel = reference.get_function("projection")
        module = cupy.RawModule(code=source, options=("--fmad=false",))
        self.fractal_kernel = module.get_function("fractal")
        self.turbulence_kernel = module.get_function("turbulence")
        self.composite_kernel = module.get_function("composite")
        self.splat_kernel = module.get_function("splat")
        self.active_splat_kernel = module.get_function("splat_active") if optimized else None
        self.resize_kernel = module.get_function("bilinear_f32")
        if fast:
            self.dense_projection_kernel = module.get_function("projection")
            weights = [decay**octave for decay, count in ((0.53, 5), (0.41, 4), (0.38, 3))
                       for octave in range(count)]
            self.projection_weights = self.cp.asarray(weights, dtype=self.cp.float64)
        self.gradients = None
        self.device_gradients = None
        self.threads = int(cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())["warpSize"]) * 4
        self.resize_tables = {}
        if optimized:
            self.configure_layer_kernels()
        if fast:
            self.configure_fast_kernels()

    def configure_fast_kernels(self):
        self.dust_sites = self.cp.ElementwiseKernel(
            "raw float32 envelope, raw float32 compression, raw float32 core, raw float32 rim, "
            "int32 x, int32 y, int32 width, int32 height",
            "float32 probability, float32 pressure, float32 dy, float32 dx",
            r"""
            const int p = y * width + x;
            const int left = p - (x > 0), right = p + (x < width - 1);
            const int top = p - (y > 0 ? width : 0), bottom = p + (y < height - 1 ? width : 0);
            const float vl = core[left] - (0.58f * rim[left]) * (1 - core[left]);
            const float vr = core[right] - (0.58f * rim[right]) * (1 - core[right]);
            const float vt = core[top] - (0.58f * rim[top]) * (1 - core[top]);
            const float vb = core[bottom] - (0.58f * rim[bottom]) * (1 - core[bottom]);
            dx = (vr - vl) * (x > 0 && x < width - 1 ? 0.5f : 1.f);
            dy = (vb - vt) * (y > 0 && y < height - 1 ? 0.5f : 1.f);
            pressure = compression[p];
            probability = powf(envelope[p], 1.7f) * (0.18f + 0.82f * pressure);
            """,
            "wallpaper_v5_dust_sites", options=("--fmad=false",),
        )
        self.rotate_hue = self.cp.ElementwiseKernel(
            "float32 red, float32 green, float32 blue, float32 turns",
            "float32 out_red, float32 out_green, float32 out_blue",
            r"""
            const float high = fmaxf(red, fmaxf(green, blue));
            const float span = high - fminf(red, fminf(green, blue));
            const float safe = fmaxf(span, 1e-8f);
            const float sector = high == red ? (green - blue) / safe :
                (high == green ? 2 + (blue - red) / safe : 4 + (red - green) / safe);
            const float hue = sector + 6 * turns;
            out_red = hue_channel(high, span, hue + 5);
            out_green = hue_channel(high, span, hue + 3);
            out_blue = hue_channel(high, span, hue + 1);
            """,
            "wallpaper_v5_rotate_hue", options=("--fmad=false",),
            preamble=r"""
            __device__ float hue_channel(float high, float span, float phase) {
                phase -= 6.f * floorf(phase * (1.f / 6.f));
                return high - span * fminf(fmaxf(fminf(phase, 4 - phase), 0.f), 1.f);
            }
            """,
        )
        self.star_candidates = self.cp.ElementwiseKernel(
            "float64 along, float64 orbit, float64 volume, float64 absolute_g, float64 longitude_origin",
            "float64 distance, float64 weight, float64 apparent_g",
            r"""
            double sin_lat, cos_lat;
            sincos(1.1 * orbit, &sin_lat, &cos_lat);
            distance = cbrt(volume);
            const double plane = distance * cos_lat;
            const double radius = sqrt(64.0 + plane * plane - (16.0 * plane) * cos(longitude_origin + along));
            const double height = 0.024 + distance * sin_lat;
            const double thin = exp(-(radius - 8.0) / 2.6 - abs(height) / 0.3);
            const double thick = 0.12 * exp(-(radius - 8.0) / 3.6 - abs(height) / 0.9);
            const double halo = 0.0051 * pow(8.0 / hypot(radius, height / 0.64), 2.77);
            weight = (thin + thick + halo) * cos_lat;
            const double dust = fmax(abs(sin_lat) / 0.11, 1e-9);
            const double extinction = 0.55 * -expm1(-distance * dust) / dust;
            apparent_g = (absolute_g + 5.0 * log10(100.0 * distance)) + extinction;
            """,
            "wallpaper_v5_star_candidates", options=("--fmad=false",),
        )
        self.cloud_base = self.cp.ElementwiseKernel(
            "float32 u, float32 orbit, float32 mist, float32 dust, float32 sx, float32 sy, "
            "float32 offset, raw float32 palette",
            "float32 core, float32 gloss, raw float32 rgb",
            r"""
            const float h = (u - offset) / 0.44f, z = orbit / 0.093f;
            const float arm_coord = (orbit + 0.028f * sinf(10 * u)) / 0.045f;
            const float core_x = (u - offset) / 0.17f, core_y = (orbit - 0.008f) / 0.047f;
            const float halo = expf(-(h * h) - z * z), arm = expf(-(arm_coord * arm_coord));
            core = expf(-(core_x * core_x) - core_y * core_y);
            const float t = fminf(fmaxf((dust - 0.04f) / 0.33f, 0.f), 1.f);
            const float cover = ((t * t) * (3 - 2 * t)) * arm;
            const float density = fminf(fmaxf((((0.035f + halo * (0.13f + 0.15f * mist)) +
                                  arm * (0.07f + 0.07f * mist)) + 0.19f * core) - 0.08f * cover, 0.f), 1.f);
            const float stops[5] = {0.f, 0.12f, 0.26f, 0.42f, 0.62f};
            int lo = density < stops[1] ? 0 : (density < stops[2] ? 1 : (density < stops[3] ? 2 : 3));
            const float blend = fminf(1.f, (density - stops[lo]) / (stops[lo + 1] - stops[lo]));
            for (int channel = 0; channel < 3; ++channel) {
                float low = palette[3 * lo + channel], high = palette[3 * (lo + 1) + channel];
                rgb[3 * i + channel] = low + (high - low) * blend;
            }
            const float g = ((0.52f * sx + sy) + 0.12f) / 0.15f;
            gloss = expf(-(g * g));
            """,
            "wallpaper_v5_cloud_base", options=("--fmad=false",),
        )
        self.relief_front = self.cp.ElementwiseKernel(
            "raw float32 coarse, raw float32 band, float32 core, float32 rim, float32 lit, "
            "int32 width, int32 height, float32 scaled_width",
            "float32 dy, float32 dx, float32 front, float32 light_side, float32 transition",
            r"""
            const int x = i % width, y = i / width;
            const auto left = i - (x > 0), right = i + (x < width - 1);
            const auto top = i - (y > 0 ? width : 0), bottom = i + (y < height - 1 ? width : 0);
            const float xscale = x > 0 && x < width - 1 ? 0.5f : 1.f;
            const float yscale = y > 0 && y < height - 1 ? 0.5f : 1.f;
            dx = (coarse[right] - coarse[left]) * xscale;
            dy = (coarse[bottom] - coarse[top]) * yscale;
            const float bx = (band[right] - band[left]) * xscale;
            const float by = (band[bottom] - band[top]) * yscale;
            const float length = hypotf(dx, dy);
            const float a = fabsf(dx * bx + dy * by) / fmaxf(length * hypotf(bx, by), 1e-10f);
            const float t = fminf(fmaxf((coarse[i] - 0.14f) / 0.25f, 0.f), 1.f);
            const float bright = (t * t) * (3 - 2 * t);
            const float z = fminf(fmaxf((length * scaled_width - 0.18f) / 1.27f, 0.f), 1.f);
            transition = fmaxf((4 * core) * (1 - core), ((4 * rim) * (1 - rim)) * (1 - core));
            front = (((bright * lit) * transition) * (0.45f + 0.55f * a)) * ((z * z) * (3 - 2 * z));
            const float side = fminf(fmaxf((bright - 0.39f) / 0.39f, 0.f), 1.f);
            light_side = (side * side) * (3 - 2 * side);
            """,
            "wallpaper_v5_relief_front", options=("--fmad=false",),
        )

    def configure_layer_kernels(self):
        self.layer_geometry = self.elementwise(
            "float64 wx, float64 wy, float64 slab, float64 folds, float64 broad, float32 stars, "
            "float64 cx, float64 cy, float64 sx, float64 sy, float64 tilt, "
            "float64 ox, float64 oy, float64 dx, float64 dy, float64 strength, float64 cavity_strength",
            "float64 envelope, float64 cavity, float64 compression, float64 damping, float64 core, float64 rim, "
            "float64 diffuse, float64 definition, float64 broad_light, float64 lit",
            r"""
            const double local_x = (wx - cx) + 0.021 * slab;
            const double local_y = ((wy - cy) + 0.034 * slab) - tilt * local_x;
            const double u = local_x / sx, v = local_y / sy;
            envelope = exp(-0.62 * pow(hypot(u, v), 2.15));
            const double cavity_radius = hypot((u - ox) / dx, ((v - oy) + 0.24 * folds) / dy);
            cavity = exp(-0.60 * (cavity_radius * cavity_radius));
            const double wall = (cavity_radius - 1.25) / 0.56;
            compression = exp(-0.5 * (wall * wall)) *
                          ((0.72 + 0.28 * tanh(2.3 * folds + slab)) * (0.56 + 0.44 * cavity_strength));
            const float star_damping = 0.074f * (1.0f - stars);
            damping = ((0.075 + (0.065 * cavity) * cavity_strength) - 0.066 * compression) + star_damping;
            const double core_x = (u + 0.36) + 0.40 * slab, core_y = v - 0.28;
            core = exp(-0.95 * (core_x * core_x + core_y * core_y));
            const double rim_t = fmin(fmax((((0.44 * u - 0.34 * v) + 0.68 * slab) + 0.42) / 0.90, 0.0), 1.0);
            rim = (rim_t * rim_t) * (3 - 2 * rim_t);
            diffuse = ((strength * envelope) * exp(0.49 * broad + 0.37 * slab)) * (1 - cavity_strength * cavity);
            const double d = fmin(fmax((diffuse - 0.11) / 0.66, 0.0), 1.0);
            const float star_definition = 0.54f + 0.46f * stars;
            definition = ((d * d) * (3 - 2 * d)) * star_definition;
            const double exposure_x = u + 0.28, exposure_y = v - 0.35;
            broad_light = exp(-0.50 * (exposure_x * exposure_x + exposure_y * exposure_y));
            const double b = fmin(fmax((broad_light - 0.25) / 0.56, 0.0), 1.0);
            lit = definition * ((b * b) * (3 - 2 * b));
            """,
            "wallpaper_v4_geometry", options=("--fmad=false",),
        )
        self.layer_density = self.elementwise(
            "float64 diffuse, float64 definition, float64 turbulent, float64 details, float64 envelope, "
            "float64 cavity, float64 dust, float64 slab, float64 compression, float64 cavity_strength, "
            "float64 broad_light, raw float64 transmission",
            "float64 opacity, float64 dust_depth, float64 visible",
            r"""
            const double gas = diffuse * exp((0.30 + 0.79 * definition) * turbulent -
                                               (0.13 + 0.75 * definition) * details);
            opacity = -expm1(-gas);
            dust_depth = (envelope * (0.105 + (0.43 * cavity_strength) * cavity)) *
                exp(((0.25 * dust + 0.37 * slab) + (0.39 + 0.30 * compression) * turbulent) + 1.35 * details);
            const auto k = 3 * i;
            const double mean = ((transmission[k] + transmission[k + 1]) + transmission[k + 2]) / 3;
            visible = ((opacity * broad_light) * mean) * exp(-dust_depth);
            """,
            "wallpaper_v4_density", options=("--fmad=false",),
        )
        self.layer_composite = self.elementwise(
            "float64 core, float64 rim, float64 definition, float64 exposure, float64 relief, "
            "float64 opacity, float64 dust, float64 cavity, float64 details, float64 cavity_strength, "
            "raw float64 bands",
            "raw float64 light, raw float64 transmission, float32 dust_column",
            r"""
            const double illumination = (0.74 + 0.43 * exposure) + 0.64 * relief;
            const double q = fmin(fmax(((0.25 + 0.39 * exposure) + 0.20 * definition) + 0.34 * relief, 0.0), 1.0);
            const double tone = (q * q) * (3 - 2 * q);
            const double rim_weight = (1 - core) * rim;
            const double f = pow((1 - core) - rim_weight, 2.3);
            const double c = pow(core, 2.3), r = pow(rim_weight, 2.3);
            const double denominator = (f + c) + r;
            double color[3];
            for (int channel = 0; channel < 3; ++channel) {
                const double base = (f * bands[channel] + c * bands[3 + channel]) + r * bands[6 + channel];
                const double shift = (f * bands[9 + channel] + c * bands[12 + channel]) + r * bands[15 + channel];
                color[channel] = (base + tone * shift) / denominator;
            }
            const double largest = fmax(color[0], fmax(color[1], color[2]));
            const double shadow = (0.48 * cavity_strength) * cavity + 1.65 * fmax(details, 0.0);
            const double weight = fmin(fmax(shadow, 0.0), 0.96);
            for (int channel = 0; channel < 3; ++channel) {
                const auto at = 3 * i + channel;
                const double base = channel == 0 ? 0.80 : (channel == 1 ? 1.0 : 1.19);
                const double selective = 0.81 + 0.82 * (1 - color[channel] / largest);
                const double loss = (1 - weight) * base + weight * selective;
                light[at] += ((transmission[at] * opacity) * color[channel]) * illumination;
                transmission[at] *= (1 - opacity) * exp(-dust * loss);
            }
            dust_column = (float)((double)dust_column + dust);
            """,
            "wallpaper_v4_composite", options=("--fmad=false",),
        )

    def elementwise(self, inputs, outputs, operation, name, options):
        if self.fast:
            inputs, outputs = (spec.replace("float64", "float32") for spec in (inputs, outputs))
            operation = fp32_cuda(operation)
            name += "_f32"
        return self.cp.ElementwiseKernel(inputs, outputs, operation, name, options=options)

    def array(self, data):
        return self.cp.asarray(data, dtype=self.dtype, order="C")

    def run(self, kernel, x, y, gradients, fields, scalars):
        if self.gradients is not gradients:
            self.gradients = gradients
            self.device_gradients = self.array(gradients)
        out = self.cp.empty(x.shape, dtype=self.dtype)
        args = (self.array(x), self.array(y), self.device_gradients,
                *(self.array(field) for field in fields), out, np.int32(out.size), *scalars)
        kernel((ceil(out.size / self.threads),), (self.threads,), args)
        return out

    def fractal(self, x, y, gradients, octaves=5, decay=0.53):
        weights = [decay**octave for octave in range(octaves)]
        return self.run(self.fractal_kernel, x, y, gradients, (weights,), (np.int32(octaves),))

    def hue_shift(self, rgb, turns):
        out = self.cp.empty_like(rgb)
        self.rotate_hue(*(rgb[..., i] for i in range(3)), self.cp.asarray(turns, dtype=self.cp.float32),
                        *(out[..., i] for i in range(3)))
        return out

    def sample_dust_sites(self, envelope, compression, bands, x, y):
        height, width = envelope.shape
        x, y = (self.cp.asarray(axis, dtype=self.cp.int32) for axis in (x, y))
        return self.dust_sites(envelope, compression, *bands, x, y, width, height)

    def projection(self, shape, gradients, geometry, coordinates=None):
        height, width = shape
        sparse = coordinates is not None
        dtype = np.float64 if sparse else np.float32
        count = len(coordinates[0]) if sparse else height * width
        out = self.cp.empty((2 if sparse else 4, count), dtype=dtype)
        px, py = coordinates if sparse else ([], [])
        kernel = self.sparse_projection_kernel if sparse else self.dense_projection_kernel
        args = (self.cp.asarray(px, dtype=np.int32), self.cp.asarray(py, dtype=np.int32),
                self.cp.asarray(gradients, dtype=dtype), self.projection_weights.astype(dtype, copy=False),
                self.cp.asarray(geometry, dtype=dtype), out,
                np.int32(count), np.int32(width), np.int32(height), np.int32(sparse))
        kernel((ceil(count / self.threads),), (self.threads,), args)
        return out if sparse else out.reshape(4, height, width)

    def turbulence(self, x, y, gradients, outer_scale, damping, step):
        frequencies = [1.0]
        for _ in range(6):
            frequencies.append(frequencies[-1] * step)
        weights = [frequency**(-1 / 3) for frequency in frequencies]
        return self.run(self.turbulence_kernel, x, y, gradients, (damping, frequencies, weights),
                        (self.dtype(outer_scale), self.dtype(step)))

    def composite(self, light, transmission, weights, bands, fields):
        size = weights[0].size
        arrays = [self.cp.stack([self.array(plane).ravel() for plane in inputs])
                  for inputs in (weights, fields)]
        args = (arrays[0], self.array(bands), arrays[1], light, transmission, np.int32(size))
        self.composite_kernel((ceil(size / self.threads),), (self.threads,), args)

    def splat(self, shape, patches, mode, field_a=None, field_b=None):
        height, width = shape
        tile = self.threads // 4
        if self.fast and mode == 2 and len(patches):
            tile //= 2
            offsets, flat = star_tile_graph(patches, width, height, tile)
        else:
            cols, rows = ceil(width / tile), ceil(height / tile)
            bins = [[] for _ in range(cols * rows)]
            for index, patch in enumerate(patches):
                reach, x, y = int(patch[3]), int(patch[17]), int(patch[18])
                for cy in range(max(0, y - reach) // tile, min(height - 1, y + reach) // tile + 1):
                    for cx in range(max(0, x - reach) // tile, min(width - 1, x + reach) // tile + 1):
                        bins[cy * cols + cx].append(index)
            offsets = np.cumsum([0, *(len(bucket) for bucket in bins)], dtype=np.int32)
            flat = np.fromiter((index for bucket in bins for index in bucket), dtype=np.int32)
        out = self.cp.empty((*shape, 3) if mode == 2 else shape, dtype=self.dtype)
        zero = self.cp.empty(0, dtype=self.dtype)
        args = (self.array(patches), self.cp.asarray(offsets), self.cp.asarray(flat),
                zero if field_a is None else self.array(field_a), zero if field_b is None else self.array(field_b),
                out, np.int32(width), np.int32(height), np.int32(tile), np.int32(19), np.int32(mode))
        if self.optimized and mode != 2:
            active = np.flatnonzero(offsets[1:] != offsets[:-1]).astype(np.int32)
            out.fill(0)
            if active.size:
                self.active_splat_kernel((active.size,), (self.threads,),
                                         (*args, self.cp.asarray(active), np.int32(cols)))
        else:
            self.splat_kernel((ceil(width * height / self.threads),), (self.threads,), args)
        return out

    def bilinear(self, field, size):
        field = self.cp.asarray(field, dtype=self.cp.float32, order="C")
        for vertical, target in ((0, size[0]), (1, size[1])):
            source = field.shape[0 if vertical else 1]
            if source == target:
                continue
            key = source, target
            if key not in self.resize_tables:
                scale = source / target
                support = max(scale, 1)
                stride = 2 * ceil(support) + 1
                bounds = np.empty((target, 2), dtype=np.int32)
                weights = np.zeros((target, stride))
                for index in range(target):
                    center = (index + 0.5) * scale
                    start = max(0, int(center - support + 0.5))
                    stop = min(source, int(center + support + 0.5))
                    taps = [max(0, 1 - abs((pos - center + 0.5) * (1 / support)))
                            for pos in range(start, stop)]
                    total = sum(taps)
                    bounds[index] = start, len(taps)
                    weights[index, :len(taps)] = [tap / total for tap in taps]
                self.resize_tables[key] = self.cp.asarray(bounds), self.cp.asarray(weights)
            bounds, weights = self.resize_tables[key]
            height, width = (target, field.shape[1]) if vertical else (field.shape[0], target)
            out = self.cp.empty((height, width), dtype=self.cp.float32)
            self.resize_kernel((ceil(out.size / self.threads),), (self.threads,),
                               (field, out, bounds, weights, np.int32(width), np.int32(height),
                                np.int32(field.shape[1]), np.int32(weights.shape[1]), np.int32(vertical)))
            field = out
        return field


class ThreadedNoise:
    name = "cpu"

    def __init__(self, pool, fractal, turbulence):
        self.pool = pool
        self.reference_fractal = fractal
        self.reference_turbulence = turbulence
        self.workers = os.process_cpu_count() or 1

    def rows(self, x, task):
        # Keep per-thread intermediates small enough to reuse cache through the octaves.
        stride = max(1, min(ceil(x.shape[0] / self.workers), 98_304 // x.shape[1]))
        result = np.empty_like(x)
        spans = (slice(start, min(start + stride, x.shape[0])) for start in range(0, x.shape[0], stride))

        def fill(rows):
            result[rows] = task(rows)

        list(self.pool.map(fill, spans))
        return result

    def fractal(self, x, y, gradients, octaves=5, decay=0.53):
        return self.rows(x, lambda rows: self.reference_fractal(x[rows], y[rows], gradients, octaves, decay))

    def turbulence(self, x, y, gradients, outer_scale, damping, step):
        return self.rows(x, lambda rows: self.reference_turbulence(x[rows], y[rows], gradients,
                                                                  outer_scale, damping[rows], step))


def cpu_pool():
    return ThreadPoolExecutor(max_workers=os.process_cpu_count() or 1)


def cuda_noise(optimized=False, fast=False):
    import cupy
    from cupy.cuda.compiler import CompileException

    try:
        return CudaNoise(cupy, optimized, fast)
    except CompileException as error:
        raise RuntimeError(f"NVRTC could not compile the wallpaper kernels: {error}") from error


def median_cloud_hue(rgb, xp=np):
    """Estimate the visible cloud's circular median on a coarse pixel grid."""
    step = max(1, min(rgb.shape[:2]) // 80)
    sample = rgb[step // 2::step, step // 2::step]
    if xp is not np:
        sample = xp.ascontiguousarray(sample).get()
    sample = sample.reshape(-1, 3).astype(np.float32, copy=False)
    peak = sample.max(axis=-1)
    spread = peak - sample.min(axis=-1)
    visible = (peak >= 26) & (peak < 225) & (spread >= 0.16 * peak)
    sample, peak, spread = sample[visible], peak[visible], spread[visible]
    if not sample.size:
        return None
    red, green, blue = sample.T
    hue = np.where(peak == red, (green - blue) / spread,
                   np.where(peak == green, 2 + (blue - red) / spread, 4 + (red - green) / spread)) / 6
    center = float(np.angle(np.exp(2j * np.pi * hue).mean()) / (2 * np.pi))
    return (center + float(np.median((hue - center + 0.5) % 1 - 0.5))) % 1
