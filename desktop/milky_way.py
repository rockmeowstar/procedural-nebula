"""Draw a synthetic optical star field along a warped Galactic midplane."""

import numpy as np
from PIL import Image


# Gaia GCNS J/A+A/649/A6/table1c, 0 < Dist50 < 0.1 kpc, measured Gmag.
# Counts in one-magnitude bins from absolute G = -3 to 22 (300,185 sources).
GCNS_COUNTS = np.array([
    7, 43, 150, 744, 1126, 2915, 6689, 9775, 12382, 13541, 15586, 21690, 34705,
    48603, 46255, 34423, 22748, 16851, 9000, 2402, 387, 112, 27, 20, 4,
])
GCNS_PROBABILITIES = GCNS_COUNTS / GCNS_COUNTS.sum()
GCNS_CDF = GCNS_PROBABILITIES.cumsum()
GCNS_CDF /= GCNS_CDF[-1]
REFERENCE_STAR_COUNT = 19_000


def stellar_density(radius, height, xp=np):
    """Relative number density at Galactocentric radius and height in kpc."""
    # Jurić et al. (2008), Table 10, with the local thin disk as the unit.
    thin = xp.exp(-(radius - 8.0) / 2.6 - xp.abs(height) / 0.3)
    thick = 0.12 * xp.exp(-(radius - 8.0) / 3.6 - xp.abs(height) / 0.9)
    halo_radius = xp.hypot(radius, height / 0.64)
    halo = 0.0051 * (8.0 / halo_radius)**2.77
    return thin + thick + halo


def weighted_sample(weights, count, rng, xp):
    """Match NumPy's weighted sampling order while scanning the CDF on CUDA."""
    weights = weights.copy()
    chosen = []
    remaining = count
    while remaining:
        thresholds = rng.random(remaining)
        cdf = xp.cumsum(weights)
        selected = xp.searchsorted(cdf, xp.asarray(thresholds) * cdf[-1], side="right").get()
        _, first = np.unique(selected, return_index=True)
        selected = selected[np.sort(first)]
        weights[selected] = 0
        chosen.append(selected)
        remaining -= selected.size
    return np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)


def sample_visible_stars(along, orbit, rng, *, xp=np, projection=None, evaluate=None,
                         star_count=REFERENCE_STAR_COUNT):
    height, width = orbit.shape
    # Sample screen positions uniformly and distances with the radial volume factor.
    x = rng.uniform(0, width, 900_000)
    y = rng.uniform(0, height, x.size)
    px, py = x.astype(int), y.astype(int)
    star_along, star_orbit = projection(px, py) if projection else (along[py, px], orbit[py, px])
    volume = xp.asarray(rng.uniform(0.03**3, 12.0**3, x.size))
    if projection is not None:
        magnitude_bin = xp.searchsorted(xp.asarray(GCNS_CDF), xp.asarray(rng.random(x.size)), side="right")
        absolute_g = magnitude_bin - 3 + xp.asarray(rng.random(x.size))
    else:
        magnitude_bin = rng.choice(GCNS_COUNTS.size, size=x.size, p=GCNS_PROBABILITIES)
        absolute_g = xp.asarray(magnitude_bin - 3 + rng.random(x.size))
    if evaluate is not None:
        distance, weights, apparent_g = evaluate(star_along, star_orbit, volume, absolute_g, np.deg2rad(60))
    else:
        longitude = np.deg2rad(60) + star_along
        latitude = 1.1 * star_orbit
        distance = xp.cbrt(volume)
        plane_distance = distance * xp.cos(latitude)
        radius = xp.sqrt(8.0**2 + plane_distance**2 - 16.0 * plane_distance * xp.cos(longitude))
        height_kpc = 0.024 + distance * xp.sin(latitude)
        weights = stellar_density(radius, height_kpc, xp) * xp.cos(latitude)
        # Integrate an illustrative exponential dust layer along the line of sight.
        dust_rate = xp.maximum(xp.abs(xp.sin(latitude)) / 0.11, 1e-9)
        extinction = 0.55 * -xp.expm1(-distance * dust_rate) / dust_rate
        apparent_g = absolute_g + 5 * xp.log10(100 * distance) + extinction
    eligible = xp.flatnonzero(apparent_g < 22.8)
    eligible_weights = weights[eligible]
    count = min(star_count, eligible.size)
    if projection is not None:
        chosen = eligible[weighted_sample(eligible_weights, count, rng, xp)].get()
    else:
        if xp is not np:
            eligible, eligible_weights = eligible.get(), eligible_weights.get()
        probabilities = eligible_weights / eligible_weights.sum()
        chosen = rng.choice(eligible, size=count, replace=False, p=probabilities)
    magnitudes, distances = apparent_g[chosen], distance[chosen]
    if xp is not np:
        magnitudes, distances = magnitudes.get(), distances.get()
    return x[chosen], y[chosen], magnitudes, distances


def star_density_map(stars, shape, *, compute=None, scene_scale=1.0):
    height, width = shape
    x, y = stars[:2]
    bin_width = 20 * scene_scale
    coarse, _, _ = np.histogram2d(y, x, bins=(max(2, int(height / bin_width)), max(2, int(width / bin_width))),
                                  range=((0, height), (0, width)))
    axis = np.arange(-6, 7)
    kernel = np.exp(-0.5 * (axis / 2.1)**2)
    kernel /= kernel.sum()
    for direction in (0, 1):
        coarse = np.apply_along_axis(lambda p: np.convolve(np.pad(p, 6, mode="edge"), kernel, mode="valid"),
                                    direction, coarse)
    low, high = np.percentile(coarse, (12, 87))
    normalized = np.clip((coarse - low) / max(high - low, 1e-6), 0, 1)
    if compute is not None:
        return compute.bilinear(normalized, (width, height))
    return np.asarray(Image.fromarray(normalized.astype(np.float32)).resize((width, height), Image.Resampling.BILINEAR))


def optical_radius(dx, dy, angle, stretch, asymmetry):
    # An equal-area elliptical core, with a gentle asymmetric perturbation.
    u = (np.cos(angle) * dx + np.sin(angle) * dy) / stretch
    v = (np.cos(angle) * dy - np.sin(angle) * dx) * stretch
    radius_sq = u**2 + v**2
    skew = u * (u**2 - 2.3 * v**2) / (1 + radius_sq)**1.5
    return radius_sq * (1 + asymmetry * skew)


def star_field(stars, dust_column, cloud_light, rng, *, compute=None, scene_scale=1.0):
    x, y, apparent_g, distance = stars
    height, width = dust_column.shape
    # Gaussian approximation to the HST WFC3/UVIS point response. Pixel scale and
    # exposure are illustrative; unresolved stars do not have rendered disks.
    sigma = 0.85
    behind_cloud = -np.expm1(-(np.maximum(distance - 0.7, 0) / 2.2)**1.5)
    px, py = x.astype(int), y.astype(int)
    local_dust = dust_column[py, px]
    local_light = cloud_light[py, px]
    if compute is not None:
        local_dust, local_light = local_dust.get(), local_light.get()
    extinction = local_dust * behind_cloud
    flux = 10**(-0.4 * (apparent_g - 18.85)) * np.exp(-extinction)
    neutral = np.array((224, 229, 238))
    luminance_weights = np.array((0.2126, 0.7152, 0.0722))
    luminance = local_light @ luminance_weights
    local_color = local_light * ((neutral @ luminance_weights) / np.maximum(luminance, 1))[..., None]
    tint = 0.47 * behind_cloud * -np.expm1(-luminance / 38)
    edge_colors = (local_color - neutral) * tint[..., None]
    angles = rng.uniform(0, 2 * np.pi, x.size)
    stretches = 1 + rng.uniform(0.035, 0.13, x.size) * (1 - 0.35 * behind_cloud)
    asymmetries = rng.uniform(0.045, 0.15, x.size) * (1 - 0.35 * behind_cloud)
    margins = np.ceil(np.where(flux > 18, 16, 6) * scene_scale)
    if compute is not None:
        patches = np.zeros((x.size, 19))
        patches[:, 0], patches[:, 1] = x, y
        patches[:, 2] = scene_scale
        patches[:, 3] = margins
        patches[:, 4], patches[:, 5] = np.cos(angles), np.sin(angles)
        patches[:, 6], patches[:, 7], patches[:, 8] = stretches, asymmetries, flux
        patches[:, 9:12] = edge_colors
        patches[:, 17], patches[:, 18] = np.rint(x), np.rint(y)
        return compute.splat((height, width), patches, 2)
    signal = np.zeros((height, width))
    colored_edges = np.zeros((height, width, 3))
    for sx, sy, photons, edge_color, angle, stretch, asymmetry, margin in zip(
        x, y, flux, edge_colors, angles, stretches, asymmetries, margins.astype(int), strict=True
    ):
        ix, iy = round(sx), round(sy)
        left, right = max(0, ix - margin), min(width, ix + margin + 1)
        top, bottom = max(0, iy - margin), min(height, iy + margin + 1)
        dy, dx = np.ogrid[top:bottom, left:right]
        radius_sq = optical_radius((dx - sx) / scene_scale, (dy - sy) / scene_scale, angle, stretch, asymmetry)
        psf = 0.951 * np.exp(-radius_sq / (2 * sigma**2)) / (2 * np.pi * sigma**2)
        psf += 0.042 * np.exp(-radius_sq / (2 * 2.2**2)) / (2 * np.pi * 2.2**2)
        psf += 0.007 * np.exp(-radius_sq / (2 * 5.2**2)) / (2 * np.pi * 5.2**2)
        received = photons * psf
        signal[top:bottom, left:right] += received
        # Taper the local color into the wings and leave the apparent core neutral.
        wing = -np.expm1(-0.31 * radius_sq**2)
        colored_edges[top:bottom, left:right] += (received * wing)[..., None] * edge_color
    brightness = -np.expm1(-signal)
    response = brightness / np.maximum(signal, np.finfo(float).tiny)
    return brightness[..., None] * neutral + response[..., None] * colored_edges
