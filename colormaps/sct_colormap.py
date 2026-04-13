#!/usr/bin/env python3
"""
SCT Colormap — Perceptually Uniform Colormaps from the Sieve of Eratosthenes
=======================================================================

Generates colormaps as geodesic paths on the simplex Δ² with monotonic
luminance ℓ. Unlike viridis/inferno (calibrated on CIELAB), these are
uniform in the Fisher metric — by construction, not by fitting.

Conservation: S + L = log₂(3) at every point guarantees no dead zones.

Usage:
    python sct_colormap.py              # Generate all colormaps + comparison
    python sct_colormap.py --demo       # Apply to real scientific data
"""

import numpy as np
import sys, os

# Add SCT engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'PT_PROJECTS', 'SCT_GRADE_APP'))
from sct_core import sct_to_rgb, GAMMAS, M_HPE, M_HPE_INV, M_SRGB_TO_XYZ, M_XYZ_TO_SRGB, LOG2_3


# ============================================================
# SIMPLEX GEOMETRY
# ============================================================

def simplex_vertices(chrom_strength=0.5):
    """
    In-gamut vertices of Δ² for colormap construction.

    chrom_strength controls how far from neutral the vertices are:
      0.0 = achromatic (grayscale)
      0.5 = moderate chromaticity (default, high readability)
      1.0 = maximum in-gamut chromaticity

    PT prescribes 3/4 luminance, 1/4 chromaticity — so moderate
    chromaticity (0.3-0.5) maximizes readability.
    """
    from sct_core import rgb_to_sct
    neutral = np.array([1/3, 1/3, 1/3])

    # Maximum in-gamut chromaticity vertices
    _, _, pi_r_max = rgb_to_sct(np.array([0.70, 0.30, 0.25]))
    _, _, pi_g_max = rgb_to_sct(np.array([0.30, 0.65, 0.30]))
    _, _, pi_b_max = rgb_to_sct(np.array([0.25, 0.35, 0.75]))

    def blend(pi_max):
        pi = neutral + chrom_strength * (pi_max - neutral)
        pi = np.maximum(pi, 1e-12)
        return pi / pi.sum()

    return {
        'red':   blend(pi_r_max),
        'green': blend(pi_g_max),
        'blue':  blend(pi_b_max),
    }


def geodesic_path(pi_start, pi_end, n_steps):
    """
    Geodesic interpolation on the simplex Δ² using the Fisher-Rao metric.
    Uses the square-root (Bhattacharyya) embedding: ξ = √π.
    The geodesic on the positive orthant of S² is a great circle arc.

    Reparameterized by arc length so that successive points are
    equidistant in Fisher metric (uniform perceptual steps).
    """
    xi_start = np.sqrt(np.maximum(pi_start, 1e-12))
    xi_end = np.sqrt(np.maximum(pi_end, 1e-12))

    xi_start = xi_start / np.linalg.norm(xi_start)
    xi_end = xi_end / np.linalg.norm(xi_end)

    cos_omega = np.clip(np.dot(xi_start, xi_end), -1, 1)
    omega = np.arccos(cos_omega)

    if omega < 1e-10:
        return np.tile(pi_start, (n_steps, 1))

    # On the unit sphere with Bhattacharyya embedding, the geodesic IS
    # parameterized by arc length (great circle). So uniform t gives
    # uniform Bhattacharyya distance. But we need uniform FISHER distance
    # which includes the γ_p weighting.
    #
    # Strategy: oversample, compute cumulative Fisher arc length,
    # then resample at uniform arc length intervals.
    n_over = n_steps * 10
    path_over = np.zeros((n_over, 3))
    for i in range(n_over):
        t = i / (n_over - 1)
        xi = (np.sin((1 - t) * omega) * xi_start + np.sin(t * omega) * xi_end) / np.sin(omega)
        pi = xi ** 2
        pi = np.maximum(pi, 1e-12)
        path_over[i] = pi / pi.sum()

    # Compute cumulative weighted Fisher arc length
    arc_len = np.zeros(n_over)
    for i in range(1, n_over):
        # γ-weighted Fisher: ds² = Σ γ_p / π_p · dπ_p²
        dpi = path_over[i] - path_over[i-1]
        pi_mid = 0.5 * (path_over[i] + path_over[i-1])
        ds2 = 0
        for k in range(3):
            ds2 += GAMMAS[k] / max(pi_mid[k], 1e-12) * dpi[k]**2
        arc_len[i] = arc_len[i-1] + np.sqrt(max(ds2, 0))

    total_len = arc_len[-1]
    if total_len < 1e-12:
        return np.tile(pi_start, (n_steps, 1))

    # Resample at uniform arc length
    target_lengths = np.linspace(0, total_len, n_steps)
    path = np.zeros((n_steps, 3))
    for i, tgt in enumerate(target_lengths):
        idx = np.searchsorted(arc_len, tgt)
        idx = min(idx, n_over - 1)
        path[i] = path_over[idx]

    return path


def make_colormap(pi_path, ell_start=0.08, ell_end=0.92, n=256):
    """
    Convert a geodesic path on Δ² + monotonic ℓ → sRGB colormap.

    Coupled reparameterization: the 4D path (ℓ, π) is oversampled,
    then the cumulative PT Fisher distance is computed at each step:
        d² = (3/4)·d_lum² + (1/4)·d_chrom²
    The path is resampled at equal Fisher distance intervals.
    This eliminates dead zones by construction.
    """
    n_over = n * 20  # oversample

    # Generate oversampled path with arcsin luminance
    theta_start = np.arcsin(np.sqrt(max(ell_start, 0.001)))
    theta_end = np.arcsin(np.sqrt(min(ell_end, 0.999)))

    ells_over = np.zeros(n_over)
    pis_over = np.zeros((n_over, 3))
    for i in range(n_over):
        t = i / (n_over - 1)
        theta = theta_start + t * (theta_end - theta_start)
        ells_over[i] = np.sin(theta) ** 2
        # Map t to pi_path index (pi_path has n points)
        pi_idx = t * (len(pi_path) - 1)
        idx = int(pi_idx)
        frac = pi_idx - idx
        if idx >= len(pi_path) - 1:
            pis_over[i] = pi_path[-1]
        else:
            pis_over[i] = (1 - frac) * pi_path[idx] + frac * pi_path[idx + 1]
            pis_over[i] = np.maximum(pis_over[i], 1e-12)
            pis_over[i] /= pis_over[i].sum()

    # Compute cumulative coupled Fisher distance
    arc_len = np.zeros(n_over)
    for i in range(1, n_over):
        # Luminance: Fisher-Bernoulli
        e1, e2 = max(ells_over[i-1], 0.001), max(ells_over[i], 0.001)
        d_lum = 2 * abs(np.arcsin(np.sqrt(e2)) - np.arcsin(np.sqrt(e1)))

        # Chromaticity: γ-weighted Fisher on Δ²
        dpi = pis_over[i] - pis_over[i-1]
        pi_mid = 0.5 * (pis_over[i] + pis_over[i-1])
        ds2_chrom = 0
        for k in range(3):
            ds2_chrom += GAMMAS[k] / max(pi_mid[k], 1e-12) * dpi[k]**2
        d_chrom = np.sqrt(max(ds2_chrom, 0))

        # PT combined: d² = (3/4)·d_lum² + (1/4)·d_chrom²
        d_total = np.sqrt(0.75 * d_lum**2 + 0.25 * d_chrom**2)
        arc_len[i] = arc_len[i-1] + d_total

    total_len = arc_len[-1]
    if total_len < 1e-12:
        # Flat path — just return uniform
        colors = np.zeros((n, 3))
        for i in range(n):
            colors[i] = np.clip(sct_to_rgb(ells_over[0], pis_over[0]), 0, 1)
        return colors

    # Resample at uniform Fisher distance
    target = np.linspace(0, total_len, n)
    colors = np.zeros((n, 3))
    for i, tgt in enumerate(target):
        idx = np.searchsorted(arc_len, tgt)
        idx = min(max(idx, 0), n_over - 1)
        # Linear interpolation between bracketing points
        if idx > 0 and idx < n_over and arc_len[idx] != arc_len[idx-1]:
            frac = (tgt - arc_len[idx-1]) / (arc_len[idx] - arc_len[idx-1])
            frac = np.clip(frac, 0, 1)
            ell = (1 - frac) * ells_over[idx-1] + frac * ells_over[idx]
            pi = (1 - frac) * pis_over[idx-1] + frac * pis_over[idx]
        else:
            ell = ells_over[idx]
            pi = pis_over[idx]
        pi = np.maximum(pi, 1e-12)
        pi /= pi.sum()
        colors[i] = np.clip(sct_to_rgb(ell, pi), 0, 1)

    return colors


# ============================================================
# PREDEFINED COLORMAPS
# ============================================================

def sct_thermal(n=256):
    """Blue → Red via geodesic. Classic thermal/heat map."""
    v = simplex_vertices(chrom_strength=0.5)
    path = geodesic_path(v['blue'], v['red'], n)
    return make_colormap(path, ell_start=0.05, ell_end=0.92)


def sct_cool(n=256):
    """Blue → Green via geodesic. Cool tones for depth/ocean."""
    v = simplex_vertices(chrom_strength=0.4)
    path = geodesic_path(v['blue'], v['green'], n)
    return make_colormap(path, ell_start=0.05, ell_end=0.90)


def sct_terrain(n=256):
    """
    Terrain/elevation: deep blue → teal → green → brown → white.

    Semantic code: ocean → coast → lowland → highland → snow/peak.
    Wide arc, Fisher-uniform, 0 parameters.
    """
    from sct_core import rgb_to_sct
    _, _, pi_ocean = rgb_to_sct(np.array([0.12, 0.20, 0.55]))
    _, _, pi_coast = rgb_to_sct(np.array([0.20, 0.50, 0.55]))
    _, _, pi_low   = rgb_to_sct(np.array([0.35, 0.60, 0.25]))
    _, _, pi_high  = rgb_to_sct(np.array([0.60, 0.42, 0.25]))
    _, _, pi_snow  = rgb_to_sct(np.array([0.88, 0.86, 0.84]))

    seg = n // 4
    segs = [
        geodesic_path(pi_ocean, pi_coast, seg),
        geodesic_path(pi_coast, pi_low, seg),
        geodesic_path(pi_low, pi_high, seg),
        geodesic_path(pi_high, pi_snow, n - 3 * seg),
    ]
    path = np.vstack(segs)
    return make_colormap(path, ell_start=0.04, ell_end=0.93)


def sct_warm(n=256):
    """Green → Red via geodesic. Warm tones for intensity/activation."""
    v = simplex_vertices(chrom_strength=0.4)
    path = geodesic_path(v['green'], v['red'], n)
    return make_colormap(path, ell_start=0.08, ell_end=0.90)


def sct_full(n=256):
    """Blue → Green → Red (two arcs). Full-spectrum scientific colormap."""
    v = simplex_vertices(chrom_strength=0.4)
    half = n // 2
    path1 = geodesic_path(v['blue'], v['green'], half)
    path2 = geodesic_path(v['green'], v['red'], n - half)
    path = np.vstack([path1, path2])
    return make_colormap(path, ell_start=0.05, ell_end=0.92)


def sct_spectrum(n=256):
    """
    Full-spectrum colormap: dark purple → blue → cyan → green → yellow.

    PT equivalent of viridis: wide chromatic arc for maximum visual
    landmarks, Fisher-reparameterized, 0 parameters.
    Traverses 4 waypoints on Δ² for rich color variation.
    """
    from sct_core import rgb_to_sct
    # 4 waypoints: purple → blue → cyan/teal → green → yellow
    _, _, pi_purple = rgb_to_sct(np.array([0.35, 0.15, 0.55]))
    _, _, pi_blue   = rgb_to_sct(np.array([0.20, 0.40, 0.70]))
    _, _, pi_teal   = rgb_to_sct(np.array([0.20, 0.60, 0.55]))
    _, _, pi_green  = rgb_to_sct(np.array([0.35, 0.65, 0.25]))
    _, _, pi_yellow = rgb_to_sct(np.array([0.80, 0.75, 0.25]))

    # Chain geodesics through waypoints
    seg = n // 4
    segs = [
        geodesic_path(pi_purple, pi_blue, seg),
        geodesic_path(pi_blue, pi_teal, seg),
        geodesic_path(pi_teal, pi_green, seg),
        geodesic_path(pi_green, pi_yellow, n - 3 * seg),
    ]
    path = np.vstack(segs)
    return make_colormap(path, ell_start=0.05, ell_end=0.92)


def sct_magma(n=256):
    """
    Dark → hot colormap: black → purple → orange → pale yellow.

    PT equivalent of inferno/magma: dark-to-bright with warm tones,
    Fisher-reparameterized, 0 parameters.
    """
    from sct_core import rgb_to_sct
    _, _, pi_dark   = rgb_to_sct(np.array([0.15, 0.10, 0.25]))
    _, _, pi_purple = rgb_to_sct(np.array([0.45, 0.15, 0.50]))
    _, _, pi_orange = rgb_to_sct(np.array([0.75, 0.40, 0.20]))
    _, _, pi_pale   = rgb_to_sct(np.array([0.90, 0.85, 0.65]))

    seg = n // 3
    segs = [
        geodesic_path(pi_dark, pi_purple, seg),
        geodesic_path(pi_purple, pi_orange, seg),
        geodesic_path(pi_orange, pi_pale, n - 2 * seg),
    ]
    path = np.vstack(segs)
    return make_colormap(path, ell_start=0.02, ell_end=0.93)


def sct_turbo(n=256):
    """
    Full rainbow: blue → cyan → green → yellow → red.

    PT equivalent of turbo/jet but Fisher-uniform: no false contours,
    no dead zones, wide chromatic arc for maximum readability.
    """
    from sct_core import rgb_to_sct
    _, _, pi_blue   = rgb_to_sct(np.array([0.20, 0.30, 0.75]))
    _, _, pi_cyan   = rgb_to_sct(np.array([0.20, 0.65, 0.65]))
    _, _, pi_green  = rgb_to_sct(np.array([0.30, 0.70, 0.25]))
    _, _, pi_yellow = rgb_to_sct(np.array([0.80, 0.75, 0.20]))
    _, _, pi_red    = rgb_to_sct(np.array([0.80, 0.25, 0.20]))

    seg = n // 4
    segs = [
        geodesic_path(pi_blue, pi_cyan, seg),
        geodesic_path(pi_cyan, pi_green, seg),
        geodesic_path(pi_green, pi_yellow, seg),
        geodesic_path(pi_yellow, pi_red, n - 3 * seg),
    ]
    path = np.vstack(segs)
    return make_colormap(path, ell_start=0.08, ell_end=0.88)


def sct_diverging(n=256):
    """Blue ← Neutral → Red. Diverging colormap for anomalies."""
    v = simplex_vertices()
    neutral = np.array([1/3, 1/3, 1/3])
    half = n // 2

    # Two halves: blue→neutral (dark→bright) and neutral→red (bright→dark)
    path1 = geodesic_path(v['blue'], neutral, half)
    cmap1 = make_colormap(path1, ell_start=0.12, ell_end=0.85, n=half)

    path2 = geodesic_path(neutral, v['red'], n - half)
    cmap2 = make_colormap(path2, ell_start=0.85, ell_end=0.12, n=n - half)

    return np.vstack([cmap1, cmap2])


def sct_medical(n=256):
    """
    Medical imaging colormap: high luminance range, subtle chromaticity.

    Stays close to the achromatic center of Δ² with a gentle
    blue-to-warm arc. Maximizes luminance contrast (the primary
    readability driver in medical imaging) while adding just enough
    chromaticity to distinguish adjacent structures.

    Designed for: MRI, CT, ultrasound, X-ray.
    """
    from sct_core import rgb_to_sct
    # Near-neutral endpoints with subtle blue→warm shift
    _, _, pi_dark = rgb_to_sct(np.array([0.25, 0.28, 0.38]))   # dark blue-gray
    _, _, pi_bright = rgb_to_sct(np.array([0.90, 0.82, 0.72]))  # warm white
    path = geodesic_path(pi_dark, pi_bright, n)
    return make_colormap(path, ell_start=0.03, ell_end=0.95)


def sct_seismic(n=256):
    """
    Geophysics colormap: diverging with high contrast.

    Blue (negative) → white (zero) → red (positive).
    Wider chromaticity than sct_diverging, designed for
    seismic amplitude, gravity anomalies, temperature departures.
    """
    from sct_core import rgb_to_sct
    _, _, pi_cold = rgb_to_sct(np.array([0.15, 0.30, 0.80]))
    _, _, pi_hot = rgb_to_sct(np.array([0.80, 0.20, 0.15]))
    neutral = np.array([1/3, 1/3, 1/3])
    half = n // 2
    path1 = geodesic_path(pi_cold, neutral, half)
    cmap1 = make_colormap(path1, ell_start=0.08, ell_end=0.92, n=half)
    path2 = geodesic_path(neutral, pi_hot, n - half)
    cmap2 = make_colormap(path2, ell_start=0.92, ell_end=0.08, n=n - half)
    return np.vstack([cmap1, cmap2])


def sct_vegetation(n=256):
    """
    Vegetation/NDVI colormap: red-brown → orange → yellow → green → dark green.

    Wide chromatic arc matching the semantic code:
      red/brown = bare soil, stress, drought
      orange = sparse, degraded
      yellow = agricultural, moderate
      green = healthy vegetation
      dark green = dense canopy, forest

    Fisher-uniform geodesic, 0 parameters.
    """
    from sct_core import rgb_to_sct
    _, _, pi_stress = rgb_to_sct(np.array([0.70, 0.25, 0.15]))   # red-brown (stress)
    _, _, pi_sparse = rgb_to_sct(np.array([0.75, 0.55, 0.20]))   # orange (sparse)
    _, _, pi_trans  = rgb_to_sct(np.array([0.80, 0.78, 0.25]))   # yellow (transition)
    _, _, pi_veg    = rgb_to_sct(np.array([0.35, 0.65, 0.20]))   # green (healthy)
    _, _, pi_dense  = rgb_to_sct(np.array([0.10, 0.45, 0.15]))   # dark green (canopy)

    seg = n // 4
    segs = [
        geodesic_path(pi_stress, pi_sparse, seg),
        geodesic_path(pi_sparse, pi_trans, seg),
        geodesic_path(pi_trans, pi_veg, seg),
        geodesic_path(pi_veg, pi_dense, n - 3 * seg),
    ]
    path = np.vstack(segs)
    return make_colormap(path, ell_start=0.10, ell_end=0.75)


# ============================================================
# FISHER UNIFORMITY METRIC (PT-weighted)
# ============================================================

def fisher_distance(rgb1, rgb2):
    """
    PT color distance: d² = (3/4)·d_lum² + (1/4)·d_chrom²

    d_lum: Fisher-Bernoulli on p=2 channel (luminance)
    d_chrom: Bhattacharyya on Δ² with γ_p weighting (chromaticity)
    Weights 3/4 and 1/4 from N/(N+1) formula.
    """
    from sct_core import rgb_to_sct
    ell1, S1, pi1 = rgb_to_sct(rgb1)
    ell2, S2, pi2 = rgb_to_sct(rgb2)

    # Chromatic: Bhattacharyya distance on simplex
    xi1 = np.sqrt(np.maximum(pi1, 1e-12))
    xi2 = np.sqrt(np.maximum(pi2, 1e-12))
    cos_d = np.clip(np.dot(xi1, xi2), -1, 1)
    d_chrom = 2 * np.arccos(cos_d)  # Factor 2 for Fisher normalization

    # Luminance: Fisher-Bernoulli distance
    ell1 = np.clip(ell1, 0.01, 0.99)
    ell2 = np.clip(ell2, 0.01, 0.99)
    d_lum = 2 * abs(np.arcsin(np.sqrt(ell2)) - np.arcsin(np.sqrt(ell1)))

    # PT-weighted combination
    d_total = np.sqrt(0.75 * d_lum**2 + 0.25 * d_chrom**2)
    return d_total, d_lum, d_chrom


def measure_uniformity(cmap_rgb):
    """
    Measure perceptual uniformity of a colormap using PT Fisher distance.
    Returns:
        cv_total: CV of total ΔE steps (lower = more uniform)
        cv_lum: CV of luminance steps
        cv_chrom: CV of chromatic steps
        mean_d: mean total step size
    """
    d_totals, d_lums, d_chroms = [], [], []
    for i in range(len(cmap_rgb) - 1):
        dt, dl, dc = fisher_distance(cmap_rgb[i], cmap_rgb[i + 1])
        d_totals.append(dt)
        d_lums.append(dl)
        d_chroms.append(dc)

    d_totals = np.array(d_totals)
    d_lums = np.array(d_lums)
    d_chroms = np.array(d_chroms)

    cv = lambda x: x.std() / max(x.mean(), 1e-10)

    return {
        'cv_total': cv(d_totals),
        'cv_lum': cv(d_lums),
        'cv_chrom': cv(d_chroms),
        'mean_total': d_totals.mean(),
        'mean_lum': d_lums.mean(),
        'mean_chrom': d_chroms.mean(),
        'dead_zone': np.sum(d_totals < d_totals.mean() * 0.2) / len(d_totals),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_colormaps():
    """Generate comparison figure: SCT colormaps vs standard ones."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    cmaps_sct = {
        'sct_spectrum': sct_spectrum(),
        'sct_turbo': sct_turbo(),
        'sct_magma': sct_magma(),
        'sct_terrain': sct_terrain(),
        'sct_vegetation': sct_vegetation(),
        'sct_medical': sct_medical(),
        'sct_diverging': sct_diverging(),
        'sct_seismic': sct_seismic(),
        'sct_thermal': sct_thermal(),
        'sct_cool': sct_cool(),
        'sct_warm': sct_warm(),
        'sct_full': sct_full(),
    }

    cmaps_std = {
        'viridis': plt.cm.viridis,
        'turbo': plt.cm.turbo,
        'inferno': plt.cm.inferno,
        'terrain': plt.cm.terrain,
        'RdYlGn': plt.cm.RdYlGn,
        'bone': plt.cm.bone,
        'coolwarm': plt.cm.coolwarm,
        'seismic': plt.cm.seismic,
        'jet': plt.cm.jet,
    }

    n_total = len(cmaps_sct) + 1 + len(cmaps_std)  # +1 for separator
    fig, axes = plt.subplots(n_total, 1, figsize=(14, n_total * 0.6 + 2))
    fig.suptitle('SCT Colormaps (0 params, Fisher-geodesic) vs Standard',
                 fontsize=14, fontweight='bold', y=0.98)

    gradient = np.linspace(0, 1, 256).reshape(1, -1)

    row = 0
    for name, cmap in cmaps_sct.items():
        ax = axes[row]
        lsc = mcolors.ListedColormap(cmap)
        ax.imshow(gradient, aspect='auto', cmap=lsc)
        m = measure_uniformity(cmap)
        ax.set_ylabel(f'{name}\nCV={m["cv_total"]:.3f}', fontsize=9, rotation=0, labelpad=100, va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1

    # Separator
    axes[row].set_visible(False)
    row += 1

    for name, cmap in cmaps_std.items():
        ax = axes[row]
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        std_rgb = cmap(np.linspace(0, 1, 256))[:, :3]
        m = measure_uniformity(std_rgb)
        ax.set_ylabel(f'{name}\nCV={m["cv_total"]:.3f}', fontsize=9, rotation=0, labelpad=100, va='center')
        ax.set_xticks([])
        ax.set_yticks([])
        row += 1

    plt.tight_layout(rect=[0.12, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), 'fig_colormap_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def demo_scientific_data():
    """Apply SCT colormaps to real scientific datasets."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    # Load real data
    elevation = np.load(os.path.join(data_dir, 'elevation_fuji_etopo.npy'))
    brain = np.load(os.path.join(data_dir, 'brain_mri_mni152.npy'))
    sst = np.load(os.path.join(data_dir, 'sst_pacific_noaa.npy'))
    ndvi = np.load(os.path.join(data_dir, 'ndvi_forest.npy'))

    sct_maps = {
        'sct_thermal': mcolors.ListedColormap(sct_thermal()),
        'sct_cool': mcolors.ListedColormap(sct_cool()),
        'sct_warm': mcolors.ListedColormap(sct_warm()),
        'sct_diverging': mcolors.ListedColormap(sct_diverging()),
        'sct_medical': mcolors.ListedColormap(sct_medical()),
        'sct_seismic': mcolors.ListedColormap(sct_seismic()),
        'sct_vegetation': mcolors.ListedColormap(sct_vegetation()),
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    fig.suptitle('SCT Colormaps on Real Scientific Data — 0 parameters, Fisher-geodesic',
                 fontsize=13, fontweight='bold')

    sct_maps['sct_spectrum'] = mcolors.ListedColormap(sct_spectrum())
    sct_maps['sct_magma'] = mcolors.ListedColormap(sct_magma())
    sct_maps['sct_turbo'] = mcolors.ListedColormap(sct_turbo())

    sct_maps['sct_terrain'] = mcolors.ListedColormap(sct_terrain())

    datasets = [
        (elevation, 'Elevation (sct_terrain)', 'sct_terrain', None),
        (brain, 'Brain MRI (sct_medical)', 'sct_medical', None),
        (sst, 'Sea Temp. (sct_turbo)', 'sct_turbo', None),
        (ndvi, 'NDVI (sct_vegetation)', 'sct_vegetation', None),
        (elevation, 'Elevation (terrain)', 'terrain', None),
        (brain, 'Brain MRI (bone)', 'bone', None),
        (sst, 'Sea Temp. (turbo)', 'turbo', None),
        (ndvi, 'NDVI (RdYlGn)', 'RdYlGn', None),
    ]

    for i, (data, title, cmap_name, vrange) in enumerate(datasets):
        ax = axes[i // 4][i % 4]
        cm = sct_maps[cmap_name] if cmap_name in sct_maps else plt.get_cmap(cmap_name)
        kwargs = {'cmap': cm, 'aspect': 'equal'}
        if vrange:
            kwargs['vmin'], kwargs['vmax'] = vrange
        im = ax.imshow(data, **kwargs)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0][0].set_ylabel('SCT\n(Fisher-geodesic)', fontsize=11, fontweight='bold')
    axes[1][0].set_ylabel('Standard', fontsize=11, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(os.path.dirname(__file__), 'fig_scientific_demo.png')
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


def print_uniformity_table():
    """Print uniformity comparison table with PT-weighted Fisher metric."""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 90)
    print("  PERCEPTUAL UNIFORMITY — PT Fisher metric: d² = (3/4)·d_lum² + (1/4)·d_chrom²")
    print("=" * 90)
    print(f"  {'Colormap':<20} {'CV_total':>9} {'CV_lum':>8} {'CV_chrom':>9} {'Dead%':>6}  {'Note'}")
    print("-" * 90)

    sct_maps = {
        'sct_thermal': sct_thermal(),
        'sct_cool': sct_cool(),
        'sct_warm': sct_warm(),
        'sct_full': sct_full(),
        'sct_diverging': sct_diverging(),
        'sct_medical': sct_medical(),
        'sct_seismic': sct_seismic(),
        'sct_vegetation': sct_vegetation(),
        'sct_spectrum': sct_spectrum(),
        'sct_magma': sct_magma(),
        'sct_turbo': sct_turbo(),
        'sct_terrain': sct_terrain(),
    }

    for name, cmap in sct_maps.items():
        m = measure_uniformity(cmap)
        print(f"  {name:<20} {m['cv_total']:9.4f} {m['cv_lum']:8.4f} {m['cv_chrom']:9.4f} {m['dead_zone']*100:5.1f}%  geodesic, 0 params")

    print("-" * 90)

    std_maps = {'jet': plt.cm.jet, 'viridis': plt.cm.viridis,
                'inferno': plt.cm.inferno, 'coolwarm': plt.cm.coolwarm,
                'turbo': plt.cm.turbo}

    for name, cmap in std_maps.items():
        rgb = cmap(np.linspace(0, 1, 256))[:, :3]
        m = measure_uniformity(rgb)
        note = 'CIELAB-fitted' if name in ('viridis', 'inferno') else 'empirical'
        print(f"  {name:<20} {m['cv_total']:9.4f} {m['cv_lum']:8.4f} {m['cv_chrom']:9.4f} {m['dead_zone']*100:5.1f}%  {note}")

    print("=" * 90)
    print("  CV = coefficient of variation (lower = more uniform). Dead% = fraction of steps < 20% of mean")
    print("  Dead zones = regions where the colormap wastes perceptual budget (invisible changes)")
    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("SCT Colormap — Simplex Color Transform Scientific Visualization")
    print()

    print_uniformity_table()

    if '--demo' in sys.argv or len(sys.argv) == 1:
        plot_colormaps()
        demo_scientific_data()
        print("\nDone. Figures saved.")
