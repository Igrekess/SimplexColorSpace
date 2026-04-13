# Colormaps

## How they work

Each SCT colormap is a geodesic path on the simplex delta-2 with coupled
luminance ramp. The construction has two steps:

1. **Chromaticity path**: multi-waypoint geodesic on delta-2, reparameterized
   by gamma-weighted Fisher arc length (equal chromatic steps).

2. **Luminance ramp**: arcsin parameterization (`ell = sin^2(theta)`) so
   that equal parameter steps produce equal Fisher-Bernoulli distances
   (equal luminance steps).

The two are then coupled: the total path (ell, pi) is reparameterized by
combined Fisher arc length:

```
d^2_total = (3/4) * d_lum^2 + (1/4) * d_chrom^2
```

This ensures uniform perceptual steps along the entire colormap.

## Available colormaps

### Sequential

| Name | Colors | Use case |
|------|--------|----------|
| `sct_spectrum` | purple -> blue -> cyan -> green -> yellow | General scientific (PT viridis) |
| `sct_turbo` | blue -> cyan -> green -> yellow -> red | Full rainbow, no false contours |
| `sct_magma` | black -> purple -> orange -> pale yellow | Dark-to-bright intensity |
| `sct_terrain` | deep blue -> teal -> green -> brown -> white | Elevation, bathymetry |
| `sct_vegetation` | red-brown -> orange -> yellow -> green -> dark green | NDVI, biomass, forest cover |
| `sct_medical` | dark blue-gray -> warm white | MRI, CT, X-ray (high lum range) |
| `sct_thermal` | blue -> red | Heat maps |
| `sct_cool` | blue -> green | Ocean depth |
| `sct_warm` | green -> red | Activation, intensity |
| `sct_full` | blue -> green -> red | Full spectrum |

### Diverging

| Name | Colors | Use case |
|------|--------|----------|
| `sct_diverging` | blue <- neutral -> red | Anomalies, T-statistics |
| `sct_seismic` | strong blue <- white -> strong red | Seismic, gravity anomalies |

## Usage with matplotlib

```python
from sct.colormaps import register_matplotlib
register_matplotlib()

import matplotlib.pyplot as plt

# Use by name
plt.imshow(data, cmap='sct_spectrum')

# Or get the array directly
from sct.colormaps import sct_vegetation
cmap_array = sct_vegetation(256)  # (256, 3) float sRGB
```

## Comparison with standard colormaps

Perceptual uniformity measured by coefficient of variation (CV) of
successive Fisher distances. Lower = more uniform.

| Colormap | CV total | Dead zones | Parameters |
|----------|----------|------------|------------|
| sct_diverging | 0.063 | 0.4% | 0 |
| sct_vegetation | 0.101 | 0% | 0 |
| sct_spectrum | 0.102 | 0% | 0 |
| sct_medical | 0.145 | 0% | 0 |
| sct_seismic | 0.175 | 0% | 0 |
| sct_turbo | 0.198 | 0% | 0 |
| sct_terrain | 0.216 | 0.4% | 0 |
| sct_magma | 0.230 | 0% | 0 |
| viridis | 0.286 | 0% | fitted |
| turbo | 0.292 | 1.6% | empirical |
| inferno | 0.634 | 2.4% | fitted |
| jet | 0.661 | 1.6% | empirical |

9 out of 12 SCT colormaps beat viridis in Fisher uniformity.

## When to use SCT colormaps vs standard ones

**Use SCT** when:
- You need a certificate of uniformity (diagnostic medical imaging)
- You need gamut portability (same colormap on sRGB and P3 displays)
- You need the GFT conservation guarantee (no dead zones)

**Use standard** when:
- Maximum chromatic contrast matters more than uniformity
- Users are already trained on viridis/RdYlGn conventions
- The colormap is decorative rather than part of the measurement chain
