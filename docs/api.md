# API Reference

## Color Space Conversions

### `scs.rgb_to_scs(rgb) -> (ell, S, pi)`

Convert sRGB [0,1] to SCS coordinates.

- **rgb**: array-like of 3 floats, sRGB values in [0, 1]
- **Returns**: tuple (ell, S, pi)
  - `ell`: luminance in [0.001, 0.999]
  - `S`: saturation (D_KL from uniform) in bits, range [0, log2(3)]
  - `pi`: numpy array (3,), simplex coordinates (pi_3, pi_5, pi_7)

```python
ell, S, pi = scs.rgb_to_scs([0.8, 0.3, 0.2])
# ell=0.183, S=0.275, pi=[0.575, 0.330, 0.095]
```

### `scs.scs_to_rgb(ell, pi) -> rgb`

Convert SCS coordinates back to sRGB [0,1].

- **ell**: luminance
- **pi**: simplex coordinates (3,)
- **Returns**: numpy array (3,), sRGB values

Round-trip is exact: `scs_to_rgb(*rgb_to_scs(x))` returns `x`.

### `scs.image_to_scs(img) -> (pi, ell, S, hue)`

Vectorized conversion: image (H,W,3) uint8 -> SCS coordinates.

- **img**: numpy array (H,W,3) uint8, sRGB image
- **Returns**: tuple
  - `pi`: (H,W,3) simplex coordinates
  - `ell`: (H,W) luminance
  - `S`: (H,W) saturation in bits
  - `hue`: (H,W) hue angle in degrees [0, 360)

### `scs.scs_to_image(pi, ell) -> img`

Vectorized inverse: SCS coordinates -> image (H,W,3) uint8.

---

## Color Difference

### `scs.delta_e(xyz1, xyz2) -> float`

SCS color difference between two CIE XYZ colors. Zero adjustable parameters.

Formula: `dE^2 = (3/4) * d_lum^2 + (1/4) * d_chrom^2`

- `d_lum`: Fisher distance on Bernoulli(ell), the p=2 channel
- `d_chrom`: Bhattacharyya distance on the simplex with gamma_p weighting
- Weights 3/4 and 1/4 derived from N/(N+1) with N=3 active primes

```python
d = scs.delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
```

### `scs.delta_e_lab(L1, a1, b1, L2, a2, b2) -> float`

Same as delta_e but from CIELAB coordinates (convenience wrapper).

---

## GFT-Conserving Adjustments

All adjustments maintain the identity S + L = log2(3).

### `scs.adjust_saturation(pi, delta_S) -> pi_new`

Adjust saturation by `delta_S` bits while preserving hue direction.

- Moves pi along a radial line toward/away from uniform (1/3, 1/3, 1/3)
- Uses binary search (40 iterations) for exact target saturation
- Hue is exactly preserved (same radial direction)

```python
pi_more_sat = scs.adjust_saturation(pi, +0.1)   # more saturated
pi_less_sat = scs.adjust_saturation(pi, -0.05)   # less saturated
```

### `scs.adjust_luminance(ell, pi, delta_L) -> (ell_new, pi_new)`

Adjust luminance while maintaining GFT balance.

- The saturation is automatically compensated using the Fisher luminance
  scaling factor: `f = sqrt(ell_new * (1-ell_new)) / sqrt(ell * (1-ell))`
- This is the exact GFT-conserving correction — not a heuristic

```python
ell_new, pi_new = scs.adjust_luminance(0.5, pi, +0.1)  # brighter
```

---

## Geodesic Hue Rotation

### `scs.rotate_hue_geodesic(pi, angle_deg) -> pi_new`

Rotate hue by `angle_deg` on the Fisher-Rao geodesic.

- Uses the square-root embedding: xi = sqrt(pi)
- Rotation around the achromatic axis (1,1,1)/sqrt(3)
- **Preserves saturation exactly** (great circle on S^2)
- Works on single points (3,) or batches (N, 3)

```python
pi_rotated = scs.rotate_hue_geodesic(pi, 45)       # single point
pi_batch = scs.rotate_hue_geodesic(pi_array, 30)   # (N, 3) batch
```

---

## Skin Tone Protection

### `scs.skin_mask(pi, S=None) -> mask`

Compute per-pixel skin tone mask from simplex coordinates.

- Skin tones are centered at hue ~30 deg on the simplex with 35 deg half-width
- Uses raised-cosine falloff: smooth transition, no hard boundary
- Requires minimum saturation (S > 0.15 bits) to exclude achromatic pixels
- No training data, no skin detection model, no calibration

```python
pi, ell, S, hue = scs.image_to_scs(img)
mask = scs.skin_mask(pi, S)   # (H, W), values 0-1
# Use: result = graded * (1-mask) + original * mask
```

---

## Color Grades

### `scs.capture_grade(before, after, lut_size=17) -> grade`

Capture a color grade from before/after images.

- Measures the displacement field (delta_pi, delta_ell) on the simplex
- Stores as a 3D LUT indexed by RGB of the before image
- Empty cells filled by nearest-neighbor interpolation
- The grade is **gamut-independent**: captured in any space, applicable in any space

### `scs.apply_grade(img, grade, intensity=1.0) -> img`

Apply a grade to an image.

- Trilinear interpolation in the grade LUT
- `intensity`: 0 = no effect, 1 = full, >1 = exaggerate, <0 = inverse

### `scs.save_grade(grade, path)` / `scs.load_grade(path) -> grade`

Save/load a grade as .scs file (JSON format).

### `scs.blend_grades(grade_a, grade_b, ratio) -> grade`

Blend two grades by linear interpolation on the simplex.

- `ratio=0`: pure A, `ratio=1`: pure B, `ratio=0.5`: equal mix

---

## GFT Conservation

### `scs.gft_check(pi) -> (S, L, total, error)`

Verify the GFT identity: S + L = log2(3).

- `S`: saturation = D_KL(pi || uniform)
- `L`: entropy = H(pi)
- `total`: S + L (should be log2(3) = 1.58496...)
- `error`: |total - log2(3)| (typically < 1e-15)

```python
S, L, total, err = scs.gft_check([0.5, 0.3, 0.2])
# total = 1.584963, err = 2.2e-16
```

---

## Colormaps

### `scs.colormaps.scs_spectrum(n=256) -> array`

Returns a (n, 3) float array of sRGB values.

Available colormaps:

| Name | Description | Best for |
|------|-------------|----------|
| `scs_spectrum` | Purple -> blue -> green -> yellow | General scientific |
| `scs_turbo` | Blue -> cyan -> green -> yellow -> red | Temperature, velocity |
| `scs_magma` | Black -> purple -> orange -> pale | Dark-to-bright intensity |
| `scs_terrain` | Blue -> teal -> green -> brown -> white | Elevation, bathymetry |
| `scs_vegetation` | Red -> orange -> yellow -> green -> dark green | NDVI, biomass, forest |
| `scs_medical` | Dark blue-gray -> warm white | MRI, CT, X-ray |
| `scs_thermal` | Blue -> red | Heat maps |
| `scs_cool` | Blue -> green | Depth, ocean |
| `scs_warm` | Green -> red | Activation, intensity |
| `scs_full` | Blue -> green -> red | Full spectrum |
| `scs_diverging` | Blue <- neutral -> red | Anomalies, T-stats |
| `scs_seismic` | Strong blue <- white -> strong red | Seismic, gravity |

### `scs.colormaps.register_matplotlib()`

Register all SCS colormaps with matplotlib so they can be used by name:

```python
from scs.colormaps import register_matplotlib
register_matplotlib()

import matplotlib.pyplot as plt
plt.imshow(data, cmap='scs_spectrum')
```

---

## Constants

| Name | Value | Source |
|------|-------|--------|
| `scs.MU_STAR` | 15 | Fixed point of sieve (T7) |
| `scs.GAMMAS` | (0.808, 0.696, 0.595) | Effective dimensions at mu*=15 |
| `scs.PRIMES` | (3, 5, 7) | Active chromatic primes |
| `scs.LOG2_3` | 1.58496... | Total informational budget |
