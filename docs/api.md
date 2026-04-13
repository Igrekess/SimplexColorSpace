# API Reference

## Color Space Conversions

### `sct.rgb_to_sct(rgb) -> (ell, S, pi)`

Convert sRGB [0,1] to SCT coordinates.

- **rgb**: array-like of 3 floats, sRGB values in [0, 1]
- **Returns**: tuple (ell, S, pi)
  - `ell`: luminance in [0.001, 0.999]
  - `S`: saturation (D_KL from uniform) in bits, range [0, log2(3)]
  - `pi`: numpy array (3,), simplex coordinates (pi_3, pi_5, pi_7)

```python
ell, S, pi = sct.rgb_to_sct([0.8, 0.3, 0.2])
# ell=0.183, S=0.275, pi=[0.575, 0.330, 0.095]
```

### `sct.sct_to_rgb(ell, pi) -> rgb`

Convert SCT coordinates back to sRGB [0,1].

- **ell**: luminance
- **pi**: simplex coordinates (3,)
- **Returns**: numpy array (3,), sRGB values

Round-trip is exact: `sct_to_rgb(*rgb_to_sct(x))` returns `x`.

### `sct.image_to_sct(img) -> (pi, ell, S, hue)`

Vectorized conversion: image (H,W,3) uint8 -> SCT coordinates.

- **img**: numpy array (H,W,3) uint8, sRGB image
- **Returns**: tuple
  - `pi`: (H,W,3) simplex coordinates
  - `ell`: (H,W) luminance
  - `S`: (H,W) saturation in bits
  - `hue`: (H,W) hue angle in degrees [0, 360)

### `sct.sct_to_image(pi, ell) -> img`

Vectorized inverse: SCT coordinates -> image (H,W,3) uint8.

---

## Color Difference

### `sct.delta_e(xyz1, xyz2) -> float`

SCT color difference between two CIE XYZ colors. Zero adjustable parameters.

Formula: `dE^2 = (3/4) * d_lum^2 + (1/4) * d_chrom^2`

- `d_lum`: Fisher distance on Bernoulli(ell), the p=2 channel
- `d_chrom`: Bhattacharyya distance on the simplex with gamma_p weighting
- Weights 3/4 and 1/4 derived from N/(N+1) with N=3 active primes

```python
d = sct.delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
```

### `sct.delta_e_lab(L1, a1, b1, L2, a2, b2) -> float`

Same as delta_e but from CIELAB coordinates (convenience wrapper).

---

## GFT-Conserving Adjustments

All adjustments maintain the identity S + L = log2(3).

### `sct.adjust_saturation(pi, delta_S) -> pi_new`

Adjust saturation by `delta_S` bits while preserving hue direction.

- Moves pi along a radial line toward/away from uniform (1/3, 1/3, 1/3)
- Uses binary search (40 iterations) for exact target saturation
- Hue is exactly preserved (same radial direction)

```python
pi_more_sat = sct.adjust_saturation(pi, +0.1)   # more saturated
pi_less_sat = sct.adjust_saturation(pi, -0.05)   # less saturated
```

### `sct.adjust_luminance(ell, pi, delta_L) -> (ell_new, pi_new)`

Adjust luminance while maintaining GFT balance.

- The saturation is automatically compensated using the Fisher luminance
  scaling factor: `f = sqrt(ell_new * (1-ell_new)) / sqrt(ell * (1-ell))`
- This is the exact GFT-conserving correction — not a heuristic

```python
ell_new, pi_new = sct.adjust_luminance(0.5, pi, +0.1)  # brighter
```

---

## Geodesic Hue Rotation

### `sct.rotate_hue_geodesic(pi, angle_deg) -> pi_new`

Rotate hue by `angle_deg` on the Fisher-Rao geodesic.

- Uses the square-root embedding: xi = sqrt(pi)
- Rotation around the achromatic axis (1,1,1)/sqrt(3)
- **Preserves saturation exactly** (great circle on S^2)
- Works on single points (3,) or batches (N, 3)

```python
pi_rotated = sct.rotate_hue_geodesic(pi, 45)       # single point
pi_batch = sct.rotate_hue_geodesic(pi_array, 30)   # (N, 3) batch
```

---

## Skin Tone Protection

### `sct.skin_mask(pi, S=None) -> mask`

Compute per-pixel skin tone mask from simplex coordinates.

- Skin tones are centered at hue ~30 deg on the simplex with 35 deg half-width
- Uses raised-cosine falloff: smooth transition, no hard boundary
- Requires minimum saturation (S > 0.15 bits) to exclude achromatic pixels
- No training data, no skin detection model, no calibration

```python
pi, ell, S, hue = sct.image_to_sct(img)
mask = sct.skin_mask(pi, S)   # (H, W), values 0-1
# Use: result = graded * (1-mask) + original * mask
```

---

## Color Grades

### `sct.capture_grade(before, after, lut_size=17) -> grade`

Capture a color grade from before/after images.

- Measures the displacement field (delta_pi, delta_ell) on the simplex
- Stores as a 3D LUT indexed by RGB of the before image
- Empty cells filled by nearest-neighbor interpolation
- The grade is **gamut-independent**: captured in any space, applicable in any space

### `sct.apply_grade(img, grade, intensity=1.0) -> img`

Apply a grade to an image.

- Trilinear interpolation in the grade LUT
- `intensity`: 0 = no effect, 1 = full, >1 = exaggerate, <0 = inverse

### `sct.save_grade(grade, path)` / `sct.load_grade(path) -> grade`

Save/load a grade as .sct file (JSON format).

### `sct.blend_grades(grade_a, grade_b, ratio) -> grade`

Blend two grades by linear interpolation on the simplex.

- `ratio=0`: pure A, `ratio=1`: pure B, `ratio=0.5`: equal mix

---

## GFT Conservation

### `sct.gft_check(pi) -> (S, L, total, error)`

Verify the GFT identity: S + L = log2(3).

- `S`: saturation = D_KL(pi || uniform)
- `L`: entropy = H(pi)
- `total`: S + L (should be log2(3) = 1.58496...)
- `error`: |total - log2(3)| (typically < 1e-15)

```python
S, L, total, err = sct.gft_check([0.5, 0.3, 0.2])
# total = 1.584963, err = 2.2e-16
```

---

## Colormaps

### `sct.colormaps.sct_spectrum(n=256) -> array`

Returns a (n, 3) float array of sRGB values.

Available colormaps:

| Name | Description | Best for |
|------|-------------|----------|
| `sct_spectrum` | Purple -> blue -> green -> yellow | General scientific |
| `sct_turbo` | Blue -> cyan -> green -> yellow -> red | Temperature, velocity |
| `sct_magma` | Black -> purple -> orange -> pale | Dark-to-bright intensity |
| `sct_terrain` | Blue -> teal -> green -> brown -> white | Elevation, bathymetry |
| `sct_vegetation` | Red -> orange -> yellow -> green -> dark green | NDVI, biomass, forest |
| `sct_medical` | Dark blue-gray -> warm white | MRI, CT, X-ray |
| `sct_thermal` | Blue -> red | Heat maps |
| `sct_cool` | Blue -> green | Depth, ocean |
| `sct_warm` | Green -> red | Activation, intensity |
| `sct_full` | Blue -> green -> red | Full spectrum |
| `sct_diverging` | Blue <- neutral -> red | Anomalies, T-stats |
| `sct_seismic` | Strong blue <- white -> strong red | Seismic, gravity |

### `sct.colormaps.register_matplotlib()`

Register all SCT colormaps with matplotlib so they can be used by name:

```python
from sct.colormaps import register_matplotlib
register_matplotlib()

import matplotlib.pyplot as plt
plt.imshow(data, cmap='sct_spectrum')
```

---

## Constants

| Name | Value | Source |
|------|-------|--------|
| `sct.MU_STAR` | 15 | Fixed point of sieve (T7) |
| `sct.GAMMAS` | (0.808, 0.696, 0.595) | Effective dimensions at mu*=15 |
| `sct.PRIMES` | (3, 5, 7) | Active chromatic primes |
| `sct.LOG2_3` | 1.58496... | Total informational budget |
