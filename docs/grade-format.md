# Grade Format (.sct — Simplex Color Translation)

## Overview

A `.sct` file stores a color grade as a displacement field on the SCT
simplex. Unlike `.cube` files (which are RGB->RGB tables tied to a specific
color space), a `.sct` grade is **gamut-independent**: it can be captured in
one color space and applied in any other.

## File structure

JSON with the following keys:

```json
{
  "version": 1,
  "lut_size": 17,
  "delta_pi": [[[ ... ]]],
  "delta_ell": [[[ ... ]]],
  "metadata": {
    "coverage": 85.3,
    "mean_delta_pi": 0.042,
    "mean_delta_ell": 0.031
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Format version (currently 1) |
| `lut_size` | int | Grid resolution N (typically 17 or 33) |
| `delta_pi` | float[N][N][N][3] | Chromaticity displacement on simplex |
| `delta_ell` | float[N][N][N] | Luminance displacement |
| `metadata.coverage` | float | Percentage of LUT cells with data (0-100) |
| `metadata.mean_delta_pi` | float | Mean absolute chromaticity shift |
| `metadata.mean_delta_ell` | float | Mean absolute luminance shift |

### How it works

The LUT is indexed by the **before** image's RGB values. For each grid
point (R, G, B):

- `delta_pi[R][G][B]` = how much to shift the simplex coordinates
- `delta_ell[R][G][B]` = how much to shift the luminance

At application time:
1. Convert input pixel to SCT: (pi, ell)
2. Look up (delta_pi, delta_ell) by trilinear interpolation in the LUT
3. Apply: pi_new = pi + intensity * delta_pi
4. Normalize pi_new on the simplex
5. Convert back to RGB

### Why it's gamut-independent

The displacement `delta_pi` lives on the simplex, not in RGB space. The
simplex coordinates are the same regardless of which primaries or transfer
curve produced the original RGB values. So a grade captured from Rec.709
footage can be applied to S-Log3 footage — the simplex displacement is
translated to the target gamut at step 5.

## Typical file sizes

| LUT size | File size |
|----------|-----------|
| 17^3 | ~3 MB |
| 33^3 | ~50 MB |

## Creating a .sct from a .cube

Use ChromaPlex (tab 3: Convert .cube) with the "Export .sct" checkbox,
or programmatically:

```python
import sct

# From before/after images
grade = sct.capture_grade(before_img, after_img, lut_size=17)
sct.save_grade(grade, "my_look.sct")

# From an existing grade
loaded = sct.load_grade("my_look.sct")
result = sct.apply_grade(photo, loaded, intensity=0.8)
```

## Blending two grades

```python
mix = sct.blend_grades(grade_a, grade_b, ratio=0.3)
# 30% of grade A + 70% of grade B (linear on the simplex)
sct.save_grade(mix, "blend_30_70.sct")
```
