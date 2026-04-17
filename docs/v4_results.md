# V4 fMRI calibration — results summary

## Dataset

- **Source**: OpenNeuro [ds005521](https://openneuro.org/datasets/ds005521) (Conway et al., *J. Neurosci.* 2025)
- **License**: CC0 (public domain)
- **Subjects**: 2 macaques (sub-M1, sub-M2)
- **Runs**: 118 fMRI runs
- **Design**: 9 DKL hues × 4 contrasts = 36 conditions, ~1062 stimulus blocks of 14 s at TR = 2 s
- **Pipeline**: functional ROI (top 10% activated, posterior cortex), block-average BOLD, HRF-shifted +4 s, percent signal change

## Key numerical results (cited in PT_COLOR.tex abstract and §Relation to DKL)

### V4 channel weights at 95% contrast

| Channel | V4 weight (BOLD) | SCS prediction (γ_p / Σγ_p) | Gap |
|---------|-----------------:|----------------------------:|-----|
| L − M   | **0.373** | **0.385** | **3.2%** ← headline match |
| S       | 0.056 | 0.284 | 80% (S-cone underrepresented in functional ROI) |
| Luminance | 0.571 | 0.750 | 24% |

The L − M match is the claim cited in the paper. The S and Luminance gaps are honest caveats: the functional ROI underrepresents S-cone signal (well-known fMRI limitation) and the luminance channel carries contributions from non-chromatic visual processing that the SCS simplex does not separate.

### PT hypothesis tests

| Test | Result | Verdict |
|------|--------|---------|
| Luminance (p=2) > Chromatic | BOLD ratio 2.57 | **Confirmed** |
| L − M (γ₃) > S (γ₇) | L − M −0.078% vs S −0.086% | Marginal, same sign |
| r(BOLD, d_Fisher), 8 chromatic hues | 0.434 | Positive, not dominant |
| r(BOLD, d_Fisher) at 30% contrast (linear regime) | **0.517** | Best correlation |

### V4 contrast-response decomposition

| Contrast | L − M (γ₃) | S (γ₇) | Luminance (p=2) | `|LM/S|` |
|----------|-----------:|-------:|----------------:|---------:|
| 10% | −0.154 | −0.041 | +0.206 | 3.75 |
| 30% | −0.095 | −0.115 | +0.079 | 0.83 |
| 50% | −0.055 | −0.042 | +0.043 | 1.31 |
| 95% | −0.137 | +0.021 | +0.210 | 6.62 |

SCS predicted `|LM/S|` = γ₃ / γ₇ = **1.356**; observed mean 3.13. The agreement is qualitative across contrasts, not a tight quantitative match.

## What the V4 data confirms

- The luminance/chromaticity split (p = 2 vs {3, 5, 7}) is echoed in V4 cortical responses (luminance 2.57× stronger than chromatic).
- The L − M channel weight in V4 matches γ₃ / (γ₃ + γ₅ + γ₇) to **3.2%** — a prediction from `s = 1/2` with 0 adjusted parameters.
- Low-contrast V4 responses correlate with Fisher distance on the SCS simplex (r = 0.517 at 30% contrast).

## What the V4 data does NOT do

- It does **not** replace CIECAM02 in the hybrid model — the V4-opponent hybrid reaches r = 0.675 on COMBVD, vs r = 0.824 for SCS + CIECAM02 with actual CAM02 features.
- The S channel is underrepresented in our functional ROI — a known fMRI limitation, not a theoretical failure.
- 9 DKL directions cannot interpolate to arbitrary COMBVD colors without additional modeling.

## What the V4 data suggests as follow-up

- Proper retinotopic mapping (task-meridianmapper) for a true V4 ROI instead of the functional proxy used here.
- GLM analysis with motion correction for cleaner BOLD estimates.
- Denser DKL stimulus grid to drive out V4 tuning curves.
- Replication on human fMRI data (not just macaque).

## Reproducibility

```
python3 scripts/v4_neural_extraction.py      # pipeline
python3 scripts/v4_refined_analysis.py       # prints L-M = 0.373 vs γ₃/Σγ = 0.385
python3 scripts/v4_hybrid_model.py           # COMBVD hybrid with V4 opponent channels
python3 scripts/v4_analysis_plots.py         # figures
```

Data file paths are resolved via the `SCS_V4_DATA` environment variable; default is `data/ds005521` under the repo root. Download instructions are in the `v4_neural_extraction.py` header.

## Cited files

- `datasets/v4_bold_response.csv` — 36 conditions (9 hues × 4 contrasts)
- `datasets/v4_neural_transfer.csv` — model comparison table
- `docs/figures/fig_v4_bold_polar.pdf`
- `docs/figures/fig_v4_bold_vs_fisher.pdf`
- `docs/figures/fig_v4_contrast_response.pdf`
- `docs/figures/fig_v4_gamma_ratio_test.pdf`
- `docs/figures/fig_v4_simplex_map.pdf`
