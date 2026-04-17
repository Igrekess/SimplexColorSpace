# SCS public benchmark — leaderboard

This leaderboard tracks reproducible submissions against three open challenges stated in the companion paper (section *Open problems and experimental hooks*). Contributions are welcome by pull request.

## How to submit

1. Fork the repo and add your run under `bench/<challenge>/<your_handle>/`.
2. Include: reproduction script, datasets used (or exact download path), random seeds, output CSV, and a one-paragraph README.
3. Submit a PR with your numbers added to the table below.

Your entry will be merged if the run reproduces end-to-end on a clean clone.

---

## Challenge 1 — MacAdam ellipse orientation

Predict MacAdam's 25 ellipse orientations with zero fitted parameters. Metric: number of wins on `|Δθ|` + RMS angle error.

| Method | Wins (/25) | RMS Δθ | Parameters | Script | Date |
|--------|-----------|--------|-----------|--------|------|
| CIELAB | 7 / 25 | 52.0° | 3 | `scripts/macadam_test.py` (CIELAB branch) | — |
| **SCS (pure)** | **18 / 25** | **37.8°** | **0** | `scripts/macadam_test.py` | 2026-04 |
| your entry | — | — | — | — | — |

---

## Challenge 2 — COMBVD dark-region correlation (L\* < 25)

Predict visual-difference (DV) ratings on COMBVD pairs restricted to the dark region (n = 176 pairs in the combined COMBVD split). Metric: Pearson r.

| Method | r (L\* < 25) | Parameters | Bootstrap 95% CI | Script | Date |
|--------|--------------|-----------|------------------|--------|------|
| CIELAB | 0.558 | 3 | — | `scripts/delta_e_scs00.py --region dark` | — |
| **SCS (pure Fisher-Bernoulli)** | **0.625** | **0** | (pending, see issue #TODO) | `scripts/delta_e_scs00.py --region dark --pure` | 2026-04 |
| your entry | — | — | — | — | — |

Note: the significance of the 0.625 / 0.558 gap has not yet been formally tested. A bootstrap CI submission is itself a welcome contribution.

---

## Challenge 3 — ΔE_SCS00 vs CIEDE2000 on full COMBVD

Predict DV on the full COMBVD (n = 3813 pairs, 5-fold CV). Metric: Pearson r with bootstrap CI.

| Method | r | Parameters (fitted) | 95% CI | p vs CIEDE2000 | Script | Date |
|--------|---|---------------------|--------|----------------|--------|------|
| CIELAB | 0.755 | 3 | — | — | — | — |
| CIEDE2000 | 0.878 | 5 | — | — | — | — |
| **ΔE_SCS00** | **0.893** | **6 (Ridge, α=1)** | **[0.885, 0.901]** | **p < 0.0001** | `scripts/delta_e_scs00.py` | 2026-04 |
| your entry | — | — | — | — | — |

---

## Open experimental predictions

The paper also states three falsifiable predictions not yet tested. These do not (yet) belong on a leaderboard — they need primary data. We flag them here so collaborators can pick one up.

### E1 — Koide-saturation JND null at 1/√2 ≈ 70.7%

**Claim.** A 2AFC just-noticeable-difference minimum at `S/S_max = 0.707`, at least 10% below the mean across `{0.60, 0.66, 0.707, 0.75, 0.80}`.

**Status.** Un-tested. Preregistration template: `bench/E1_koide_jnd/README.md` (TODO).

**Falsification.** Flat JND curve, or minimum displaced by more than ±3%.

### E2 — Chromatic-discrimination ceiling at 3×5×7 = 105

**Claim.** Maximum-likelihood effective chromatic dimensionality (Stubbs & Stubbs method on metameric isoluminant sets) saturates near 105 states per channel configuration.

**Status.** Un-tested. Requires colorimeter-calibrated display under ISO 19264-1 conditions.

**Falsification.** Effective dimensionality significantly different from 105 (e.g. > 200 or < 50).

### E3 — Tetrachromat fourth channel from p = 11

**Claim.** Functional tetrachromats show a partial fourth chromatic channel with predicted weight `γ_11 / Σ γ_p`.

**Status.** Un-tested. Requires genotype-screened recruitment (~2% of women).

**Falsification.** No separation in tetrachromat performance vs trichromat controls on SCS-selected metamer pairs.

---

## Replication / extension directions we particularly want

- **Line-element replication** with γ_p weights on Vos–Walraven or Wyszecki–Stiles line-element fits (MacAdam, Brown, Wyszecki ellipses).
- **CIECAM02 ablation**: which contribution of SCS+CAM02's r = 0.824 comes from the γ_p weighting vs the Fisher geodesic vs the interaction term.
- **Stockman–Sharpe vs HPE**: re-running the full COMBVD pipeline with Stockman–Sharpe 2° cone fundamentals instead of HPE.
- **Chromatic-adaptation extension**: an SCS-compatible D-factor analogue.
- **Individual-observer modelling**: Hofer et al. 2005 cone-mosaic variability on the γ_p structure.

---

## Contact

Open an issue on this repo, or email the maintainer (address in the paper).
