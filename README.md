# Sieve Color Space (SCS)

[Français](README_FR.md)

An information-geometric substrate for color derived from a single assumption (`s = 1/2`) on a prime-gap dynamical system — the [Persistence Theory](https://zenodo.org/records/19583187) framework. SCS is proposed as a principled geometric *layer* in a factored color-appearance architecture, **alongside** existing tools (CIECAM02, CIEDE2000, iCAM, CAM02-UCS, J<sub>z</sub>A<sub>z</sub>B<sub>z</sub>), not as a replacement for them.

**What is derived with zero fitted parameters:** the three-channel simplex, the `γ₃ > γ₅ > γ₇` ordering that matches the L>M>S cone-bandwidth ordering, the Fisher metric on the simplex, the Bhattacharyya and Fisher–Bernoulli geodesics, and the saturation–luminance sum rule.

**What remains fitted / measured:** the hybrid color-difference metrics (SCS+CIECAM02 and ΔE_SCS00) use Ridge-regressed weights on top of the derived geometry. The paper labels these as hybrids throughout. Chromatic adaptation, spectral sensitivity beyond the HPE cone-fundamental linearization, viewing conditions, and flare are **not** modeled by SCS (see *Scope and limits* in the paper); CIECAM02 and CIEDE2000 remain the appropriate tools where those effects dominate.

[**Quick online demo**](https://igrekess.github.io/SieveColorSpace/demonstration/demo.html)
· [English PDF](PT_COLOR.pdf)
· [French PDF](PT_COLOR_FR.pdf)
· [**Leaderboard**](leaderboard.md) — open benchmark, pull requests welcome
· [V4 fMRI results](docs/v4_results.md) — biological validation

---

## Reproducing the paper's headline numbers

Each claim in the paper is reproducible from a specific script in `scripts/`. A clean clone + `pip install -e .` + ~1 min of CPU time gives you the four key numbers:

```bash
# 1. MacAdam ellipse orientation — SCS wins 18/25, RMS Δθ = 37.8°
python3 scripts/macadam_test.py

# 2. Pure SCS color difference on COMBVD — r ≈ 0.500 (zero fitted weights)
python3 scripts/delta_e_scs.py

# 3. SCS + CIECAM02 hybrid on COMBVD — r = 0.824 (6 fitted weights)
python3 scripts/scs_cam02_hybrid.py

# 4. ΔE_SCS00 vs CIEDE2000 on COMBVD — r = 0.893 vs 0.878, p < 0.0001
python3 scripts/delta_e_scs00.py

# 5. V4 fMRI channel weights — L−M = 0.37 vs SCS prediction γ₃/Σγ = 0.385
python3 scripts/v4_summary.py      # quick reproduction from pre-computed CSV
python3 scripts/v4_refined_analysis.py  # full pipeline (needs OpenNeuro ds005521)
```

Each script prints its result next to the paper's cited value. If your run disagrees, open an issue — we want to know.

---

## Honest performance on COMBVD (3813 pairs, 5-fold CV)

| Method | Fitted weights | r | vs CIEDE2000 |
|--------|---------------:|--:|:--|
| SCS pure (single-feature `delta_e_scs`) | **0** | **0.492** | geometric baseline |
| CIELAB | 3 | 0.755 | –14% |
| SCS pure + Ridge (d_lum, d_chrom) | 2 | 0.642 | –27% |
| SCS + CIECAM02 hybrid | 6 | 0.824 | –6% |
| **CIEDE2000** | 5 | 0.878 | reference |
| **ΔE_SCS00** (CIEDE2000 + Fisher–Bernoulli) | 5 | **0.893** | **+1.8%**, p < 0.0001 |

The pure SCS metric (zero fitted parameters) sits **below** CIELAB globally. It wins in two specific regimes:
- **Dark region** (L\* < 25): r = 0.625 vs CIELAB's 0.558
- **MacAdam ellipse orientation**: 18/25 wins, RMS Δθ = 37.8° vs CIELAB's 52.0°

The value of SCS as a working metric is as an **additive information channel** on top of cortical models (CIECAM02, CIEDE2000) — never as a replacement for them.

The `ΔE_SCS00` formula:
```
ΔE_SCS00 = w₀ + w₁·ΔE₀₀ + w₂·d_lum + w₃·ΔE₀₀² + w₄·ΔE₀₀·d_lum + w₅·d_lum²
```
`d_lum = 2 |arcsin(√ℓ₁) − arcsin(√ℓ₂)|` is derived from `s = 1/2` with zero fitted parameters. The six weights `w₀…w₅` are Ridge-regressed on COMBVD (`α = 1`, 5-fold CV, seed 42). The key term is the interaction `ΔE₀₀ · d_lum` — information the cortical model alone does not carry.

---

## Install

```bash
pip install -e .
```

Or from PyPI:
```bash
pip install scs
```

Required dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `colour-science` (for CIECAM02 in the hybrid fitter). `matplotlib` for figures.

---

## Quick start

```python
from scs import delta_e, to_scs, fisher_luminance, gft_check

# Color difference (pure geodesic, 0 parameters)
d = delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
print(f"ΔE_SCS = {d:.4f}")

# XYZ → SCS coordinates
c = to_scs([0.95, 1.0, 1.09])
print(f"ℓ={c.ell:.2f}  S={c.S:.3f} bits  θ={c.hue:.0f}°")
print(f"π = ({c.pi[0]:.3f}, {c.pi[1]:.3f}, {c.pi[2]:.3f})")

# Fisher–Bernoulli luminance geodesic (used in ΔE_SCS00)
d_lum = fisher_luminance(Y1=50, Y2=45)

# From CIELAB values
from scs import delta_e_lab
d = delta_e_lab(50, 25, 10, 52, 28, 8)

# S + L sum rule (generic identity on any 3-outcome distribution)
S, L, total, err = gft_check([0.4, 0.35, 0.25])
print(f"S + L = {total:.6f} (err = {err:.1e})")
```

### API summary

| Function | Purpose |
|----------|---------|
| `delta_e(xyz1, xyz2)` | SCS color difference (pure geodesic, 0 parameters) |
| `delta_e_lab(L1,a1,b1, L2,a2,b2)` | Same, from CIELAB inputs |
| `to_scs(xyz)` | XYZ → `SCSColor(ell, S, hue, pi)` |
| `fisher_luminance(Y1, Y2)` | Fisher–Bernoulli geodesic `d_lum` |
| `saturation(pi)` | Kullback–Leibler divergence from uniform |
| `luminance_entropy(pi)` | Shannon entropy `H(π)` |
| `gft_check(pi)` | Verify `S + L = log 3` |

### Constants (all derived from `s = 1/2` at `μ* = 15`)

| Constant | Value | Source |
|----------|-------|--------|
| `MU_STAR` | 15 | Unique fixed point (Theorem T5) |
| `GAMMAS` | (0.808, 0.696, 0.595) | Anomalous dimensions at `μ* = 15` |
| `PRIMES` | (3, 5, 7) | Active primes |
| `Q_REL` | 13/15 | Vertex branch of the sieve bifurcation |
| `Q_THERM` | e^(−1/15) | Edge branch |
| `W_LUM` | 3/4 | Luminance weight `N/(N+1)` |
| `W_CHROM` | 1/4 | Chromaticity weight `1/(N+1)` |

---

## Scripts

All scripts live in `scripts/`. Run from anywhere; they resolve data paths from the repo root.

| Script | What it does | Paper claim it reproduces |
|--------|--------------|---------------------------|
| `scs.py` | Core module: coordinates, metrics, sum rule, self-test | — |
| `scs_companion.py` | PT quantities, numerical verification, figure generation | — |
| `delta_e_scs.py` | Pure SCS color difference (combined metric: Fisher + Fubini–Study + bifurcation) | `r = 0.492` baseline |
| `delta_e_scs00.py` | Hybrid ΔE_SCS00 = CIEDE2000 + Fisher–Bernoulli | **r = 0.893 vs 0.878**, p < 0.0001 |
| `scs_cam02_hybrid.py` | SCS + CIECAM02 hybrid fitter (Ridge, 5-fold CV) | **r = 0.824** |
| `macadam_test.py` | MacAdam 25-ellipse validation with combined metric | **18/25 wins, RMS 37.8°** |
| `v4_summary.py` | V4 channel-weight reproduction from pre-computed CSV | L−M ≈ 0.37 vs γ₃/Σγ = 0.385 |
| `v4_neural_extraction.py` | Full V4 fMRI extraction (requires OpenNeuro ds005521) | Generates `v4_bold_response.csv` |
| `v4_refined_analysis.py` | Ridge-regression V4 → SCS channel mapping | L−M = 0.373 (3.2% gap) |
| `v4_hybrid_model.py` | V4 opponent channels as CAM02 proxy in COMBVD hybrid | r = 0.675 (V4-based, below full CAM02) |
| `v4_analysis_plots.py` | V4 figures | — |
| `scs_ciede2000_analysis.py` | SCS vs CIEDE2000 comparative analysis | — |
| `exploratory/` | Scratch R&D: `model20_deep_analysis`, `push_beyond_ciede2000`, `pt_matrix` | Not load-bearing |

---

## Article

The paper is **self-contained**: the appendix provides derivations of all Persistence Theory results used (T1 forbidden transitions, `s = 1/2`, the sum rule, the vertex–edge bifurcation, holonomy, anomalous dimensions, active primes, fixed point). A reader does not need to adopt PT as a whole to evaluate the SCS claims — see the explicit "Reading this paper without a commitment to the broader framework" paragraph in the introduction.

- [English PDF](PT_COLOR.pdf)
- [French PDF](PT_COLOR_FR.pdf)

---

## Derivation chain

```
s = 1/2  →  T₁ forbidden transitions  →  T₃ = antidiag(1, 1)
         →  T₅  →  μ* = 15,  active primes {3, 5, 7}
         →  holonomy  →  sin² θ_p,  α_EM ≈ 1/137
         →  Rydberg → Balmer  →  visible window 380–656 nm
         →  γ_p hierarchy: γ₃ > γ₅ > γ₇   (matches L>M>S bandwidth)
         →  Fisher metric (Čencov, unique up to a constant)  →  SCS
```

---

## Open problems (collaboration invited)

The paper states three falsifiable predictions, each tractable in a three-month experiment. Protocol sketches and preregistration hooks are in [`leaderboard.md`](leaderboard.md).

- **E1** — Koide-saturation JND null at `S/S_max = 1/√2 ≈ 70.7%` (2AFC discrimination).
- **E2** — Chromatic-dimensionality ceiling at `3 × 5 × 7 = 105` states (maximum-likelihood method on metameric sets).
- **E3** — Tetrachromat fourth-channel signature predicted from `γ₁₁`.

The paper also explicitly flags directions where the authors need help (chromatic adaptation, individual-observer modeling, psychophysical protocol design). Correspondence, replications, counter-experiments, and forked repositories are actively invited — open an issue or PR.

---

## Documentation

- [API Reference](docs/api.md) — functions, classes, constants
- [Theory](docs/theory.md) — mathematical foundations (condensed)
- [Colormaps](docs/colormaps.md) — 12 Fisher-geodesic colormaps
- [Grade format](docs/grade-format.md) — `.scs` file format for color grading
- [V4 fMRI results](docs/v4_results.md) — biological validation details
- [Leaderboard](leaderboard.md) — benchmark submissions and open predictions

---

## References

Y. Senez, *"The Sieve Color Space: A First-Principles Color Space from the Sieve of Eratosthenes"* (2026). [PDF](PT_COLOR.pdf) · [Zenodo](https://zenodo.org/records/19614967)

Y. Senez, *"Persistence Theory: Mathematical Foundations of Prime Gap Dynamics"*, preprint (2026). [Zenodo](https://zenodo.org/records/19583187)

Key adjacent literature cited in the paper: Wyszecki & Stiles (2000), Vos & Walraven (1972), Derrington–Krauskopf–Lennie (1984), MacLeod–Boynton (1979), Wuerger et al. (2002), Koenderink (2010), Fairchild (2013), Hofer et al. (2005), Amari (1985), Fairchild & Johnson iCAM (2004), Luo et al. CAM02-UCS (2006), Safdar et al. JzAzBz (2017), Conway et al. (2025, OpenNeuro ds005521).

## License

MIT — Yan Senez
