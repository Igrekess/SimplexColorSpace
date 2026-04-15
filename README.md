# PT Sieve Color Space (SCS)

[Francais](README_FR.md)

A first-principles color space derived from
[Persistence Theory](https://zenodo.org/records/19520809), a mathematical framework based on prime gap dynamics.

**Single input:** `s = 1/2`. **Zero adjustable parameters.**

All ratios and structures are derived.

[**For a quick online demo :**](https://igrekess.github.io/SimplexColorSpace/demonstration/demo.html)[ Click here](https://igrekess.github.io/SimplexColorSpace/demonstration/demo.html)

**New (April 2026):** The SCS00 formula (CIEDE2000 + Fisher-Bernoulli geodesic)
**surpasses CIEDE2000** on COMBVD (r = 0.893 vs 0.878, p < 0.0001).

The article is now self-contained with a full mathematical appendix.
Read the [English PDF](article/PT_COLOR.pdf) or the [French PDF](article/PT_COLOR_FR.pdf).


## Install

```bash
pip install scs
```

Or from this repo:

```bash
pip install -e .
```

## Package: `scs`

The core Python package provides color coordinates, color differences,
and conservation law verification — all derived from `s = 1/2` with
zero adjustable parameters.

```python
from scs import delta_e, to_scs, fisher_luminance, gft_check

# --- Color difference (0 parameters) ---
d = delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
print(f"ΔE_SCS = {d:.4f}")

# --- Convert XYZ to SCS coordinates ---
c = to_scs([0.95, 1.0, 1.09])
print(f"ℓ={c.ell:.2f}, S={c.S:.3f} bits, θ={c.hue:.0f}°")
print(f"π = ({c.pi[0]:.3f}, {c.pi[1]:.3f}, {c.pi[2]:.3f})")

# --- Fisher-Bernoulli luminance geodesic (used in SCS00) ---
d_lum = fisher_luminance(Y1=50, Y2=45)

# --- From CIELAB values ---
from scs import delta_e_lab
d = delta_e_lab(50, 25, 10, 52, 28, 8)

# --- GFT conservation: S + L = log₂(3), always ---
S, L, total, err = gft_check([0.4, 0.35, 0.25])
print(f"S + L = {total:.6f} (err = {err:.1e})")
```

### API summary

| Function | Purpose |
|----------|---------|
| `delta_e(xyz1, xyz2)` | SCS color difference (0 parameters) |
| `delta_e_lab(L1,a1,b1, L2,a2,b2)` | Same, from CIELAB inputs |
| `to_scs(xyz)` | XYZ → `SCSColor(ell, S, hue, pi)` |
| `fisher_luminance(Y1, Y2)` | Fisher-Bernoulli geodesic d_lum |
| `saturation(pi)` | D_KL(π \|\| uniform) |
| `luminance_entropy(pi)` | H(π) |
| `gft_check(pi)` | Verify S + L = log₂(3) |

### Constants (all derived, not chosen)

| Constant | Value | Source |
|----------|-------|--------|
| `MU_STAR` | 15 | Unique fixed point (T5) |
| `GAMMAS` | (0.808, 0.696, 0.595) | Anomalous dimensions at μ*=15 |
| `PRIMES` | (3, 5, 7) | Active primes |
| `Q_REL` | 13/15 | Vertex branch |
| `Q_THERM` | e^(-1/15) | Edge branch |
| `W_LUM` | 3/4 | Luminance weight N/(N+1) |
| `W_CHROM` | 1/4 | Chromaticity weight 1/(N+1) |

## Performance

| Method | Parameters | r (COMBVD) | vs CIEDE2000 |
|--------|-----------|------------|--------------|
| SCS pure | **0** | 0.583 | geometric baseline |
| CIELAB | 3 | 0.755 | -14% |
| SCS + CAM02 | 6 | 0.824 | -6% |
| **CIEDE2000** | **5** | **0.878** | **reference** |
| **SCS00** | **5** | **0.893** | **+1.8%** |

The SCS00 formula combines CIEDE2000's cortical model with the
Fisher-Bernoulli luminance geodesic derived from PT:

```
SCS00 = w0 + w1*DE00 + w2*d_lum + w3*DE00^2 + w4*DE00*d_lum + w5*d_lum^2
```

where `d_lum = 2|arcsin(sqrt(l1)) - arcsin(sqrt(l2))|` has **zero adjustable
parameters** (derived from s = 1/2). The key term is `DE00 * d_lum`:
the interaction between the cortical metric and the retinal geodesic.

On MacAdam's 25 discrimination ellipses, the SCS metric wins **18/25**
with zero parameters versus CIELAB's three.

## Article

The paper is **self-contained**: an appendix provides complete proofs
of all Persistence Theory results (T1, s = 1/2, GFT, bifurcation,
holonomy, anomalous dimensions, active primes, fixed point).

- [English PDF](article/PT_COLOR.pdf)
- [French PDF](article/PT_COLOR_FR.pdf)

## Scripts

| Script | Purpose |
|--------|---------|
| `delta_e_scs00.py` | **SCS00 formula** — CIEDE2000 + Fisher geodesic |
| `delta_e_scs.py` | Pure SCS color difference (0 parameters) |
| `scs_companion.py` | PT quantities, figures, numerical verification |
| `macadam_test.py` | MacAdam ellipse validation |
| `scs_ciede2000_analysis.py` | SCS vs CIEDE2000 comparative analysis |
| `push_beyond_ciede2000.py` | Systematic exploration of 20+ models |
| `model20_deep_analysis.py` | Deep analysis of the SCS00 model |

## Derivation chain

```
s = 1/2 → T₁ → T₃ = antidiag(1,1)
        → T₅ → μ* = 15, active primes {3,5,7}
        → holonomy → sin²θ_p, α_EM ≈ 1/137
        → Rydberg → Balmer → 380–656 nm (visible window)
        → γ_p hierarchy → Red (p=3) > Green (p=5) > Blue (p=7)
        → Fisher metric (Čencov, unique) → SCS
```

## Documentation

- [API Reference](docs/api.md) — all functions and classes
- [Theory](docs/theory.md) — mathematical foundations
- [Colormaps](docs/colormaps.md) — the 12 Fisher-geodesic colormaps
- [Grade Format](docs/grade-format.md) — the .scs file format

## References

Y. Senez, [*"The Sieve Color Space: A First-Principles Color Space
from the Sieve of Eratosthenes"* (2026).](article/PT_COLOR.pdf)

Y. Senez, [*"Persistence Theory: Mathematical Foundations of Prime Gap
Dynamics"*, preprint (2026).](https://zenodo.org/records/19520809)

## License

MIT — Yan Senez 
