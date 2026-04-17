#!/usr/bin/env python3
"""
ΔE_SCS00 — Enhanced Color Difference: CIEDE2000 + Fisher Correction
=====================================================================

The SCS00 formula combines CIEDE2000's empirical cortical model with
the Fisher-Bernoulli luminance geodesic derived from Persistence Theory.

Formula:
    ΔE²_SCS00 = w₁·ΔE₀₀ + w₂·d_lum + w₃·ΔE₀₀² + w₄·ΔE₀₀·d_lum + w₅·d_lum²

where:
    ΔE₀₀  = standard CIEDE2000 color difference
    d_lum = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|  (Fisher-Bernoulli geodesic)
    ℓ     = Y/100 (normalized luminance)

The 5 weights are regressed on COMBVD (3813 pairs).
The geodesic d_lum has ZERO adjustable parameters (derived from s = 1/2).

Performance (5-fold CV on COMBVD):
    CIEDE2000:  r = 0.878
    ΔE_SCS00:   r = 0.893  (+1.8%)

Why it works:
    CIEDE2000's SL polynomial saturates at ~1.75 for L*→0.
    The Fisher geodesic arcsin(√ℓ) has sensitivity ∝ 1/√(ℓ(1-ℓ)),
    which diverges correctly in the dark.  The interaction term
    ΔE₀₀·d_lum captures the supra-linear coupling between luminance
    and chrominance that CIEDE2000 treats additively.

Usage:
    from delta_e_scs00 import delta_e_scs00, fit_scs00, SCS00Model

    # Fit on COMBVD (or your own dataset)
    model = fit_scs00()

    # Predict
    de = delta_e_scs00(Lab1, Lab2, xyz1, xyz2, model)

References:
    - PT_COLOR.tex, Section 8 (CIEDE2000 + SCS)
    - CIE Technical Report 142-2001 (CIEDE2000)
    - Čencov (1982), Fisher metric uniqueness
"""

import numpy as np
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ============================================================
# CIEDE2000 (CIE standard, self-contained)
# ============================================================

def ciede2000(Lab1, Lab2, kL=1, kC=1, kH=1):
    """
    CIEDE2000 color difference (CIE Technical Report 142-2001).

    Parameters:
        Lab1, Lab2: CIELAB (L*, a*, b*) as 3-vectors or (N,3) arrays
        kL, kC, kH: parametric factors (default 1:1:1)

    Returns:
        dE00: CIEDE2000 distance (scalar or 1D array)
    """
    L1, a1, b1 = np.asarray(Lab1).T if np.ndim(Lab1) > 1 else Lab1
    L2, a2, b2 = np.asarray(Lab2).T if np.ndim(Lab2) > 1 else Lab2

    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2
    C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7)))

    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = np.where(C1p * C2p == 0, 0.0,
           np.where(np.abs(h2p - h1p) <= 180, h2p - h1p,
           np.where(h2p - h1p > 180, h2p - h1p - 360, h2p - h1p + 360)))
    dHp = 2 * np.sqrt(np.maximum(C1p * C2p, 0)) * np.sin(np.radians(dhp / 2))

    L_bar_p = (L1 + L2) / 2
    C_bar_p = (C1p + C2p) / 2

    h_sum = h1p + h2p
    h_bar_p = np.where(C1p * C2p == 0, h_sum,
               np.where(np.abs(h1p - h2p) <= 180, h_sum / 2,
               np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2)))

    T = (1 - 0.17 * np.cos(np.radians(h_bar_p - 30))
           + 0.24 * np.cos(np.radians(2 * h_bar_p))
           + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
           - 0.20 * np.cos(np.radians(4 * h_bar_p - 63)))

    L50sq = (L_bar_p - 50)**2
    SL = 1 + 0.015 * L50sq / np.sqrt(20 + L50sq)
    SC = 1 + 0.045 * C_bar_p
    SH = 1 + 0.015 * C_bar_p * T

    Cb7 = C_bar_p**7
    RC = 2 * np.sqrt(Cb7 / (Cb7 + 25**7))
    d_theta = 30 * np.exp(-((h_bar_p - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    tL = dLp / (kL * SL)
    tC = dCp / (kC * SC)
    tH = dHp / (kH * SH)

    dE2 = tL**2 + tC**2 + tH**2 + RT * tC * tH
    return np.sqrt(np.maximum(dE2, 0))


# ============================================================
# FISHER-BERNOULLI LUMINANCE GEODESIC
# ============================================================

def fisher_luminance(Y1, Y2, Y_ref=100.0):
    """
    Fisher-Bernoulli luminance geodesic.

    d_lum = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|

    where ℓ = Y/Y_ref ∈ (0, 1).

    This is the UNIQUE geodesic distance on the Bernoulli manifold
    (variance-stabilizing transform for p=2 binary channel).
    Zero adjustable parameters — derived from s = 1/2.
    """
    ell1 = np.clip(np.asarray(Y1) / Y_ref, 1e-10, 1 - 1e-10)
    ell2 = np.clip(np.asarray(Y2) / Y_ref, 1e-10, 1 - 1e-10)
    return 2 * np.abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))


# ============================================================
# SCS00 MODEL
# ============================================================

@dataclass
class SCS00Model:
    """Fitted SCS00 model coefficients."""
    w: np.ndarray       # 5 polynomial weights
    intercept: float
    r_cv: float         # cross-validated correlation
    r_ciede2000: float  # CIEDE2000 baseline correlation
    n_train: int

    def __repr__(self):
        return (f"SCS00Model(r={self.r_cv:.4f}, "
                f"ΔE₀₀_ref={self.r_ciede2000:.4f}, "
                f"n={self.n_train})")


def _build_features(de00, d_lum):
    """Build degree-2 polynomial features from (ΔE₀₀, d_lum)."""
    de00 = np.asarray(de00, dtype=float)
    d_lum = np.asarray(d_lum, dtype=float)
    return np.column_stack([
        de00,               # ΔE₀₀
        d_lum,              # d_lum
        de00**2,            # ΔE₀₀²
        de00 * d_lum,       # ΔE₀₀ · d_lum  (the KEY interaction)
        d_lum**2,           # d_lum²
    ])


def fit_scs00(combvd_path=None):
    """
    Fit the SCS00 model on COMBVD.

    Returns:
        SCS00Model with fitted weights.
    """
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from scipy import stats

    if combvd_path is None:
        # Datasets live under the repo root, not under scripts/
        _here = os.path.dirname(os.path.abspath(__file__))
        _repo = os.path.dirname(_here)
        combvd_path = os.path.join(_repo, 'datasets', 'COMBVD_3813.csv')
        if not os.path.exists(combvd_path):
            # fallback to scripts/datasets (legacy layout) for backward compat
            combvd_path = os.path.join(_here, 'datasets', 'COMBVD_3813.csv')

    df = pd.read_csv(combvd_path)
    N = len(df)
    dv = df['DV'].values

    # Compute CIEDE2000
    Lab1 = df[['L1', 'a1', 'b1']].values
    Lab2 = df[['L2', 'a2', 'b2']].values
    de00 = np.array([ciede2000(Lab1[i], Lab2[i]) for i in range(N)])

    # Compute Fisher luminance
    d_lum = fisher_luminance(df['Y1'].values, df['Y2'].values)

    # Build features
    X = _build_features(de00, d_lum)

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(N)
    for tr, te in kf.split(X):
        sc = StandardScaler()
        m = Ridge(alpha=1.0).fit(sc.fit_transform(X[tr]), dv[tr])
        y_pred[te] = m.predict(sc.transform(X[te]))

    r_cv = stats.pearsonr(y_pred, dv)[0]
    r_00 = stats.pearsonr(de00, dv)[0]

    # Fit on full data for final weights
    sc_full = StandardScaler()
    X_sc = sc_full.fit_transform(X)
    m_full = Ridge(alpha=1.0).fit(X_sc, dv)

    # Convert to raw (unstandardized) weights
    w_raw = m_full.coef_ / sc_full.scale_
    b_raw = m_full.intercept_ - np.sum(m_full.coef_ * sc_full.mean_ / sc_full.scale_)

    return SCS00Model(
        w=w_raw,
        intercept=b_raw,
        r_cv=r_cv,
        r_ciede2000=r_00,
        n_train=N,
    )


def delta_e_scs00(Lab1, Lab2, Y1, Y2, model=None):
    """
    Compute ΔE_SCS00 color difference.

    Parameters:
        Lab1, Lab2: CIELAB (L*, a*, b*)
        Y1, Y2: CIE Y tristimulus (luminance)
        model: fitted SCS00Model (if None, uses default weights)

    Returns:
        ΔE_SCS00 (float or array)
    """
    de00 = ciede2000(Lab1, Lab2)
    d_lum = fisher_luminance(Y1, Y2)
    X = _build_features(de00, d_lum)

    if model is None:
        model = _DEFAULT_MODEL

    return np.maximum(model.intercept + X @ model.w, 0)


# ============================================================
# VALIDATION AND REPORTING
# ============================================================

def validate(verbose=True):
    """
    Run full validation on COMBVD.
    Returns dict with all metrics.
    """
    import pandas as pd
    from scipy import stats
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)
    data_path = os.path.join(_repo, 'datasets', 'COMBVD_3813.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(_here, 'datasets', 'COMBVD_3813.csv')
    df = pd.read_csv(data_path)
    N = len(df)
    dv = df['DV'].values

    Lab1 = df[['L1', 'a1', 'b1']].values
    Lab2 = df[['L2', 'a2', 'b2']].values
    de00 = np.array([ciede2000(Lab1[i], Lab2[i]) for i in range(N)])
    d_lum = fisher_luminance(df['Y1'].values, df['Y2'].values)
    de_lab = np.sqrt(np.sum((Lab2 - Lab1)**2, axis=1))

    X = _build_features(de00, d_lum)
    feat_names = ['ΔE₀₀', 'd_lum', 'ΔE₀₀²', 'ΔE₀₀·d_lum', 'd_lum²']

    # CV prediction
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(N)
    for tr, te in kf.split(X):
        sc = StandardScaler()
        m = Ridge(alpha=1.0).fit(sc.fit_transform(X[tr]), dv[tr])
        y_pred[te] = m.predict(sc.transform(X[te]))

    r_scs00 = stats.pearsonr(y_pred, dv)[0]
    r_de00 = stats.pearsonr(de00, dv)[0]
    r_lab = stats.pearsonr(de_lab, dv)[0]

    # Per-region analysis
    L_mid = (df['L1'].values + df['L2'].values) / 2
    regions = {
        'dark (L*<25)': L_mid < 25,
        'mid (25≤L*<75)': (L_mid >= 25) & (L_mid < 75),
        'light (L*≥75)': L_mid >= 75,
    }

    # Full fit for weights
    sc_full = StandardScaler()
    m_full = Ridge(alpha=1.0).fit(sc_full.fit_transform(X), dv)
    beta = m_full.coef_
    w_raw = m_full.coef_ / sc_full.scale_
    b_raw = m_full.intercept_ - np.sum(m_full.coef_ * sc_full.mean_ / sc_full.scale_)

    # Bootstrap significance
    n_boot = 2000
    rng = np.random.RandomState(42)
    dr_boot = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        r_00_b = stats.pearsonr(de00[idx], dv[idx])[0]
        r_20_b = stats.pearsonr(y_pred[idx], dv[idx])[0]
        dr_boot[b] = r_20_b - r_00_b

    if verbose:
        print("=" * 65)
        print("ΔE_SCS00 — Validation Report")
        print("=" * 65)

        print(f"\n  Dataset: COMBVD ({N} pairs, 5-fold CV)")

        print(f"\n  ┌─────────────────────────┬────────┬────────┐")
        print(f"  │ Method                  │   r    │ #param │")
        print(f"  ├─────────────────────────┼────────┼────────┤")
        print(f"  │ CIELAB ΔE*ab            │ {r_lab:.4f} │      3 │")
        print(f"  │ CIEDE2000 ΔE₀₀          │ {r_de00:.4f} │      5 │")
        print(f"  │ ΔE_SCS00 (this work)    │ {r_scs00:.4f} │      5 │")
        print(f"  └─────────────────────────┴────────┴────────┘")

        print(f"\n  Improvement over CIEDE2000: Δr = +{r_scs00 - r_de00:.4f}")
        ci_lo, ci_hi = np.percentile(dr_boot, [2.5, 97.5])
        p_val = np.mean(dr_boot <= 0)
        print(f"  Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"  P(Δr ≤ 0) = {p_val:.4f}  {'(significant)' if p_val < 0.05 else ''}")

        print(f"\n  Per-region performance:")
        for name, mask in regions.items():
            n = mask.sum()
            if n < 10:
                continue
            r1 = stats.pearsonr(de00[mask], dv[mask])[0]
            r2 = stats.pearsonr(y_pred[mask], dv[mask])[0]
            print(f"    {name:20s} (n={n:4d}): "
                  f"CIEDE2000 r={r1:.3f}  SCS00 r={r2:.3f}  Δ={r2-r1:+.3f}")

        print(f"\n  Formula:")
        print(f"    ΔE_SCS00 = {b_raw:.4f}")
        for name, w in zip(feat_names, w_raw):
            print(f"               + {w:+.4f} · {name}")

        print(f"\n  Standardized weights (|β|):")
        for name, b in sorted(zip(feat_names, beta), key=lambda x: -abs(x[1])):
            print(f"    {name:15s}: β = {b:+.4f}")

        print(f"\n  Physical interpretation:")
        print(f"    ΔE₀₀         : CIEDE2000 cortical model (empirical)")
        print(f"    d_lum        : Fisher-Bernoulli geodesic (derived, s=1/2)")
        print(f"    ΔE₀₀·d_lum   : luminance×chrominance coupling (key term)")
        print(f"    ΔE₀₀², d_lum²: nonlinear sensitivity scaling")

    return {
        'r_scs00': r_scs00, 'r_ciede2000': r_de00, 'r_cielab': r_lab,
        'weights': w_raw, 'intercept': b_raw, 'beta': beta,
        'dr_ci': (ci_lo, ci_hi), 'p_value': p_val,
        'y_pred': y_pred, 'de00': de00, 'd_lum': d_lum,
    }


# Default model (fitted on COMBVD at import time if available)
try:
    _DEFAULT_MODEL = fit_scs00()
except Exception:
    _DEFAULT_MODEL = None


if __name__ == '__main__':
    validate(verbose=True)
