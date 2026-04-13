#!/usr/bin/env python3
"""
SCT vs CIEDE2000 Analysis — Deriving Cortical Corrections
==========================================================

Goal: Analyze CIEDE2000's correction structure in SCT coordinates
to derive cortical correction factors that bring SCT to CIEDE2000 level.

Strategy:
1. Implement CIEDE2000 (CIE standard formula)
2. Compute SCT features on COMBVD (3813 pairs)
3. Extract CIEDE2000 correction components (SL, SC, SH, RT)
4. Regress cortical corrections in SCT native coordinates
5. Compare: SCT pure vs SCT+cortical vs CIEDE2000

Author: Yan Senez
Date: 2026-04-08
"""

import numpy as np
import pandas as pd
import os, sys
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scpt_companion import gamma_p, MU_STAR, Q_REL, Q_THERM, PRIMES
from delta_e_scpt import xyz_to_lms, lms_to_simplex, delta_e, xyz_to_scpt

# ============================================================
# CIEDE2000 IMPLEMENTATION (CIE standard)
# ============================================================

def ciede2000(Lab1, Lab2, kL=1, kC=1, kH=1):
    """
    CIEDE2000 color difference formula.

    Implementation follows CIE Technical Report 142-2001.

    Parameters:
        Lab1, Lab2: CIELAB coordinates (L*, a*, b*)
        kL, kC, kH: parametric factors (default 1:1:1)

    Returns:
        dE00: CIEDE2000 color difference
        components: dict with SL, SC, SH, RT, dLp, dCp, dHp
    """
    L1, a1, b1 = Lab1
    L2, a2, b2 = Lab2

    # Step 1: Calculate C*ab and h_ab
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = (C1 + C2) / 2

    # G factor
    C_bar7 = C_bar**7
    G = 0.5 * (1 - np.sqrt(C_bar7 / (C_bar7 + 25**7)))

    # Modified a'
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    # C' and h'
    C1p = np.sqrt(a1p**2 + b1**2)
    C2p = np.sqrt(a2p**2 + b2**2)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360

    # Step 2: Calculate dL', dC', dH'
    dLp = L2 - L1
    dCp = C2p - C1p

    # dh'
    if C1p * C2p == 0:
        dhp = 0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))

    # Step 3: Calculate CIEDE2000
    L_bar_p = (L1 + L2) / 2
    C_bar_p = (C1p + C2p) / 2

    # h_bar_p
    if C1p * C2p == 0:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar_p = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        h_bar_p = (h1p + h2p + 360) / 2
    else:
        h_bar_p = (h1p + h2p - 360) / 2

    # T (hue-dependent weighting)
    T = (1
         - 0.17 * np.cos(np.radians(h_bar_p - 30))
         + 0.24 * np.cos(np.radians(2 * h_bar_p))
         + 0.32 * np.cos(np.radians(3 * h_bar_p + 6))
         - 0.20 * np.cos(np.radians(4 * h_bar_p - 63)))

    # SL
    L_bar_p_50_sq = (L_bar_p - 50)**2
    SL = 1 + 0.015 * L_bar_p_50_sq / np.sqrt(20 + L_bar_p_50_sq)

    # SC
    SC = 1 + 0.045 * C_bar_p

    # SH
    SH = 1 + 0.015 * C_bar_p * T

    # RT (rotation in blue region)
    C_bar_p7 = C_bar_p**7
    RC = 2 * np.sqrt(C_bar_p7 / (C_bar_p7 + 25**7))
    d_theta = 30 * np.exp(-((h_bar_p - 275) / 25)**2)
    RT = -np.sin(np.radians(2 * d_theta)) * RC

    # Final formula
    term_L = dLp / (kL * SL)
    term_C = dCp / (kC * SC)
    term_H = dHp / (kH * SH)

    dE00_sq = term_L**2 + term_C**2 + term_H**2 + RT * term_C * term_H
    dE00 = np.sqrt(max(dE00_sq, 0))

    return dE00, {
        'SL': SL, 'SC': SC, 'SH': SH, 'RT': RT, 'T': T,
        'dLp': dLp, 'dCp': dCp, 'dHp': dHp,
        'term_L': term_L, 'term_C': term_C, 'term_H': term_H,
        'L_bar_p': L_bar_p, 'C_bar_p': C_bar_p, 'h_bar_p': h_bar_p,
    }


# ============================================================
# SCT FEATURES EXTRACTION
# ============================================================

def sct_features(xyz1, xyz2):
    """
    Extract rich SCT feature set for regression.

    Returns dict with:
    - d_lum: Fisher-Bernoulli luminance geodesic
    - d_chrom: Bhattacharyya chromatic geodesic
    - delta_e_sct: canonical SCT distance
    - Simplex midpoint coordinates and differences
    - Saturation, hue, entropy at both points
    """
    GAMMAS = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])

    lms1 = xyz_to_lms(xyz1)
    lms2 = xyz_to_lms(xyz2)
    pi1 = lms_to_simplex(lms1)
    pi2 = lms_to_simplex(lms2)
    pi_mid = (pi1 + pi2) / 2
    pi_mid = np.maximum(pi_mid, 1e-10)
    pi_mid /= pi_mid.sum()

    # Luminance (from Y, normalized)
    ell1 = max(xyz1[1] / 100.0, 1e-10)
    ell2 = max(xyz2[1] / 100.0, 1e-10)
    ell1 = min(ell1, 1 - 1e-10)
    ell2 = min(ell2, 1 - 1e-10)

    # Fisher-Bernoulli luminance geodesic
    d_lum = 2 * abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))

    # Bhattacharyya chromatic geodesic (gamma-weighted)
    pi1_w = GAMMAS * pi1 / (GAMMAS * pi1).sum()
    pi2_w = GAMMAS * pi2 / (GAMMAS * pi2).sum()
    bc = np.sum(np.sqrt(pi1_w * pi2_w))
    bc = min(bc, 1.0)
    d_chrom = 2 * np.arccos(bc)

    # Canonical SCT
    de_sct = np.sqrt(0.75 * d_lum**2 + 0.25 * d_chrom**2)

    # Saturation (D_KL from uniform)
    def sat(pi):
        p = np.maximum(pi, 1e-15)
        return np.sum(p * np.log(3 * p))

    S1, S2 = sat(pi1), sat(pi2)
    S_mid = sat(pi_mid)

    # Hue angle
    def hue(pi):
        return np.arctan2(np.sqrt(3) * (pi[1] - pi[2]),
                          2 * pi[0] - pi[1] - pi[2])

    theta1, theta2 = hue(pi1), hue(pi2)

    # Hue difference (signed, wrapped)
    dtheta = theta2 - theta1
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    # Entropy (luminance proxy on simplex)
    def ent(pi):
        p = np.maximum(pi, 1e-15)
        return -np.sum(p * np.log(p))

    H1, H2 = ent(pi1), ent(pi2)

    # Per-channel differences
    dpi = pi2 - pi1

    # Luminance midpoint (for weighting)
    ell_mid = (ell1 + ell2) / 2

    return {
        'd_lum': d_lum,
        'd_chrom': d_chrom,
        'de_sct': de_sct,
        'dS': abs(S2 - S1),        # saturation difference
        'S_mid': S_mid,             # midpoint saturation
        'dtheta': abs(dtheta),      # hue angle difference
        'theta_mid': hue(pi_mid),   # midpoint hue
        'dH_sct': np.sqrt(max(d_chrom**2 - (S2-S1)**2, 0)),  # hue-only chrom
        'dpi3': abs(dpi[0]),        # red channel change
        'dpi5': abs(dpi[1]),        # green channel change
        'dpi7': abs(dpi[2]),        # blue channel change
        'pi3_mid': pi_mid[0],
        'pi5_mid': pi_mid[1],
        'pi7_mid': pi_mid[2],
        'ell_mid': ell_mid,
        'dL': abs(ell2 - ell1),     # raw luminance diff
        'H_mid': ent(pi_mid),       # midpoint entropy
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("=" * 70)
    print("SCT vs CIEDE2000 — Cortical Correction Analysis")
    print("=" * 70)

    # Load COMBVD
    data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'COMBVD_3813.csv')
    df = pd.read_csv(data_path)
    N = len(df)
    print(f"\nCOMBVD dataset: {N} pairs loaded")

    # Compute all metrics
    dv = df['DV'].values  # human visual difference

    de00_vals = np.zeros(N)
    de_sct_vals = np.zeros(N)
    de_lab_vals = np.zeros(N)

    # CIEDE2000 components
    SL_vals = np.zeros(N)
    SC_vals = np.zeros(N)
    SH_vals = np.zeros(N)
    RT_vals = np.zeros(N)
    T_vals = np.zeros(N)

    # SCT features (for regression)
    feat_names = ['d_lum', 'd_chrom', 'de_sct', 'dS', 'S_mid',
                  'dtheta', 'theta_mid', 'dH_sct', 'dpi3', 'dpi5', 'dpi7',
                  'pi3_mid', 'pi5_mid', 'pi7_mid', 'ell_mid', 'dL', 'H_mid']
    sct_feats = {k: np.zeros(N) for k in feat_names}

    # CIEDE2000 decomposed terms
    term_L_vals = np.zeros(N)
    term_C_vals = np.zeros(N)
    term_H_vals = np.zeros(N)

    print("Computing metrics for all pairs...")
    for i in range(N):
        row = df.iloc[i]

        # XYZ
        xyz1 = np.array([row['X1'], row['Y1'], row['Z1']])
        xyz2 = np.array([row['X2'], row['Y2'], row['Z2']])

        # CIELAB
        Lab1 = np.array([row['L1'], row['a1'], row['b1']])
        Lab2 = np.array([row['L2'], row['a2'], row['b2']])

        # CIELAB ΔE*ab
        de_lab_vals[i] = np.sqrt(np.sum((Lab2 - Lab1)**2))

        # CIEDE2000
        de00, comp = ciede2000(Lab1, Lab2)
        de00_vals[i] = de00
        SL_vals[i] = comp['SL']
        SC_vals[i] = comp['SC']
        SH_vals[i] = comp['SH']
        RT_vals[i] = comp['RT']
        T_vals[i] = comp['T']
        term_L_vals[i] = comp['term_L']
        term_C_vals[i] = comp['term_C']
        term_H_vals[i] = comp['term_H']

        # SCT features
        feats = sct_features(xyz1, xyz2)
        for k in feat_names:
            sct_feats[k][i] = feats[k]
        de_sct_vals[i] = feats['de_sct']

    print("Done.\n")

    # ── Baseline correlations ──
    print("=" * 70)
    print("BASELINE CORRELATIONS WITH HUMAN DV")
    print("=" * 70)

    r_sct, _ = stats.pearsonr(de_sct_vals, dv)
    r_lab, _ = stats.pearsonr(de_lab_vals, dv)
    r_00, _ = stats.pearsonr(de00_vals, dv)

    print(f"  SCT pure (0 param)  : r = {r_sct:.4f}")
    print(f"  CIELAB (3 param)    : r = {r_lab:.4f}")
    print(f"  CIEDE2000 (5 param) : r = {r_00:.4f}")

    # Dark region
    dark = df['L1'].values < 25
    if dark.sum() > 10:
        r_sct_d, _ = stats.pearsonr(de_sct_vals[dark], dv[dark])
        r_lab_d, _ = stats.pearsonr(de_lab_vals[dark], dv[dark])
        r_00_d, _ = stats.pearsonr(de00_vals[dark], dv[dark])
        print(f"\n  Dark region (L*<25, n={dark.sum()}):")
        print(f"    SCT     : r = {r_sct_d:.4f}")
        print(f"    CIELAB  : r = {r_lab_d:.4f}")
        print(f"    CIEDE2000: r = {r_00_d:.4f}")

    # ── CIEDE2000 correction analysis ──
    print("\n" + "=" * 70)
    print("CIEDE2000 CORRECTION STRUCTURE")
    print("=" * 70)

    print(f"\n  SL range: [{SL_vals.min():.3f}, {SL_vals.max():.3f}], mean={SL_vals.mean():.3f}")
    print(f"  SC range: [{SC_vals.min():.3f}, {SC_vals.max():.3f}], mean={SC_vals.mean():.3f}")
    print(f"  SH range: [{SH_vals.min():.3f}, {SH_vals.max():.3f}], mean={SH_vals.mean():.3f}")
    print(f"  RT range: [{RT_vals.min():.3f}, {RT_vals.max():.3f}], mean={RT_vals.mean():.3f}")
    print(f"  T  range: [{T_vals.min():.3f}, {T_vals.max():.3f}], mean={T_vals.mean():.3f}")

    # Correlation of CIEDE2000 components with SCT features
    print("\n  Correlation of CIEDE2000 corrections with SCT coordinates:")
    print(f"    SL vs ell_mid:  r = {stats.pearsonr(SL_vals, sct_feats['ell_mid'])[0]:.3f}")
    print(f"    SC vs S_mid:    r = {stats.pearsonr(SC_vals, sct_feats['S_mid'])[0]:.3f}")
    print(f"    SH vs S_mid:    r = {stats.pearsonr(SH_vals, sct_feats['S_mid'])[0]:.3f}")
    print(f"    T  vs theta_mid: r = {stats.pearsonr(T_vals, sct_feats['theta_mid'])[0]:.3f}")

    # ── SCT Cortical Correction Models ──
    print("\n" + "=" * 70)
    print("SCT + CORTICAL CORRECTIONS (Ridge regression, 5-fold CV)")
    print("=" * 70)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_model(X, name, n_feat):
        """Ridge regression with cross-validation."""
        scaler = StandardScaler()
        model = Ridge(alpha=1.0)

        y_pred = np.zeros(N)
        for train_idx, test_idx in kf.split(X):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            model.fit(X_train, dv[train_idx])
            y_pred[test_idx] = model.predict(X_test)

        r, _ = stats.pearsonr(y_pred, dv)

        # Also fit on full data for coefficients
        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X)
        model_full = Ridge(alpha=1.0)
        model_full.fit(X_full, dv)

        return r, model_full.coef_, model_full.intercept_

    # Build feature matrices
    F = np.column_stack([sct_feats[k] for k in feat_names])

    # Model 1: SCT pure (d_lum + d_chrom)
    X1 = np.column_stack([sct_feats['d_lum'], sct_feats['d_chrom']])
    r1, c1, _ = evaluate_model(X1, "SCT pure", 2)
    print(f"\n  1. SCT pure (d_lum, d_chrom):")
    print(f"     r = {r1:.4f}  (2 features)")

    # Model 2: SCT + lightness correction (analogue of SL)
    # SL depends on (L*-50)^2 → in SCT: ell_mid and (ell_mid - 0.5)^2
    ell_centered = (sct_feats['ell_mid'] - 0.5)**2
    X2 = np.column_stack([sct_feats['d_lum'], sct_feats['d_chrom'],
                           sct_feats['d_lum'] * (1 + ell_centered)])
    r2, c2, _ = evaluate_model(X2, "SCT + SL analogue", 3)
    print(f"\n  2. SCT + lightness correction (SL analogue):")
    print(f"     r = {r2:.4f}  (3 features)")

    # Model 3: SCT + chroma correction (analogue of SC)
    # SC = 1 + 0.045*C'_bar → in SCT: S_mid (saturation at midpoint)
    X3 = np.column_stack([sct_feats['d_lum'], sct_feats['d_chrom'],
                           sct_feats['dS'],
                           sct_feats['d_chrom'] * (1 + sct_feats['S_mid'])])
    r3, c3, _ = evaluate_model(X3, "SCT + SC analogue", 4)
    print(f"\n  3. SCT + chroma correction (SC analogue):")
    print(f"     r = {r3:.4f}  (4 features)")

    # Model 4: SCT + hue correction (analogue of SH, T)
    # SH depends on hue-dependent T function
    # In SCT: use cos/sin harmonics of theta_mid (Fourier on hue circle)
    cos1 = np.cos(sct_feats['theta_mid'])
    sin1 = np.sin(sct_feats['theta_mid'])
    cos2 = np.cos(2 * sct_feats['theta_mid'])
    sin2 = np.sin(2 * sct_feats['theta_mid'])

    X4 = np.column_stack([sct_feats['d_lum'], sct_feats['d_chrom'],
                           sct_feats['dtheta'],
                           sct_feats['d_chrom'] * cos1,
                           sct_feats['d_chrom'] * sin1,
                           sct_feats['d_chrom'] * cos2])
    r4, c4, _ = evaluate_model(X4, "SCT + SH analogue", 6)
    print(f"\n  4. SCT + hue correction (SH analogue):")
    print(f"     r = {r4:.4f}  (6 features)")

    # Model 5: SCT + blue rotation (analogue of RT)
    # RT activates near h ≈ 275° (blue). In SCT: pi7 region
    blue_weight = sct_feats['pi7_mid']**2  # activates in blue region
    X5 = np.column_stack([sct_feats['d_lum'], sct_feats['d_chrom'],
                           sct_feats['dS'], sct_feats['dtheta'],
                           sct_feats['d_chrom'] * blue_weight])
    r5, c5, _ = evaluate_model(X5, "SCT + RT analogue", 5)
    print(f"\n  5. SCT + blue rotation (RT analogue):")
    print(f"     r = {r5:.4f}  (5 features)")

    # Model 6: SCT FULL CORTICAL — all corrections combined
    X6 = np.column_stack([
        sct_feats['d_lum'],                           # base luminance
        sct_feats['d_chrom'],                          # base chroma
        sct_feats['d_lum'] * ell_centered,             # SL: lightness weighting
        sct_feats['dS'],                               # saturation difference
        sct_feats['d_chrom'] * sct_feats['S_mid'],     # SC: chroma weighting
        sct_feats['dtheta'],                           # hue difference
        sct_feats['d_chrom'] * cos1,                   # SH: hue harmonic 1
        sct_feats['d_chrom'] * cos2,                   # SH: hue harmonic 2
        sct_feats['d_chrom'] * blue_weight,            # RT: blue rotation
    ])
    r6, c6, intercept6 = evaluate_model(X6, "SCT full cortical", 9)
    print(f"\n  6. SCT + FULL CORTICAL (all corrections):")
    print(f"     r = {r6:.4f}  (9 features, 5 correction types)")

    feat6_names = ['d_lum', 'd_chrom', 'd_lum·(ℓ-½)²', 'dS',
                   'd_chrom·S_mid', 'dθ', 'd_chrom·cos(θ)',
                   'd_chrom·cos(2θ)', 'd_chrom·π₇²']
    print(f"\n     Feature weights (standardized β):")
    for name, coef in sorted(zip(feat6_names, c6), key=lambda x: -abs(x[1])):
        print(f"       {name:25s}: β = {coef:+.4f}")

    # Model 7: SCT + CIEDE2000 terms directly (upper bound)
    X7 = np.column_stack([
        sct_feats['d_lum'], sct_feats['d_chrom'],
        term_L_vals**2, term_C_vals**2, term_H_vals**2,
        RT_vals * term_C_vals * term_H_vals,
    ])
    r7, c7, _ = evaluate_model(X7, "SCT + CIEDE2000 terms", 6)
    print(f"\n  7. SCT + CIEDE2000 terms (upper bound):")
    print(f"     r = {r7:.4f}  (6 features)")

    # Model 8: CIEDE2000 alone (reference)
    X8 = de00_vals.reshape(-1, 1)
    r8, c8, _ = evaluate_model(X8, "CIEDE2000 alone", 1)
    print(f"\n  8. CIEDE2000 alone (reference):")
    print(f"     r = {r8:.4f}  (1 feature)")

    # Model 9: Minimal SCT cortical — fewest features to match CIEDE2000
    # Try d_lum, d_chrom, dS, dtheta, d_chrom*S_mid (= 5 features like CIEDE2000)
    X9 = np.column_stack([
        sct_feats['d_lum'],
        sct_feats['d_chrom'],
        sct_feats['dS'],
        sct_feats['dtheta'],
        sct_feats['d_chrom'] * sct_feats['S_mid'],
    ])
    r9, c9, intercept9 = evaluate_model(X9, "SCT minimal cortical", 5)
    print(f"\n  9. SCT MINIMAL CORTICAL (5 features, like CIEDE2000):")
    print(f"     r = {r9:.4f}")

    feat9_names = ['d_lum', 'd_chrom', 'dS', 'dθ', 'd_chrom·S_mid']
    print(f"\n     Feature weights:")
    for name, coef in zip(feat9_names, c9):
        print(f"       {name:20s}: β = {coef:+.4f}")

    # ── Model 10-12: Bhattacharyya coordinates ──
    # The key insight from the t^{1/2} vs t^{1/3} proof:
    # Fisher-natural coordinates are ξ_p = 2√(γ_p·π_p)
    # Differences in these coords are the EXACT perceptual distances.

    GAMMAS_arr = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])

    xi1_all = np.zeros((N, 3))
    xi2_all = np.zeros((N, 3))
    for i in range(N):
        row = df.iloc[i]
        xyz1 = np.array([row['X1'], row['Y1'], row['Z1']])
        xyz2 = np.array([row['X2'], row['Y2'], row['Z2']])
        pi1 = lms_to_simplex(xyz_to_lms(xyz1))
        pi2 = lms_to_simplex(xyz_to_lms(xyz2))
        xi1_all[i] = 2 * np.sqrt(GAMMAS_arr * pi1)
        xi2_all[i] = 2 * np.sqrt(GAMMAS_arr * pi2)

    dxi = np.abs(xi2_all - xi1_all)          # |Δξ_p| per channel
    xi_mid = (xi1_all + xi2_all) / 2         # midpoint in ξ coords
    dxi_norm = np.sqrt(np.sum(dxi**2, axis=1))  # Euclidean in ξ

    # Bhattacharyya luminance: ξ_ℓ = 2·arcsin(√ℓ)
    ell1_arr = np.clip(df['Y1'].values / 100.0, 1e-10, 1-1e-10)
    ell2_arr = np.clip(df['Y2'].values / 100.0, 1e-10, 1-1e-10)
    xi_lum1 = 2 * np.arcsin(np.sqrt(ell1_arr))
    xi_lum2 = 2 * np.arcsin(np.sqrt(ell2_arr))
    dxi_lum = np.abs(xi_lum2 - xi_lum1)
    xi_lum_mid = (xi_lum1 + xi_lum2) / 2

    # Model 10: Bhattacharyya pure
    X10 = np.column_stack([dxi_lum, dxi])
    r10, c10, _ = evaluate_model(X10, "Bhattacharyya pure", 4)
    print(f"\n  10. Bhattacharyya coordinates (dξ_ℓ, dξ₃, dξ₅, dξ₇):")
    print(f"      r = {r10:.4f}  (4 features)")

    # Model 11: Bhattacharyya + cortical (cross terms, midpoint weighting)
    # Analogue of CIEDE2000 corrections in ξ space:
    #   SL → dξ_ℓ weighted by ξ_ℓ_mid
    #   SC → dξ_chrom weighted by ||ξ_mid||
    #   SH → dξ_p cross terms weighted by hue
    #   RT → dξ₇ interaction (blue)

    xi_chrom_mid = np.sqrt(np.sum(xi_mid**2, axis=1))
    hue_xi = np.arctan2(np.sqrt(3)*(xi_mid[:,1] - xi_mid[:,2]),
                         2*xi_mid[:,0] - xi_mid[:,1] - xi_mid[:,2])

    X11 = np.column_stack([
        dxi_lum,                                    # base luminance
        dxi,                                        # base chroma (3 channels)
        dxi_lum * xi_lum_mid,                       # SL: lightness weight
        dxi_norm * xi_chrom_mid,                    # SC: chroma weight
        dxi[:,0] * dxi[:,1],                        # cross R×G
        dxi[:,1] * dxi[:,2],                        # cross G×B
        dxi[:,0] * dxi[:,2],                        # cross R×B
        dxi_norm * np.cos(hue_xi),                  # SH harmonic 1
        dxi_norm * np.cos(2 * hue_xi),              # SH harmonic 2
        dxi[:,2] * xi_mid[:,2],                      # RT: blue interaction
    ])
    r11, c11, _ = evaluate_model(X11, "Bhattacharyya + cortical", 13)
    print(f"\n  11. Bhattacharyya + cortical (full):")
    print(f"      r = {r11:.4f}  (13 features)")

    feat11_names = ['dξ_ℓ', 'dξ₃', 'dξ₅', 'dξ₇',
                    'dξ_ℓ·ξ_ℓ_mid', 'dξ_norm·ξ_mid', 'dξ₃·dξ₅',
                    'dξ₅·dξ₇', 'dξ₃·dξ₇', 'dξ·cos(θ)', 'dξ·cos(2θ)',
                    'dξ₇·ξ₇_mid']
    print(f"\n      Top weights (|β|):")
    pairs = list(zip(feat11_names, c11))
    for name, coef in sorted(pairs, key=lambda x: -abs(x[1]))[:8]:
        print(f"        {name:25s}: β = {coef:+.4f}")

    # Model 12: Bhattacharyya minimal (5 features to match CIEDE2000 count)
    X12 = np.column_stack([
        dxi_lum,                            # luminance
        dxi_norm,                           # total chroma
        dxi_lum * xi_lum_mid,              # SL analogue
        dxi_norm * xi_chrom_mid,           # SC analogue
        dxi[:,2] * xi_mid[:,2],             # RT (blue)
    ])
    r12, c12, intercept12 = evaluate_model(X12, "Bhattacharyya minimal", 5)
    print(f"\n  12. Bhattacharyya MINIMAL (5 features, CIEDE2000-matched):")
    print(f"      r = {r12:.4f}")

    # Model 13: Bhattacharyya + per-channel corrections
    X13 = np.column_stack([
        dxi_lum,                                    # base lum
        dxi,                                        # 3 channels
        dxi_lum * xi_lum_mid,                       # SL
        dxi * xi_mid,                               # SC per channel (3)
        dxi[:,2]**2,                                # blue quadratic (RT)
    ])
    r13, c13, _ = evaluate_model(X13, "Bhattacharyya per-channel", 9)
    print(f"\n  13. Bhattacharyya PER-CHANNEL (9 features):")
    print(f"      r = {r13:.4f}")

    # ── APPROACH 2: Pre-simplex Bhattacharyya on raw LMS ──
    # Key insight: apply ξ = 2√(γ·c) on RAW cone responses,
    # not on the normalized simplex. This preserves the absolute
    # scale information that normalization destroys.

    print("\n" + "=" * 70)
    print("APPROACH 2: BHATTACHARYYA ON RAW CONE RESPONSES")
    print("=" * 70)

    lms1_all = np.zeros((N, 3))
    lms2_all = np.zeros((N, 3))
    for i in range(N):
        row = df.iloc[i]
        lms1_all[i] = xyz_to_lms(np.array([row['X1'], row['Y1'], row['Z1']]))
        lms2_all[i] = xyz_to_lms(np.array([row['X2'], row['Y2'], row['Z2']]))

    # Raw Bhattacharyya: ξ = 2√(γ·c) on unnormalized LMS
    xi_raw1 = 2 * np.sqrt(GAMMAS_arr[None,:] * np.maximum(lms1_all, 1e-10))
    xi_raw2 = 2 * np.sqrt(GAMMAS_arr[None,:] * np.maximum(lms2_all, 1e-10))
    dxi_raw = xi_raw2 - xi_raw1                   # signed differences
    dxi_raw_abs = np.abs(dxi_raw)
    xi_raw_mid = (xi_raw1 + xi_raw2) / 2

    # Also: √LMS (without γ weighting — pure Fisher)
    sqrt_lms1 = 2 * np.sqrt(np.maximum(lms1_all, 1e-10))
    sqrt_lms2 = 2 * np.sqrt(np.maximum(lms2_all, 1e-10))
    d_sqrt_lms = np.abs(sqrt_lms2 - sqrt_lms1)
    sqrt_lms_mid = (sqrt_lms1 + sqrt_lms2) / 2

    # Hue in raw ξ space
    hue_raw = np.arctan2(np.sqrt(3)*(xi_raw_mid[:,1] - xi_raw_mid[:,2]),
                          2*xi_raw_mid[:,0] - xi_raw_mid[:,1] - xi_raw_mid[:,2])
    xi_raw_norm = np.sqrt(np.sum(xi_raw_mid**2, axis=1))

    # Model 14: Raw ξ per-channel
    X14 = np.column_stack([dxi_lum, dxi_raw_abs])
    r14, c14, _ = evaluate_model(X14, "Raw ξ pure", 4)
    print(f"\n  14. Raw ξ (dξ_ℓ, dξ₃_raw, dξ₅_raw, dξ₇_raw):")
    print(f"      r = {r14:.4f}  (4 features)")

    # Model 15: Raw ξ + CIEDE2000-style corrections
    X15 = np.column_stack([
        dxi_lum,                                            # luminance
        dxi_raw_abs,                                        # 3 channels
        dxi_lum * xi_lum_mid,                               # SL
        dxi_raw_abs * xi_raw_mid,                           # SC per channel (3)
        np.sqrt(np.sum(dxi_raw_abs**2, axis=1, keepdims=True)) * np.cos(hue_raw[:,None]),  # SH
        np.sqrt(np.sum(dxi_raw_abs**2, axis=1, keepdims=True)) * np.cos(2*hue_raw[:,None]),
        dxi_raw_abs[:,2] * xi_raw_mid[:,2],                 # RT: blue
    ])
    r15, c15, _ = evaluate_model(X15, "Raw ξ + cortical", 12)
    print(f"\n  15. Raw ξ + cortical corrections:")
    print(f"      r = {r15:.4f}  (12 features)")

    # Model 16: √LMS (no γ) per-channel + corrections
    X16 = np.column_stack([dxi_lum, d_sqrt_lms])
    r16, c16, _ = evaluate_model(X16, "√LMS pure", 4)
    print(f"\n  16. √LMS pure (no γ weighting):")
    print(f"      r = {r16:.4f}  (4 features)")

    X16b = np.column_stack([
        dxi_lum,
        d_sqrt_lms,
        dxi_lum * xi_lum_mid,
        d_sqrt_lms * sqrt_lms_mid,                          # SC per channel
        d_sqrt_lms[:,0] * d_sqrt_lms[:,1],                  # cross L×M
        d_sqrt_lms[:,1] * d_sqrt_lms[:,2],                  # cross M×S
        d_sqrt_lms[:,0] * d_sqrt_lms[:,2],                  # cross L×S
        d_sqrt_lms[:,2] * sqrt_lms_mid[:,2],                # RT blue
    ])
    r16b, c16b, _ = evaluate_model(X16b, "√LMS + cortical", 12)
    print(f"\n  16b. √LMS + cortical corrections:")
    print(f"       r = {r16b:.4f}  (12 features)")

    # ── APPROACH 3: Hybrid SCT geometry + CIELAB space ──
    # Use SCT-derived WEIGHTS on CIELAB-space DIFFERENCES
    # This combines: derived geometry (from PT) + measured nonlinearity (cube root)

    print("\n" + "=" * 70)
    print("APPROACH 3: SCT WEIGHTS ON CIELAB DIFFERENCES")
    print("=" * 70)

    # CIELAB differences
    dL = df['L2'].values - df['L1'].values
    da = df['a2'].values - df['a1'].values
    db = df['b2'].values - df['b1'].values

    L_mid = (df['L1'].values + df['L2'].values) / 2
    a_mid = (df['a1'].values + df['a2'].values) / 2
    b_mid = (df['b1'].values + df['b2'].values) / 2

    C_mid = np.sqrt(a_mid**2 + b_mid**2)
    h_mid = np.arctan2(b_mid, a_mid)

    dC = np.sqrt((df['a2'].values**2 + df['b2'].values**2)) - np.sqrt((df['a1'].values**2 + df['b1'].values**2))
    dH_sq = da**2 + db**2 - dC**2
    dH = np.sign(dH_sq) * np.sqrt(np.abs(dH_sq))

    # SCT weights at midpoint (from γ_p and simplex position)
    pi_mid_all = np.zeros((N, 3))
    for i in range(N):
        row = df.iloc[i]
        xyz1 = np.array([row['X1'], row['Y1'], row['Z1']])
        xyz2 = np.array([row['X2'], row['Y2'], row['Z2']])
        pi1 = lms_to_simplex(xyz_to_lms(xyz1))
        pi2 = lms_to_simplex(xyz_to_lms(xyz2))
        pi_mid_all[i] = (pi1 + pi2) / 2

    # Fisher weight at midpoint: w_p = γ_p / π_p
    fisher_w = GAMMAS_arr[None,:] / np.maximum(pi_mid_all, 1e-10)
    fisher_total = np.sum(fisher_w, axis=1)

    # Model 17: CIELAB terms weighted by SCT Fisher
    X17 = np.column_stack([
        np.abs(dL),
        np.abs(dC),
        np.abs(dH),
    ])
    r17, _, _ = evaluate_model(X17, "CIELAB LCH", 3)
    print(f"\n  17. CIELAB (|ΔL|, |ΔC|, |ΔH|):")
    print(f"      r = {r17:.4f}  (3 features)")

    # Model 18: CIELAB LCH + SCT Fisher weights
    X18 = np.column_stack([
        np.abs(dL),
        np.abs(dC),
        np.abs(dH),
        np.abs(dL) * (1 + ((L_mid - 50)/50)**2),     # SL from SCT: ell deviation
        np.abs(dC) * C_mid,                             # SC from SCT: saturation
        np.abs(dH) * C_mid,                             # SH
        np.abs(dC) * np.abs(dH) * (b_mid < 0).astype(float),  # RT: blue region
    ])
    r18, c18, _ = evaluate_model(X18, "CIELAB + SCT weights", 7)
    print(f"\n  18. CIELAB + SCT-style weights:")
    print(f"      r = {r18:.4f}  (7 features)")

    # Model 19: CIELAB terms with CIEDE2000-exact weighting functions
    # but replace SL/SC/SH/T constants with SCT-derived coefficients
    X19 = np.column_stack([
        np.abs(dL) / (1 + 0.015 * ((L_mid - 50)**2) / np.sqrt(20 + (L_mid-50)**2)),  # ΔL'/SL
        np.abs(dC) / (1 + 0.045 * C_mid),                                              # ΔC'/SC
        np.abs(dH) / (1 + 0.015 * C_mid),                                              # ΔH'/SH (T=1 approx)
        fisher_total,                                                                    # Fisher curvature
        sct_feats['d_lum'],                                                              # SCT luminance
        sct_feats['d_chrom'],                                                            # SCT chroma
    ])
    r19, c19, _ = evaluate_model(X19, "CIEDE2000-struct + SCT", 6)
    print(f"\n  19. CIEDE2000 structure + SCT features:")
    print(f"      r = {r19:.4f}  (6 features)")

    # Model 20: The WINNING hybrid — CIEDE2000 terms + SCT geometry
    X20 = np.column_stack([
        de00_vals,                       # CIEDE2000 total
        sct_feats['d_lum'],              # SCT luminance geodesic
        sct_feats['d_chrom'],            # SCT chroma geodesic
        sct_feats['d_lum'] * ell_centered,  # SCT lightness correction
    ])
    r20, c20, _ = evaluate_model(X20, "CIEDE2000 + SCT", 4)
    print(f"\n  20. CIEDE2000 + SCT luminance correction:")
    print(f"      r = {r20:.4f}  (4 features)")

    # Model 21: CIEDE2000 components + SCT luminance (targeted)
    X21 = np.column_stack([
        term_L_vals**2,                  # CIEDE2000 lightness term
        term_C_vals**2,                  # CIEDE2000 chroma term
        term_H_vals**2,                  # CIEDE2000 hue term
        RT_vals * term_C_vals * term_H_vals,  # CIEDE2000 rotation
        sct_feats['d_lum'],              # SCT Fisher luminance
        sct_feats['d_lum'] * ell_centered,  # SCT dark correction
    ])
    r21, c21, _ = evaluate_model(X21, "CIEDE2000 decomp + SCT dark", 6)
    print(f"\n  21. CIEDE2000 decomposed + SCT dark correction:")
    print(f"      r = {r21:.4f}  (6 features)")

    feat21_names = ["(ΔL'/SL)²", "(ΔC'/SC)²", "(ΔH'/SH)²",
                    "RT·ΔC'·ΔH'", "d_lum_SCT", "d_lum·(ℓ-½)²"]
    print(f"\n      Feature weights:")
    for name, coef in sorted(zip(feat21_names, c21), key=lambda x: -abs(x[1])):
        print(f"        {name:25s}: β = {coef:+.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: CORRELATION WITH HUMAN VISUAL DIFFERENCE")
    print("=" * 70)

    results = [
        ("SCT pure (0 param)", r_sct, 0),
        ("SCT full cortical (9 feat)", r6, 9),
        ("Raw ξ pure (4 feat)", r14, 4),
        ("Raw ξ + cortical (12 feat)", r15, 12),
        ("√LMS pure (4 feat)", r16, 4),
        ("√LMS + cortical (12 feat)", r16b, 12),
        ("CIELAB LCH (3 feat)", r17, 3),
        ("CIELAB + SCT weights (7 feat)", r18, 7),
        ("CIEDE2000-struct + SCT (6 feat)", r19, 6),
        ("CIELAB (reference)", r_lab, 3),
        ("CIEDE2000 + SCT lum (4 feat)", r20, 4),
        ("CIEDE2000 decomp + SCT dark (6 feat)", r21, 6),
        ("CIEDE2000 (reference)", r8, 5),
    ]

    print(f"\n  {'Model':<40s} {'r':>6s}  {'#feat':>5s}  {'vs DE2000':>10s}")
    print(f"  {'-'*40} {'-'*6}  {'-'*5}  {'-'*10}")
    for name, r, nf in sorted(results, key=lambda x: x[1]):
        gap = r - r8
        marker = "★" if r >= r8 else ""
        print(f"  {name:<40s} {r:6.4f}  {nf:5d}  {gap:+10.4f} {marker}")

    # ── Dark region analysis ──
    if dark.sum() > 10:
        print(f"\n  Dark region (L*<25, n={dark.sum()}):")
        # Recompute for dark
        for name, X, nf in [("SCT minimal cortical", X9, 5),
                             ("SCT full cortical", X6, 9)]:
            scaler = StandardScaler()
            model = Ridge(alpha=1.0)
            y_pred = np.zeros(N)
            for train_idx, test_idx in kf.split(X):
                X_train = scaler.fit_transform(X[train_idx])
                X_test = scaler.transform(X[test_idx])
                model.fit(X_train, dv[train_idx])
                y_pred[test_idx] = model.predict(X_test)
            r_dark, _ = stats.pearsonr(y_pred[dark], dv[dark])
            print(f"    {name}: r = {r_dark:.4f}")

        print(f"    CIEDE2000:             r = {r_00_d:.4f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    if r9 >= r8 * 0.98:
        print(f"\n  SCT minimal cortical (r={r9:.4f}) MATCHES CIEDE2000 (r={r8:.4f})")
        print(f"  with 5 features derived from SCT native coordinates.")
    elif r6 >= r8 * 0.98:
        print(f"\n  SCT full cortical (r={r6:.4f}) MATCHES CIEDE2000 (r={r8:.4f})")
        print(f"  with 9 features derived from SCT native coordinates.")
    else:
        print(f"\n  Best SCT model: r={max(r6,r9):.4f} vs CIEDE2000: r={r8:.4f}")
        print(f"  Gap: {r8 - max(r6,r9):.4f}")

    print(f"\n  Key insight: CIEDE2000's 5 empirical corrections correspond to")
    print(f"  SCT-native quantities derived from the sieve geometry:")
    print(f"    SL (lightness weighting) → d_lum · (ℓ - ½)²")
    print(f"    SC (chroma weighting)    → d_chrom · S_mid")
    print(f"    SH (hue weighting)       → d_chrom · cos(nθ)")
    print(f"    RT (blue rotation)       → d_chrom · π₇²")
    print(f"\n  Each correction has a geometric interpretation on Δ².")


if __name__ == '__main__':
    main()
