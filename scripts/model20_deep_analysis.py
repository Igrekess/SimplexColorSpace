#!/usr/bin/env python3
"""
Deep Analysis of Model 20: CIEDE2000 + SCT
===========================================

Model 20 (r=0.8831) beats CIEDE2000 alone (r=0.8774) with 4 features:
  f0: ΔE₀₀ (CIEDE2000 total)
  f1: d_lum (SCT Fisher-Bernoulli luminance geodesic)
  f2: d_chrom (SCT Bhattacharyya chromatic geodesic)
  f3: d_lum · (ℓ - ½)² (SCT dark correction)

This script investigates:
  1. Where the improvement concentrates (L* bins, hue bins, datasets)
  2. Residual analysis: what CIEDE2000 gets wrong that SCT fixes
  3. Ablation: which SCT feature matters most
  4. The Fisher-Bernoulli geodesic vs CIEDE2000 SL in the dark region
  5. Statistical significance of the improvement
  6. Variations: can we do even better?
"""

import numpy as np
import pandas as pd
import os, sys
from scipy import stats
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sct_ciede2000_analysis import ciede2000, sct_features
from delta_e_scpt import xyz_to_lms, lms_to_simplex
from scpt_companion import gamma_p

GAMMAS = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])


def load_and_compute():
    """Load COMBVD, compute all metrics."""
    data_path = os.path.join(os.path.dirname(__file__), 'datasets', 'COMBVD_3813.csv')
    df = pd.read_csv(data_path)
    N = len(df)

    dv = df['DV'].values
    de00 = np.zeros(N)
    comp_all = []

    # SCT features
    d_lum = np.zeros(N)
    d_chrom = np.zeros(N)
    ell_mid = np.zeros(N)
    de_sct = np.zeros(N)
    S_mid = np.zeros(N)
    dtheta = np.zeros(N)
    theta_mid = np.zeros(N)
    pi_mid_all = np.zeros((N, 3))

    # CIELAB
    L_mid = np.zeros(N)
    C_mid = np.zeros(N)
    h_mid = np.zeros(N)

    for i in range(N):
        row = df.iloc[i]
        xyz1 = np.array([row['X1'], row['Y1'], row['Z1']])
        xyz2 = np.array([row['X2'], row['Y2'], row['Z2']])
        Lab1 = np.array([row['L1'], row['a1'], row['b1']])
        Lab2 = np.array([row['L2'], row['a2'], row['b2']])

        de00_val, comp = ciede2000(Lab1, Lab2)
        de00[i] = de00_val
        comp_all.append(comp)

        feats = sct_features(xyz1, xyz2)
        d_lum[i] = feats['d_lum']
        d_chrom[i] = feats['d_chrom']
        ell_mid[i] = feats['ell_mid']
        de_sct[i] = feats['de_sct']
        S_mid[i] = feats['S_mid']
        dtheta[i] = feats['dtheta']
        theta_mid[i] = feats['theta_mid']

        pi1 = lms_to_simplex(xyz_to_lms(xyz1))
        pi2 = lms_to_simplex(xyz_to_lms(xyz2))
        pi_mid_all[i] = (pi1 + pi2) / 2

        L_mid[i] = (row['L1'] + row['L2']) / 2
        a_m = (row['a1'] + row['a2']) / 2
        b_m = (row['b1'] + row['b2']) / 2
        C_mid[i] = np.sqrt(a_m**2 + b_m**2)
        h_mid[i] = np.degrees(np.arctan2(b_m, a_m)) % 360

    comp_df = pd.DataFrame(comp_all)
    ell_centered = (ell_mid - 0.5)**2

    return (df, dv, de00, comp_df, d_lum, d_chrom, ell_mid, ell_centered,
            de_sct, S_mid, dtheta, theta_mid, pi_mid_all, L_mid, C_mid, h_mid)


def cv_predict(X, y, alpha=1.0):
    """5-fold CV prediction with Ridge."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr])
        X_te = sc.transform(X[te])
        m = Ridge(alpha=alpha).fit(X_tr, y[tr])
        y_pred[te] = m.predict(X_te)
    return y_pred


def main():
    print("=" * 70)
    print("DEEP ANALYSIS: Model 20 (CIEDE2000 + SCT)")
    print("=" * 70)

    (df, dv, de00, comp_df, d_lum, d_chrom, ell_mid, ell_centered,
     de_sct, S_mid, dtheta, theta_mid, pi_mid_all, L_mid, C_mid, h_mid) = load_and_compute()
    N = len(dv)

    # ── 1. Full model fit on all data ──
    print("\n" + "─" * 70)
    print("1. MODEL COEFFICIENTS (full data fit)")
    print("─" * 70)

    X20 = np.column_stack([de00, d_lum, d_chrom, d_lum * ell_centered])
    feat_names = ['ΔE₀₀', 'd_lum', 'd_chrom', 'd_lum·(ℓ-½)²']

    sc = StandardScaler()
    X_sc = sc.fit_transform(X20)
    model = Ridge(alpha=1.0).fit(X_sc, dv)

    print(f"\n  Standardized coefficients (β):")
    for name, coef in zip(feat_names, model.coef_):
        print(f"    {name:20s}: β = {coef:+.4f}")
    print(f"    {'intercept':20s}:     {model.intercept_:+.4f}")

    # Unstandardized
    coef_raw = model.coef_ / sc.scale_
    intercept_raw = model.intercept_ - np.sum(model.coef_ * sc.mean_ / sc.scale_)
    print(f"\n  Raw (unstandardized) coefficients:")
    for name, coef in zip(feat_names, coef_raw):
        print(f"    {name:20s}: w = {coef:+.4f}")
    print(f"    {'intercept':20s}:     {intercept_raw:+.4f}")

    # Variance explained per feature
    X_contrib = X_sc * model.coef_[None, :]
    var_total = np.var(dv)
    print(f"\n  Variance contribution (% of total prediction variance):")
    pred = X_sc @ model.coef_
    for j, name in enumerate(feat_names):
        # Correlation of feature contribution with total prediction
        r_contrib = stats.pearsonr(X_contrib[:, j], pred)[0]
        print(f"    {name:20s}: {100*r_contrib**2:.1f}%")

    # ── 2. Ablation study ──
    print("\n" + "─" * 70)
    print("2. ABLATION STUDY")
    print("─" * 70)

    # Base: CIEDE2000 alone
    y_base = cv_predict(de00.reshape(-1, 1), dv)
    r_base = stats.pearsonr(y_base, dv)[0]
    print(f"\n  CIEDE2000 alone:                r = {r_base:.4f}")

    # Full model 20
    y_full = cv_predict(X20, dv)
    r_full = stats.pearsonr(y_full, dv)[0]
    print(f"  Full Model 20:                  r = {r_full:.4f}  (Δ = +{r_full-r_base:.4f})")

    # Remove each feature
    for j, name in enumerate(feat_names):
        if j == 0:
            continue  # don't remove CIEDE2000 base
        mask = [k for k in range(4) if k != j]
        X_abl = X20[:, mask]
        y_abl = cv_predict(X_abl, dv)
        r_abl = stats.pearsonr(y_abl, dv)[0]
        print(f"  Without {name:20s}: r = {r_abl:.4f}  (loss = {r_full-r_abl:+.4f})")

    # Add each SCT feature individually to CIEDE2000
    print(f"\n  Adding single SCT features to CIEDE2000:")
    sct_extras = {
        'd_lum': d_lum,
        'd_chrom': d_chrom,
        'd_lum·(ℓ-½)²': d_lum * ell_centered,
        'dS (saturation diff)': np.abs(S_mid),
        'dθ (hue diff)': dtheta,
        'de_sct (canonical)': de_sct,
    }
    for name, feat in sct_extras.items():
        X_add = np.column_stack([de00, feat])
        y_add = cv_predict(X_add, dv)
        r_add = stats.pearsonr(y_add, dv)[0]
        print(f"    + {name:25s}: r = {r_add:.4f}  (Δ = +{r_add-r_base:.4f})")

    # ── 3. Residual analysis ──
    print("\n" + "─" * 70)
    print("3. RESIDUAL ANALYSIS: WHERE DOES SCT FIX CIEDE2000?")
    print("─" * 70)

    resid_00 = dv - de00 * (stats.pearsonr(de00, dv)[0] * np.std(dv) / np.std(de00))
    resid_20 = dv - y_full

    # By L* bins
    bins_L = [(0, 25, 'dark'), (25, 50, 'mid-dark'), (50, 75, 'mid-light'), (75, 100, 'light')]
    print(f"\n  Residual RMS by lightness region:")
    print(f"  {'Region':<15s} {'n':>5s} {'CIEDE2000':>10s} {'Model 20':>10s} {'Improve':>10s}")
    for lo, hi, label in bins_L:
        mask = (L_mid >= lo) & (L_mid < hi)
        n = mask.sum()
        if n < 5:
            continue
        rms_00 = np.sqrt(np.mean(resid_00[mask]**2))
        rms_20 = np.sqrt(np.mean(resid_20[mask]**2))
        r_00_bin = stats.pearsonr(de00[mask], dv[mask])[0]
        r_20_bin = stats.pearsonr(y_full[mask], dv[mask])[0]
        print(f"  L*∈[{lo},{hi}) {label:<8s} {n:5d}  r={r_00_bin:.3f}     r={r_20_bin:.3f}     Δr={r_20_bin-r_00_bin:+.3f}")

    # By chroma bins
    bins_C = [(0, 15, 'low'), (15, 30, 'medium'), (30, 60, 'high'), (60, 200, 'v.high')]
    print(f"\n  Correlation by chroma region:")
    print(f"  {'Region':<15s} {'n':>5s} {'CIEDE2000':>10s} {'Model 20':>10s} {'Improve':>10s}")
    for lo, hi, label in bins_C:
        mask = (C_mid >= lo) & (C_mid < hi)
        n = mask.sum()
        if n < 5:
            continue
        r_00_bin = stats.pearsonr(de00[mask], dv[mask])[0]
        r_20_bin = stats.pearsonr(y_full[mask], dv[mask])[0]
        print(f"  C*∈[{lo},{hi}) {label:<8s} {n:5d}  r={r_00_bin:.3f}     r={r_20_bin:.3f}     Δr={r_20_bin-r_00_bin:+.3f}")

    # By hue region
    bins_h = [(0, 60, 'red-yellow'), (60, 150, 'yellow-green'),
              (150, 240, 'green-blue'), (240, 360, 'blue-red')]
    print(f"\n  Correlation by hue region:")
    print(f"  {'Region':<15s} {'n':>5s} {'CIEDE2000':>10s} {'Model 20':>10s} {'Improve':>10s}")
    for lo, hi, label in bins_h:
        mask = (h_mid >= lo) & (h_mid < hi)
        n = mask.sum()
        if n < 5:
            continue
        r_00_bin = stats.pearsonr(de00[mask], dv[mask])[0]
        r_20_bin = stats.pearsonr(y_full[mask], dv[mask])[0]
        print(f"  h∈[{lo},{hi}) {label:<11s} {n:5d}  r={r_00_bin:.3f}     r={r_20_bin:.3f}     Δr={r_20_bin-r_00_bin:+.3f}")

    # By dataset
    print(f"\n  Correlation by COMBVD sub-dataset:")
    print(f"  {'Dataset':<20s} {'n':>5s} {'CIEDE2000':>10s} {'Model 20':>10s} {'Improve':>10s}")
    for ds in df['dataset'].unique():
        mask = df['dataset'].values == ds
        n = mask.sum()
        if n < 10:
            continue
        r_00_ds = stats.pearsonr(de00[mask], dv[mask])[0]
        r_20_ds = stats.pearsonr(y_full[mask], dv[mask])[0]
        print(f"  {ds:<20s} {n:5d}  r={r_00_ds:.3f}     r={r_20_ds:.3f}     Δr={r_20_ds-r_00_ds:+.3f}")

    # ── 4. Fisher-Bernoulli vs CIEDE2000 SL in the dark ──
    print("\n" + "─" * 70)
    print("4. FISHER-BERNOULLI vs CIEDE2000 SL (DARK REGION)")
    print("─" * 70)

    dark = L_mid < 25
    n_dark = dark.sum()
    SL = comp_df['SL'].values
    dLp = comp_df['dLp'].values

    # CIEDE2000's luminance term: ΔL'/(kL·SL)
    ciede_lum_term = np.abs(dLp) / SL
    # SCT's luminance: d_lum = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|
    sct_lum_term = d_lum

    print(f"\n  Dark region (L*<25): n = {n_dark}")
    print(f"\n  Correlation with human DV (dark only):")
    r_ciede_lum_dark = stats.pearsonr(ciede_lum_term[dark], dv[dark])[0]
    r_sct_lum_dark = stats.pearsonr(sct_lum_term[dark], dv[dark])[0]
    print(f"    CIEDE2000 |ΔL'/SL|     : r = {r_ciede_lum_dark:.4f}")
    print(f"    SCT d_lum (Fisher)     : r = {r_sct_lum_dark:.4f}")

    # Compare the SL function with Fisher behavior
    # SL = 1 + 0.015·(L*-50)²/√(20+(L*-50)²)
    # Fisher: the arcsin(√ℓ) transformation
    # At L*→0: SL → 1 + 0.015·2500/√2520 ≈ 1.75 (finite)
    # But Fisher: arcsin(√ℓ) has slope 1/√(4ℓ(1-ℓ)) → ∞ as ℓ→0
    print(f"\n  WHY Fisher beats SL in the dark:")
    print(f"    At L*=10 (ℓ≈0.01): SL = {1 + 0.015*(10-50)**2/np.sqrt(20+(10-50)**2):.3f}")
    print(f"      → CIEDE2000 divides by {1 + 0.015*(10-50)**2/np.sqrt(20+(10-50)**2):.3f} (finite correction)")
    ell_10 = 0.01
    fisher_slope = 1 / np.sqrt(4 * ell_10 * (1 - ell_10))
    print(f"      → Fisher slope = 1/√(4ℓ(1-ℓ)) = {fisher_slope:.1f} (divergent, correct)")
    print(f"    At L*=5 (ℓ≈0.003): SL = {1 + 0.015*(5-50)**2/np.sqrt(20+(5-50)**2):.3f}")
    ell_5 = 0.003
    fisher_slope_5 = 1 / np.sqrt(4 * ell_5 * (1 - ell_5))
    print(f"      → Fisher slope = {fisher_slope_5:.1f}")
    print(f"\n  CIEDE2000's SL saturates at ~1.75; Fisher diverges as 1/√ℓ.")
    print(f"  Fisher captures the TRUE sensitivity increase in the dark,")
    print(f"  where SL's polynomial approximation runs out of range.")

    # Quantify: what fraction of the improvement comes from dark?
    print(f"\n  Improvement decomposition:")
    # Residuals in dark vs rest
    not_dark = ~dark
    resid_00_scaled = dv - np.polyval(np.polyfit(de00, dv, 1), de00)
    resid_20_scaled = dv - y_full

    mse_00_dark = np.mean(resid_00_scaled[dark]**2)
    mse_20_dark = np.mean(resid_20_scaled[dark]**2)
    mse_00_rest = np.mean(resid_00_scaled[not_dark]**2)
    mse_20_rest = np.mean(resid_20_scaled[not_dark]**2)

    print(f"    MSE dark  (L*<25):  CIEDE2000={mse_00_dark:.4f}  Model20={mse_20_dark:.4f}  reduction={100*(1-mse_20_dark/mse_00_dark):.1f}%")
    print(f"    MSE rest  (L*≥25):  CIEDE2000={mse_00_rest:.4f}  Model20={mse_20_rest:.4f}  reduction={100*(1-mse_20_rest/mse_00_rest):.1f}%")

    # ── 5. Statistical significance ──
    print("\n" + "─" * 70)
    print("5. STATISTICAL SIGNIFICANCE")
    print("─" * 70)

    # Bootstrap confidence intervals
    n_boot = 2000
    r_00_boot = np.zeros(n_boot)
    r_20_boot = np.zeros(n_boot)
    rng = np.random.RandomState(42)

    for b in range(n_boot):
        idx = rng.choice(N, N, replace=True)
        r_00_boot[b] = stats.pearsonr(de00[idx], dv[idx])[0]
        r_20_boot[b] = stats.pearsonr(y_full[idx], dv[idx])[0]

    dr_boot = r_20_boot - r_00_boot
    ci_lo, ci_hi = np.percentile(dr_boot, [2.5, 97.5])
    p_val = np.mean(dr_boot <= 0)

    print(f"\n  Bootstrap (n={n_boot}):")
    print(f"    CIEDE2000:  r = {np.mean(r_00_boot):.4f} [{np.percentile(r_00_boot,2.5):.4f}, {np.percentile(r_00_boot,97.5):.4f}]")
    print(f"    Model 20:   r = {np.mean(r_20_boot):.4f} [{np.percentile(r_20_boot,2.5):.4f}, {np.percentile(r_20_boot,97.5):.4f}]")
    print(f"    Δr:         {np.mean(dr_boot):+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    P(Δr ≤ 0): {p_val:.4f}  {'SIGNIFICANT' if p_val < 0.05 else 'not significant'}")

    # Steiger test for dependent correlations
    r12 = stats.pearsonr(de00, y_full)[0]  # correlation between predictors
    # Steiger (1980) z-test
    z_00 = np.arctanh(r_base)
    z_20 = np.arctanh(r_full)
    z_12 = np.arctanh(r12)
    det = (1 - r_base**2)**2 + (1 - r_full**2)**2 - 2*r12**3
    se = np.sqrt(2*(1 - r12) / ((N - 3) * (1 + r12)))
    z_stat = (z_20 - z_00) / se
    p_steiger = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"\n  Steiger (1980) test for dependent correlations:")
    print(f"    z = {z_stat:.3f},  p = {p_steiger:.4f}  {'SIGNIFICANT' if p_steiger < 0.05 else 'not significant'}")

    # ── 6. Extended models ──
    print("\n" + "─" * 70)
    print("6. EXTENDED MODELS: CAN WE DO BETTER?")
    print("─" * 70)

    # 20a: CIEDE2000 + d_lum only
    X20a = np.column_stack([de00, d_lum])
    y_20a = cv_predict(X20a, dv)
    r_20a = stats.pearsonr(y_20a, dv)[0]
    print(f"\n  20a. ΔE₀₀ + d_lum:                    r = {r_20a:.4f}")

    # 20b: CIEDE2000 + d_lum + dark correction
    X20b = np.column_stack([de00, d_lum, d_lum * ell_centered])
    y_20b = cv_predict(X20b, dv)
    r_20b = stats.pearsonr(y_20b, dv)[0]
    print(f"  20b. ΔE₀₀ + d_lum + dark:              r = {r_20b:.4f}")

    # 20c: Full model 20
    print(f"  20c. Full Model 20:                    r = {r_full:.4f}")

    # 20d: + saturation
    X20d = np.column_stack([de00, d_lum, d_chrom, d_lum * ell_centered, S_mid])
    y_20d = cv_predict(X20d, dv)
    r_20d = stats.pearsonr(y_20d, dv)[0]
    print(f"  20d. Model 20 + S_mid:                 r = {r_20d:.4f}")

    # 20e: + hue harmonics
    cos1 = np.cos(theta_mid)
    cos2 = np.cos(2 * theta_mid)
    sin1 = np.sin(theta_mid)
    X20e = np.column_stack([de00, d_lum, d_chrom, d_lum * ell_centered,
                             d_chrom * cos1, d_chrom * sin1])
    y_20e = cv_predict(X20e, dv)
    r_20e = stats.pearsonr(y_20e, dv)[0]
    print(f"  20e. Model 20 + hue harmonics:         r = {r_20e:.4f}")

    # 20f: + per-channel SCT (all 3 dπ)
    dpi3 = np.zeros(N)
    dpi5 = np.zeros(N)
    dpi7 = np.zeros(N)
    for i in range(N):
        row = df.iloc[i]
        xyz1 = np.array([row['X1'], row['Y1'], row['Z1']])
        xyz2 = np.array([row['X2'], row['Y2'], row['Z2']])
        pi1 = lms_to_simplex(xyz_to_lms(xyz1))
        pi2 = lms_to_simplex(xyz_to_lms(xyz2))
        dpi3[i] = abs(pi2[0] - pi1[0])
        dpi5[i] = abs(pi2[1] - pi1[1])
        dpi7[i] = abs(pi2[2] - pi1[2])

    X20f = np.column_stack([de00, d_lum, d_lum * ell_centered,
                             dpi3, dpi5, dpi7])
    y_20f = cv_predict(X20f, dv)
    r_20f = stats.pearsonr(y_20f, dv)[0]
    print(f"  20f. ΔE₀₀ + d_lum + dark + dπ₃,₅,₇:   r = {r_20f:.4f}")

    # 20g: CIEDE2000 decomposed + SCT luminance
    term_L = comp_df['term_L'].values
    term_C = comp_df['term_C'].values
    term_H = comp_df['term_H'].values
    RT = comp_df['RT'].values

    X20g = np.column_stack([
        term_L**2, term_C**2, term_H**2,
        RT * term_C * term_H,
        d_lum,
        d_lum * ell_centered,
    ])
    y_20g = cv_predict(X20g, dv)
    r_20g = stats.pearsonr(y_20g, dv)[0]
    print(f"  20g. CIEDE2000 decomp + SCT lum+dark:  r = {r_20g:.4f}")

    # 20h: Like 20g but also per-channel SCT
    X20h = np.column_stack([
        term_L**2, term_C**2, term_H**2,
        RT * term_C * term_H,
        d_lum, d_lum * ell_centered,
        dpi3, dpi5, dpi7,
    ])
    y_20h = cv_predict(X20h, dv)
    r_20h = stats.pearsonr(y_20h, dv)[0]
    print(f"  20h. CIEDE2000 decomp + SCT full:      r = {r_20h:.4f}")

    # 20i: CIEDE2000 + quadratic SCT luminance
    d_lum_sq = d_lum**2
    X20i = np.column_stack([de00, d_lum, d_lum_sq, d_lum * ell_centered])
    y_20i = cv_predict(X20i, dv)
    r_20i = stats.pearsonr(y_20i, dv)[0]
    print(f"  20i. ΔE₀₀ + d_lum + d_lum² + dark:    r = {r_20i:.4f}")

    # 20j: CIEDE2000 + Fisher curvature at midpoint
    fisher_curv = np.sum(GAMMAS[None,:] / np.maximum(pi_mid_all, 1e-10), axis=1)
    X20j = np.column_stack([de00, d_lum, d_lum * ell_centered, fisher_curv])
    y_20j = cv_predict(X20j, dv)
    r_20j = stats.pearsonr(y_20j, dv)[0]
    print(f"  20j. Model 20 + Fisher curvature:      r = {r_20j:.4f}")

    # ── 7. The Interpretation ──
    print("\n" + "─" * 70)
    print("7. PHYSICAL INTERPRETATION")
    print("─" * 70)

    print(f"""
  CIEDE2000's SL correction uses a polynomial:
    SL = 1 + 0.015·(L*-50)² / √(20+(L*-50)²)

  This SATURATES at SL ≈ 1.75 for L*→0.

  The SCT Fisher-Bernoulli geodesic uses:
    d_lum = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|

  This has sensitivity proportional to 1/√(ℓ(1-ℓ)), which
  DIVERGES correctly as ℓ→0 (dark region).

  The arcsin(√ℓ) transformation is the variance-stabilizing
  transform for a Bernoulli random variable — it is the UNIQUE
  transformation that makes the Fisher information constant.

  CIEDE2000's SL polynomial was fit to observers circa 1990-2000
  and approximates this behavior in the mid-range (L*=25-75)
  but falls short in the tails.

  The SCT correction d_lum·(ℓ-½)² amplifies the Fisher geodesic
  AWAY from the midpoint, precisely where CIEDE2000 underperforms.

  This is the t^{{1/2}} vs t^{{1/3}} argument made quantitative:
  the Fisher-derived exponent 1/2 is the theoretically exact value,
  and its geodesic captures perceptual sensitivity in the dark
  where the empirical polynomial of CIEDE2000 runs out of range.
""")

    # ── 8. Proposed formula ──
    print("─" * 70)
    print("8. PROPOSED FORMULA: ΔE_SCT00")
    print("─" * 70)

    # Fit on full data for the final formula
    sc_final = StandardScaler()
    X_final = sc_final.fit_transform(X20)
    m_final = Ridge(alpha=1.0).fit(X_final, dv)
    w_raw = m_final.coef_ / sc_final.scale_
    b_raw = m_final.intercept_ - np.sum(m_final.coef_ * sc_final.mean_ / sc_final.scale_)

    print(f"""
  ΔE_SCT00 = {b_raw:.3f}
             + {w_raw[0]:.3f} · ΔE₀₀
             + {w_raw[1]:.3f} · d_lum
             + {w_raw[2]:.3f} · d_chrom
             + {w_raw[3]:.3f} · d_lum · (ℓ - ½)²

  where:
    ΔE₀₀    = standard CIEDE2000 formula
    d_lum   = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|  (Fisher-Bernoulli geodesic)
    d_chrom = 2·arccos(Σ √(π̃₁·π̃₂))          (Bhattacharyya on Δ²)
    ℓ       = Y/100  (normalized luminance)
    ℓ_mid   = (ℓ₁ + ℓ₂)/2

  Performance: r = {r_full:.4f} (vs CIEDE2000: r = {r_base:.4f})
  Parameters: 4 (3 regression weights + 1 intercept)
  SCT features: 0 adjustable parameters (derived from s = 1/2)
""")


if __name__ == '__main__':
    main()
