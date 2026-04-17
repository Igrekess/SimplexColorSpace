#!/usr/bin/env python3
"""
Push Beyond CIEDE2000 — Systematic exploration
================================================

Current best: Model 20i at r = 0.8849 (ΔE₀₀ + d_lum + d_lum² + dark)
Target: find the minimal SCS-derived model that CLEARLY surpasses CIEDE2000.

Strategies:
  A. Fisher-LAB: replace CIELAB's t^{1/3} with t^{1/2} everywhere
  B. Modified CIEDE2000: replace SL with Fisher weighting, keep SC/SH/RT
  C. Per-channel Fisher geodesics on raw LMS (not simplex-normalized)
  D. Opponent channels in Fisher coordinates
  E. Polynomial / interaction features
  F. Full SCS-native ΔE from scratch (no CIELAB dependency)
"""

import numpy as np
import pandas as pd
import os, sys
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scs_ciede2000_analysis import ciede2000
from delta_e_scs import xyz_to_lms, lms_to_simplex, M_HPE
from scs_companion import gamma_p, MU_STAR, Q_REL, PRIMES

GAMMAS = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])
G3, G5, G7 = GAMMAS


def load_data():
    path = os.path.join(os.path.dirname(__file__), 'datasets', 'COMBVD_3813.csv')
    df = pd.read_csv(path)
    N = len(df)

    dv = df['DV'].values
    xyz1 = df[['X1','Y1','Z1']].values
    xyz2 = df[['X2','Y2','Z2']].values
    Lab1 = df[['L1','a1','b1']].values
    Lab2 = df[['L2','a2','b2']].values

    # Precompute LMS
    lms1 = np.array([xyz_to_lms(x) for x in xyz1])
    lms2 = np.array([xyz_to_lms(x) for x in xyz2])

    # CIEDE2000
    de00 = np.zeros(N)
    comp_all = []
    for i in range(N):
        d, c = ciede2000(Lab1[i], Lab2[i])
        de00[i] = d
        comp_all.append(c)
    comp = pd.DataFrame(comp_all)

    return df, N, dv, xyz1, xyz2, Lab1, Lab2, lms1, lms2, de00, comp


def cv_predict(X, y, alpha=1.0):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    yp = np.zeros(len(y))
    for tr, te in kf.split(X):
        sc = StandardScaler()
        m = Ridge(alpha=alpha).fit(sc.fit_transform(X[tr]), y[tr])
        yp[te] = m.predict(sc.transform(X[te]))
    return yp


def evaluate(X, y, name):
    yp = cv_predict(X, y)
    r = stats.pearsonr(yp, y)[0]
    return r, yp


def main():
    print("=" * 70)
    print("PUSH BEYOND CIEDE2000")
    print("=" * 70)

    df, N, dv, xyz1, xyz2, Lab1, Lab2, lms1, lms2, de00, comp = load_data()

    # Reference
    r_00 = stats.pearsonr(de00, dv)[0]
    print(f"\n  CIEDE2000 reference: r = {r_00:.4f}")

    results = []

    def test(X, name):
        r, yp = evaluate(X, dv, name)
        results.append((name, r, X.shape[1]))
        return r, yp

    # ════════════════════════════════════════════════════════════
    # A. FISHER-LAB: replace t^{1/3} with t^{1/2}
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("A. FISHER-LAB SPACE")
    print("=" * 70)

    # CIELAB uses f(t) = t^{1/3} on X/Xn, Y/Yn, Z/Zn
    # Fisher uses f(t) = t^{1/2}
    # Let's create Fisher-Lab: same structure, different nonlinearity
    Xn, Yn, Zn = 95.047, 100.0, 108.883  # D65

    def fisher_f(t):
        """Fisher nonlinearity: t^{1/2} (variance-stabilizing)."""
        t = np.maximum(t, 1e-10)
        return np.sqrt(t)

    def cielab_f(t):
        """CIELAB nonlinearity: t^{1/3}."""
        t = np.maximum(t, 1e-10)
        delta = 6/29
        return np.where(t > delta**3, t**(1/3), t/(3*delta**2) + 4/29)

    # Fisher-Lab coordinates
    def to_fisher_lab(xyz):
        fx = fisher_f(xyz[0] / Xn)
        fy = fisher_f(xyz[1] / Yn)
        fz = fisher_f(xyz[2] / Zn)
        L_f = 116 * fy - 16
        a_f = 500 * (fx - fy)
        b_f = 200 * (fy - fz)
        return np.array([L_f, a_f, b_f])

    fLab1 = np.array([to_fisher_lab(x) for x in xyz1])
    fLab2 = np.array([to_fisher_lab(x) for x in xyz2])
    dfLab = np.abs(fLab2 - fLab1)

    r_flab, _ = test(dfLab, "A1. Fisher-Lab |ΔL_f|,|Δa_f|,|Δb_f|")
    print(f"  A1. Fisher-Lab pure:           r = {r_flab:.4f}")

    # Fisher-Lab with CIEDE2000-style corrections
    fL_mid = (fLab1[:,0] + fLab2[:,0]) / 2
    fa_mid = (fLab1[:,1] + fLab2[:,1]) / 2
    fb_mid = (fLab1[:,2] + fLab2[:,2]) / 2
    fC_mid = np.sqrt(fa_mid**2 + fb_mid**2)
    fh_mid = np.arctan2(fb_mid, fa_mid)

    fdL = fLab2[:,0] - fLab1[:,0]
    fC1 = np.sqrt(fLab1[:,1]**2 + fLab1[:,2]**2)
    fC2 = np.sqrt(fLab2[:,1]**2 + fLab2[:,2]**2)
    fdC = fC2 - fC1
    fdH_sq = (fLab2[:,1]-fLab1[:,1])**2 + (fLab2[:,2]-fLab1[:,2])**2 - fdC**2
    fdH = np.sign(fdH_sq) * np.sqrt(np.abs(fdH_sq))

    X_a2 = np.column_stack([
        np.abs(fdL), np.abs(fdC), np.abs(fdH),
        np.abs(fdL) * (1 + ((fL_mid-50)/50)**2),  # SL analogue
        np.abs(fdC) * fC_mid,                       # SC analogue
        np.abs(fdH) * fC_mid,                       # SH analogue
    ])
    r_a2, _ = test(X_a2, "A2. Fisher-Lab + corrections")
    print(f"  A2. Fisher-Lab + LCH corrections: r = {r_a2:.4f}")

    # Fisher-Lab + CIEDE2000 combined
    X_a3 = np.column_stack([de00, np.abs(fdL), np.abs(fdC), np.abs(fdH)])
    r_a3, _ = test(X_a3, "A3. CIEDE2000 + Fisher-Lab LCH")
    print(f"  A3. CIEDE2000 + Fisher-Lab LCH: r = {r_a3:.4f}")

    # ════════════════════════════════════════════════════════════
    # B. MODIFIED CIEDE2000: swap SL for Fisher
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("B. MODIFIED CIEDE2000: FISHER SL")
    print("=" * 70)

    # Idea: in CIEDE2000, replace ΔL'/SL with Fisher luminance geodesic
    ell1 = np.clip(xyz1[:,1] / 100.0, 1e-10, 1-1e-10)
    ell2 = np.clip(xyz2[:,1] / 100.0, 1e-10, 1-1e-10)
    d_lum_fisher = 2 * np.abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))

    term_C = comp['term_C'].values
    term_H = comp['term_H'].values
    RT = comp['RT'].values

    X_b1 = np.column_stack([
        d_lum_fisher,                    # Fisher luminance (replaces ΔL'/SL)
        np.abs(term_C),                  # CIEDE2000 chroma term
        np.abs(term_H),                  # CIEDE2000 hue term
    ])
    r_b1, _ = test(X_b1, "B1. Fisher-SL + CIEDE2000 CH")
    print(f"  B1. Fisher-SL + CIEDE2000 CH:    r = {r_b1:.4f}")

    X_b2 = np.column_stack([
        d_lum_fisher,
        term_C**2, term_H**2,
        RT * term_C * term_H,
    ])
    r_b2, _ = test(X_b2, "B2. Fisher-SL + CIEDE2000 CH²+RT")
    print(f"  B2. Fisher-SL + CIEDE2000 CH²+RT: r = {r_b2:.4f}")

    # The key hybrid: CIEDE2000 total + Fisher luminance
    ell_mid = (ell1 + ell2) / 2
    ell_centered = (ell_mid - 0.5)**2

    X_b3 = np.column_stack([de00, d_lum_fisher])
    r_b3, _ = test(X_b3, "B3. ΔE₀₀ + d_lum_Fisher")
    print(f"  B3. ΔE₀₀ + d_lum_Fisher:         r = {r_b3:.4f}")

    # ════════════════════════════════════════════════════════════
    # C. PER-CHANNEL FISHER GEODESICS ON RAW LMS
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("C. PER-CHANNEL FISHER ON RAW LMS")
    print("=" * 70)

    # arcsin(√(c/c_max)) per channel — Fisher geodesic on each cone
    # This preserves absolute scale (no simplex normalization)
    lms_max = np.maximum(lms1.max(axis=0), lms2.max(axis=0))

    def fisher_lms(lms, lms_ref):
        """Fisher geodesic coordinates per channel."""
        t = np.clip(lms / np.maximum(lms_ref, 1e-10), 1e-10, 1-1e-10)
        return 2 * np.arcsin(np.sqrt(t))

    # Use D65 white as reference
    lms_white = xyz_to_lms(np.array([Xn, Yn, Zn]))
    fl1 = fisher_lms(lms1, lms_white[None,:])
    fl2 = fisher_lms(lms2, lms_white[None,:])
    dfl = fl2 - fl1  # signed differences
    dfl_abs = np.abs(dfl)
    fl_mid = (fl1 + fl2) / 2

    X_c1 = np.column_stack([dfl_abs])
    r_c1, _ = test(X_c1, "C1. Fisher-LMS pure")
    print(f"  C1. Fisher-LMS per-channel:      r = {r_c1:.4f}")

    # With γ weighting
    dfl_gamma = dfl_abs * GAMMAS[None,:]
    X_c1g = np.column_stack([dfl_gamma])
    r_c1g, _ = test(X_c1g, "C1g. Fisher-LMS γ-weighted")
    print(f"  C1g. Fisher-LMS γ-weighted:      r = {r_c1g:.4f}")

    # + CIEDE2000-style corrections
    fl_norm = np.sqrt(np.sum(dfl_abs**2, axis=1))
    fl_mid_norm = np.sqrt(np.sum(fl_mid**2, axis=1))

    X_c2 = np.column_stack([
        dfl_abs,                           # 3 channels
        dfl_abs * fl_mid,                  # SC per channel (3)
        dfl_abs[:,0] * dfl_abs[:,1],       # L×M interaction
        dfl_abs[:,1] * dfl_abs[:,2],       # M×S interaction
        dfl_abs[:,0] * dfl_abs[:,2],       # L×S interaction
    ])
    r_c2, _ = test(X_c2, "C2. Fisher-LMS + corrections")
    print(f"  C2. Fisher-LMS + corrections:    r = {r_c2:.4f}")

    # CIEDE2000 + Fisher-LMS channels
    X_c3 = np.column_stack([de00, dfl_abs])
    r_c3, _ = test(X_c3, "C3. ΔE₀₀ + Fisher-LMS")
    print(f"  C3. ΔE₀₀ + Fisher-LMS channels:  r = {r_c3:.4f}")

    X_c4 = np.column_stack([de00, dfl_gamma])
    r_c4, _ = test(X_c4, "C4. ΔE₀₀ + Fisher-LMS·γ")
    print(f"  C4. ΔE₀₀ + Fisher-LMS·γ:         r = {r_c4:.4f}")

    # ════════════════════════════════════════════════════════════
    # D. OPPONENT CHANNELS IN FISHER COORDINATES
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("D. OPPONENT CHANNELS IN FISHER SPACE")
    print("=" * 70)

    # Opponent: (L-M), S-(L+M)/2, (L+M+S)/3 in Fisher-LMS space
    opp_lm1 = fl1[:,0] - fl1[:,1]  # L-M
    opp_lm2 = fl2[:,0] - fl2[:,1]
    opp_s1 = fl1[:,2] - (fl1[:,0] + fl1[:,1])/2  # S-(L+M)/2
    opp_s2 = fl2[:,2] - (fl2[:,0] + fl2[:,1])/2
    opp_lum1 = (fl1[:,0] + fl1[:,1] + fl1[:,2]) / 3  # luminance
    opp_lum2 = (fl2[:,0] + fl2[:,1] + fl2[:,2]) / 3

    d_opp_lm = np.abs(opp_lm2 - opp_lm1)
    d_opp_s = np.abs(opp_s2 - opp_s1)
    d_opp_lum = np.abs(opp_lum2 - opp_lum1)

    X_d1 = np.column_stack([d_opp_lum, d_opp_lm, d_opp_s])
    r_d1, _ = test(X_d1, "D1. Fisher-opponent pure")
    print(f"  D1. Fisher-opponent pure:         r = {r_d1:.4f}")

    # With γ weighting on opponents
    d_opp_lm_g = d_opp_lm * G3  # L-M weighted by γ₃
    d_opp_s_g = d_opp_s * G7    # S weighted by γ₇
    d_opp_lum_g = d_opp_lum      # luminance unweighted

    X_d1g = np.column_stack([d_opp_lum_g, d_opp_lm_g, d_opp_s_g])
    r_d1g, _ = test(X_d1g, "D1g. Fisher-opponent γ-weighted")
    print(f"  D1g. Fisher-opponent γ-weighted:  r = {r_d1g:.4f}")

    # CIEDE2000 + Fisher opponents
    X_d2 = np.column_stack([de00, d_opp_lum, d_opp_lm, d_opp_s])
    r_d2, _ = test(X_d2, "D2. ΔE₀₀ + Fisher-opponents")
    print(f"  D2. ΔE₀₀ + Fisher-opponents:     r = {r_d2:.4f}")

    # With interactions
    opp_lm_mid = (opp_lm1 + opp_lm2) / 2
    opp_s_mid = (opp_s1 + opp_s2) / 2

    X_d3 = np.column_stack([
        de00,
        d_opp_lum, d_opp_lm, d_opp_s,
        d_opp_lm * np.abs(opp_lm_mid),   # chroma-weighted L-M
        d_opp_s * np.abs(opp_s_mid),      # chroma-weighted S
        d_opp_lum * ell_centered,         # dark correction
    ])
    r_d3, _ = test(X_d3, "D3. ΔE₀₀ + Fisher-opp + corrections")
    print(f"  D3. ΔE₀₀ + Fisher-opp + corr:   r = {r_d3:.4f}")

    # ════════════════════════════════════════════════════════════
    # E. POLYNOMIAL / INTERACTION FEATURES
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("E. POLYNOMIAL AND INTERACTION FEATURES")
    print("=" * 70)

    # Take best base (ΔE₀₀ + d_lum) and add polynomial interactions
    X_base = np.column_stack([de00, d_lum_fisher])

    # Degree 2 polynomial
    poly2 = PolynomialFeatures(degree=2, include_bias=False)
    X_e1 = poly2.fit_transform(X_base)
    r_e1, _ = test(X_e1, "E1. (ΔE₀₀, d_lum) degree-2")
    print(f"  E1. (ΔE₀₀, d_lum) degree-2:     r = {r_e1:.4f}  ({X_e1.shape[1]} feat)")

    # Best combo + polynomial
    X_base2 = np.column_stack([de00, d_lum_fisher, dfl_abs[:,2]])  # + Fisher S-cone
    poly2b = PolynomialFeatures(degree=2, include_bias=False)
    X_e2 = poly2b.fit_transform(X_base2)
    r_e2, _ = test(X_e2, "E2. (ΔE₀₀, d_lum, dS_fisher) deg-2")
    print(f"  E2. (ΔE₀₀, d_lum, dξ_S) deg-2:  r = {r_e2:.4f}  ({X_e2.shape[1]} feat)")

    # Full Fisher-LMS + ΔE₀₀ polynomial
    X_base3 = np.column_stack([de00, dfl_abs])
    poly2c = PolynomialFeatures(degree=2, include_bias=False)
    X_e3 = poly2c.fit_transform(X_base3)
    r_e3, _ = test(X_e3, "E3. (ΔE₀₀, dξ_L, dξ_M, dξ_S) deg-2")
    print(f"  E3. (ΔE₀₀, dξ_L,M,S) deg-2:     r = {r_e3:.4f}  ({X_e3.shape[1]} feat)")

    # ════════════════════════════════════════════════════════════
    # F. FULL SCS-NATIVE ΔE (NO CIELAB DEPENDENCY)
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("F. FULL SCS-NATIVE (NO CIELAB)")
    print("=" * 70)

    # Can we match CIEDE2000 without any CIELAB/CIEDE2000 input?
    # Use only Fisher-LMS coordinates + corrections

    X_f1 = np.column_stack([
        dfl_abs,                               # 3 Fisher-LMS channels
        dfl_abs * fl_mid,                      # midpoint weighting (3)
        dfl_abs[:,0] * dfl_abs[:,1],           # L×M
        dfl_abs[:,1] * dfl_abs[:,2],           # M×S
        dfl_abs[:,0] * dfl_abs[:,2],           # L×S
    ])
    r_f1, _ = test(X_f1, "F1. Fisher-LMS full (no CIELAB)")
    print(f"  F1. Fisher-LMS full:              r = {r_f1:.4f}  (9 feat, no CIELAB)")

    # + γ weighting + hue harmonics
    hue_fl = np.arctan2(np.sqrt(3)*(fl_mid[:,1] - fl_mid[:,2]),
                         2*fl_mid[:,0] - fl_mid[:,1] - fl_mid[:,2])
    X_f2 = np.column_stack([
        dfl_gamma,                              # 3 γ-weighted channels
        dfl_gamma * fl_mid,                     # midpoint weighting (3)
        fl_norm * np.cos(hue_fl),               # hue harmonic 1
        fl_norm * np.sin(hue_fl),               # hue harmonic 2
        fl_norm * np.cos(2*hue_fl),             # hue harmonic 3
        dfl_abs[:,2] * fl_mid[:,2],             # blue interaction
    ])
    r_f2, _ = test(X_f2, "F2. Fisher-LMS-γ + hue (no CIELAB)")
    print(f"  F2. Fisher-LMS-γ + hue:           r = {r_f2:.4f}  (10 feat, no CIELAB)")

    # Fisher-LMS polynomial (no CIELAB at all)
    poly_f = PolynomialFeatures(degree=2, include_bias=False)
    X_f3 = poly_f.fit_transform(dfl_abs)
    r_f3, _ = test(X_f3, "F3. Fisher-LMS degree-2 (no CIELAB)")
    print(f"  F3. Fisher-LMS degree-2:          r = {r_f3:.4f}  ({X_f3.shape[1]} feat, no CIELAB)")

    # γ-weighted Fisher-LMS polynomial
    X_f4 = poly_f.fit_transform(dfl_gamma)
    r_f4, _ = test(X_f4, "F4. Fisher-LMS-γ degree-2 (no CIELAB)")
    print(f"  F4. Fisher-LMS-γ degree-2:        r = {r_f4:.4f}  ({X_f4.shape[1]} feat, no CIELAB)")

    # Fisher-opponent polynomial
    X_opp_base = np.column_stack([d_opp_lum, d_opp_lm, d_opp_s])
    X_f5 = poly_f.fit_transform(X_opp_base)
    r_f5, _ = test(X_f5, "F5. Fisher-opponent degree-2 (no CIELAB)")
    print(f"  F5. Fisher-opponent degree-2:     r = {r_f5:.4f}  ({X_f5.shape[1]} feat, no CIELAB)")

    # ════════════════════════════════════════════════════════════
    # GRAND SUMMARY
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GRAND SUMMARY")
    print("=" * 70)

    results.sort(key=lambda x: x[1])
    print(f"\n  {'Model':<50s} {'r':>6s} {'#f':>4s} {'vs ΔE₀₀':>8s}")
    print(f"  {'-'*50} {'-'*6} {'-'*4} {'-'*8}")
    for name, r, nf in results:
        gap = r - r_00
        marker = " ★" if r > r_00 else ""
        print(f"  {name:<50s} {r:6.4f} {nf:4d} {gap:+8.4f}{marker}")

    print(f"\n  CIEDE2000 reference:               r = {r_00:.4f}")

    # Find best that beats CIEDE2000
    beats = [(n, r, f) for n, r, f in results if r > r_00]
    if beats:
        print(f"\n  Models that beat CIEDE2000:")
        for name, r, nf in sorted(beats, key=lambda x: -x[1]):
            print(f"    {name:<48s} r = {r:.4f} (+{r-r_00:.4f}), {nf} feat")

    # Best SCS-only (no CIELAB)
    scs_only = [(n, r, f) for n, r, f in results if 'no CIELAB' in n]
    if scs_only:
        print(f"\n  Best SCS-native (no CIELAB dependency):")
        for name, r, nf in sorted(scs_only, key=lambda x: -x[1])[:3]:
            print(f"    {name:<48s} r = {r:.4f}, {nf} feat")


if __name__ == '__main__':
    main()
