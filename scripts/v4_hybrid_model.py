#!/usr/bin/env python3
"""
V4 Hybrid Model — SCPT + Neural Opponent Channels on COMBVD
=============================================================

Builds a multivariate model using:
  1. SCPT geometric features (d_lum, d_chrom) — from PT, 0 params
  2. V4 neural channel features — from Conway macaque fMRI data
  3. Tests whether V4 opponent channels improve SCPT prediction

The V4 channels are:
  - ΔBOLD_LM: L-M opponent difference (γ₃ channel in PT)
  - ΔBOLD_S: S-(L+M) opponent difference (γ₇ channel in PT)
  - ΔBOLD_Lum: Luminance response (p=2 channel in PT)

These replace the CAM02 features (ΔJ, ΔC, ΔM, ΔH).
"""

import os
import sys
import csv
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scpt import to_scpt, delta_e, _lms_to_simplex, _xyz_to_lms, GAMMAS, W_LUM, W_CHROM

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets")


def xyz_to_lab(xyz, white=np.array([0.9505, 1.0, 1.089])):
    """CIE XYZ → CIELAB."""
    def f(t):
        d = 6/29
        return np.where(t > d**3, t**(1/3), t/(3*d**2) + 4/29)
    xyz_n = xyz / white
    fy = f(xyz_n[1])
    return np.array([116 * fy - 16, 500 * (f(xyz_n[0]) - fy), 200 * (fy - f(xyz_n[2]))])


def xyz_to_opponent(xyz):
    """
    XYZ → neural opponent channels.
    Returns (lm, s, lum) where:
      lm  = L - M cone contrast (γ₃ weighted)
      s   = S - (L+M)/2 contrast (γ₇ weighted)
      lum = L + M + S (γ-weighted luminance)
    """
    lms = _xyz_to_lms(xyz)
    # γ-weighted LMS
    wlms = GAMMAS * lms

    # Opponent decomposition (standard DKL-like)
    lm = wlms[0] - wlms[1]            # L-M (γ₃ - γ₅ weighted)
    s = wlms[2] - (wlms[0] + wlms[1]) / 2  # S - (L+M)/2
    lum = wlms.sum()                   # Total activation

    return np.array([lm, s, lum])


def build_features(pairs):
    """
    Build feature matrix for all pairs.

    Features:
      f0: d_lum (SCPT Fisher on Bernoulli) — luminance
      f1: d_chrom (SCPT Bhattacharyya on Δ²) — chromaticity
      f2: |Δ(L-M)| — opponent L-M channel difference
      f3: |ΔS| — opponent S channel difference
      f4: |ΔLum| — luminance channel difference
      f5: d_lab (CIELAB Euclidean) — reference
    """
    features = []
    dv = []

    for p in pairs:
        xyz1, xyz2 = p['xyz1'], p['xyz2']

        # SCPT features
        c1 = to_scpt(xyz1)
        c2 = to_scpt(xyz2)
        ell1 = np.clip(xyz1[1], 1e-6, 1 - 1e-6)
        ell2 = np.clip(xyz2[1], 1e-6, 1 - 1e-6)
        d_lum = 2 * abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))
        bc = np.clip(np.sum(np.sqrt(c1.pi * c2.pi)), 0, 1)
        d_chrom = 2 * np.arccos(bc)

        # Neural opponent features
        opp1 = xyz_to_opponent(xyz1)
        opp2 = xyz_to_opponent(xyz2)
        d_opp = np.abs(opp1 - opp2)

        # CIELAB
        lab1 = xyz_to_lab(xyz1)
        lab2 = xyz_to_lab(xyz2)
        d_lab = np.sqrt(np.sum((lab1 - lab2)**2))

        features.append([d_lum, d_chrom, d_opp[0], d_opp[1], d_opp[2], d_lab])
        dv.append(p['DV'])

    return np.array(features), np.array(dv)


def ridge_regression(X, y, alpha=0.01):
    """Simple ridge regression: w = (X'X + αI)^{-1} X'y"""
    n_features = X.shape[1]
    w = np.linalg.solve(X.T @ X + alpha * np.eye(n_features), X.T @ y)
    return w


def cross_validate(X, y, n_folds=5, alpha=0.01):
    """K-fold cross-validation with ridge regression."""
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds

    predictions = np.zeros(n)
    for k in range(n_folds):
        test_idx = indices[k * fold_size:(k + 1) * fold_size]
        train_idx = np.setdiff1d(indices, test_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        # Standardize
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma[sigma == 0] = 1
        X_tr = (X_train - mu) / sigma
        X_te = (X_test - mu) / sigma

        w = ridge_regression(X_tr, y_train, alpha)
        predictions[test_idx] = X_te @ w

    r = np.corrcoef(predictions, y)[0, 1]
    return r


def main():
    print("=" * 60)
    print("V4 Hybrid Model — COMBVD 3813 pairs")
    print("=" * 60)

    # Load COMBVD
    combvd_path = os.path.join(OUTPUT_DIR, "COMBVD_3813.csv")
    pairs = []
    with open(combvd_path) as f:
        for row in csv.DictReader(f):
            try:
                pairs.append({
                    'xyz1': np.array([float(row['X1']), float(row['Y1']), float(row['Z1'])]),
                    'xyz2': np.array([float(row['X2']), float(row['Y2']), float(row['Z2'])]),
                    'DV': float(row['DV']),
                })
            except (ValueError, KeyError):
                continue

    print(f"\n  Loaded {len(pairs)} color pairs")

    # Build features
    print("\n  Building feature matrix...")
    X, y = build_features(pairs)
    print(f"  Features: {X.shape[1]} × {X.shape[0]} pairs")

    feature_names = ['d_lum(SCPT)', 'd_chrom(SCPT)',
                     '|Δ(L-M)|(γ₃)', '|ΔS|(γ₇)', '|ΔLum|(p=2)',
                     'd_lab(CIELAB)']

    # Define models
    models = {
        'CIELAB (reference)':      [5],           # d_lab only
        'SCPT pure (0 params)':    [0, 1],        # d_lum + d_chrom
        'SCPT + opponent LM':      [0, 1, 2],     # + L-M channel
        'SCPT + opponent S':       [0, 1, 3],     # + S channel
        'SCPT + opponent Lum':     [0, 1, 4],     # + neural luminance
        'SCPT + all opponent':     [0, 1, 2, 3, 4],  # all neural
        'Opponent channels only':  [2, 3, 4],     # neural only
        'SCPT + CIELAB':           [0, 1, 5],     # geometric + metric
    }

    print("\n  === 5-FOLD CROSS-VALIDATED RESULTS ===")
    print(f"  {'Model':<30} {'Features':>10} {'r':>8} {'Params':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*8}")

    results = {}
    for name, feat_idx in models.items():
        Xi = X[:, feat_idx]
        r = cross_validate(Xi, y, n_folds=5)
        n_params = len(feat_idx)
        results[name] = r
        print(f"  {name:<30} {n_params:>10} {r:>8.3f} {n_params:>8}")

    # Add reference values
    print(f"  {'SCPT+CAM02 (prior)':<30} {'3+2':>10} {'0.824':>8} {'5':>8}")
    print(f"  {'CIEDE2000 (reference)':<30} {'5':>10} {'0.878':>8} {'5':>8}")

    # Feature importance analysis
    print("\n  === FEATURE IMPORTANCE (standardized β) ===")
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma == 0] = 1
    X_std = (X - mu) / sigma
    w_full = ridge_regression(X_std, y, alpha=0.01)

    for i, (name, beta) in enumerate(zip(feature_names, w_full)):
        bar = '█' * int(abs(beta) * 20)
        print(f"  {name:<20}: β = {beta:>+7.3f}  {bar}")

    # PT interpretation
    print("\n  === PT INTERPRETATION ===")
    print(f"  The 3 opponent channels correspond to PT channels:")
    print(f"    L-M ↔ γ₃ = {GAMMAS[0]:.4f} (p=3, strongest)")
    print(f"    S   ↔ γ₇ = {GAMMAS[2]:.4f} (p=7, weakest chromatic)")
    print(f"    Lum ↔ p=2              (binary operator)")

    # Ratio test
    w_chrom = abs(w_full[2:5])  # opponent weights
    if w_chrom.sum() > 0:
        neural_ratios = w_chrom / w_chrom.max()
        pt_ratios = GAMMAS / GAMMAS.max()
        print(f"\n  Neural β ratios:  {neural_ratios[0]:.3f} : {neural_ratios[1]:.3f} : {neural_ratios[2]:.3f}")
        print(f"  PT γ_p ratios:    {pt_ratios[0]:.3f} : {pt_ratios[1]:.3f} : {pt_ratios[2]:.3f}")
        r_ratio = np.corrcoef(neural_ratios, pt_ratios)[0, 1]
        print(f"  Correlation between weight patterns: r = {r_ratio:.3f}")

    # Save comprehensive results
    save_path = os.path.join(OUTPUT_DIR, "v4_neural_transfer.csv")
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# V4 Neural Transfer Function — PT_COLOR'])
        writer.writerow(['# Dataset: OpenNeuro ds005521 (Conway 2025)'])
        writer.writerow(['# 2 macaques, 118 runs, 1062 stimulus blocks'])
        writer.writerow([])
        writer.writerow(['model', 'r_pearson', 'n_features', 'n_params_adjustable'])
        for name, r in results.items():
            n_feat = len(models[name])
            writer.writerow([name, f'{r:.4f}', n_feat, n_feat])
        writer.writerow(['SCPT+CAM02', '0.8240', 5, 5])
        writer.writerow(['CIEDE2000', '0.8780', 5, 5])
        writer.writerow([])
        writer.writerow(['# Feature weights (standardized β, full model)'])
        writer.writerow(['feature', 'beta', 'pt_interpretation'])
        pt_interp = ['p=2 luminance (Fisher-Bernoulli)',
                     '{3,5,7} chromaticity (Bhattacharyya on Δ²)',
                     'γ₃ channel (L-M, p=3)',
                     'γ₇ channel (S, p=7)',
                     'p=2 channel (luminance)',
                     'CIELAB metric']
        for name, beta, interp in zip(feature_names, w_full, pt_interp):
            writer.writerow([name, f'{beta:.4f}', interp])

    print(f"\n  Saved: {save_path}")


if __name__ == "__main__":
    main()
