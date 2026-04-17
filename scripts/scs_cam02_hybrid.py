#!/usr/bin/env python3
"""
SCS + CIECAM02 hybrid — COMBVD fit and cross-validated evaluation
==================================================================

This script fits and evaluates the SCS + CIECAM02 hybrid color-difference
metric on the COMBVD dataset (3813 pairs; BFD, RIT-DuPont, Leeds, Witt).
It produces the r = 0.824 headline figure quoted in the paper's abstract
and Section on "SCS as an additive information channel".

The 6 features fitted (Ridge, alpha=1, 5-fold cross-validation):

  SCS features (0 parameters, derived from s = 1/2):
    f1 = d_lum    Fisher-Bernoulli geodesic on p=2 (luminance)
                  d_lum = 2 | arcsin(sqrt(Y1)) - arcsin(sqrt(Y2)) |
    f2 = d_chrom  Bhattacharyya geodesic on the Delta^2 simplex
                  d_chrom = 2 arccos( sum_p sqrt(pi1_p * pi2_p) )

  CIECAM02 features (computed via colour-science library):
    f3 = |DeltaJ|   lightness
    f4 = |DeltaC|   chroma
    f5 = |DeltaM|   colorfulness
    f6 = |DeltaH|   hue (great-circle on CAM02 hue wheel)

Reference points on the same COMBVD split (reported in the paper):
    CIELAB     r = 0.755   (3 CIE params)
    CIEDE2000  r = 0.878   (5 CIE params)
    SCS (pure) r = 0.500   (0 params, Fisher on simplex)
    SCS+CAM02  r = 0.824   (6 regressed weights)   <-- this script

Usage:
    python3 scs_cam02_hybrid.py                 # full fit + 5-fold CV
    python3 scs_cam02_hybrid.py --bootstrap 500 # add bootstrap CI

Data:
    datasets/COMBVD_3813.csv
Cited at: PT_COLOR.tex abstract and §Discussion "SCS + CIECAM02 hybrid"
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "datasets"
COMBVD = DATA_DIR / "COMBVD_3813.csv"

# ---------------------------------------------------------------------------
# SCS constants (derived in PT_COLOR.tex §The Sieve Foundation)
#   gamma_p at mu* = 15 on the active primes {3, 5, 7}
# ---------------------------------------------------------------------------
GAMMAS = np.array([0.80799, 0.69575, 0.59509])  # gamma_3, gamma_5, gamma_7

# Hunt-Pointer-Estevez (HPE) matrix: sRGB linear -> LMS
# (D65 white point, matches §Physical bridge)
XYZ_TO_LMS_HPE = np.array([
    [ 0.38971,  0.68898, -0.07868],
    [-0.22981,  1.18340,  0.04641],
    [ 0.00000,  0.00000,  1.00000],
])


def xyz_to_simplex(xyz: np.ndarray) -> np.ndarray:
    """
    CIE XYZ -> SCS barycentric coordinates (pi_3, pi_5, pi_7) on Delta^2.

    Implements the pipeline described in PT_COLOR.tex §Coordinates:
      XYZ -> LMS (via HPE) -> gamma-weighted simplex coordinates.
    """
    xyz = np.asarray(xyz, dtype=float)
    lms = XYZ_TO_LMS_HPE @ xyz
    lms = np.clip(lms, 1e-9, None)  # guard against tiny negatives from matrix
    weighted = GAMMAS * lms
    return weighted / weighted.sum()


def scs_d_lum(y1: float, y2: float) -> float:
    """
    Fisher-Bernoulli geodesic on the luminance channel (p = 2).

    d_lum = 2 |arcsin(sqrt(Y1)) - arcsin(sqrt(Y2))|

    Y values are clipped to (eps, 1-eps) for numerical stability.
    """
    y1c = np.clip(y1, 1e-6, 1 - 1e-6)
    y2c = np.clip(y2, 1e-6, 1 - 1e-6)
    return 2.0 * abs(np.arcsin(np.sqrt(y1c)) - np.arcsin(np.sqrt(y2c)))


def scs_d_chrom(pi1: np.ndarray, pi2: np.ndarray) -> float:
    """
    Bhattacharyya (Hellinger) geodesic on the chromatic simplex.

    d_chrom = 2 * arccos( sum_p sqrt(pi1_p * pi2_p) )

    This is the great-circle distance on the positive orthant of the
    unit sphere (the Fisher metric made Euclidean in sqrt-coordinates).
    """
    bc = np.clip(np.sum(np.sqrt(pi1 * pi2)), 0.0, 1.0)
    return 2.0 * np.arccos(bc)


# ---------------------------------------------------------------------------
# CIECAM02 via colour-science
# ---------------------------------------------------------------------------
def cam02_features(
    xyz1: np.ndarray,
    xyz2: np.ndarray,
    xyz_w: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Return (|dJ|, |dC|, |dM|, dH) for a pair of XYZ stimuli under CIECAM02
    with standard average-surround viewing conditions.

    Uses the colour-science (v0.4.x) CIECAM02 implementation. Hue
    differences dH use the great-circle arc on the hue wheel
    (shortest angular distance, in degrees scaled to match CAM02 dH
    conventions: dH = 2 * sqrt(C1 * C2) * sin((h2-h1)/2)).
    """
    import colour

    # Standard CIECAM02 viewing conditions (average surround, adapting lum 80 nt)
    L_A = 80.0
    Y_b = 20.0
    surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]

    spec1 = colour.XYZ_to_CIECAM02(
        XYZ=xyz1 * 100.0,  # colour-science expects XYZ scaled to [0, 100]
        XYZ_w=xyz_w * 100.0,
        L_A=L_A,
        Y_b=Y_b,
        surround=surround,
    )
    spec2 = colour.XYZ_to_CIECAM02(
        XYZ=xyz2 * 100.0,
        XYZ_w=xyz_w * 100.0,
        L_A=L_A,
        Y_b=Y_b,
        surround=surround,
    )

    dJ = abs(spec1.J - spec2.J)
    dC = abs(spec1.C - spec2.C)
    dM = abs(spec1.M - spec2.M)

    # Hue-difference convention of CIEDE/CAM: dH = 2 sqrt(C1 C2) sin(dh/2)
    dh_rad = np.deg2rad(spec2.h - spec1.h)
    # wrap to [-pi, pi]
    dh_rad = (dh_rad + np.pi) % (2 * np.pi) - np.pi
    dH = 2.0 * np.sqrt(max(spec1.C * spec2.C, 0.0)) * np.sin(dh_rad / 2.0)
    dH = abs(dH)

    return dJ, dC, dM, dH


# ---------------------------------------------------------------------------
# CIELAB baseline
# ---------------------------------------------------------------------------
def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """CIE XYZ -> CIELAB (full piecewise, matches CIE 1976)."""
    def f(t: np.ndarray) -> np.ndarray:
        delta = 6.0 / 29.0
        return np.where(t > delta ** 3, t ** (1.0 / 3.0), t / (3.0 * delta ** 2) + 4.0 / 29.0)
    xn = xyz / white
    fx, fy, fz = f(xn[0]), f(xn[1]), f(xn[2])
    return np.array([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)])


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_combvd(path: Path) -> list[dict]:
    """Load COMBVD as a list of dict rows with xyz1/xyz2/xyzw/DV."""
    pairs = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pairs.append({
                    "dataset": row.get("dataset", ""),
                    "xyz1": np.array([float(row["X1"]), float(row["Y1"]), float(row["Z1"])]),
                    "xyz2": np.array([float(row["X2"]), float(row["Y2"]), float(row["Z2"])]),
                    "xyz_w": np.array([float(row["Xw"]), float(row["Yw"]), float(row["Zw"])]),
                    "L1": float(row["L1"]),
                    "DV": float(row["DV"]),
                })
            except (ValueError, KeyError):
                continue
    return pairs


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------
def build_features(pairs: list[dict], verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build the 7-feature matrix:
      [d_lum, d_chrom, dJ, dC, dM, dH, d_lab]

    The last column (d_lab = CIELAB Euclidean DeltaE*ab) is included only
    as a baseline reference column; the hybrid fit uses columns 0-5.
    """
    import colour  # imported here to keep top-level import optional

    L_A = 80.0
    Y_b = 20.0
    surround = colour.VIEWING_CONDITIONS_CIECAM02["Average"]

    feature_names = ["d_lum", "d_chrom", "dJ", "dC", "dM", "dH", "d_lab"]
    X = np.zeros((len(pairs), 7), dtype=float)
    y = np.zeros(len(pairs), dtype=float)

    for i, p in enumerate(pairs):
        xyz1, xyz2, xyz_w = p["xyz1"], p["xyz2"], p["xyz_w"]

        # SCS features
        pi1 = xyz_to_simplex(xyz1)
        pi2 = xyz_to_simplex(xyz2)
        X[i, 0] = scs_d_lum(xyz1[1], xyz2[1])
        X[i, 1] = scs_d_chrom(pi1, pi2)

        # CAM02 features (batch would be faster but keep it simple/auditable)
        spec1 = colour.XYZ_to_CIECAM02(xyz1 * 100.0, xyz_w * 100.0, L_A, Y_b, surround)
        spec2 = colour.XYZ_to_CIECAM02(xyz2 * 100.0, xyz_w * 100.0, L_A, Y_b, surround)
        X[i, 2] = abs(spec1.J - spec2.J)
        X[i, 3] = abs(spec1.C - spec2.C)
        X[i, 4] = abs(spec1.M - spec2.M)
        dh = np.deg2rad(spec2.h - spec1.h)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi
        X[i, 5] = abs(2.0 * np.sqrt(max(spec1.C * spec2.C, 0.0)) * np.sin(dh / 2.0))

        # CIELAB Euclidean reference
        lab1 = xyz_to_lab(xyz1, xyz_w)
        lab2 = xyz_to_lab(xyz2, xyz_w)
        X[i, 6] = np.sqrt(np.sum((lab1 - lab2) ** 2))

        y[i] = p["DV"]

        if verbose and (i + 1) % 500 == 0:
            print(f"    ... {i + 1}/{len(pairs)} pairs featurized", flush=True)

    return X, y, feature_names


# ---------------------------------------------------------------------------
# Cross-validated Ridge regression
# ---------------------------------------------------------------------------
def cv_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    K-fold cross-validated Ridge regression with feature standardization
    (fit on train fold, applied to test fold).

    Returns (r_pearson, cv_predictions).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    fold_size = n // n_folds
    predictions = np.zeros(n, dtype=float)

    for k in range(n_folds):
        lo, hi = k * fold_size, (k + 1) * fold_size if k < n_folds - 1 else n
        test = idx[lo:hi]
        train = np.setdiff1d(idx, test)

        scaler = StandardScaler().fit(X[train])
        Xtr = scaler.transform(X[train])
        Xte = scaler.transform(X[test])

        model = Ridge(alpha=alpha).fit(Xtr, y[train])
        predictions[test] = model.predict(Xte)

    r = float(np.corrcoef(predictions, y)[0, 1])
    return r, predictions


def standardized_betas(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Full-data standardized Ridge coefficients (for feature-importance table)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = Ridge(alpha=alpha).fit(Xs, y)
    return model.coef_


# ---------------------------------------------------------------------------
# Bootstrap CI on r (paired-pairs resampling)
# ---------------------------------------------------------------------------
def bootstrap_r_ci(
    predictions: np.ndarray,
    y: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI on Pearson r by paired bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(y)
    rs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ii = rng.integers(0, n, n)
        rs[b] = np.corrcoef(predictions[ii], y[ii])[0, 1]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="Bootstrap samples for r CI (0 = off).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge alpha.")
    parser.add_argument("--folds", type=int, default=5,
                        help="CV folds.")
    args = parser.parse_args()

    print("=" * 64)
    print("  SCS + CIECAM02 hybrid on COMBVD (reproduces abstract r = 0.824)")
    print("=" * 64)

    if not COMBVD.exists():
        print(f"[FATAL] COMBVD dataset not found at {COMBVD}", file=sys.stderr)
        return 1

    pairs = load_combvd(COMBVD)
    print(f"  Loaded {len(pairs)} COMBVD pairs")
    print(f"  Datasets: {sorted(set(p['dataset'] for p in pairs))}")
    print()
    print("  Computing features (SCS + CIECAM02, may take ~30 s)...")

    X, y, names = build_features(pairs)

    # Fit configurations
    configs = {
        "CIELAB (ref)":         [6],                # d_lab only, no fit
        "SCS pure (0 weights)": [0, 1],             # d_lum + d_chrom, 2 weights
        "CAM02 alone (4 w)":    [2, 3, 4, 5],       # CAM features, 4 weights
        "SCS + CAM02 (6 w)":    [0, 1, 2, 3, 4, 5],  # hybrid, 6 weights
    }

    print()
    print(f"  {args.folds}-fold cross-validated Ridge (alpha = {args.alpha})")
    print(f"  {'Model':<24} {'features':>10} {'weights':>8} {'r':>8}")
    print(f"  {'-' * 24} {'-' * 10} {'-' * 8} {'-' * 8}")

    results: dict[str, tuple[float, np.ndarray]] = {}
    for name, cols in configs.items():
        Xi = X[:, cols]
        if name == "CIELAB (ref)":
            # CIELAB as d_lab is a single column with no intercept-free fit;
            # report correlation of the raw feature against DV
            r = float(np.corrcoef(Xi[:, 0], y)[0, 1])
            preds = Xi[:, 0]
        else:
            r, preds = cv_ridge(Xi, y, alpha=args.alpha, n_folds=args.folds)
        results[name] = (r, preds)
        print(f"  {name:<24} {len(cols):>10d} {len(cols):>8d} {r:>8.3f}")

    # Reference values (not computed here; for context)
    print(f"  {'CIEDE2000 (reference)':<24} {'5':>10} {'5':>8} {'0.878':>8}")

    # Headline claim
    r_hybrid = results["SCS + CAM02 (6 w)"][0]
    print()
    print(f"  SCS + CAM02 hybrid r = {r_hybrid:.4f}")
    print(f"  Paper claim (abstract): r = 0.824")
    diff = abs(r_hybrid - 0.824)
    status = "MATCH (within 0.01)" if diff < 0.01 else ("CLOSE" if diff < 0.02 else "MISMATCH")
    print(f"  |fit - claim| = {diff:.4f}  -> {status}")

    # Feature importance
    print()
    print("  === STANDARDIZED BETAS (full-data Ridge) ===")
    betas = standardized_betas(X[:, :6], y, alpha=args.alpha)
    for name, b in zip(names[:6], betas):
        bar = "#" * int(abs(b) * 30)
        print(f"  {name:<10}  beta = {b:>+7.3f}  {bar}")

    # Bootstrap CI on the hybrid
    if args.bootstrap > 0:
        print()
        print(f"  Bootstrap 95% CI on r (n = {args.bootstrap})...")
        preds_hybrid = results["SCS + CAM02 (6 w)"][1]
        lo, hi = bootstrap_r_ci(preds_hybrid, y, n_boot=args.bootstrap)
        print(f"  r = {r_hybrid:.3f}  95% CI = [{lo:.3f}, {hi:.3f}]")

    # Save machine-readable results
    out = DATA_DIR / "scs_cam02_hybrid_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# SCS + CIECAM02 hybrid, COMBVD 3813 pairs"])
        w.writerow(["# Cross-validation: 5-fold, Ridge alpha=1, seed=42"])
        w.writerow(["# Generated by scs_cam02_hybrid.py"])
        w.writerow([])
        w.writerow(["model", "features", "weights", "r_pearson"])
        for name, (r, _) in results.items():
            cols = configs[name]
            w.writerow([name, len(cols), len(cols), f"{r:.4f}"])
        w.writerow(["CIEDE2000_reference", 5, 5, "0.8780"])
        w.writerow([])
        w.writerow(["# Standardized Ridge betas (SCS + CAM02 hybrid)"])
        w.writerow(["feature", "beta"])
        for name, b in zip(names[:6], betas):
            w.writerow([name, f"{b:.4f}"])

    print()
    print(f"  Saved: {out}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
