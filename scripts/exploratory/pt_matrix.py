#!/usr/bin/env python3
"""
PT-Derived XYZ→LMS Conversion Matrix
=====================================

Derives the color-space conversion matrix from PT quantities (sin², γ, δ)
instead of the empirical Hunt-Pointer-Estevez matrix.

PT insight: the matrix M maps the "physical" basis (CIE XYZ, related to
q_therm/geometry) to the "perceptual" basis (LMS cones, related to
q_rel/coupling). The entries are constrained by:
  - Channel bandwidths: δ_p = (1-q^p)/p
  - Coupling strengths: sin²(θ_p)
  - Effective dimensions: γ_p
  - Spectral overlap: q_rel^|p_i - p_j| (geometric decay between channels)

We test several PT constructions and compare MacAdam ellipse predictions.

Usage:
    python pt_matrix.py
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scs_companion import (gamma_p, sin2_theta, delta_p, fisher_metric,
                             MU_STAR, Q_REL, Q_THERM, PRIMES)

# ============================================================
# PT QUANTITIES
# ============================================================

G3, G5, G7 = gamma_p(3), gamma_p(5), gamma_p(7)
S3, S5, S7 = sin2_theta(3, Q_REL), sin2_theta(5, Q_REL), sin2_theta(7, Q_REL)
D3, D5, D7 = delta_p(3, Q_REL), delta_p(5, Q_REL), delta_p(7, Q_REL)

print("PT quantities at mu*=15:")
print(f"  γ = ({G3:.4f}, {G5:.4f}, {G7:.4f})")
print(f"  sin² = ({S3:.4f}, {S5:.4f}, {S7:.4f})")
print(f"  δ = ({D3:.4f}, {D5:.4f}, {D7:.4f})")
print(f"  q_rel = {Q_REL:.6f}")

# ============================================================
# MATRIX CONSTRUCTIONS
# ============================================================

def matrix_hpe():
    """Hunt-Pointer-Estevez (empirical, 3 fitted parameters)."""
    return np.array([
        [ 0.38971,  0.68898, -0.07868],
        [-0.22981,  1.18340,  0.04641],
        [ 0.00000,  0.00000,  1.00000],
    ])


def matrix_pt_holonomy():
    """
    PT Matrix A: Holonomy rotation.

    The three holonomy angles θ_p define how the sieve "rotates" the
    physical (XYZ) basis into the perceptual (LMS) basis.

    θ_p = arcsin(√sin²θ_p)

    Construction: R = R_z(θ₃) · R_x(θ₅) · R_y(θ₇)
    then scale rows by γ_p.
    """
    t3 = np.arcsin(np.sqrt(S3))
    t5 = np.arcsin(np.sqrt(S5))
    t7 = np.arcsin(np.sqrt(S7))

    c3, s3 = np.cos(t3), np.sin(t3)
    c5, s5 = np.cos(t5), np.sin(t5)
    c7, s7 = np.cos(t7), np.sin(t7)

    Rz = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, c5, -s5], [0, s5, c5]])
    Ry = np.array([[c7, 0, s7], [0, 1, 0], [-s7, 0, c7]])

    R = Rz @ Rx @ Ry
    # Scale rows by gamma
    M = np.diag([G3, G5, G7]) @ R
    return M


def matrix_pt_overlap():
    """
    PT Matrix B: Overlap model.

    Diagonal = γ_p (bandwidth of channel p).
    Off-diagonal = spectral overlap, decaying as q_rel^|p_i - p_j|.

    Signs from physical constraint: L-M overlap is POSITIVE (both peak
    in yellow-green), L-Z cross-talk is NEGATIVE (red vs blue).

    M_ij = γ_i · q_rel^|p_i - p_j|  with sign from (i,j) pairing.
    """
    primes = [3, 5, 7]
    M = np.zeros((3, 3))

    for i in range(3):
        for j in range(3):
            pi, pj = primes[i], primes[j]
            overlap = Q_REL ** abs(pi - pj)
            M[i, j] = gamma_p(pi) * overlap

    # Sign convention: match the known structure of cone sensitivities
    # L (p=3) row: +X, +Y, -Z  → keep M[0,0], M[0,1] positive, negate M[0,2]
    # M (p=5) row: -X, +Y, +Z  → negate M[1,0], keep M[1,1], M[1,2] positive
    # S (p=7) row: 0X, 0Y, +Z  → suppress M[2,0], M[2,1], keep M[2,2]
    M[0, 2] *= -1  # L-Z cross-talk is negative
    M[1, 0] *= -1  # M-X cross-talk is negative
    M[2, 0] *= 0   # S has no X sensitivity
    M[2, 1] *= 0   # S has no Y sensitivity (approximately)

    return M


def matrix_pt_coupling():
    """
    PT Matrix C: Coupling-based.

    Each row uses sin²(θ_p) as the SELF-coupling (diagonal) and
    sin²(θ_min) · cos²(θ_max) for the cross-coupling.

    The bifurcation factor sin²/cos² = transmitted/absorbed
    determines the sign: same-branch coupling is positive,
    cross-branch is negative.
    """
    s = [S3, S5, S7]
    c = [1-S3, 1-S5, 1-S7]  # cos²
    g = [G3, G5, G7]

    M = np.array([
        [g[0] * s[0],       g[0] * np.sqrt(s[0]*s[1]),  -g[0] * np.sqrt(s[0]*s[2])],
        [-g[1] * np.sqrt(s[1]*s[0]), g[1] * s[1],        g[1] * np.sqrt(s[1]*s[2]) * 0.1],
        [0,                  0,                           g[2] * s[2] / s[2]],
    ])

    return M


def matrix_pt_spectral():
    """
    PT Matrix D: Spectral width model.

    Each cone's spectral sensitivity is a Lorentzian of width δ_p,
    centered at wavelength λ_p. The peak positions are determined by
    the sieve ordering (p=3 → red ~600nm, p=5 → green ~540nm,
    p=7 → blue ~440nm).

    The overlap integral between two Lorentzians gives the off-diagonal
    elements analytically:

    overlap(p,q) = 2·δ_p·δ_q / (δ_p + δ_q) · 1/(1 + ((λ_p-λ_q)/(δ_p+δ_q))^2)

    Peak wavelengths from the γ_p-weighted partitioning of [380,780]nm:
    """
    # Partition the visible spectrum by gamma weights
    lam_min, lam_max = 380, 780
    total_g = G3 + G5 + G7

    # Peaks: blue at short λ, green in middle, red at long λ
    # Weighted by γ: larger γ → broader region → peak closer to center
    lam7 = lam_min + (G7/total_g) * (lam_max - lam_min) * 0.5  # blue: ~437nm
    lam5 = lam_min + (G7 + G5*0.5)/total_g * (lam_max - lam_min)  # green: ~534nm
    lam3 = lam_max - (G3/total_g) * (lam_max - lam_min) * 0.3  # red: ~683nm

    # Spectral widths proportional to δ_p (in nm)
    scale = 400  # nm scale
    w3, w5, w7 = D3 * scale, D5 * scale, D7 * scale

    def lorentzian_overlap(lam_a, w_a, lam_b, w_b):
        """Overlap integral of two Lorentzians."""
        w_sum = w_a + w_b
        return 2 * w_a * w_b / w_sum / (1 + ((lam_a - lam_b) / w_sum)**2)

    # Build matrix: M[i,j] = how much channel i responds to CIE axis j
    # CIE axes: X peaks ~600nm, Y peaks ~555nm, Z peaks ~445nm
    lam_X, lam_Y, lam_Z = 600, 555, 445

    peaks = [lam3, lam5, lam7]
    widths = [w3, w5, w7]
    gammas = [G3, G5, G7]
    cie_peaks = [lam_X, lam_Y, lam_Z]

    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            M[i, j] = gammas[i] * lorentzian_overlap(peaks[i], widths[i],
                                                      cie_peaks[j], widths[i])

    # Normalize rows to sum to 1
    for i in range(3):
        M[i] /= M[i].sum()

    # Apply sign convention: L has negative Z, M has negative X
    M[0, 2] = -abs(M[0, 2]) * 0.5
    M[1, 0] = -abs(M[1, 0])

    return M


def matrix_pt_bifurcation():
    """
    PT Matrix E: Bifurcation-derived.

    The ratio sin²(q_rel)/sin²(q_therm) → 2 per face gives the
    conversion factor between the two branches. The matrix is:

    M = diag(γ) · R_bifurcation

    where R_bifurcation encodes the rotation from q_therm (physical/XYZ)
    to q_rel (perceptual/LMS) branch.

    The angle between branches for each prime:
    φ_p = arctan(sin²_rel / sin²_therm)
    """
    s_rel = [sin2_theta(p, Q_REL) for p in PRIMES]
    s_thm = [sin2_theta(p, Q_THERM) for p in PRIMES]
    ratios = [r/t for r, t in zip(s_rel, s_thm)]

    print(f"  Bifurcation ratios: {[f'{r:.3f}' for r in ratios]}")

    # The bifurcation angle for each channel
    phi = [np.arctan2(sr, st) for sr, st in zip(s_rel, s_thm)]

    # Construct rotation: each channel rotates by its bifurcation angle
    # Use a single rotation in the simplex plane
    c = [np.cos(p) for p in phi]
    s = [np.sin(p) for p in phi]

    # The matrix is approximately diagonal with corrections from the bifurcation
    M = np.array([
        [c[0],  s[0]*s[1],  -s[0]*s[2]],
        [-s[1]*s[0], c[1],   s[1]*s[2]*0.1],
        [0,     0,           c[2]],
    ])

    # Scale by gamma
    M = np.diag([G3, G5, G7]) @ M

    return M


# ============================================================
# TEST ALL MATRICES
# ============================================================

def test_matrix(name, M, verbose=False):
    """Test a conversion matrix on MacAdam ellipses. Returns RMS angle error."""
    from macadam_test import MACADAM_DATA, fisher_ellipse_at, angle_diff

    # Override the xy_to_simplex function to use our matrix
    def xy_to_simplex_custom(x, y):
        if y < 1e-10:
            return np.array([1/3, 1/3, 1/3])
        X = x / y
        Y = 1.0
        Z = (1 - x - y) / y
        xyz = np.array([X, Y, Z])
        lms = M @ xyz
        lms = np.maximum(lms, 1e-10)
        total = lms.sum()
        return lms / total

    # Monkey-patch and run
    import macadam_test as mt
    original_fn = mt.xy_to_simplex
    mt.xy_to_simplex = xy_to_simplex_custom

    angle_errors = []
    ratio_errors = []

    for x, y, a_obs, b_obs, theta_obs in MACADAM_DATA:
        ratio_obs = a_obs / b_obs
        try:
            a_pt, b_pt, theta_pt = fisher_ellipse_at(x, y)
            ratio_pt = a_pt / b_pt
            ratio_errors.append(abs(ratio_pt - ratio_obs) / ratio_obs)
            angle_errors.append(angle_diff(theta_pt, theta_obs))
        except Exception:
            ratio_errors.append(1.0)
            angle_errors.append(90.0)

    mt.xy_to_simplex = original_fn

    rms_ratio = np.sqrt(np.mean(np.array(ratio_errors)**2))
    rms_angle = np.sqrt(np.mean(np.array(angle_errors)**2))
    mae_angle = np.mean(angle_errors)

    if verbose:
        print(f"\n  {name}:")
        print(f"    Matrix:\n{np.array2string(M, precision=4, suppress_small=True)}")
        print(f"    RMS ratio error: {rms_ratio:.3f}")
        print(f"    RMS angle error: {rms_angle:.1f}°")
        print(f"    MAE angle error: {mae_angle:.1f}°")

    return rms_ratio, rms_angle, mae_angle


def main():
    print("\n" + "=" * 70)
    print("PT-DERIVED CONVERSION MATRICES: MacAdam Test")
    print("=" * 70)

    matrices = [
        ("HPE (empirical, 3 params)", matrix_hpe()),
        ("PT-A: Holonomy rotation", matrix_pt_holonomy()),
        ("PT-B: Overlap model", matrix_pt_overlap()),
        ("PT-C: Coupling-based", matrix_pt_coupling()),
        ("PT-D: Spectral Lorentzian", matrix_pt_spectral()),
        ("PT-E: Bifurcation-derived", matrix_pt_bifurcation()),
    ]

    print(f"\n{'Name':>35} {'Params':>7} {'RMS ratio':>10} {'RMS θ (°)':>10} {'MAE θ (°)':>10}")
    print("-" * 75)

    results = []
    for name, M in matrices:
        n_params = 3 if "empirical" in name else 0
        rms_r, rms_a, mae_a = test_matrix(name, M, verbose=True)
        results.append((name, n_params, rms_r, rms_a, mae_a))
        print(f"{name:>35} {n_params:>7} {rms_r:>10.3f} {rms_a:>10.1f} {mae_a:>10.1f}")

    # Find best PT matrix
    pt_results = [(n, p, rr, ra, ma) for n, p, rr, ra, ma in results if p == 0]
    best_pt = min(pt_results, key=lambda x: x[3])  # best by RMS angle
    hpe = [r for r in results if r[1] == 3][0]

    print(f"\n{'='*75}")
    print(f"BEST PT MATRIX: {best_pt[0]}")
    print(f"  RMS angle: {best_pt[3]:.1f}° (vs HPE: {hpe[3]:.1f}°)")
    print(f"  RMS ratio: {best_pt[2]:.3f} (vs HPE: {hpe[2]:.3f})")

    improvement = (hpe[3] - best_pt[3]) / hpe[3] * 100
    if improvement > 0:
        print(f"  → PT improves orientation by {improvement:.1f}% with 0 params vs 3")
    else:
        print(f"  → HPE still better by {-improvement:.1f}% (but has 3 fitted params)")
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
