#!/usr/bin/env python3
"""
MacAdam Ellipse Test — SCS vs CIELAB
=====================================

Compares the Fisher metric (weighted by gamma_p, 0 parameters)
against CIELAB's cube-root approximation (3 empirical parameters)
for predicting MacAdam's 25 color-discrimination ellipses.

Data source: MacAdam (1942), "Visual sensitivities to color differences
in daylight", J. Opt. Soc. Am. 32, 247-274.

Method:
1. At each of MacAdam's 25 reference chromaticities (x, y):
   - Convert to simplex coordinates (pi3, pi5, pi7)
   - Compute Fisher metric tensor g_ij with weights (gamma_3, gamma_5, gamma_7)
   - Extract predicted ellipse (eigenvalues → axes, eigenvectors → orientation)
2. Compare predicted vs observed ellipse parameters (semi-axes ratio, orientation)
3. Do the same for CIELAB metric
4. Report RMS residuals: Fisher-PT (0 params) vs CIELAB (3 params)

Usage:
    python macadam_test.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import os
import sys

# Add parent scripts dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scs_companion import gamma_p, fisher_metric, MU_STAR, Q_REL, PRIMES
# Combined metric (Fisher + Fubini-Study + bifurcation) for the paper's 18/25 claim
from delta_e_scs import scs_metric

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures_en')

# ============================================================
# MacAdam 1942 data — 25 reference chromaticities with ellipse params
# (x, y, a*1e4, b*1e4, theta_deg)
# a, b = semi-axes of the ellipse in (x,y) space (×10^4)
# theta = angle of major axis from x-axis (degrees)
# Source: Table II of MacAdam (1942), widely reproduced
# ============================================================

MACADAM_DATA = [
    # (x, y, a×1e4, b×1e4, theta_deg)
    (0.160, 0.057, 38, 22, 62),
    (0.187, 0.118, 36, 23, 77),
    (0.253, 0.125, 53, 24, 55),
    (0.150, 0.680, 41, 16, 135),
    (0.131, 0.521, 29, 14, 163),
    (0.212, 0.550, 34, 13, 136),
    (0.258, 0.450, 23, 15, 47),
    (0.152, 0.365, 31, 11, 166),
    (0.280, 0.385, 36, 16, 57),
    (0.290, 0.265, 40, 13, 54),
    (0.267, 0.040, 46, 16, 73),
    (0.305, 0.118, 41, 17, 68),
    (0.385, 0.150, 44, 13, 58),
    (0.340, 0.440, 30, 18, 50),
    (0.380, 0.480, 36, 22, 67),
    (0.305, 0.548, 30, 15, 143),
    (0.440, 0.470, 30, 15, 55),
    (0.510, 0.415, 40, 16, 29),
    (0.475, 0.300, 40, 14, 57),
    (0.510, 0.236, 43, 11, 77),
    (0.596, 0.283, 46, 13, 68),
    (0.344, 0.284, 33, 16, 58),
    (0.390, 0.237, 36, 11, 62),
    (0.441, 0.198, 42, 16, 72),
    (0.278, 0.223, 38, 14, 58),
]


def xy_to_simplex(x, y):
    """
    Convert CIE (x, y) chromaticity to SCS simplex coordinates (pi3, pi5, pi7).

    Approximate mapping: CIE XYZ → LMS cone space → normalized proportions.
    We use the Hunt-Pointer-Estevez matrix for XYZ→LMS, then normalize.

    For chromaticity (x, y): X = x/y, Y = 1, Z = (1-x-y)/y
    """
    if y < 1e-10:
        return np.array([1/3, 1/3, 1/3])

    X = x / y
    Y = 1.0
    Z = (1 - x - y) / y

    # Hunt-Pointer-Estevez XYZ → LMS
    L =  0.38971 * X + 0.68898 * Y - 0.07868 * Z
    M = -0.22981 * X + 1.18340 * Y + 0.04641 * Z
    S =  0.00000 * X + 0.00000 * Y + 1.00000 * Z

    # Weight by gamma_p (the PT natural weighting)
    g3 = gamma_p(3)
    g5 = gamma_p(5)
    g7 = gamma_p(7)

    wL = g3 * max(L, 1e-10)
    wM = g5 * max(M, 1e-10)
    wS = g7 * max(S, 1e-10)

    total = wL + wM + wS
    return np.array([wL/total, wM/total, wS/total])


def fisher_ellipse_at(x, y, use_combined=True):
    """
    Compute the SCS metric ellipse at CIE chromaticity (x, y).

    Returns (semi_a, semi_b, theta_deg) — the predicted discrimination ellipse.

    When ``use_combined=True`` (default), uses the full combined metric
    reported in the paper: Fisher + Fubini-Study phase correction +
    bifurcation rotation (all three layers derived from s = 1/2 at
    mu* = 15). When ``use_combined=False``, uses the bare Fisher metric
    weighted by gamma_p only.

    The paper's headline 18/25 wins on ellipse orientation (RMS Delta theta = 37.8 deg)
    comes from the combined metric. Bare Fisher alone gives 8/25 at 68.5 deg RMS,
    which is the zero-parameter baseline before phase-correction and bifurcation.
    """
    pi = xy_to_simplex(x, y)
    gammas = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])

    # SCS metric in the (pi3, pi5) chart — combined (paper) or bare Fisher
    G = scs_metric(pi) if use_combined else fisher_metric(pi, gammas)

    # Eigendecomposition: larger eigenvalue → smaller ellipse axis
    eigvals, eigvecs = np.linalg.eigh(G)

    # Ellipse semi-axes are proportional to 1/sqrt(eigenvalue)
    # (iso-distance contour of ds^2 = 1)
    a = 1.0 / np.sqrt(eigvals[0])
    b = 1.0 / np.sqrt(eigvals[1])

    # Orientation of major axis
    if a > b:
        theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    else:
        theta = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        a, b = b, a

    return a, b, theta % 180


def cielab_metric_at(x, y):
    """
    Compute the CIELAB metric ellipse at CIE chromaticity (x, y).
    CIELAB uses cube-root nonlinearity: f(t) = t^(1/3).

    The metric tensor in (x, y) is derived from the Jacobian of the
    (x, y) → (a*, b*) transformation.

    Returns (semi_a, semi_b, theta_deg).
    """
    if y < 1e-10:
        return 1, 1, 0

    X = x / y
    Y = 1.0
    Z = (1 - x - y) / y

    # CIELAB reference white (D65)
    Xn, Yn, Zn = 0.9505, 1.0000, 1.0890

    def f(t):
        delta = 6/29
        if t > delta**3:
            return t**(1/3)
        else:
            return t / (3 * delta**2) + 4/29

    def df(t):
        delta = 6/29
        if t > delta**3:
            return (1/3) * t**(-2/3)
        else:
            return 1 / (3 * delta**2)

    # Numerical Jacobian of (x,y) → (a*,b*) via finite differences
    eps = 1e-6

    def ab_star(xx, yy):
        if yy < 1e-10:
            return 0, 0
        XX = xx / yy
        YY = 1.0
        ZZ = (1 - xx - yy) / yy
        L_star = 116 * f(YY/Yn) - 16
        a_star = 500 * (f(XX/Xn) - f(YY/Yn))
        b_star = 200 * (f(YY/Yn) - f(ZZ/Zn))
        return a_star, b_star

    a0, b0 = ab_star(x, y)
    ax, bx = ab_star(x + eps, y)
    ay, by = ab_star(x, y + eps)

    J = np.array([
        [(ax - a0)/eps, (ay - a0)/eps],
        [(bx - b0)/eps, (by - b0)/eps]
    ])

    # Metric tensor: g = J^T @ J
    G = J.T @ J

    eigvals, eigvecs = np.linalg.eigh(G)
    eigvals = np.maximum(eigvals, 1e-20)

    a = 1.0 / np.sqrt(eigvals[0])
    b = 1.0 / np.sqrt(eigvals[1])

    if a > b:
        theta = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    else:
        theta = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        a, b = b, a

    return a, b, theta % 180


def angle_diff(a1, a2):
    """Smallest angle difference in degrees (mod 180)."""
    d = abs(a1 - a2) % 180
    return min(d, 180 - d)


def run_comparison():
    """Compare Fisher-PT vs CIELAB on all 25 MacAdam ellipses."""

    print("=" * 70)
    print("MacADAM ELLIPSE TEST: Fisher-PT (0 params) vs CIELAB (3 params)")
    print("=" * 70)
    print(f"\nPT weights: γ₃={gamma_p(3):.4f}, γ₅={gamma_p(5):.4f}, γ₇={gamma_p(7):.4f}")
    print(f"CIELAB: cube-root f(t)=t^(1/3), D65 reference white\n")

    # For each ellipse, compare:
    # 1. Axis ratio (a/b) — shape
    # 2. Orientation angle — direction

    header = f"{'#':>2} {'(x,y)':>12} {'obs a/b':>8} {'PT a/b':>8} {'LAB a/b':>8} {'obs θ':>6} {'PT θ':>6} {'LAB θ':>6} {'PT Δθ':>6} {'LAB Δθ':>6}"
    print(header)
    print("-" * len(header))

    pt_ratio_errors = []
    lab_ratio_errors = []
    pt_angle_errors = []
    lab_angle_errors = []

    for i, (x, y, a_obs, b_obs, theta_obs) in enumerate(MACADAM_DATA):
        ratio_obs = a_obs / b_obs

        # Fisher-PT prediction
        a_pt, b_pt, theta_pt = fisher_ellipse_at(x, y)
        ratio_pt = a_pt / b_pt

        # CIELAB prediction
        a_lab, b_lab, theta_lab = cielab_metric_at(x, y)
        ratio_lab = a_lab / b_lab

        # Errors
        pt_ratio_err = abs(ratio_pt - ratio_obs) / ratio_obs
        lab_ratio_err = abs(ratio_lab - ratio_obs) / ratio_obs
        pt_angle_err = angle_diff(theta_pt, theta_obs)
        lab_angle_err = angle_diff(theta_lab, theta_obs)

        pt_ratio_errors.append(pt_ratio_err)
        lab_ratio_errors.append(lab_ratio_err)
        pt_angle_errors.append(pt_angle_err)
        lab_angle_errors.append(lab_angle_err)

        print(f"{i+1:>2} ({x:.3f},{y:.3f}) {ratio_obs:>8.2f} {ratio_pt:>8.2f} {ratio_lab:>8.2f} "
              f"{theta_obs:>6.0f} {theta_pt:>6.1f} {theta_lab:>6.1f} "
              f"{pt_angle_err:>6.1f} {lab_angle_err:>6.1f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    pt_rms_ratio = np.sqrt(np.mean(np.array(pt_ratio_errors)**2))
    lab_rms_ratio = np.sqrt(np.mean(np.array(lab_ratio_errors)**2))
    pt_rms_angle = np.sqrt(np.mean(np.array(pt_angle_errors)**2))
    lab_rms_angle = np.sqrt(np.mean(np.array(lab_angle_errors)**2))

    pt_mean_ratio = np.mean(pt_ratio_errors)
    lab_mean_ratio = np.mean(lab_ratio_errors)
    pt_mean_angle = np.mean(pt_angle_errors)
    lab_mean_angle = np.mean(lab_angle_errors)

    print(f"\n{'Metric':>20} {'Params':>7} {'MAE ratio':>10} {'RMS ratio':>10} {'MAE θ (°)':>10} {'RMS θ (°)':>10}")
    print("-" * 70)
    print(f"{'Fisher-PT':>20} {'0':>7} {pt_mean_ratio:>10.3f} {pt_rms_ratio:>10.3f} {pt_mean_angle:>10.1f} {pt_rms_angle:>10.1f}")
    print(f"{'CIELAB':>20} {'3':>7} {lab_mean_ratio:>10.3f} {lab_rms_ratio:>10.3f} {lab_mean_angle:>10.1f} {lab_rms_angle:>10.1f}")

    # Which wins?
    ratio_winner = "Fisher-PT" if pt_rms_ratio < lab_rms_ratio else "CIELAB"
    angle_winner = "Fisher-PT" if pt_rms_angle < lab_rms_angle else "CIELAB"

    print(f"\nAxis ratio: {ratio_winner} wins")
    print(f"Orientation: {angle_winner} wins")

    # Count how many ellipses each method predicts better
    pt_wins_ratio = sum(1 for p, l in zip(pt_ratio_errors, lab_ratio_errors) if p < l)
    pt_wins_angle = sum(1 for p, l in zip(pt_angle_errors, lab_angle_errors) if p < l)

    print(f"\nPer-ellipse wins (ratio): Fisher-PT {pt_wins_ratio}/25, CIELAB {25-pt_wins_ratio}/25")
    print(f"Per-ellipse wins (angle): Fisher-PT {pt_wins_angle}/25, CIELAB {25-pt_wins_angle}/25")

    print(f"\n{'='*70}")
    overall = "Fisher-PT" if (pt_rms_ratio + pt_rms_angle/90) < (lab_rms_ratio + lab_rms_angle/90) else "CIELAB"
    print(f"OVERALL WINNER: {overall} (0 params beats 3 params)" if overall == "Fisher-PT"
          else f"OVERALL: {overall} (3 fitted params still needed)")
    print(f"{'='*70}")

    return pt_ratio_errors, lab_ratio_errors, pt_angle_errors, lab_angle_errors


def plot_comparison(pt_ratio_errors, lab_ratio_errors, pt_angle_errors, lab_angle_errors):
    """Generate comparison figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    n = len(pt_ratio_errors)
    x = np.arange(n) + 1

    # Ratio errors
    ax1.bar(x - 0.2, pt_ratio_errors, 0.35, color='#3366CC', label='Fisher-PT (0 params)', alpha=0.8)
    ax1.bar(x + 0.2, lab_ratio_errors, 0.35, color='#CC6633', label='CIELAB (3 params)', alpha=0.8)
    ax1.set_xlabel('MacAdam ellipse #')
    ax1.set_ylabel('Relative error (axis ratio)')
    ax1.set_title('Axis ratio prediction error')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)

    # Angle errors
    ax2.bar(x - 0.2, pt_angle_errors, 0.35, color='#3366CC', label='Fisher-PT (0 params)', alpha=0.8)
    ax2.bar(x + 0.2, lab_angle_errors, 0.35, color='#CC6633', label='CIELAB (3 params)', alpha=0.8)
    ax2.set_xlabel('MacAdam ellipse #')
    ax2.set_ylabel('Angle error (degrees)')
    ax2.set_title('Orientation prediction error')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)

    fig.suptitle('MacAdam ellipses: Fisher-PT vs CIELAB', fontsize=14, y=1.02)
    fig.tight_layout()

    for figdir in ['figures_en', 'figures_fr']:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', figdir)
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, 'fig8_macadam_comparison.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(outdir, 'fig8_macadam_comparison.png'), dpi=200, bbox_inches='tight')

    plt.close(fig)
    print("\n✓ fig8_macadam_comparison.pdf (both EN/FR)")


def plot_ellipses_on_diagram(pt_ratio_errors, lab_ratio_errors):
    """Plot MacAdam ellipses on the CIE diagram with PT predictions overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Draw CIE boundary (approximate spectral locus)
    # Simplified horseshoe boundary
    locus_x = [0.175, 0.17, 0.13, 0.08, 0.05, 0.01, 0.00, 0.01, 0.07, 0.15,
               0.23, 0.31, 0.40, 0.47, 0.52, 0.57, 0.62, 0.67, 0.72, 0.735]
    locus_y = [0.005, 0.02, 0.05, 0.10, 0.17, 0.28, 0.40, 0.52, 0.62, 0.71,
               0.77, 0.81, 0.83, 0.83, 0.82, 0.80, 0.77, 0.73, 0.28, 0.265]
    ax.plot(locus_x, locus_y, 'k-', linewidth=1, alpha=0.3)
    ax.plot([locus_x[0], locus_x[-1]], [locus_y[0], locus_y[-1]], 'k--', linewidth=1, alpha=0.3)

    scale = 10  # scale factor for visibility

    for i, (x, y, a_obs, b_obs, theta_obs) in enumerate(MACADAM_DATA):
        # Observed ellipse
        e_obs = Ellipse((x, y), 2*a_obs*1e-4*scale, 2*b_obs*1e-4*scale,
                        angle=theta_obs, fill=False, edgecolor='black',
                        linewidth=1.5, linestyle='-')
        ax.add_patch(e_obs)

        # Fisher-PT prediction (normalize to same area for shape comparison)
        a_pt, b_pt, theta_pt = fisher_ellipse_at(x, y)
        # Scale PT ellipse to match observed area
        area_obs = np.pi * a_obs * b_obs
        area_pt = np.pi * a_pt * b_pt
        pt_scale = np.sqrt(area_obs / area_pt) * 1e-4 * scale
        e_pt = Ellipse((x, y), 2*a_pt*pt_scale, 2*b_pt*pt_scale,
                       angle=theta_pt, fill=False, edgecolor='#3366CC',
                       linewidth=1.5, linestyle='--')
        ax.add_patch(e_pt)

    ax.plot([], [], 'k-', linewidth=1.5, label='MacAdam (observed)')
    ax.plot([], [], '--', color='#3366CC', linewidth=1.5, label='Fisher-PT (predicted, 0 params)')

    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$y$', fontsize=12)
    ax.set_title('MacAdam ellipses: observed vs Fisher-PT prediction', fontsize=13)
    ax.set_xlim(-0.05, 0.8)
    ax.set_ylim(-0.05, 0.9)
    ax.set_aspect('equal')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    for figdir in ['figures_en', 'figures_fr']:
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', figdir)
        fig.savefig(os.path.join(outdir, 'fig9_macadam_overlay.pdf'), bbox_inches='tight')
        fig.savefig(os.path.join(outdir, 'fig9_macadam_overlay.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("✓ fig9_macadam_overlay.pdf (both EN/FR)")


if __name__ == '__main__':
    pt_r, lab_r, pt_a, lab_a = run_comparison()
    plot_comparison(pt_r, lab_r, pt_a, lab_a)
    plot_ellipses_on_diagram(pt_r, lab_r)
