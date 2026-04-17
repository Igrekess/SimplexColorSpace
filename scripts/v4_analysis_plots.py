#!/usr/bin/env python3
"""
V4 Neural Analysis — Visualization & PT Hypothesis Tests
=========================================================

Creates publication-quality figures for the PT_COLOR article showing:
1. V4 BOLD response as a function of DKL hue angle
2. Correlation with Fisher distance on SCS simplex
3. Contrast response function (CRF) by hue
4. γ_p ratio test: does V4 show the 0.808:0.696:0.595 pattern?
5. Polar plot: V4 tuning curve vs SCS prediction
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

# PT constants
GAMMAS = np.array([0.80761, 0.69632, 0.59547])
GAMMA_RATIOS = GAMMAS / GAMMAS[0]  # 1.000, 0.862, 0.737

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "datasets", "v4_bold_response.csv")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

DKL_LABELS = {
    0: 'L-M+\n(red)',
    45: 'Day+\n(orange)',
    90: 'S+\n(blue)',
    135: 'ADay+\n(green)',
    180: 'L-M-\n(cyan)',
    225: 'Day-\n(blue)',
    270: 'S-\n(violet)',
    315: 'ADay-\n(mag.)',
}


def load_data():
    """Load V4 BOLD response data."""
    data = []
    with open(DATA_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'hue': int(row['hue_index']),
                'angle': int(row['hue_dkl_deg']),
                'contrast': float(row['contrast']),
                'bold': float(row['bold_v4_mean']),
                'bold_sem': float(row['bold_v4_std']),
                'n': int(row['n_runs']),
                'pi3': float(row['scs_pi3']),
                'pi5': float(row['scs_pi5']),
                'pi7': float(row['scs_pi7']),
                'fisher': float(row['fisher_distance']),
            })
    return data


def fig1_bold_vs_hue(data):
    """
    Figure 1: V4 BOLD response by DKL hue angle.
    Polar plot showing the chromatic tuning of V4.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                              subplot_kw={'projection': 'polar'})

    contrasts = [0.30, 0.95]
    titles = ['Low contrast (30%)', 'High contrast (95%)']

    for ax, ct, title in zip(axes, contrasts, titles):
        chrom = [d for d in data if d['hue'] <= 8 and d['contrast'] == ct]
        if not chrom:
            continue

        angles = np.array([np.radians(d['angle']) for d in chrom])
        bold = np.array([d['bold'] for d in chrom])
        sems = np.array([d['bold_sem'] for d in chrom])

        # Close the polar plot
        angles = np.append(angles, angles[0])
        bold = np.append(bold, bold[0])
        sems = np.append(sems, sems[0])

        # Plot with error band
        ax.fill_between(angles, bold - sems, bold + sems, alpha=0.2, color='steelblue')
        ax.plot(angles, bold, 'o-', color='steelblue', markersize=6, linewidth=2)

        # Add zero line
        ax.plot(np.linspace(0, 2*np.pi, 100), np.zeros(100), 'k--', alpha=0.3, linewidth=0.5)

        ax.set_title(title, pad=20, fontsize=13)
        ax.set_thetagrids([0, 45, 90, 135, 180, 225, 270, 315],
                          labels=['L-M+', '45°', 'S+', '135°',
                                  'L-M-', '225°', 'S-', '315°'])

    fig.suptitle('V4 BOLD Response by DKL Hue Angle (macaque fMRI)', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def fig2_bold_vs_fisher(data):
    """
    Figure 2: Scatter plot — BOLD vs Fisher distance from achromatic.
    The PT hypothesis test.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all conditions (color by contrast)
    ax = axes[0]
    chrom = [d for d in data if d['hue'] <= 8]
    colors_map = {0.10: '#99d8c9', 0.30: '#41ae76', 0.50: '#006d2c', 0.95: '#00441b'}

    for ct in sorted(set(d['contrast'] for d in chrom)):
        subset = [d for d in chrom if d['contrast'] == ct]
        x = [d['fisher'] for d in subset]
        y = [d['bold'] for d in subset]
        ax.scatter(x, y, c=colors_map.get(ct, 'gray'), label=f'{ct:.0%}',
                   s=60, alpha=0.8, edgecolors='white', linewidth=0.5)

    # Regression line
    all_x = np.array([d['fisher'] for d in chrom])
    all_y = np.array([d['bold'] for d in chrom])
    if len(all_x) > 2 and all_x.std() > 0:
        z = np.polyfit(all_x, all_y, 1)
        xline = np.linspace(all_x.min(), all_x.max(), 50)
        ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5, linewidth=1.5)
        r = np.corrcoef(all_x, all_y)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes,
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Fisher distance from achromatic (SCS)', fontsize=11)
    ax.set_ylabel('BOLD V4 (% signal change)', fontsize=11)
    ax.set_title('All conditions', fontsize=12)
    ax.legend(title='Contrast', fontsize=9)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # Right: per-hue (averaged over contrasts)
    ax = axes[1]
    hue_data = defaultdict(lambda: {'bold': [], 'fisher': []})
    for d in chrom:
        hue_data[d['hue']]['bold'].append(d['bold'])
        hue_data[d['hue']]['fisher'].append(d['fisher'])

    hue_bold = []
    hue_fisher = []
    hue_labels = []
    for h in range(1, 9):
        if h in hue_data:
            hue_bold.append(np.mean(hue_data[h]['bold']))
            hue_fisher.append(np.mean(hue_data[h]['fisher']))
            angle = (h - 1) * 45
            hue_labels.append(f'{angle}°')

    if len(hue_bold) >= 3:
        hue_bold = np.array(hue_bold)
        hue_fisher = np.array(hue_fisher)

        ax.scatter(hue_fisher, hue_bold, s=100, c='steelblue',
                   edgecolors='navy', linewidth=1.5, zorder=5)
        for i, label in enumerate(hue_labels):
            ax.annotate(label, (hue_fisher[i], hue_bold[i]),
                       textcoords="offset points", xytext=(8, 5), fontsize=9)

        # Regression
        z = np.polyfit(hue_fisher, hue_bold, 1)
        xline = np.linspace(hue_fisher.min(), hue_fisher.max(), 50)
        ax.plot(xline, np.polyval(z, xline), 'r--', alpha=0.5, linewidth=1.5)
        r = np.corrcoef(hue_fisher, hue_bold)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.3f} (n=8 hues)', transform=ax.transAxes,
                fontsize=12, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Mean Fisher distance (SCS)', fontsize=11)
    ax.set_ylabel('Mean BOLD V4 (% signal change)', fontsize=11)
    ax.set_title('Per-hue average', fontsize=12)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    fig.suptitle('Test: BOLD_V4 ∝ Fisher(γ_p) ?', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def fig3_contrast_response(data):
    """
    Figure 3: Contrast response function by DKL axis.
    Groups: L-M axis (hues 1,5), S axis (hues 3,7), intermediate (2,4,6,8), luminance (9).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = {
        'L-M axis (γ₃)': [1, 5],
        'S axis (γ₇)': [3, 7],
        'Intermediate': [2, 4, 6, 8],
        'Luminance (p=2)': [9],
    }
    colors = ['#e41a1c', '#377eb8', '#999999', '#ff7f00']
    markers = ['o', 's', 'D', '^']

    contrasts = sorted(set(d['contrast'] for d in data))

    for (name, hues), color, marker in zip(groups.items(), colors, markers):
        means = []
        sems = []
        for ct in contrasts:
            vals = [d['bold'] for d in data if d['hue'] in hues and d['contrast'] == ct]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / max(np.sqrt(len(vals)), 1))
            else:
                means.append(np.nan)
                sems.append(0)

        means = np.array(means)
        sems = np.array(sems)
        ax.errorbar(contrasts, means, yerr=sems, fmt=f'{marker}-',
                     color=color, label=name, markersize=8, linewidth=2,
                     capsize=4, capthick=1.5)

    ax.set_xlabel('Contrast (fraction of max gamut)', fontsize=12)
    ax.set_ylabel('BOLD V4 (% signal change)', fontsize=12)
    ax.set_title('V4 Contrast Response Function by DKL Axis', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xticks(contrasts)
    ax.set_xticklabels([f'{c:.0%}' for c in contrasts])

    # Add PT annotation
    ax.text(0.98, 0.02,
            'PT predicts: L-M (γ₃=0.808) > S (γ₇=0.595)\n'
            'i.e., L-M axis should produce stronger V4 response',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def fig4_gamma_ratio_test(data):
    """
    Figure 4: The γ_p ratio test.
    Does V4 response at L-M / S axes follow the PT ratio 0.808/0.595 = 1.358?
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    contrasts = sorted(set(d['contrast'] for d in data))

    # L-M axis: hues 1 (0°) and 5 (180°)
    # S axis: hues 3 (90°) and 7 (270°)
    lm_response = []
    s_response = []

    for ct in contrasts:
        lm_vals = [abs(d['bold']) for d in data if d['hue'] in [1, 5] and d['contrast'] == ct]
        s_vals = [abs(d['bold']) for d in data if d['hue'] in [3, 7] and d['contrast'] == ct]
        if lm_vals and s_vals:
            lm_response.append(np.mean(lm_vals))
            s_response.append(np.mean(s_vals))

    if len(lm_response) > 0 and len(s_response) > 0:
        lm_response = np.array(lm_response)
        s_response = np.array(s_response)

        # Observed ratio
        safe_s = np.maximum(s_response, 0.001)
        observed_ratio = lm_response / safe_s

        # PT predicted ratio
        pt_ratio = GAMMAS[0] / GAMMAS[2]  # γ₃/γ₇ = 1.356

        ax.bar(range(len(contrasts)), observed_ratio, alpha=0.7,
               color='steelblue', label='Observed |BOLD_LM| / |BOLD_S|')
        ax.axhline(pt_ratio, color='red', linestyle='--', linewidth=2,
                   label=f'PT predicted: γ₃/γ₇ = {pt_ratio:.3f}')
        ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xticks(range(len(contrasts)))
        ax.set_xticklabels([f'{c:.0%}' for c in contrasts])
        ax.set_xlabel('Contrast', fontsize=12)
        ax.set_ylabel('|BOLD(L-M)| / |BOLD(S)| ratio', fontsize=12)
        ax.set_title('γ_p Ratio Test: L-M vs S axis V4 response', fontsize=13)
        ax.legend(fontsize=11)

        # Summary
        mean_ratio = np.mean(observed_ratio)
        ax.text(0.98, 0.98,
                f'Mean observed: {mean_ratio:.3f}\n'
                f'PT predicted: {pt_ratio:.3f}\n'
                f'Deviation: {abs(mean_ratio - pt_ratio)/pt_ratio*100:.1f}%',
                transform=ax.transAxes, fontsize=10, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    return fig


def fig5_simplex_map(data):
    """
    Figure 5: The 8 DKL hues mapped onto the SCS simplex Δ².
    Shows where each stimulus falls on the γ-weighted probability simplex.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Simplex vertices in 2D (equilateral triangle)
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    labels = [f'L (γ₃={GAMMAS[0]:.3f})',
              f'M (γ₅={GAMMAS[1]:.3f})',
              f'S (γ₇={GAMMAS[2]:.3f})']

    # Draw simplex
    triangle = plt.Polygon(v, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(triangle)

    for i, (vx, label) in enumerate(zip(v, labels)):
        offset = [(-0.12, -0.05), (0.05, -0.05), (0, 0.05)][i]
        ax.annotate(label, vx, xytext=(vx[0]+offset[0], vx[1]+offset[1]),
                   fontsize=10, fontweight='bold')

    # Plot achromatic center
    center = v.mean(axis=0)
    ax.plot(*center, 'k+', markersize=15, markeredgewidth=2)
    ax.annotate('achromatic', center, xytext=(center[0]+0.05, center[1]-0.04),
               fontsize=9, color='gray')

    # Plot each hue at max contrast
    chrom_max = [d for d in data if d['hue'] <= 8 and d['contrast'] == 0.95]
    if not chrom_max:
        chrom_max = [d for d in data if d['hue'] <= 8 and d['contrast'] == 0.50]

    cmap = plt.cm.hsv
    for d in chrom_max:
        pi = np.array([d['pi3'], d['pi5'], d['pi7']])
        # Barycentric to Cartesian
        xy = pi[0] * v[0] + pi[1] * v[1] + pi[2] * v[2]
        color = cmap(d['angle'] / 360)
        size = max(abs(d['bold']) * 200, 30)
        ax.scatter(*xy, s=size, c=[color], edgecolors='black', linewidth=1,
                   zorder=5, alpha=0.8)
        ax.annotate(f"{d['angle']}°", xy, xytext=(5, 5),
                   textcoords='offset points', fontsize=8)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.set_title('DKL hues on SCS simplex Δ² (size ∝ |BOLD|)', fontsize=13)
    ax.axis('off')

    plt.tight_layout()
    return fig


def main():
    """Generate all figures."""
    data = load_data()
    print(f"Loaded {len(data)} data points from {DATA_PATH}")

    os.makedirs(FIG_DIR, exist_ok=True)

    figs = [
        ('fig_v4_bold_polar.pdf', fig1_bold_vs_hue),
        ('fig_v4_bold_vs_fisher.pdf', fig2_bold_vs_fisher),
        ('fig_v4_contrast_response.pdf', fig3_contrast_response),
        ('fig_v4_gamma_ratio_test.pdf', fig4_gamma_ratio_test),
        ('fig_v4_simplex_map.pdf', fig5_simplex_map),
    ]

    for fname, func in figs:
        print(f"  Generating {fname}...")
        fig = func(data)
        path = os.path.join(FIG_DIR, fname)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"    Saved: {path}")

    print(f"\n{len(figs)} figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
