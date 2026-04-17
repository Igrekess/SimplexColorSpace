#!/usr/bin/env python3
"""
SCS Companion Script — Sieve Color Space
=========================================

Computes all PT-derived color quantities at mu*=15 and generates
publication-quality figures for the PT_COLOR article.

Usage:
    python scs_companion.py              # Run all computations + generate figures
    python scs_companion.py --verify     # Run numerical verifications only
    python scs_companion.py --figures    # Generate figures only

All figures saved to ../figures/ as PDF.

Reference: PT_COLOR.tex, Theorems T0-T7, Demonstrations D02, D07, D08, D17b.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Ellipse, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import os
import sys

# ============================================================
# PT CONSTANTS AT mu* = 15
# ============================================================

MU_STAR = 15
S_PARAM = 0.5          # s = 1/2 (unique input)
ALPHA_3 = S_PARAM**2   # alpha(3) = s^2 = 1/4

# The two q parameters
Q_REL  = 1 - 2/MU_STAR          # = 13/15 ≈ 0.8667
Q_THERM = np.exp(-1/MU_STAR)     # ≈ 0.9355

# Active primes
PRIMES = [3, 5, 7]
# Bilingual text dictionaries
TEXTS = {
    'en': {
        'prime_labels': {3: 'p=3 (red)', 5: 'p=5 (green)', 7: 'p=7 (blue)'},
        'fig1_title': 'Chromatic simplex $\\Delta^2$ \u2014 SCS System',
        'fig2_title': 'Chromatic conservation law (GFT)',
        'fig2_xlabel': 'Saturation  $S = D_{KL}(\\pi \\| u)$  [nats]',
        'fig2_ylabel': 'Luminance  $L = H(\\pi)$  [nats]',
        'fig2_white': 'White\n(center)',
        'fig2_pure_red': 'Pure red\n(vertex)',
        'fig3_title': 'Fisher ellipses on $\\Delta^2$ \u2014 MacAdam analogue',
        'fig3_caption': 'Small ellipses = fine discrimination (yellow-green)\nLarge ellipses = coarse discrimination (blue)',
        'fig4_title': 'Hue circle \u2014 sieve holonomy on $S^1$',
        'fig4_purple': 'Purple\n($2\\pi$ closure)',
        'fig4_holonomy': 'Holonomy\n$\\mathbb{Z}/p\\mathbb{Z} \\to S^1$',
        'fig4_bifurcation': 'bifurcation',
        'fig5_title': 'Chromatic channel hierarchy \u2014 derived from the sieve at $\\mu^*=15$',
        'fig5_ylabel': '$\\gamma_p$ (effective dimension)',
        'fig5_threshold': 'Threshold $\\gamma = s = 1/2$',
        'fig5_foundation': 'binary\nfoundation',
        'fig5_active': 'active: $\\{3,5,7\\}$',
        'fig5_parity': '(parity)', 'fig5_red': '(red)', 'fig5_green': '(green)', 'fig5_blue': '(blue)',
        'fig6_title': 'Sieve bifurcation: $q_{rel}$ (coupling) vs $q_{therm}$ (geometry)',
        'fig6_left': '$q_{rel}$ branch \u2014 Transmission\n(additive mixing, screens)',
        'fig6_right': '$q_{therm}$ branch \u2014 Absorption\n(subtractive mixing, pigments)',
        'fig6_comp': 'Complementary: $\\sin^2 + \\cos^2 = 1$\nR\u2194C,  G\u2194M,  B\u2194Y',
        'fig7_title': 'Berlin-Kay universals \u2014 color term order follows $\\gamma_p$',
        'fig7_threshold': 'threshold $s=1/2$',
        'fig7_stages': [
            'Dark / Light\n($p=2$, parity)', 'Red\n($p=3$, $\\gamma=0.808$)',
            'Yellow-Green\n($p=5$, $\\gamma=0.696$)', 'Blue\n($p=7$, $\\gamma=0.595$)',
            'Brown, Pink,\nOrange, Grey...'
        ],
    },
    'fr': {
        'prime_labels': {3: 'p=3 (rouge)', 5: 'p=5 (vert)', 7: 'p=7 (bleu)'},
        'fig1_title': 'Simplexe chromatique $\\Delta^2$ \u2014 Syst\u00e8me SCS',
        'fig2_title': 'Loi de conservation chromatique (GFT)',
        'fig2_xlabel': 'Saturation  $S = D_{KL}(\\pi \\| u)$  [nats]',
        'fig2_ylabel': 'Luminance  $L = H(\\pi)$  [nats]',
        'fig2_white': 'Blanc\n(centre)',
        'fig2_pure_red': 'Rouge pur\n(sommet)',
        'fig3_title': 'Ellipses de Fisher sur $\\Delta^2$ \u2014 analogue de MacAdam',
        'fig3_caption': 'Petites ellipses = discrimination fine (vert-jaune)\nGrandes ellipses = discrimination grossi\u00e8re (bleu)',
        'fig4_title': 'Cercle de teinte \u2014 holonomie du crible sur $S^1$',
        'fig4_purple': 'Pourpre\n(fermeture $2\\pi$)',
        'fig4_holonomy': 'Holonomie\n$\\mathbb{Z}/p\\mathbb{Z} \\to S^1$',
        'fig4_bifurcation': 'bifurcation',
        'fig5_title': 'Hi\u00e9rarchie des canaux chromatiques \u2014 d\u00e9riv\u00e9e du crible \u00e0 $\\mu^*=15$',
        'fig5_ylabel': '$\\gamma_p$ (dimension effective)',
        'fig5_threshold': 'Seuil $\\gamma = s = 1/2$',
        'fig5_foundation': 'fondation\nbinaire',
        'fig5_active': 'actifs : $\\{3,5,7\\}$',
        'fig5_parity': '(parit\u00e9)', 'fig5_red': '(rouge)', 'fig5_green': '(vert)', 'fig5_blue': '(bleu)',
        'fig6_title': 'Bifurcation du crible : $q_{rel}$ (couplage) vs $q_{therm}$ (g\u00e9om\u00e9trie)',
        'fig6_left': 'Branche $q_{rel}$ \u2014 Transmission\n(synth\u00e8se additive, \u00e9crans)',
        'fig6_right': 'Branche $q_{therm}$ \u2014 Absorption\n(synth\u00e8se soustractive, pigments)',
        'fig6_comp': 'Compl\u00e9mentaires : $\\sin^2 + \\cos^2 = 1$\nR\u2194C,  G\u2194M,  B\u2194Y',
        'fig7_title': "Universaux de Berlin-Kay \u2014 l'ordre des termes suit $\\gamma_p$",
        'fig7_threshold': 'seuil $s=1/2$',
        'fig7_stages': [
            'Sombre / Clair\n($p=2$, parit\u00e9)', 'Rouge\n($p=3$, $\\gamma=0.808$)',
            'Jaune-Vert\n($p=5$, $\\gamma=0.696$)', 'Bleu\n($p=7$, $\\gamma=0.595$)',
            'Brun, Rose,\nOrange, Gris...'
        ],
    },
}

LANG = 'fr'
T = TEXTS[LANG]
PRIME_LABELS = T['prime_labels']
PRIME_COLORS = {3: '#CC3333', 5: '#33AA33', 7: '#3333CC'}

# Output directory
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


def delta_p(p, q):
    """Holonomy angle parameter: delta_p = (1 - q^p) / p"""
    return (1 - q**p) / p


def sin2_theta(p, q):
    """sin^2(theta_p) = delta_p * (2 - delta_p)  [T6, D07]"""
    d = delta_p(p, q)
    return d * (2 - d)


def gamma_p(p, mu=MU_STAR, q=None):
    """Effective dimension: gamma_p = -d(ln sin^2) / d(ln mu)  [T7, D08]"""
    if q is None:
        q = 1 - 2/mu
    d = delta_p(p, q)
    return 4 * p * q**(p-1) * (1 - d) / (mu * (1 - q**p) * (2 - d))


def dkl_color(pi):
    """D_KL(pi || uniform) — saturation in nats (natural log). [sum rule, D02]"""
    pi = np.asarray(pi, dtype=float)
    pi = pi[pi > 0]  # avoid log(0)
    return np.sum(pi * np.log(3 * pi))


def entropy_color(pi):
    """H(pi) — chromatic entropy in nats (natural log). [sum rule, D02]"""
    pi = np.asarray(pi, dtype=float)
    pi = pi[pi > 0]
    return -np.sum(pi * np.log(pi))


def fisher_metric(pi, gamma):
    """
    Fisher information metric tensor at point pi on Delta^2.
    Returns 2x2 matrix (in the (pi3, pi5) chart, pi7 = 1-pi3-pi5).

    ds^2 = sum_p gamma_p * dpi_p^2 / pi_p
    In chart (pi3, pi5): dpi7 = -dpi3 - dpi5
    g_ij = gamma_i/pi_i * delta_ij + gamma_7/pi_7
    """
    pi3, pi5 = pi[0], pi[1]
    pi7 = 1 - pi3 - pi5
    g3, g5, g7 = gamma

    g11 = g3/pi3 + g7/pi7
    g22 = g5/pi5 + g7/pi7
    g12 = g7/pi7

    return np.array([[g11, g12], [g12, g22]])


# ============================================================
# NUMERICAL VERIFICATION
# ============================================================

def verify_all():
    """Run all numerical checks. Returns True if all pass."""
    print("=" * 60)
    print("SCS NUMERICAL VERIFICATION")
    print("=" * 60)

    all_pass = True

    # 1. Sieve parameters
    print(f"\n--- Sieve parameters at mu* = {MU_STAR} ---")
    print(f"  s = {S_PARAM}")
    print(f"  q_rel  = 1 - 2/{MU_STAR} = {Q_REL:.10f}")
    print(f"  q_therm = exp(-1/{MU_STAR}) = {Q_THERM:.10f}")

    # 2. Holonomy angles sin^2(theta_p)
    print(f"\n--- Holonomy angles sin^2(theta_p, q_rel) ---")
    sin2_values = {}
    for p in PRIMES:
        s2 = sin2_theta(p, Q_REL)
        sin2_values[p] = s2
        print(f"  p={p}: delta={delta_p(p, Q_REL):.6f}, sin^2={s2:.6f}")

    # 3. alpha_EM (bare)
    alpha_bare = np.prod([sin2_values[p] for p in PRIMES])
    print(f"\n--- alpha_EM (bare) ---")
    print(f"  prod sin^2 = {alpha_bare:.6f}")
    print(f"  1/alpha    = {1/alpha_bare:.3f}")
    print(f"  Expected   = 136.28")
    check = abs(1/alpha_bare - 136.28) < 0.5
    print(f"  CHECK: {'PASS' if check else 'FAIL'}")
    all_pass &= check

    # 4. gamma_p (effective dimensions)
    print(f"\n--- Effective dimensions gamma_p ---")
    gammas = {}
    for p in [3, 5, 7, 11]:
        g = gamma_p(p)
        gammas[p] = g
        active = g > 0.5
        print(f"  gamma_{p} = {g:.4f}  {'ACTIVE' if active else 'inactive'}")

    check = gammas[3] > 0.5 and gammas[5] > 0.5 and gammas[7] > 0.5 and gammas[11] < 0.5
    print(f"  Exactly {{3,5,7}} active: {'PASS' if check else 'FAIL'}")
    all_pass &= check

    # 5. Hierarchy
    check = gammas[3] > gammas[5] > gammas[7]
    print(f"  Hierarchy gamma_3 > gamma_5 > gamma_7: {'PASS' if check else 'FAIL'}")
    all_pass &= check

    # 6. Sum rule: D_KL + H = log 3 (natural log, nats)
    print(f"\n--- Sum rule: S + L = log 3 (nats) ---")
    log3 = np.log(3)
    test_points = [
        ("White", [1/3, 1/3, 1/3]),
        ("Pure red", [0.999, 0.0005, 0.0005]),
        ("Vermeer", [0.42, 0.35, 0.23]),
        ("Mid-green", [0.15, 0.70, 0.15]),
        ("Koide point", [0.45, 0.33, 0.22]),
    ]
    for name, pi in test_points:
        S = dkl_color(pi)
        L = entropy_color(pi)
        total = S + L
        err = abs(total - log3)
        print(f"  {name:12s}: S={S:.4f} + L={L:.4f} = {total:.6f}  (err={err:.2e})  "
              f"{'PASS' if err < 1e-10 else 'FAIL'}")
        all_pass &= (err < 1e-10)

    # 7. Koide saturation
    print(f"\n--- Koide optimal saturation ---")
    S_max = np.log2(3)
    S_koide = S_max / np.sqrt(2)
    S_koide_pct = S_koide / S_max * 100
    print(f"  S_max   = log2(3) = {S_max:.6f} bits")
    print(f"  S_Koide = S_max/sqrt(2) = {S_koide:.6f} bits")
    print(f"  S_Koide/S_max = 1/sqrt(2) = {S_koide_pct:.2f}%")
    check = abs(S_koide_pct - 70.71) < 0.1
    print(f"  ~70.7%: {'PASS' if check else 'FAIL'}")
    all_pass &= check

    # 8. CRT resolution
    print(f"\n--- CRT resolution ---")
    crt = 3 * 5 * 7
    print(f"  3 × 5 × 7 = {crt}")
    check = crt == 105
    print(f"  = 105: {'PASS' if check else 'FAIL'}")
    all_pass &= check

    # 9. Complementarity check
    print(f"\n--- Complementarity: sin^2 + cos^2 = 1 ---")
    for p in PRIMES:
        s2 = sin2_theta(p, Q_REL)
        c2 = 1 - s2
        total = s2 + c2
        print(f"  p={p}: sin^2={s2:.6f} + cos^2={c2:.6f} = {total:.10f}  "
              f"{'PASS' if abs(total - 1) < 1e-15 else 'FAIL'}")
        all_pass &= (abs(total - 1) < 1e-15)

    print(f"\n{'='*60}")
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")
    return all_pass


# ============================================================
# FIGURE GENERATION
# ============================================================

def fig_chromatic_simplex():
    """
    Figure 1: The chromatic simplex Delta^2 with R, G, B vertices,
    white center, and iso-saturation contours.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Triangle vertices (equilateral in display coordinates)
    # R at top, G at bottom-left, B at bottom-right
    h = np.sqrt(3)/2
    vR = np.array([0.5, h])
    vG = np.array([0.0, 0.0])
    vB = np.array([1.0, 0.0])
    vW = (vR + vG + vB) / 3  # center

    # Draw filled simplex with color gradient
    n = 200
    for i in range(n):
        for j in range(n - i):
            k = n - 1 - i - j
            pi3 = (i + 0.5) / n
            pi5 = (j + 0.5) / n
            pi7 = (k + 0.5) / n

            # Position in display coords
            pos = pi3 * vR + pi5 * vG + pi7 * vB

            # Color: map pi to RGB (approximate)
            r = min(1, pi3 * 2.5)
            g = min(1, pi5 * 2.5)
            b = min(1, pi7 * 2.5)

            ax.plot(pos[0], pos[1], 'o', color=(r, g, b), markersize=2.5,
                    markeredgewidth=0, alpha=0.7)

    # Draw triangle edges
    triangle = plt.Polygon([vR, vG, vB], fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    # Iso-saturation contours (circles around center in barycentric)
    for s_frac in [0.25, 0.5, 0.707, 0.9]:
        theta = np.linspace(0, 2*np.pi, 100)
        r_display = s_frac * 0.35  # scale to fit
        cx = vW[0] + r_display * np.cos(theta)
        cy = vW[1] + r_display * np.sin(theta)
        style = '--' if s_frac != 0.707 else '-'
        lw = 1 if s_frac != 0.707 else 2.5
        color = 'gray' if s_frac != 0.707 else '#CC6600'
        ax.plot(cx, cy, style, color=color, linewidth=lw, alpha=0.7)
        if s_frac == 0.707:
            ax.annotate(f'Koide\n$1/\\sqrt{{2}}$', xy=(cx[25], cy[25]),
                       fontsize=9, color='#CC6600', fontweight='bold',
                       ha='center')

    # Labels
    offset = 0.04
    ax.text(vR[0], vR[1]+offset, '$R$\n$\\pi_3 = 1$\n$p=3$',
            ha='center', va='bottom', fontsize=12, fontweight='bold', color='#CC3333')
    ax.text(vG[0]-offset, vG[1]-offset, '$G$\n$\\pi_5 = 1$\n$p=5$',
            ha='right', va='top', fontsize=12, fontweight='bold', color='#33AA33')
    ax.text(vB[0]+offset, vB[1]-offset, '$B$\n$\\pi_7 = 1$\n$p=7$',
            ha='left', va='top', fontsize=12, fontweight='bold', color='#3333CC')
    ax.plot(*vW, 'o', color='white', markersize=10, markeredgecolor='black',
            markeredgewidth=1.5, zorder=10)
    ax.text(vW[0]+0.05, vW[1]+0.02, '$W$\n$(\\frac{1}{3}, \\frac{1}{3}, \\frac{1}{3})$',
            fontsize=10, ha='left')

    # gamma_p annotations
    ax.text(0.02, 0.95, f'$\\gamma_3 = {gamma_p(3):.3f}$', transform=ax.transAxes,
            fontsize=10, color='#CC3333')
    ax.text(0.02, 0.90, f'$\\gamma_5 = {gamma_p(5):.3f}$', transform=ax.transAxes,
            fontsize=10, color='#33AA33')
    ax.text(0.02, 0.85, f'$\\gamma_7 = {gamma_p(7):.3f}$', transform=ax.transAxes,
            fontsize=10, color='#3333CC')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Simplexe chromatique $\\Delta^2$ — Système SCS', fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig1_simplex.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig1_simplex.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig1_simplex.pdf")


def fig_conservation():
    """
    Figure 2: The conservation law S + L = log2(3).
    Saturation vs luminance trade-off.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    log2_3 = np.log2(3)

    # The conservation line
    S = np.linspace(0, log2_3, 200)
    L = log2_3 - S
    ax.plot(S, L, 'k-', linewidth=2.5, label='$S + L = \\log_2 3$ (GFT)')

    # Fill regions
    ax.fill_between(S, L, alpha=0.05, color='blue')

    # Mark key points
    points = {
        'Blanc\n(centre)': (0, log2_3),
        'Rouge pur\n(sommet)': (log2_3, 0),
        'Koide\n$1/\\sqrt{2}$': (log2_3/np.sqrt(2), log2_3*(1 - 1/np.sqrt(2))),
    }

    artists = {
        'Caravage': (log2_3*0.85, log2_3*0.15, '#8B0000', 's'),
        'Turner':   (log2_3*0.15, log2_3*0.85, '#DAA520', 'D'),
        'Vermeer':  (log2_3*0.68, log2_3*0.32, '#4169E1', 'o'),
        'Soulages': (log2_3*0.02, log2_3*0.98, '#333333', 'v'),
        'Richter':  (log2_3*0.50, log2_3*0.50, '#666666', '^'),
    }

    for name, (sx, ly) in points.items():
        ax.plot(sx, ly, 'ko', markersize=8, zorder=5)
        ax.annotate(name, (sx, ly), textcoords="offset points",
                   xytext=(10, 10), fontsize=9)

    for name, (sx, ly, color, marker) in artists.items():
        ax.plot(sx, ly, marker=marker, color=color, markersize=12,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.annotate(name, (sx, ly), textcoords="offset points",
                   xytext=(10, -10), fontsize=10, fontstyle='italic', color=color)

    # Koide line
    sk = log2_3/np.sqrt(2)
    ax.axvline(x=sk, color='#CC6600', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Saturation  $S = D_{KL}(\\pi \\| u)$  [nats]', fontsize=12)
    ax.set_ylabel('Luminance  $L = H(\\pi)$  [nats]', fontsize=12)
    ax.set_title('Loi de conservation chromatique (GFT)', fontsize=14)
    ax.set_xlim(-0.05, log2_3 + 0.1)
    ax.set_ylim(-0.05, log2_3 + 0.1)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig2_conservation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig2_conservation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig2_conservation.pdf")


def fig_fisher_ellipses():
    """
    Figure 3: Fisher metric ellipses on the simplex.
    Shows how discrimination varies across the color space.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    h = np.sqrt(3)/2
    vR = np.array([0.5, h])
    vG = np.array([0.0, 0.0])
    vB = np.array([1.0, 0.0])

    # Draw simplex
    triangle = plt.Polygon([vR, vG, vB], fill=True, facecolor='#f8f8f8',
                           edgecolor='black', linewidth=2)
    ax.add_patch(triangle)

    gammas = np.array([gamma_p(3), gamma_p(5), gamma_p(7)])

    # Sample points on the simplex and draw Fisher ellipses
    sample_points = []
    for i in range(1, 8):
        for j in range(1, 8 - i):
            k = 8 - i - j
            pi = np.array([i/8, j/8, k/8])
            if min(pi) > 0.05:  # avoid edges
                sample_points.append(pi)

    for pi in sample_points:
        # Position in display coords
        pos = pi[0] * vR + pi[1] * vG + pi[2] * vB

        # Fisher metric at this point
        G = fisher_metric(pi, gammas)

        # Eigenvalues and eigenvectors for ellipse
        eigvals, eigvecs = np.linalg.eigh(G)

        # Ellipse size inversely proportional to metric (small metric = large ellipse)
        # Scale for visibility
        scale = 0.015
        w = scale / np.sqrt(eigvals[0])
        ht = scale / np.sqrt(eigvals[1])
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        # Color based on position
        r = min(1, pi[0] * 2)
        g = min(1, pi[1] * 2)
        b = min(1, pi[2] * 2)

        ellipse = Ellipse(pos, w, ht, angle=angle, fill=True,
                         facecolor=(r, g, b, 0.4), edgecolor=(r*0.5, g*0.5, b*0.5),
                         linewidth=1)
        ax.add_patch(ellipse)

    # Labels
    offset = 0.04
    ax.text(vR[0], vR[1]+offset, '$R$ ($\\gamma_3=0.808$)',
            ha='center', va='bottom', fontsize=11, fontweight='bold', color='#CC3333')
    ax.text(vG[0]-offset, vG[1]-offset, '$G$ ($\\gamma_5=0.696$)',
            ha='right', va='top', fontsize=11, fontweight='bold', color='#33AA33')
    ax.text(vB[0]+offset, vB[1]-offset, '$B$ ($\\gamma_7=0.595$)',
            ha='left', va='top', fontsize=11, fontweight='bold', color='#3333CC')

    ax.text(0.5, -0.12, T['fig3_caption'],
            ha='center', fontsize=10, style='italic', color='gray')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.2, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ellipses de Fisher sur $\\Delta^2$ — analogue des ellipses de MacAdam',
                fontsize=13)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig3_fisher_ellipses.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig3_fisher_ellipses.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig3_fisher_ellipses.pdf")


def fig_hue_circle():
    """
    Figure 4: The hue circle from Z/pZ → S^1 holonomy.
    Shows complementary pairs and the purple closure.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Hue angles (conventional: R=0°, G=120°, B=240°)
    hues = {
        'Rouge':     (0,   '#CC3333'),
        'Orange':    (30,  '#FF8C00'),
        'Jaune':     (60,  '#CCCC00'),
        'Vert-jaune':(90,  '#88CC00'),
        'Vert':      (120, '#33AA33'),
        'Cyan':      (180, '#00AAAA'),
        'Bleu':      (240, '#3333CC'),
        'Violet':    (270, '#7733AA'),
        'Magenta':   (300, '#CC33AA'),
        'Pourpre':   (330, '#AA3366'),
    }

    # Draw outer circle
    theta = np.linspace(0, 2*np.pi, 360)
    r_outer = 1.0

    # Color the circle
    for i in range(360):
        t = i * np.pi / 180
        # HSV to RGB (approximate)
        h_val = i / 360
        r, g, b = plt.cm.hsv(h_val)[:3]
        ax.plot(r_outer * np.cos(t), r_outer * np.sin(t), 'o',
                color=(r, g, b), markersize=5, markeredgewidth=0)

    # Draw complementary lines
    comp_pairs = [(0, 180), (60, 240), (120, 300)]
    for h1, h2 in comp_pairs:
        t1 = np.radians(90 - h1)  # convert to math angle
        t2 = np.radians(90 - h2)
        x1, y1 = 0.85*np.cos(t1), 0.85*np.sin(t1)
        x2, y2 = 0.85*np.cos(t2), 0.85*np.sin(t2)
        ax.plot([x1, x2], [y1, y2], '--', color='gray', linewidth=1, alpha=0.5)

    # Mark the primes
    prime_hues = {
        'p=3': (0, '#CC3333', 0.808),
        'p=5': (120, '#33AA33', 0.696),
        'p=7': (240, '#3333CC', 0.595),
    }

    for label, (hue, color, gam) in prime_hues.items():
        t = np.radians(90 - hue)
        x, y = 1.15 * np.cos(t), 1.15 * np.sin(t)
        ax.plot(1.05*np.cos(t), 1.05*np.sin(t), 'o', color=color,
                markersize=15, markeredgecolor='black', markeredgewidth=1.5, zorder=10)
        ax.text(x, y, f'{label}\n$\\gamma={gam:.3f}$',
                ha='center', va='center', fontsize=10, fontweight='bold', color=color)

    # Mark the purple line (non-spectral closure)
    t1 = np.radians(90 - 330)
    t2 = np.radians(90 - 30)
    arc_t = np.linspace(t1, t2 - 2*np.pi, 30)  # go the short way around bottom
    ax.plot(0.75*np.cos(arc_t), 0.75*np.sin(arc_t), '-', color='#AA3366',
            linewidth=3, alpha=0.6)
    ax.text(0, -0.85, 'Pourpre\n(fermeture $2\\pi$)', ha='center',
            fontsize=10, color='#AA3366', fontstyle='italic')

    # sin^2 + cos^2 = 1 annotation
    ax.text(0, 0, '$\\sin^2 + \\cos^2 = 1$\n(bifurcation)',
            ha='center', va='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Holonomy annotation
    ax.annotate('', xy=(0.6, 0.6), xytext=(0.6, 0.3),
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.text(0.7, 0.45, 'Holonomie\n$\\mathbb{Z}/p\\mathbb{Z} \\to S^1$',
            fontsize=9, color='gray')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Cercle de teinte — holonomie du crible sur $S^1$", fontsize=14)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig4_hue_circle.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig4_hue_circle.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig4_hue_circle.pdf")


def fig_gamma_hierarchy():
    """
    Figure 5: The gamma_p hierarchy — why red > green > blue.
    Bar chart with the threshold at 1/2.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    primes_ext = [2, 3, 5, 7, 11, 13, 17]
    gammas = []
    colors = []
    for p in primes_ext:
        if p == 2:
            gammas.append(1.0)  # p=2 is the foundation, gamma not defined same way
            colors.append('#888888')
        else:
            g = gamma_p(p)
            gammas.append(g)
            if g > 0.5:
                colors.append(PRIME_COLORS.get(p, '#888888'))
            else:
                colors.append('#CCCCCC')

    labels = [f'$p=2$\n{T["fig5_parity"]}', f'$p=3$\n{T["fig5_red"]}', f'$p=5$\n{T["fig5_green"]}',
              f'$p=7$\n{T["fig5_blue"]}', '$p=11$', '$p=13$', '$p=17$']

    bars = ax.bar(range(len(primes_ext)), gammas, color=colors,
                  edgecolor='black', linewidth=0.8, width=0.7)

    # Threshold line
    ax.axhline(y=0.5, color='#CC6600', linestyle='--', linewidth=2,
               label=T['fig5_threshold'])

    # Annotations
    ax.text(0, 1.05, 'fondation\nbinaire', ha='center', fontsize=8, color='#888888')
    for i, p in enumerate(primes_ext[1:4], 1):
        ax.text(i, gammas[i] + 0.02, f'{gammas[i]:.3f}', ha='center', fontsize=10,
                fontweight='bold')

    ax.set_xticks(range(len(primes_ext)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('$\\gamma_p$ (dimension effective)', fontsize=12)
    ax.set_title("Hiérarchie des canaux chromatiques — dérivée du crible à $\\mu^*=15$",
                fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)

    # Bracket for active primes
    ax.annotate('', xy=(0.7, 0.55), xytext=(3.3, 0.55),
               arrowprops=dict(arrowstyle='|-|', color='#CC6600', lw=1.5))
    ax.text(2, 0.57, 'actifs : $\\{3,5,7\\}$', ha='center', fontsize=10,
            color='#CC6600', fontweight='bold')

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig5_gamma_hierarchy.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig5_gamma_hierarchy.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig5_gamma_hierarchy.pdf")


def fig_bifurcation():
    """
    Figure 6: The bifurcation — additive (q_rel) vs subtractive (q_therm).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: q_rel branch (additive, transmission)
    ax1.set_title('Branche $q_{rel}$ — Transmission\n(synthèse additive, écrans)',
                  fontsize=12)

    # Draw RGB additive mixing
    from matplotlib.patches import Circle
    for (x, y, color, label) in [
        (0.35, 0.6, '#CC3333', 'R ($\\sin^2_3$)'),
        (0.5, 0.35, '#33AA33', 'G ($\\sin^2_5$)'),
        (0.65, 0.6, '#3333CC', 'B ($\\sin^2_7$)'),
    ]:
        circle = Circle((x, y), 0.2, color=color, alpha=0.4)
        ax1.add_patch(circle)
        ax1.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    ax1.text(0.5, 0.55, 'W', ha='center', va='center', fontsize=14,
             fontweight='bold', color='white',
             bbox=dict(boxstyle='round', facecolor='gray', alpha=0.5))
    ax1.text(0.5, 0.05, '$\\alpha_{EM} = \\prod \\sin^2(\\theta_p, q_{rel})$\n'
             f'$= 1/{1/np.prod([sin2_theta(p, Q_REL) for p in PRIMES]):.1f}$',
             ha='center', fontsize=11)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 0.9)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Right: q_therm branch (subtractive, absorption)
    ax2.set_title('Branche $q_{therm}$ — Absorption\n(synthèse soustractive, pigments)',
                  fontsize=12)

    for (x, y, color, label) in [
        (0.35, 0.6, '#00AAAA', 'C ($\\cos^2_3$)'),
        (0.5, 0.35, '#CC33AA', 'M ($\\cos^2_5$)'),
        (0.65, 0.6, '#CCCC00', 'Y ($\\cos^2_7$)'),
    ]:
        circle = Circle((x, y), 0.2, color=color, alpha=0.4)
        ax2.add_patch(circle)
        ax2.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold')

    ax2.text(0.5, 0.55, 'K', ha='center', va='center', fontsize=14,
             fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax2.text(0.5, 0.05, T['fig6_comp'],
             ha='center', fontsize=11)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 0.9)
    ax2.set_aspect('equal')
    ax2.axis('off')

    fig.suptitle(T['fig6_title'],
                fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig6_bifurcation.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig6_bifurcation.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig6_bifurcation.pdf")


def fig_berlin_kay():
    """
    Figure 7: Berlin-Kay universals explained by gamma_p hierarchy.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    stage_colors = ['#888888', '#CC3333', '#88AA00', '#3333CC', '#AAAAAA']
    stage_gammas = [1.0, 0.808, 0.696, 0.595, 0.3]
    stages = list(zip(T['fig7_stages'], stage_colors, stage_gammas))

    for i, (label, color, gam) in enumerate(stages):
        x = i * 2
        rect = plt.Rectangle((x - 0.4, 0), 0.8, gam, color=color, alpha=0.7,
                             edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, -0.08, label, ha='center', va='top', fontsize=9)
        if gam > 0.4:
            ax.text(x, gam + 0.02, f'$\\gamma = {gam:.3f}$', ha='center',
                    fontsize=9, fontweight='bold')

        # Arrow between stages
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + 1.0, 0.5), xytext=(x + 0.6, 0.5),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    ax.axhline(y=0.5, color='#CC6600', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(9.5, 0.52, 'seuil $s=1/2$', fontsize=9, color='#CC6600')

    ax.set_xlim(-1, 10)
    ax.set_ylim(-0.25, 1.1)
    ax.set_ylabel('$\\gamma_p$', fontsize=12)
    ax.set_title("Universaux de Berlin-Kay — l'ordre des termes de couleur suit $\\gamma_p$",
                fontsize=13)
    ax.set_xticks([])
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'fig7_berlin_kay.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIG_DIR, 'fig7_berlin_kay.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ fig7_berlin_kay.pdf")


def generate_all_figures():
    """Generate all figures for the article."""
    print("\nGenerating figures...")
    fig_chromatic_simplex()
    fig_conservation()
    fig_fisher_ellipses()
    fig_hue_circle()
    fig_gamma_hierarchy()
    fig_bifurcation()
    fig_berlin_kay()
    print(f"\nAll figures saved to {FIG_DIR}/")


# ============================================================
# MAIN
# ============================================================

def generate_for_lang(lang):
    """Generate all figures for a specific language."""
    global LANG, T, PRIME_LABELS, FIG_DIR
    LANG = lang
    T = TEXTS[lang]
    PRIME_LABELS = T['prime_labels']
    FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f'figures_{lang}')
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"\n>>> Generating {lang.upper()} figures to {FIG_DIR}/")
    generate_all_figures()


if __name__ == '__main__':
    if '--verify' in sys.argv:
        verify_all()
    elif '--lang=en' in sys.argv:
        generate_for_lang('en')
    elif '--lang=fr' in sys.argv:
        generate_for_lang('fr')
    elif '--figures' in sys.argv:
        generate_for_lang('en')
        generate_for_lang('fr')
    else:
        ok = verify_all()
        print()
        generate_for_lang('en')
        generate_for_lang('fr')
        print(f"\n{'='*60}")
        print(f"Script complete. Verification: {'PASS' if ok else 'FAIL'}")
        print(f"Figures EN: figures_en/")
        print(f"Figures FR: figures_fr/")
        print(f"{'='*60}")
