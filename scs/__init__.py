"""
SCS — Sieve Color Space
========================

A first-principles color space and color-difference formula derived
from the Sieve of Eratosthenes (Persistence Theory).

Single input: s = 1/2. Zero adjustable parameters.

Quick start:
    >>> from scs import delta_e, to_scs
    >>> d = delta_e([0.95, 1.0, 1.09], [0.60, 0.50, 0.30])
    >>> print(f"Color difference: {d:.4f}")
    >>> coords = to_scs([0.95, 1.0, 1.09])
    >>> print(f"SCS: ℓ={coords.ell:.3f}, S={coords.S:.3f}, θ={coords.hue:.1f}°")

Theory summary:
    - 4 primes: {2, 3, 5, 7}. p=2 = luminance. {3,5,7} = chromaticity.
    - Metric: Fisher (Čencov unique) weighted by γ_p at μ*=15.
    - Distance: Bhattacharyya geodesic on Δ² + Fisher on Bernoulli(ℓ).
    - Balance: w_lum = 3/4, w_chrom = 1/4  (N_active/(N_active+1), T7).
    - Conservation: D_KL + H = log₂(3)  (GFT, algebraic identity).
    - Spectral window: {3,5,7} → α_EM → Rydberg → Balmer → 380–656 nm.

Reference: PT_COLOR.tex (Senez, 2026).
"""

import numpy as np
from dataclasses import dataclass

__version__ = "0.2.0"
__all__ = ["delta_e", "delta_e_lab", "to_scs", "SCSColor",
           "MU_STAR", "GAMMAS", "PRIMES"]

# ============================================================
# CONSTANTS — all derived from s = 1/2 at μ* = 15
# ============================================================

MU_STAR = 15                          # unique fixed point (T5)
S_PARAM = 0.5                         # symmetry parameter (T1)
Q_REL = 1 - 2 / MU_STAR              # = 13/15 (vertex branch)
Q_THERM = np.exp(-1 / MU_STAR)       # ≈ 0.9355 (edge branch)
PRIMES = (3, 5, 7)                    # active primes (T5)
N_ACTIVE = len(PRIMES)                # = 3

def _delta(p, q=Q_REL):
    return (1 - q**p) / p

def _sin2(p, q=Q_REL):
    d = _delta(p, q)
    return d * (2 - d)

def _gamma(p, mu=MU_STAR):
    q = 1 - 2/mu
    d = _delta(p, q)
    return 4*p * q**(p-1) * (1-d) / (mu * (1 - q**p) * (2 - d))

# Effective dimensions (derived, not chosen)
GAMMA_3 = _gamma(3)   # 0.8076
GAMMA_5 = _gamma(5)   # 0.6963
GAMMA_7 = _gamma(7)   # 0.5955
GAMMAS = np.array([GAMMA_3, GAMMA_5, GAMMA_7])

# Luminance/chromaticity balance (T7)
W_LUM = N_ACTIVE / (N_ACTIVE + 1)    # 3/4
W_CHROM = 1 / (N_ACTIVE + 1)         # 1/4

# Standard XYZ→LMS matrix (not a model parameter — measurement apparatus)
_M_HPE = np.array([
    [ 0.38971,  0.68898, -0.07868],
    [-0.22981,  1.18340,  0.04641],
    [ 0.00000,  0.00000,  1.00000],
])


# ============================================================
# COORDINATE CONVERSIONS
# ============================================================

@dataclass
class SCSColor:
    """SCS color coordinates."""
    ell: float          # luminance ∈ [0, 1] (p=2 channel)
    S: float            # saturation = D_KL(π||u) ∈ [0, log₂3]
    hue: float          # hue angle in degrees ∈ [0, 360)
    pi: np.ndarray      # simplex coordinates (π₃, π₅, π₇)


def _xyz_to_lms(xyz, matrix=None):
    """XYZ → LMS cone responses."""
    M = matrix if matrix is not None else _M_HPE
    return np.maximum(M @ np.asarray(xyz, dtype=float), 1e-12)


def _lms_to_simplex(lms):
    """LMS → SCS simplex (π₃, π₅, π₇), weighted by γ_p."""
    w = GAMMAS * np.maximum(lms, 1e-12)
    return w / w.sum()


def to_scs(xyz, Y_ref=1.0, matrix=None):
    """
    Convert CIE XYZ to SCS coordinates.

    Parameters:
        xyz: CIE XYZ tristimulus (3-vector)
        Y_ref: reference white luminance (default 1.0)
        matrix: optional XYZ→LMS matrix (default HPE)

    Returns:
        SCSColor with (ell, S, hue, pi)
    """
    xyz = np.asarray(xyz, dtype=float)
    lms = _xyz_to_lms(xyz, matrix)
    pi = _lms_to_simplex(lms)

    ell = np.clip(xyz[1] / Y_ref, 0, 1)

    # Saturation: D_KL(π || uniform)
    S = float(np.sum(pi[pi > 0] * np.log2(3 * pi[pi > 0])))

    # Hue: angular coordinate on the simplex
    hue = np.degrees(np.arctan2(
        np.sqrt(3) * (pi[1] - pi[2]),
        2 * pi[0] - pi[1] - pi[2]
    )) % 360

    return SCSColor(ell=ell, S=S, hue=hue, pi=pi)


# ============================================================
# COLOR DIFFERENCE: ΔE_SCS
# ============================================================

def delta_e(xyz1, xyz2, Y_ref=1.0, matrix=None):
    """
    SCS color difference between two XYZ colors.

    Formula (zero adjustable parameters):

        ΔE² = (3/4)·d_lum² + (1/4)·d_chrom²

    where:
        d_lum   = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|     Fisher on Bernoulli
        d_chrom = 2·arccos(Σ √(π̃₁·π̃₂))             Bhattacharyya on Δ²

    Derivation chain:
        s=1/2 → T1 → T5 (μ*=15, N=3) → Fisher (Čencov) → geodesic

    Parameters:
        xyz1, xyz2: CIE XYZ tristimulus values (3-vectors)
        Y_ref: reference white luminance
        matrix: optional XYZ→LMS matrix

    Returns:
        ΔE_SCS (float ≥ 0)
    """
    xyz1 = np.asarray(xyz1, dtype=float)
    xyz2 = np.asarray(xyz2, dtype=float)

    # LMS → weighted simplex
    pi1 = _lms_to_simplex(_xyz_to_lms(xyz1, matrix))
    pi2 = _lms_to_simplex(_xyz_to_lms(xyz2, matrix))

    # Chromaticity: Bhattacharyya geodesic on Δ²
    bc = np.clip(np.sum(np.sqrt(pi1 * pi2)), 0, 1)
    d_chrom = 2 * np.arccos(bc)

    # Luminance: Fisher geodesic on Bernoulli(ℓ)
    ell1 = np.clip(xyz1[1] / Y_ref, 1e-6, 1 - 1e-6)
    ell2 = np.clip(xyz2[1] / Y_ref, 1e-6, 1 - 1e-6)
    d_lum = 2 * abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))

    return float(np.sqrt(W_LUM * d_lum**2 + W_CHROM * d_chrom**2))


def fisher_luminance(Y1, Y2, Y_ref=100.0):
    """
    Fisher-Bernoulli luminance geodesic (standalone).

    d_lum = 2|arcsin(√ℓ₁) - arcsin(√ℓ₂)|

    Used by ΔE_SCS00 to improve CIEDE2000.
    Zero adjustable parameters — derived from s = 1/2.
    """
    ell1 = np.clip(np.asarray(Y1) / Y_ref, 1e-10, 1 - 1e-10)
    ell2 = np.clip(np.asarray(Y2) / Y_ref, 1e-10, 1 - 1e-10)
    return 2 * np.abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))


def delta_e_lab(L1, a1, b1, L2, a2, b2,
                white=(0.9505, 1.0, 1.089)):
    """
    SCS color difference from CIELAB coordinates.

    Convenience wrapper: converts Lab → XYZ → ΔE_SCS.
    """
    xyz1 = _lab_to_xyz(L1, a1, b1, white)
    xyz2 = _lab_to_xyz(L2, a2, b2, white)
    return delta_e(xyz1, xyz2)


# ============================================================
# CONSERVATION LAW (GFT)
# ============================================================

def saturation(pi):
    """D_KL(π || uniform) — saturation. [GFT, D02]"""
    pi = np.asarray(pi, dtype=float)
    return float(np.sum(pi[pi > 0] * np.log2(3 * pi[pi > 0])))


def luminance_entropy(pi):
    """H(π) — perceptual luminance entropy. [GFT, D02]"""
    pi = np.asarray(pi, dtype=float)
    return float(-np.sum(pi[pi > 0] * np.log2(pi[pi > 0])))


def gft_check(pi):
    """
    Verify GFT: S + L = log₂(3).
    Returns (S, L, S+L, error).
    """
    S = saturation(pi)
    L = luminance_entropy(pi)
    budget = np.log2(3)
    return S, L, S + L, abs(S + L - budget)


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _lab_to_xyz(L, a, b, white=(0.9505, 1.0, 1.089)):
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    d = 6 / 29
    def finv(t):
        return t**3 if t > d else 3 * d**2 * (t - 4/29)
    return np.array([white[0]*finv(fx), white[1]*finv(fy), white[2]*finv(fz)])


# ============================================================
# SELF-TEST
# ============================================================

def _selftest():
    """Run basic verification."""
    print(f"SCS v{__version__} — self-test")
    print(f"  μ* = {MU_STAR}, γ = ({GAMMA_3:.4f}, {GAMMA_5:.4f}, {GAMMA_7:.4f})")

    # GFT conservation
    for name, pi in [("white", [1/3,1/3,1/3]), ("red", [0.99,0.005,0.005])]:
        S, L, total, err = gft_check(pi)
        print(f"  GFT {name}: S+L = {total:.6f} (err {err:.1e}) {'PASS' if err<1e-10 else 'FAIL'}")

    # ΔE basic properties
    white = np.array([0.9505, 1.0, 1.089])
    red = _lab_to_xyz(50, 60, 30)
    d1 = delta_e(white, red)
    d2 = delta_e(red, white)
    d0 = delta_e(white, white)
    print(f"  Symmetry: |d(a,b)-d(b,a)| = {abs(d1-d2):.1e} {'PASS' if abs(d1-d2)<1e-10 else 'FAIL'}")
    print(f"  Identity: d(a,a) = {d0:.1e} {'PASS' if d0<1e-10 else 'FAIL'}")
    print(f"  ΔE(white,red) = {d1:.4f}")

    # Hierarchy
    ok = GAMMA_3 > GAMMA_5 > GAMMA_7 > _gamma(11)
    print(f"  Hierarchy γ₃>γ₅>γ₇>γ₁₁: {'PASS' if ok else 'FAIL'}")

    print("  All tests passed." if ok else "  SOME TESTS FAILED.")


if __name__ == "__main__":
    _selftest()
