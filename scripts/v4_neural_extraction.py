#!/usr/bin/env python3
"""
V4 Neural Transfer Function Extraction
=======================================

Pipeline:
  1. Load NIfTI BOLD data for each DKL run
  2. Parse event TSV → condition (hue, contrast) per block
  3. Extract mean BOLD per block (14s = 7 TRs, HRF shift +4s = +2 TRs)
  4. Define V4 ROI (anatomical: posterior occipital cortex)
  5. Average across runs/sessions/subjects → transfer function
  6. Map DKL hues → SCT coordinates
  7. Test PT hypothesis: BOLD_V4 ∝ Fisher(γ_p)?

Dataset: OpenNeuro ds005521 (Conway lab, Harvard/NIH)
  2 macaques, 118 runs, 10 DKL hues × 5 saturations

Author: PT_COLOR project
"""

import os
import sys
import csv
import glob
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

# OpenNeuro dataset ds005521 — download with:
#   pip install openneuro-py
#   openneuro download --dataset ds005521 --target-dir data/ds005521
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SCT_V4_DATA", os.path.join(SCRIPT_DIR, "data", "ds005521"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets")
TR = 2.0           # seconds
BLOCK_DUR = 14.0   # seconds per stimulus block
HRF_DELAY = 4.0    # seconds to peak of hemodynamic response
N_TRS_BLOCK = int(BLOCK_DUR / TR)  # 7 TRs per block
HRF_SHIFT = int(HRF_DELAY / TR)    # 2 TRs shift for HRF peak

# DKL hue angles (Derrington-Krauskopf-Lennie)
# From Conway et al. (J. Neurosci. 2025, 45(2):e1673232024), Table 1
# Index 1-8: 8 equiluminant hues at 45° spacing in DKL chromatic plane
# Index 9: LUMINANCE axis (LMS achromatic, not a chromatic hue!)
DKL_ANGLES_DEG = {
    1: 0,     # L-M+ (pinkish-red)
    2: 45,    # Daylight+ (orange)
    3: 90,    # S+ (blue-ish/purple-ish)
    4: 135,   # Antidaylight+ (green)
    5: 180,   # L-M- (cyan)
    6: 225,   # Daylight- (blue)
    7: 270,   # S- (purple/violet)
    8: 315,   # Antidaylight- (magenta)
    9: None,  # LUMINANCE (LMS achromatic) — p=2 channel in PT
}

# Hue 9 is achromatic luminance, not a DKL chromatic direction
CHROMATIC_HUES = list(range(1, 9))  # 1-8 only
LUMINANCE_HUE = 9

# Contrasts used in the experiment
CONTRASTS = [0.10, 0.30, 0.50, 0.95]

# ============================================================
# STEP 1-2: LOAD EVENTS
# ============================================================

def load_events(events_tsv):
    """Parse BIDS events TSV → list of (onset, duration, color, contrast)."""
    events = []
    with open(events_tsv) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                onset = float(row['onset'])
                duration = float(row['duration'])
                color = int(row['color'])
                contrast = float(row['contrast'])
                events.append((onset, duration, color, contrast))
            except (ValueError, KeyError):
                continue
    return events


# ============================================================
# STEP 3: EXTRACT BOLD PER CONDITION
# ============================================================

def extract_bold_per_condition(nifti_path, events, mask=None):
    """
    Extract mean BOLD signal per stimulus condition.

    For each non-fixation block:
      - Identify TRs during stimulus (shifted by HRF delay)
      - Average BOLD within ROI mask during those TRs
      - Return dict: (color, contrast) → list of BOLD values

    Returns: dict[(color, contrast)] → [bold_value, ...]
    """
    import nibabel as nib

    img = nib.load(nifti_path)
    data = img.get_fdata()  # shape: (x, y, z, t)

    n_trs = data.shape[-1]

    # Apply mask if provided
    if mask is not None:
        # Average over ROI voxels → timeseries
        roi_voxels = data[mask > 0]  # shape: (n_voxels, t)
        timeseries = roi_voxels.mean(axis=0)  # shape: (t,)
    else:
        # FUNCTIONAL ROI: select visually-responsive voxels
        # Strategy: voxels in posterior cortex with high variance
        # (color-responsive voxels have stimulus-locked fluctuations)
        z_dim = data.shape[2]
        z_start = 0
        z_end = int(z_dim * 0.45)  # posterior 45% (V1-V4 in macaque)

        roi_data = data[:, :, z_start:z_end, :]

        # Brain mask: above noise floor
        mean_signal = roi_data.mean(axis=-1)
        nonzero = mean_signal[mean_signal > 0]
        if len(nonzero) == 0:
            return {}
        threshold = np.percentile(nonzero, 30)
        brain_mask = mean_signal > threshold

        if brain_mask.sum() < 10:
            return {}

        # Among brain voxels, select top 20% by temporal variance
        # (these are the most stimulus-responsive)
        brain_ts = roi_data[brain_mask]  # (n_voxels, t)
        var_per_voxel = brain_ts.var(axis=1)
        var_threshold = np.percentile(var_per_voxel, 80)
        responsive = var_per_voxel >= var_threshold

        timeseries = brain_ts[responsive].mean(axis=0)

    # Normalize timeseries (percent signal change from run mean)
    run_mean = timeseries.mean()
    if run_mean == 0:
        return {}
    psc = (timeseries - run_mean) / run_mean * 100

    # Extract per condition
    conditions = defaultdict(list)
    for onset, duration, color, contrast in events:
        if color == 0:  # skip fixation
            continue

        # Convert onset to TR index, shifted by HRF
        tr_start = int(onset / TR) + HRF_SHIFT
        tr_end = tr_start + N_TRS_BLOCK

        if tr_start >= 0 and tr_end <= n_trs:
            block_bold = psc[tr_start:tr_end].mean()
            conditions[(color, contrast)].append(block_bold)

    return conditions


# ============================================================
# STEP 4: V4 ROI (Anatomical approximation)
# ============================================================

def make_posterior_occipital_mask(shape, fraction=0.35):
    """
    Create anatomical V4 proxy mask.

    V4 in macaque is in posterior occipital cortex.
    We take the posterior `fraction` of slices and create
    a conservative brain mask.

    For proper analysis, use retinotopic mapping data.
    This is a first approximation.
    """
    mask = np.zeros(shape[:3], dtype=bool)
    z_end = int(shape[2] * fraction)
    mask[:, :, :z_end] = True
    return mask


# ============================================================
# STEP 5: DKL → SCT MAPPING
# ============================================================

def dkl_to_lms_direction(angle_deg, contrast=1.0):
    """
    Convert DKL angle + contrast to LMS cone activations.

    DKL space (Derrington-Krauskopf-Lennie):
      - 0°/180°: L-M axis (red-green, isoluminant)
      - 90°/270°: S-(L+M) axis (blue-yellow, isoluminant)

    The DKL basis vectors in cone-contrast space:
      L-M:    dL/L = +a,  dM/M = -a,  dS/S = 0     (isoluminant: L+M=const)
      S-(LM): dL/L = 0,   dM/M = 0,   dS/S = +b     (S modulation only)

    For macaque with equal L,M,S background:
      a_max ≈ 0.12 for L-M (from Conway Table 1, 95% contrast)
      b_max ≈ 0.85 for S   (S cones are much more easily modulated)

    Returns LMS activations relative to an adapted state.
    """
    theta = np.radians(angle_deg)

    # DKL chromatic plane
    lm_weight = np.cos(theta)
    s_weight = np.sin(theta)

    # Cone contrast amplitudes from Conway Table 1 (approximate)
    # At 95% contrast: L-M axis has ΔL/L ≈ 12%, ΔS/S ≈ 85%
    a_lm = 0.12    # L-M max cone contrast
    b_s = 0.85     # S max cone contrast

    # Cone contrasts for this DKL direction
    cc_L = +lm_weight * a_lm
    cc_M = -lm_weight * a_lm
    cc_S = s_weight * b_s

    # Apply contrast scaling
    bg = np.array([1.0, 1.0, 1.0])
    lms = bg * (1.0 + contrast * np.array([cc_L, cc_M, cc_S]))

    # Ensure positive
    lms = np.maximum(lms, 0.01)

    return lms


def dkl_to_opponent_channels(angle_deg, contrast=1.0):
    """
    Decompose a DKL stimulus into the 3 PT-predicted opponent channels.

    Returns: (lm_signal, s_signal, lum_signal)
    where each is the activation of that channel.

    In PT:
      - L-M ↔ γ₃ channel (p=3, strongest)
      - S-(L+M) ↔ γ₇ channel (p=7, weakest chromatic)
      - Luminance ↔ p=2 channel (binary operator)

    For isoluminant DKL stimuli, lum_signal ≈ 0.
    """
    theta = np.radians(angle_deg)
    lm = np.cos(theta) * contrast   # L-M channel activation
    s = np.sin(theta) * contrast    # S channel activation
    lum = 0.0                       # isoluminant → no luminance
    return lm, s, lum


def lms_to_sct_simplex(lms):
    """LMS → SCT simplex π = (π₃, π₅, π₇), weighted by γ_p."""
    # Import gamma values from SCT
    gammas = np.array([0.80761, 0.69632, 0.59547])  # γ₃, γ₅, γ₇
    w = gammas * np.maximum(lms, 1e-12)
    return w / w.sum()


def fisher_distance_from_achromatic(pi):
    """
    Bhattacharyya/Fisher geodesic distance from achromatic point.

    On the simplex Δ², the Fisher-Rao geodesic distance is:
      d(π, u) = 2·arccos(Σ √(π_i · u_i))

    where u = (1/3, 1/3, 1/3) is the achromatic (uniform) point.

    This is EXACTLY the SCT d_chrom formula — the same distance
    that appears in the color difference ΔE_SCT.

    PT predicts: V4 response should scale with this distance,
    because it measures the information content (saturation)
    of the chromatic stimulus on the simplex T³.
    """
    u = np.array([1/3, 1/3, 1/3])
    bc = np.clip(np.sum(np.sqrt(pi * u)), 0, 1)
    return 2 * np.arccos(bc)


def fisher_anisotropy(pi):
    """
    Anisotropy of the Fisher metric at π, weighted by γ_p.

    Measures how much the stimulus activates each channel
    relative to the PT prediction:
      A(π) = Σ_p γ_p · |π_p - 1/3| / (1/3)

    If V4 follows PT, the activation pattern should be
    proportional to γ_p (0.808:0.696:0.595).
    """
    gammas = np.array([0.80761, 0.69632, 0.59547])
    deviations = np.abs(pi - 1/3) / (1/3)
    return np.sum(gammas * deviations)


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline():
    """Execute the full V4 neural extraction pipeline."""

    print("=" * 60)
    print("V4 Neural Transfer Function — PT_COLOR")
    print("=" * 60)

    # Find all available NIfTI files (not broken symlinks)
    nifti_files = sorted(glob.glob(
        f"{DATA_DIR}/sub-*/ses-*/func/*task-dkl*_bold.nii.gz"
    ))
    # Filter to actual files (not broken annex symlinks)
    available = [f for f in nifti_files if os.path.isfile(f) and os.path.getsize(f) > 1000]
    print(f"\nAvailable NIfTI files: {len(available)} / {len(nifti_files)}")

    if len(available) == 0:
        print("ERROR: No NIfTI files downloaded yet. Run download script first.")
        sys.exit(1)

    # Group by subject
    by_subject = defaultdict(list)
    for f in available:
        sub = os.path.basename(f).split('_')[0]  # sub-M1 or sub-M2
        by_subject[sub].append(f)

    for sub, files in by_subject.items():
        print(f"  {sub}: {len(files)} runs")

    # ---- EXTRACT BOLD PER CONDITION ----
    print("\n--- Extracting BOLD per condition ---")

    all_conditions = defaultdict(list)  # (sub, color, contrast) → [bold]
    global_conditions = defaultdict(list)  # (color, contrast) → [bold]

    for i, nifti_path in enumerate(available):
        # Find matching events file
        events_path = nifti_path.replace("_bold.nii.gz", "_events.tsv")
        if not os.path.exists(events_path):
            print(f"  [{i+1}/{len(available)}] SKIP (no events): {os.path.basename(nifti_path)}")
            continue

        print(f"  [{i+1}/{len(available)}] Processing {os.path.basename(nifti_path)}...")

        events = load_events(events_path)
        try:
            cond_bold = extract_bold_per_condition(nifti_path, events)
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

        sub = os.path.basename(nifti_path).split('_')[0]
        for (color, contrast), values in cond_bold.items():
            for v in values:
                all_conditions[(sub, color, contrast)].append(v)
                global_conditions[(color, contrast)].append(v)

        n_conds = len(cond_bold)
        n_blocks = sum(len(v) for v in cond_bold.values())
        print(f"    {n_conds} conditions, {n_blocks} blocks extracted")

    # ---- BUILD TRANSFER FUNCTION ----
    print("\n--- Building neural transfer function ---")

    results = []
    for color in range(1, 10):
        for contrast in CONTRASTS:
            key = (color, contrast)
            bold_values = global_conditions.get(key, [])

            if len(bold_values) == 0:
                continue

            bold_mean = np.mean(bold_values)
            bold_std = np.std(bold_values) / np.sqrt(len(bold_values))  # SEM
            n_runs = len(bold_values)

            # DKL → SCT mapping
            dkl_angle = DKL_ANGLES_DEG.get(color)
            if dkl_angle is not None:
                # Chromatic hue (1-8)
                lms = dkl_to_lms_direction(dkl_angle, contrast)
                pi = lms_to_sct_simplex(lms)
                fisher = fisher_distance_from_achromatic(pi)
            else:
                # Luminance (hue 9) — achromatic on simplex
                pi = np.array([1/3, 1/3, 1/3])
                fisher = 0.0  # no chromatic distance
                dkl_angle = -1  # flag as luminance

            results.append({
                'hue_index': color,
                'hue_dkl_deg': dkl_angle,
                'contrast': contrast,
                'bold_v4_mean': bold_mean,
                'bold_v4_std': bold_std,
                'n_runs': n_runs,
                'sct_pi3': pi[0],
                'sct_pi5': pi[1],
                'sct_pi7': pi[2],
                'fisher_distance': fisher,
            })

            print(f"  Hue {color} ({dkl_angle:>3}°) × {contrast:.2f}: "
                  f"BOLD = {bold_mean:+.3f} ± {bold_std:.3f} % "
                  f"(n={n_runs}) | F={fisher:.2f}")

    # ---- SAVE RESULTS ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outpath = os.path.join(OUTPUT_DIR, "v4_bold_response.csv")

    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'hue_index', 'hue_dkl_deg', 'contrast',
            'bold_v4_mean', 'bold_v4_std', 'n_runs',
            'sct_pi3', 'sct_pi5', 'sct_pi7', 'fisher_distance'
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n--- Saved: {outpath} ({len(results)} rows) ---")

    # ---- PT HYPOTHESIS TEST ----
    print("\n" + "=" * 60)
    print("TEST: BOLD_V4 ∝ Fisher_metric(γ_p)?")
    print("=" * 60)

    # Group by contrast level for cleaner analysis
    for ct in CONTRASTS:
        subset = [r for r in results if r['contrast'] == ct]
        if len(subset) < 3:
            continue

        bold = np.array([r['bold_v4_mean'] for r in subset])
        fisher = np.array([r['fisher_distance'] for r in subset])

        # Pearson correlation
        if bold.std() > 0 and fisher.std() > 0:
            r = np.corrcoef(bold, fisher)[0, 1]
            print(f"\n  Contrast {ct:.2f}: r(BOLD, Fisher) = {r:.3f} (n={len(subset)})")

            # Check γ_p ratios in BOLD pattern
            # If PT is right, the hue-dependent BOLD should correlate with
            # how much each γ_p channel is modulated
            print(f"    BOLD range: [{bold.min():.3f}, {bold.max():.3f}]")
            print(f"    Fisher range: [{fisher.min():.2f}, {fisher.max():.2f}]")

    # Global correlation across all conditions
    if len(results) > 5:
        all_bold = np.array([r['bold_v4_mean'] for r in results])
        all_fisher = np.array([r['fisher_distance'] for r in results])
        all_contrast = np.array([r['contrast'] for r in results])

        if all_bold.std() > 0 and all_fisher.std() > 0:
            r_global = np.corrcoef(all_bold, all_fisher)[0, 1]
            r_contrast = np.corrcoef(all_bold, all_contrast)[0, 1]
            print(f"\n  GLOBAL: r(BOLD, Fisher) = {r_global:.3f}")
            print(f"  GLOBAL: r(BOLD, contrast) = {r_contrast:.3f}")
            print(f"  (Contrast should dominate; hue modulation is the PT test)")

    # ---- PER-HUE ANALYSIS (collapsing across contrast) ----
    print("\n--- Per-hue BOLD profile (averaged over contrasts) ---")
    gammas = np.array([0.80761, 0.69632, 0.59547])
    print(f"  PT γ_p ratios: {gammas[0]/gammas[0]:.3f} : {gammas[1]/gammas[0]:.3f} : {gammas[2]/gammas[0]:.3f}")

    hue_bold = []
    hue_fisher = []
    hue_aniso = []

    # === CHROMATIC HUES (1-8) ===
    print("\n  CHROMATIC HUES (equiluminant, DKL plane):")
    for color in CHROMATIC_HUES:
        subset = [r for r in results if r['hue_index'] == color]
        if not subset:
            continue
        mean_bold = np.mean([r['bold_v4_mean'] for r in subset])
        mean_fisher = np.mean([r['fisher_distance'] for r in subset])
        dkl_angle = subset[0]['hue_dkl_deg']

        # Compute anisotropy (γ-weighted deviation from achromatic)
        pi_at_max = np.array([
            [r['sct_pi3'], r['sct_pi5'], r['sct_pi7']]
            for r in subset if r['contrast'] == 0.95
        ])
        if len(pi_at_max) > 0:
            aniso = fisher_anisotropy(pi_at_max.mean(axis=0))
        else:
            aniso = fisher_anisotropy(np.array([subset[0]['sct_pi3'],
                                                 subset[0]['sct_pi5'],
                                                 subset[0]['sct_pi7']]))

        # Which cone is most modulated at this angle?
        lms = dkl_to_lms_direction(dkl_angle, 0.95)
        cone_mod = np.abs(lms - 1.0)  # deviation from background
        dominant = ['L', 'M', 'S'][np.argmax(cone_mod)]

        hue_bold.append(mean_bold)
        hue_fisher.append(mean_fisher)
        hue_aniso.append(aniso)

        print(f"    Hue {color} ({dkl_angle:>3}°): BOLD = {mean_bold:+.3f}% | "
              f"d_Fisher = {mean_fisher:.4f} | aniso = {aniso:.3f} | cone={dominant}")

    # === LUMINANCE HUE (9 = LMS achromatic) ===
    lum_subset = [r for r in results if r['hue_index'] == LUMINANCE_HUE]
    if lum_subset:
        lum_bold = np.mean([r['bold_v4_mean'] for r in lum_subset])
        chrom_bold_mean = np.mean(hue_bold) if hue_bold else 0
        print(f"\n  LUMINANCE (hue 9, LMS achromatic):")
        print(f"    BOLD = {lum_bold:+.3f}% (vs chromatic mean {chrom_bold_mean:+.3f}%)")
        print(f"    → PT: luminance = p=2 channel (binary operator)")
        print(f"    → Luminance/chromatic ratio: {abs(lum_bold)/max(abs(chrom_bold_mean),0.001):.2f}")

    # === CORRELATIONS (chromatic hues only) ===
    if len(hue_bold) >= 5:
        hue_bold = np.array(hue_bold)
        hue_fisher = np.array(hue_fisher)
        hue_aniso = np.array(hue_aniso)
        r_hf = np.corrcoef(hue_bold, hue_fisher)[0, 1]
        r_ha = np.corrcoef(hue_bold, hue_aniso)[0, 1]
        print(f"\n  CHROMATIC CORRELATIONS (n={len(hue_bold)} hues):")
        print(f"    r(hue-BOLD, Fisher distance) = {r_hf:.3f}")
        print(f"    r(hue-BOLD, γ-anisotropy)    = {r_ha:.3f}")
        print(f"    → r > 0.5: PT geometry shapes V4 cortical response")
        print(f"    → r ~ 0: V4 independent of simplex geometry")

        # DKL axis analysis: L-M vs S
        lm_hues = [0, 4]   # indices in hue_bold: hue 1 (0°) and hue 5 (180°)
        s_hues = [2, 6]     # hue 3 (90°) and hue 7 (270°)
        if len(hue_bold) >= 8:
            lm_resp = np.mean([hue_bold[i] for i in lm_hues])
            s_resp = np.mean([hue_bold[i] for i in s_hues])
            print(f"\n  AXIS ANALYSIS:")
            print(f"    L-M axis (hues 1,5):  BOLD = {lm_resp:+.3f}%")
            print(f"    S axis (hues 3,7):    BOLD = {s_resp:+.3f}%")
            print(f"    PT predicts: L-M > S (γ₃=0.808 > γ₇=0.595)")
            print(f"    Observed: {'L-M > S ✓' if lm_resp > s_resp else 'S > L-M ✗'}")

    return results


# ============================================================
# STEP 7: HYBRID MODEL COMPARISON
# ============================================================

def compare_with_cam02(v4_results_path=None):
    """
    Compare SCT+V4_neural vs SCT+CAM02 on COMBVD 3813 pairs.

    This function:
      1. Loads the V4 neural transfer function
      2. For each COMBVD pair, computes neural-informed features
      3. Fits regression and compares correlations
    """
    import os

    if v4_results_path is None:
        v4_results_path = os.path.join(OUTPUT_DIR, "v4_bold_response.csv")

    combvd_path = os.path.join(OUTPUT_DIR, "COMBVD_3813.csv")
    if not os.path.exists(combvd_path):
        print(f"COMBVD dataset not found at {combvd_path}")
        return

    # Load V4 response
    v4_data = {}
    with open(v4_results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['hue_index']), float(row['contrast']))
            v4_data[key] = {
                'bold': float(row['bold_v4_mean']),
                'fisher': float(row['fisher_distance']),
                'pi': np.array([float(row['sct_pi3']),
                               float(row['sct_pi5']),
                               float(row['sct_pi7'])])
            }

    print(f"\nLoaded {len(v4_data)} V4 conditions")
    print("(Full hybrid model comparison requires mapping COMBVD colors → DKL → V4)")
    print("This will be implemented once V4 data extraction is complete.")


if __name__ == "__main__":
    results = run_pipeline()

    if results:
        print("\n\n" + "=" * 60)
        print("NEXT: Hybrid model comparison")
        print("=" * 60)
        compare_with_cam02()
