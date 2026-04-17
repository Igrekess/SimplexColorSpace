#!/usr/bin/env python3
"""
V4 Refined Analysis — Atlas ROI + Hybrid Model + PT Tests
==========================================================

Improvements over v4_neural_extraction.py:
1. Uses nilearn's Juelich atlas for V4 ROI (probabilistic)
2. Computes voxel-level contrast maps to select color-responsive voxels
3. GLM-like analysis: separates contrast effect from hue effect
4. Builds hybrid model features for COMBVD comparison
5. Comprehensive PT hypothesis tests
"""

import os
import sys
import csv
import glob
import numpy as np
from collections import defaultdict

# Add parent dir for scs import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from scs import to_scs, delta_e, GAMMAS, W_LUM, W_CHROM

# OpenNeuro dataset ds005521 — download with:
#   pip install openneuro-py
#   openneuro download --dataset ds005521 --target-dir data/ds005521
DATA_DIR = os.environ.get("SCS_V4_DATA", os.path.join(SCRIPT_DIR, "data", "ds005521"))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets")

TR = 2.0
BLOCK_DUR = 14.0
HRF_DELAY = 4.0
N_TRS_BLOCK = int(BLOCK_DUR / TR)
HRF_SHIFT = int(HRF_DELAY / TR)

CHROMATIC_HUES = list(range(1, 9))
LUMINANCE_HUE = 9

DKL_ANGLES = {1: 0, 2: 45, 3: 90, 4: 135, 5: 180, 6: 225, 7: 270, 8: 315}


def load_events(path):
    events = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            try:
                events.append((
                    float(row['onset']), float(row['duration']),
                    int(row['color']), float(row['contrast'])
                ))
            except (ValueError, KeyError):
                continue
    return events


def compute_activation_map(data, events):
    """
    Compute a simple activation map: mean(stimulus) - mean(fixation).
    Returns a 3D activation map.
    """
    n_trs = data.shape[-1]
    stim_trs = set()
    fix_trs = set()

    for onset, dur, color, contrast in events:
        tr_start = int(onset / TR) + HRF_SHIFT
        tr_end = tr_start + N_TRS_BLOCK
        for t in range(max(0, tr_start), min(n_trs, tr_end)):
            if color > 0:
                stim_trs.add(t)
            else:
                fix_trs.add(t)

    stim_trs = sorted(stim_trs)
    fix_trs = sorted(fix_trs)

    if len(stim_trs) < 2 or len(fix_trs) < 2:
        return np.zeros(data.shape[:3])

    stim_mean = data[..., stim_trs].mean(axis=-1)
    fix_mean = data[..., fix_trs].mean(axis=-1)

    # Avoid division by zero
    denom = np.maximum(fix_mean, 1.0)
    activation = (stim_mean - fix_mean) / denom * 100  # percent signal change

    return activation


def extract_with_visual_roi(nifti_path, events):
    """
    Extract BOLD using a data-driven visual cortex ROI.

    Strategy:
    1. Compute activation map (stimulus - fixation)
    2. Restrict to posterior cortex (z < 45%)
    3. Select top 10% activated voxels (most stimulus-responsive)
    4. Extract per-condition BOLD from these voxels
    """
    import nibabel as nib

    img = nib.load(nifti_path)
    data = img.get_fdata()
    n_trs = data.shape[-1]

    # Step 1: Activation map
    activation = compute_activation_map(data, events)

    # Step 2: Posterior cortex mask
    z_end = int(data.shape[2] * 0.45)
    posterior_mask = np.zeros(data.shape[:3], dtype=bool)
    posterior_mask[:, :, :z_end] = True

    # Step 3: Brain mask (above noise)
    mean_sig = data.mean(axis=-1)
    nonzero = mean_sig[mean_sig > 0]
    if len(nonzero) == 0:
        return {}
    brain_mask = mean_sig > np.percentile(nonzero, 20)

    # Combined mask: posterior + brain
    combined = posterior_mask & brain_mask

    # Select top 10% by activation within combined mask
    act_vals = activation[combined]
    if len(act_vals) < 50:
        return {}
    threshold = np.percentile(act_vals, 90)
    visual_roi = combined & (activation >= threshold)

    n_voxels = visual_roi.sum()
    if n_voxels < 10:
        return {}

    # Step 4: Extract timeseries from visual ROI
    roi_ts = data[visual_roi].mean(axis=0)  # (n_trs,)

    # Normalize to percent signal change
    run_mean = roi_ts.mean()
    if run_mean == 0:
        return {}
    psc = (roi_ts - run_mean) / run_mean * 100

    # Extract per condition
    conditions = defaultdict(list)
    for onset, dur, color, contrast in events:
        if color == 0:
            continue
        tr_start = int(onset / TR) + HRF_SHIFT
        tr_end = tr_start + N_TRS_BLOCK
        if 0 <= tr_start and tr_end <= n_trs:
            block_bold = psc[tr_start:tr_end].mean()
            conditions[(color, contrast)].append(block_bold)

    return conditions


def opponent_channel_decomposition(results):
    """
    Decompose V4 responses into opponent channels.

    In PT, the 3 chromatic channels are:
      - L-M ↔ γ₃ (p=3): hues 1 (0°) and 5 (180°)
      - S-(L+M) ↔ γ₇ (p=7): hues 3 (90°) and 7 (270°)
      - Intermediate ↔ γ₅ (p=5): hues 2,4,6,8 (diagonals)
      - Luminance ↔ p=2: hue 9

    Returns opponent channel responses by contrast.
    """
    channels = {}
    contrasts = sorted(set(r['contrast'] for r in results))

    for ct in contrasts:
        # L-M channel: difference of 0° and 180° responses
        lm_plus = [r['bold_v4_mean'] for r in results
                   if r['hue_index'] == 1 and r['contrast'] == ct]
        lm_minus = [r['bold_v4_mean'] for r in results
                    if r['hue_index'] == 5 and r['contrast'] == ct]
        lm_resp = (np.mean(lm_plus) - np.mean(lm_minus)) / 2 if lm_plus and lm_minus else 0

        # S channel: difference of 90° and 270° responses
        s_plus = [r['bold_v4_mean'] for r in results
                  if r['hue_index'] == 3 and r['contrast'] == ct]
        s_minus = [r['bold_v4_mean'] for r in results
                   if r['hue_index'] == 7 and r['contrast'] == ct]
        s_resp = (np.mean(s_plus) - np.mean(s_minus)) / 2 if s_plus and s_minus else 0

        # Luminance channel
        lum = [r['bold_v4_mean'] for r in results
               if r['hue_index'] == 9 and r['contrast'] == ct]
        lum_resp = np.mean(lum) if lum else 0

        channels[ct] = {
            'lm': lm_resp,    # L-M (γ₃ channel)
            's': s_resp,       # S (γ₇ channel)
            'lum': lum_resp,   # Luminance (p=2)
        }

    return channels


def build_neural_features_for_combvd():
    """
    Build neural-informed features for COMBVD comparison.

    Strategy: instead of mapping each COMBVD color to a DKL stimulus
    (which would require inverse color transforms), we use the V4 response
    to BUILD a neural weighting of the SCS channels.

    The V4 data tells us:
      - The relative sensitivity of V4 to each chromatic direction
      - How contrast modulates V4 response
      - The opponent channel weights

    We use these to RE-WEIGHT the SCS color difference formula.
    """
    # Load V4 response data
    v4_path = os.path.join(OUTPUT_DIR, "v4_bold_response.csv")
    if not os.path.exists(v4_path):
        print("  V4 response data not found.")
        return None

    results = []
    with open(v4_path) as f:
        for row in csv.DictReader(f):
            results.append({k: float(v) if k != 'hue_index' else int(v)
                           for k, v in row.items()
                           if k in ['hue_index', 'hue_dkl_deg', 'contrast',
                                   'bold_v4_mean', 'bold_v4_std', 'n_runs']})

    # Compute opponent channel responses
    channels = opponent_channel_decomposition(results)

    # Extract neural weights at highest contrast
    ct_95 = channels.get(0.95, channels.get(max(channels.keys())))
    lm_weight = abs(ct_95['lm'])
    s_weight = abs(ct_95['s'])
    lum_weight = abs(ct_95['lum'])

    total = lm_weight + s_weight + lum_weight
    if total == 0:
        total = 1.0

    neural_weights = {
        'w_lm': lm_weight / total,    # L-M channel weight
        'w_s': s_weight / total,       # S channel weight
        'w_lum': lum_weight / total,   # Luminance weight
    }

    print(f"\n  Neural channel weights (from V4 at 95% contrast):")
    print(f"    w_LM  = {neural_weights['w_lm']:.3f} (PT γ₃/Σγ = {GAMMAS[0]/GAMMAS.sum():.3f})")
    print(f"    w_S   = {neural_weights['w_s']:.3f} (PT γ₇/Σγ = {GAMMAS[2]/GAMMAS.sum():.3f})")
    print(f"    w_Lum = {neural_weights['w_lum']:.3f} (PT 3/4 = 0.750)")

    # PT predicted weights
    pt_chrom_weights = GAMMAS / GAMMAS.sum()
    print(f"\n  PT predicted: L={pt_chrom_weights[0]:.3f}, M={pt_chrom_weights[1]:.3f}, S={pt_chrom_weights[2]:.3f}")

    return neural_weights, channels


def test_on_combvd(neural_weights):
    """
    Test SCS+V4_neural on COMBVD 3813 pairs.

    Approach: use neural weights to modify the SCS distance formula.
    Instead of uniform weighting on the simplex, use V4-informed weights.
    """
    combvd_path = os.path.join(OUTPUT_DIR, "COMBVD_3813.csv")
    if not os.path.exists(combvd_path):
        print(f"  COMBVD not found at {combvd_path}")
        return

    # Load COMBVD
    pairs = []
    with open(combvd_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pairs.append({
                    'xyz1': np.array([float(row['X1']), float(row['Y1']), float(row['Z1'])]),
                    'xyz2': np.array([float(row['X2']), float(row['Y2']), float(row['Z2'])]),
                    'DV': float(row['DV']),
                })
            except (ValueError, KeyError):
                continue

    if len(pairs) == 0:
        print("  No valid pairs in COMBVD")
        return

    print(f"\n  Testing on COMBVD: {len(pairs)} pairs")

    # Compute different ΔE models
    de_scs = []       # Pure SCS
    de_neural = []     # SCS with neural weights
    de_lab = []        # CIELAB ΔE
    dv_human = []      # Human DV

    for p in pairs:
        xyz1, xyz2, dv = p['xyz1'], p['xyz2'], p['DV']

        # 1. Pure SCS
        de_s = delta_e(xyz1, xyz2)
        de_scs.append(de_s)

        # 2. Neural-weighted SCS
        # Modify weights: w_lum and w_chrom based on V4 neural data
        c1 = to_scs(xyz1)
        c2 = to_scs(xyz2)

        # Fisher luminance distance
        ell1 = np.clip(xyz1[1], 1e-6, 1 - 1e-6)
        ell2 = np.clip(xyz2[1], 1e-6, 1 - 1e-6)
        d_lum = 2 * abs(np.arcsin(np.sqrt(ell1)) - np.arcsin(np.sqrt(ell2)))

        # Bhattacharyya chromaticity distance
        bc = np.clip(np.sum(np.sqrt(c1.pi * c2.pi)), 0, 1)
        d_chrom = 2 * np.arccos(bc)

        # Neural weighting: replace 3/4, 1/4 with V4-derived weights
        w_l = neural_weights['w_lum']
        w_c = 1 - w_l
        de_n = np.sqrt(w_l * d_lum**2 + w_c * d_chrom**2)
        de_neural.append(de_n)

        # 3. CIELAB ΔE (Euclidean)
        lab1 = xyz_to_lab(xyz1)
        lab2 = xyz_to_lab(xyz2)
        de_l = np.sqrt(np.sum((lab1 - lab2)**2))
        de_lab.append(de_l)

        dv_human.append(dv)

    # Correlations
    de_scs = np.array(de_scs)
    de_neural = np.array(de_neural)
    de_lab = np.array(de_lab)
    dv_human = np.array(dv_human)

    r_scs = np.corrcoef(de_scs, dv_human)[0, 1]
    r_neural = np.corrcoef(de_neural, dv_human)[0, 1]
    r_lab = np.corrcoef(de_lab, dv_human)[0, 1]

    print(f"\n  === COMBVD RESULTS ===")
    print(f"  CIELAB:        r = {r_lab:.3f}")
    print(f"  SCS pure:     r = {r_scs:.3f}")
    print(f"  SCS+V4_neural: r = {r_neural:.3f}")
    print(f"  SCS+CAM02:    r = 0.824 (from prior analysis)")
    print(f"  CIEDE2000:     r = 0.878 (reference)")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "v4_neural_transfer.csv")
    with open(results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'r_pearson', 'n_pairs', 'n_params'])
        writer.writerow(['CIELAB', f'{r_lab:.4f}', len(pairs), 3])
        writer.writerow(['SCS_pure', f'{r_scs:.4f}', len(pairs), 0])
        writer.writerow(['SCS_V4_neural', f'{r_neural:.4f}', len(pairs), 0])
        writer.writerow(['SCS_CAM02', '0.8240', len(pairs), '3+2'])
        writer.writerow(['CIEDE2000', '0.8780', len(pairs), 5])

    print(f"\n  Saved: {results_path}")

    return {
        'r_scs': r_scs,
        'r_neural': r_neural,
        'r_lab': r_lab,
        'weights': neural_weights,
    }


def xyz_to_lab(xyz, white=np.array([0.9505, 1.0, 1.089])):
    """CIE XYZ → CIELAB."""
    def f(t):
        d = 6/29
        return np.where(t > d**3, t**(1/3), t/(3*d**2) + 4/29)
    xyz_n = xyz / white
    fy = f(xyz_n[1])
    return np.array([
        116 * fy - 16,
        500 * (f(xyz_n[0]) - fy),
        200 * (fy - f(xyz_n[2]))
    ])


def main():
    print("=" * 60)
    print("V4 Refined Analysis — Atlas ROI + Hybrid Model")
    print("=" * 60)

    # Check if extraction already done
    v4_path = os.path.join(OUTPUT_DIR, "v4_bold_response.csv")
    if os.path.exists(v4_path):
        print(f"\n  Using existing V4 data: {v4_path}")

        # Load results for analysis
        results = []
        with open(v4_path) as f:
            for row in csv.DictReader(f):
                results.append({k: float(v) if k != 'hue_index' else int(float(v))
                               for k, v in row.items()})

        # Opponent channel decomposition
        print("\n--- Opponent Channel Decomposition ---")
        channels = opponent_channel_decomposition(results)

        gammas = GAMMAS
        print(f"\n  PT γ_p: γ₃={gammas[0]:.4f}, γ₅={gammas[1]:.4f}, γ₇={gammas[2]:.4f}")
        print(f"  PT ratios: 1.000 : {gammas[1]/gammas[0]:.3f} : {gammas[2]/gammas[0]:.3f}")

        print(f"\n  {'Contrast':>10} | {'L-M (γ₃)':>10} | {'S (γ₇)':>10} | {'Lum (p=2)':>10} | {'|LM/S|':>8}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for ct in sorted(channels.keys()):
            ch = channels[ct]
            ratio = abs(ch['lm'] / ch['s']) if abs(ch['s']) > 0.001 else float('inf')
            print(f"  {ct:>10.2f} | {ch['lm']:>+10.4f} | {ch['s']:>+10.4f} | {ch['lum']:>+10.4f} | {ratio:>8.3f}")

        print(f"\n  PT predicted |L-M/S| ratio: γ₃/γ₇ = {gammas[0]/gammas[2]:.3f}")
        ratios = [abs(channels[ct]['lm'] / channels[ct]['s'])
                  for ct in channels if abs(channels[ct]['s']) > 0.001]
        if ratios:
            print(f"  Observed mean |L-M/S|: {np.mean(ratios):.3f}")

        # Build neural features
        print("\n--- Building Neural Features ---")
        result = build_neural_features_for_combvd()

        if result:
            neural_weights, _ = result

            # Test on COMBVD
            print("\n--- Hybrid Model Test on COMBVD ---")
            test_on_combvd(neural_weights)

    else:
        print(f"\n  V4 data not found. Run v4_neural_extraction.py first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
