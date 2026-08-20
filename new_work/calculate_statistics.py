#!/usr/bin/env python3
import os
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "mamba_results")
RESULT_DIR = os.path.join(OUTPUT_DIR, "results")
CACHE_DIR = os.path.join(OUTPUT_DIR, "non_leaked_probs")
TEST_PATHS = os.path.join(OUTPUT_DIR, "test_set_paths.txt")
UNETR_JSON = os.path.join(RESULT_DIR, "unetr_test_results.json")
FIGURES_DIR = os.path.join(WORKSPACE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

VOXEL_VOL_ML = (1.5 * 1.5 * 2.0) / 1000.0 # 0.0045 mL

def compute_dice(pred, gt):
    sum_pred = np.sum(pred)
    sum_gt = np.sum(gt)
    if sum_gt == 0:
        return 1.0 if sum_pred == 0 else 0.0
    if sum_pred == 0:
        return 0.0
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (sum_pred + sum_gt)

def bootstrap_ci(data, num_bootstraps=10000, ci=0.95):
    means = []
    n = len(data)
    np.random.seed(42)
    for _ in range(num_bootstraps):
        resample = np.random.choice(data, size=n, replace=True)
        means.append(np.mean(resample))
    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100
    return np.percentile(means, lower_pct), np.percentile(means, upper_pct)

def main():
    print("📊 Loading results and computing comparative statistics...")

    # 1. Load UNETR results
    if not os.path.exists(UNETR_JSON):
        print(f"❌ Error: UNETR results file not found at {UNETR_JSON}. Please run evaluate_unetr_test.py first.")
        return
    
    with open(UNETR_JSON, 'r') as f:
        unetr_data = json.load(f)
    
    unetr_cases = {r["case_name"]: r for r in unetr_data["results"]}

    # 2. Load test case paths
    test_files = []
    with open(TEST_PATHS) as f:
        for line in f:
            parts = line.strip().split("  |  ")
            if len(parts) == 2:
                test_files.append({"image": parts[0].strip(), "label": parts[1].strip()})

    # 3. Compute Mamba-UNet results at optimal P=0.45
    mamba_results = {}
    print("Evaluating Mamba-UNet predictions at optimal P=0.45...")
    for idx, item in enumerate(test_files):
        case_name = os.path.basename(item["label"]).split(".nii")[0]
        prob_fg = np.load(os.path.join(CACHE_DIR, f"{case_name}_prob.npy"))
        gt_np = np.load(os.path.join(CACHE_DIR, f"{case_name}_gt.npy"))

        pred_binary = (prob_fg >= 0.45).astype(np.uint8)
        corrected_dice = compute_dice(pred_binary, gt_np)
        
        pred_fg_voxels = np.sum(pred_binary > 0)
        gt_fg_voxels = np.sum(gt_np > 0)
        
        pred_vol_ml = pred_fg_voxels * VOXEL_VOL_ML
        gt_vol_ml = gt_fg_voxels * VOXEL_VOL_ML

        mamba_results[case_name] = {
            "corrected_dice": corrected_dice,
            "pred_vol_ml": pred_vol_ml,
            "gt_vol_ml": gt_vol_ml
        }

    # 4. Print comparative table
    print("\n" + "="*95)
    print(f"{'Case Name':<25} | {'GT Vol (mL)':<12} | {'UNETR Dice':<10} | {'Mamba Dice':<10} | {'UNETR Vol':<10} | {'Mamba Vol':<10}")
    print("="*95)
    
    unetr_dices = []
    mamba_dices = []
    
    gt_vols = []
    unetr_vols = []
    mamba_vols = []
    
    for case_name in sorted(mamba_results.keys()):
        m_res = mamba_results[case_name]
        u_res = unetr_cases[case_name]
        
        unetr_dices.append(u_res["corrected_dice"])
        mamba_dices.append(m_res["corrected_dice"])
        
        gt_vols.append(m_res["gt_vol_ml"])
        unetr_vols.append(u_res["pred_vol_ml"])
        mamba_vols.append(m_res["pred_vol_ml"])
        
        print(f"{case_name:<25} | {m_res['gt_vol_ml']:<12.2f} | {u_res['corrected_dice']:<10.4f} | {m_res['corrected_dice']:<10.4f} | {u_res['pred_vol_ml']:<10.2f} | {m_res['pred_vol_ml']:<10.2f}")
    print("="*95)

    # 5. Wilcoxon Signed-Rank Test on Dice Scores
    wilc_dice = stats.wilcoxon(mamba_dices, unetr_dices, alternative='greater')
    print(f"\nWilcoxon signed-rank test (Mamba-UNet > UNETR) p-value: {wilc_dice.pvalue:.6f}")

    # 6. Bootstrap Confidence Intervals
    u_ci_l, u_ci_u = bootstrap_ci(unetr_dices)
    m_ci_l, m_ci_u = bootstrap_ci(mamba_dices)
    print(f"UNETR Mean corrected Dice      : {np.mean(unetr_dices):.4f} (95% Bootstrap CI: [{u_ci_l:.4f}, {u_ci_u:.4f}])")
    print(f"Mamba-UNet Mean corrected Dice : {np.mean(mamba_dices):.4f} (95% Bootstrap CI: [{m_ci_l:.4f}, {m_ci_u:.4f}])")

    # 7. Volume Correlation
    # Pearson
    u_pearson_r, u_pearson_p = stats.pearsonr(gt_vols, unetr_vols)
    m_pearson_r, m_pearson_p = stats.pearsonr(gt_vols, mamba_vols)
    # Spearman
    u_spearman_r, u_spearman_p = stats.spearmanr(gt_vols, unetr_vols)
    m_spearman_r, m_spearman_p = stats.spearmanr(gt_vols, mamba_vols)
    
    # Absolute Volume Discrepancy
    u_abs_err = np.abs(np.array(unetr_vols) - np.array(gt_vols))
    m_abs_err = np.abs(np.array(mamba_vols) - np.array(gt_vols))
    wilc_vol = stats.wilcoxon(m_abs_err, u_abs_err, alternative='less')
    
    print("\nVolume Delineation Accuracy:")
    print(f"  UNETR:      Pearson r = {u_pearson_r:.4f} (p={u_pearson_p:.6f}) | Spearman rho = {u_spearman_r:.4f}")
    print(f"  Mamba-UNet: Pearson r = {m_pearson_r:.4f} (p={m_pearson_p:.6f}) | Spearman rho = {m_spearman_r:.4f}")
    print(f"  Mean Absolute Volume Discrepancy: UNETR = {np.mean(u_abs_err):.2f} mL, Mamba-UNet = {np.mean(m_abs_err):.2f} mL")
    print(f"  Wilcoxon test on volume discrepancy (Mamba-UNet < UNETR) p-value: {wilc_vol.pvalue:.6f}")

    # 8. Clinical Decision Concordance (>70 mL Mechanical Thrombectomy rule)
    print("\nClinical Decision Concordance (>70 mL Infarct Core Volume):")
    # GT positive if >70 mL
    gt_thrombectomy = np.array(gt_vols) > 70.0
    u_thrombectomy = np.array(unetr_vols) > 70.0
    m_thrombectomy = np.array(mamba_vols) > 70.0
    
    def eval_clinical_concordance(pred, gt):
        tp = np.sum((pred == True) & (gt == True))
        tn = np.sum((pred == False) & (gt == False))
        fp = np.sum((pred == True) & (gt == False))
        fn = np.sum((pred == False) & (gt == True))
        accuracy = (tp + tn) / len(gt)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
        return accuracy, sens, spec, tp, tn, fp, fn

    u_acc, u_sens, u_spec, u_tp, u_tn, u_fp, u_fn = eval_clinical_concordance(u_thrombectomy, gt_thrombectomy)
    m_acc, m_sens, m_spec, m_tp, m_tn, m_fp, m_fn = eval_clinical_concordance(m_thrombectomy, gt_thrombectomy)

    print(f"  UNETR Baseline Concordance:  Accuracy = {u_acc*100:.1f}% | Sens = {u_sens*100:.1f}% | Spec = {u_spec*100:.1f}% (TP={u_tp}, TN={u_tn}, FP={u_fp}, FN={u_fn})")
    print(f"  Mamba-UNet Concordance:      Accuracy = {m_acc*100:.1f}% | Sens = {m_sens*100:.1f}% | Spec = {m_spec*100:.1f}% (TP={m_tp}, TN={m_tn}, FP={m_fp}, FN={m_fn})")

    # 9. Bland-Altman Plot for Mamba-UNet
    print("\n📈 Generating Bland-Altman Volume Discrepancy Plot...")
    
    # Calculate means and differences
    means = (np.array(gt_vols) + np.array(mamba_vols)) / 2.0
    diffs = np.array(mamba_vols) - np.array(gt_vols) # Prediction - Ground Truth
    
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff
    
    plt.figure(figsize=(8, 6))
    plt.scatter(means, diffs, color='#1f77b4', edgecolors='k', alpha=0.8, s=60, label="Subject Cases")
    
    # Add horizontal lines
    plt.axhline(mean_diff, color='r', linestyle='-', linewidth=2, label=f"Mean Bias ({mean_diff:.2f} mL)")
    plt.axhline(loa_upper, color='g', linestyle='--', linewidth=1.5, label=f"+1.96 SD Limit ({loa_upper:.2f} mL)")
    plt.axhline(loa_lower, color='g', linestyle='--', linewidth=1.5, label=f"-1.96 SD Limit ({loa_lower:.2f} mL)")
    plt.axhline(0, color='gray', linestyle=':', linewidth=1)
    
    plt.title("Bland-Altman Plot: Mamba-UNet Infarct Volume Delineation", fontsize=14, pad=15)
    plt.xlabel("Mean Core Volume: (Ground Truth + Prediction) / 2 [mL]", fontsize=12)
    plt.ylabel("Volume Discrepancy: Prediction - Ground Truth [mL]", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle=":")
    plt.legend(loc="upper right", fontsize=10)
    
    out_img = os.path.join(FIGURES_DIR, "bland_altman.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Bland-Altman plot saved to {out_img}")

    # 10. Save stats to JSON for LaTeX retrieval
    stats_out = os.path.join(RESULT_DIR, "comparative_statistics.json")
    with open(stats_out, "w") as f:
        json.dump({
            "unetr": {
                "mean_dice": float(np.mean(unetr_dices)),
                "ci_lower": float(u_ci_l),
                "ci_upper": float(u_ci_u),
                "pearson_r": float(u_pearson_r),
                "pearson_p": float(u_pearson_p),
                "spearman_rho": float(u_spearman_r),
                "mean_abs_err": float(np.mean(u_abs_err)),
                "clinical_acc": float(u_acc),
                "clinical_sens": float(u_sens),
                "clinical_spec": float(u_spec)
            },
            "mamba": {
                "mean_dice": float(np.mean(mamba_dices)),
                "ci_lower": float(m_ci_l),
                "ci_upper": float(m_ci_u),
                "pearson_r": float(m_pearson_r),
                "pearson_p": float(m_pearson_p),
                "spearman_rho": float(m_spearman_r),
                "mean_abs_err": float(np.mean(m_abs_err)),
                "clinical_acc": float(m_acc),
                "clinical_sens": float(m_sens),
                "clinical_spec": float(m_spec)
            },
            "wilcoxon_dice_p": float(wilc_dice.pvalue),
            "wilcoxon_vol_p": float(wilc_vol.pvalue),
            "bland_altman": {
                "bias": float(mean_diff),
                "loa_upper": float(loa_upper),
                "loa_lower": float(loa_lower)
            }
        }, f, indent=2)
    print(f"Comparative stats JSON saved to {stats_out}")

if __name__ == "__main__":
    main()
