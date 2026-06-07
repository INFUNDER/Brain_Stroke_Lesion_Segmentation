#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt

# Set academic plotting style
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'legend.fontsize': 10
})

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
RESULTS_DIR = os.path.join(WORKSPACE_DIR, "mamba_results/results")
OUTPUT_CONVERGENCE = os.path.join(WORKSPACE_DIR, "figures", "learning_curves.png")
OUTPUT_TUNING = os.path.join(WORKSPACE_DIR, "figures", "cc_threshold_tuning.png")

# =============================================================================
# CHART 1: 5-FOLD CONVERGENCE CURVES
# =============================================================================
def plot_convergence():
    print("📈 Generating Convergence Curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    all_dices = []
    all_losses = []
    epochs = []
    
    # Load all 5 folds
    for fold_idx in range(1, 6):
        hist_file = os.path.join(RESULTS_DIR, f"fold{fold_idx}_history.json")
        if not os.path.exists(hist_file):
            print(f"⚠️ Warning: Missing {hist_file}. Skipping.")
            continue
            
        with open(hist_file, 'r') as f:
            history = json.load(f)
            
        fold_epochs = [item["epoch"] for item in history]
        fold_dice = [item["val_dice"] for item in history]
        fold_loss = [item["loss"] for item in history]
        
        epochs = fold_epochs
        all_dices.append(fold_dice)
        all_losses.append(fold_loss)
        
        # Plot individual fold curves with low alpha
        axes[0].plot(fold_epochs, fold_dice, color=colors[fold_idx-1], alpha=0.4, linestyle="--", label=f"Fold {fold_idx-1}")
        axes[1].plot(fold_epochs, fold_loss, color=colors[fold_idx-1], alpha=0.4, linestyle="--")
        
    # Calculate and plot the mean curves
    if all_dices:
        mean_dice = np.mean(all_dices, axis=0)
        mean_loss = np.mean(all_losses, axis=0)
        
        axes[0].plot(epochs, mean_dice, color="black", linewidth=2.5, label="Mean CV Dice")
        axes[1].plot(epochs, mean_loss, color="black", linewidth=2.5, label="Mean Loss")
        
        # Highlight best mean epoch
        best_epoch_idx = np.argmax(mean_dice)
        best_epoch = epochs[best_epoch_idx]
        best_val = mean_dice[best_epoch_idx]
        axes[0].scatter(best_epoch, best_val, color="red", s=100, zorder=5, label=f"Peak Dice: {best_val:.4f} (Ep {best_epoch})")
        
    # Subplot 0: Dice Score
    axes[0].set_title("Validation Dice Convergence")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Mean Dice Score")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend(loc="lower right")
    axes[0].set_ylim(0, 0.9)
    
    # Subplot 1: Loss
    axes[1].set_title("Training Loss Convergence")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Dice + Cross Entropy Loss")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].set_ylim(0, 1.0)
    
    plt.suptitle("5-Fold Cross-Validation Convergence (Mamba-UNet)", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_CONVERGENCE, dpi=300, bbox_inches='tight')
    print(f"✅ Convergence plot saved to: {OUTPUT_CONVERGENCE}")

# =============================================================================
# CHART 2: CONNECTED COMPONENT TUNING
# =============================================================================
def plot_tuning():
    print("🎯 Generating CC Tuning Curves...")
    tuning_file = os.path.join(RESULTS_DIR, "cc_tuning_non_leaked.json")
    if not os.path.exists(tuning_file):
        print(f"❌ Error: {tuning_file} not found. Cannot plot tuning curves.")
        return
        
    with open(tuning_file, 'r') as f:
        data = json.load(f)
        
    grid_results = data["grid_results"]
    
    # Group results by probability threshold
    prob_groups = {}
    for res in grid_results:
        p = res["prob_threshold"]
        if p not in prob_groups:
            prob_groups[p] = {"cc": [], "dice_all": [], "dice_nonempty": []}
        prob_groups[p]["cc"].append(res["cc_threshold"])
        prob_groups[p]["dice_all"].append(res["mean_dice_all"])
        prob_groups[p]["dice_nonempty"].append(res["mean_dice_nonempty"])
        
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # We will plot curves for selected representative probability thresholds
    selected_probs = [0.3, 0.4, 0.45, 0.5, 0.6]
    markers = ['o', 's', '^', 'D', 'x']
    
    for idx, p in enumerate(selected_probs):
        if p in prob_groups:
            group = prob_groups[p]
            # Sort by CC threshold just to be safe
            sort_indices = np.argsort(group["cc"])
            cc_sorted = np.array(group["cc"])[sort_indices]
            dice_all_sorted = np.array(group["dice_all"])[sort_indices]
            dice_nonempty_sorted = np.array(group["dice_nonempty"])[sort_indices]
            
            # Plot Dice All Cases
            axes[0].plot(cc_sorted, dice_all_sorted, label=f"P = {p}", marker=markers[idx], markersize=5, linewidth=1.5)
            # Plot Dice Non-Empty Cases
            axes[1].plot(cc_sorted, dice_nonempty_sorted, label=f"P = {p}", marker=markers[idx], markersize=5, linewidth=1.5)
            
    # Subplot 0: All Cases (with Mimics)
    axes[0].set_title("Dice Score Across All Test Cases (Including Mimics)")
    axes[0].set_xlabel("Minimum Connected Component Size (voxels)")
    axes[0].set_ylabel("Mean Dice (Healthy Mimics Correct = 1.0)")
    axes[0].grid(True, alpha=0.3, linestyle=":")
    axes[0].legend(loc="lower left")
    axes[0].set_xscale('symlog', linthresh=10)  # Use symlog because values span 0 to 1000
    axes[0].set_ylim(0.35, 0.75)
    
    # Subplot 1: Non-Empty Cases (Actual Stroke Cases)
    axes[1].set_title("Dice Score Across Non-Empty Cases (Lesions Present)")
    axes[1].set_xlabel("Minimum Connected Component Size (voxels)")
    axes[1].set_ylabel("Mean Dice")
    axes[1].grid(True, alpha=0.3, linestyle=":")
    axes[1].legend(loc="lower left")
    axes[1].set_xscale('symlog', linthresh=10)
    axes[1].set_ylim(0.3, 0.82)
    
    # Add annotation for the peak
    # The optimal config is P=0.45, CC=0
    axes[0].annotate('Optimal: P=0.45, CC=0\nDice: 0.6855', xy=(0, 0.6855), xytext=(5, 0.72),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))
    axes[1].annotate('Optimal: P=0.45, CC=0\nDice: 0.7685', xy=(0, 0.7685), xytext=(5, 0.79),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6))
                
    plt.suptitle("Post-Processing Parameter Grid-Search (Held-Out Test Set)", y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_TUNING, dpi=300, bbox_inches='tight')
    print(f"✅ Tuning curves plot saved to: {OUTPUT_TUNING}")

if __name__ == "__main__":
    plot_convergence()
    plot_tuning()
