#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

WORKSPACE_DIR = "/home/ronit.28010/Brain_Stroke"
OUTPUT_UNETR = os.path.join(WORKSPACE_DIR, "unetr_architecture.png")
OUTPUT_MAMBA = os.path.join(WORKSPACE_DIR, "mamba_unet_architecture.png")

# Set font styling for diagrams
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10,
    'axes.titlesize': 12
})

def draw_unetr():
    print("🎨 Drawing UNETR Architecture Schematic...")
    # Initialize Canvas
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    
    # Colors
    c_input = "#D0021B" # Red
    c_conv = "#4A90E2"  # Blue
    c_vit = "#BD10E0"   # Purple
    c_dec = "#50E3C2"   # Teal
    c_arrow = "#4A4A4A"
    
    # 1. Input Volume
    ax.add_patch(patches.Rectangle((2, 18), 6, 8, facecolor=c_input, alpha=0.8, edgecolor="black", linewidth=1.5))
    ax.text(5, 22, "Input\n(3D MRI)\n3xHxWxD", ha="center", va="center", color="white", weight="bold", fontsize=9)
    
    # Arrow to Patch Embedding
    ax.annotate("", xy=(12, 22), xytext=(8, 22), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    
    # 2. Patch & Position Embedding
    ax.add_patch(patches.Rectangle((12, 16), 8, 12, facecolor="#F5A623", alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(16, 22, "3D Patch\nEmbedding\n&\nPosition\nEmbed", ha="center", va="center", color="white", weight="bold", fontsize=9)
    
    # Arrow to Transformer Encoder
    ax.annotate("", xy=(24, 22), xytext=(20, 22), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    
    # 3. Transformer Encoder (stacked blocks)
    for i in range(4):
        x_offset = 24 + i * 8
        ax.add_patch(patches.Rectangle((x_offset, 14), 6, 16, facecolor=c_vit, alpha=0.75, edgecolor="black", lw=1.2))
        ax.text(x_offset + 3, 22, f"ViT Block\nLayer {i*3+3}", ha="center", va="center", color="white", weight="bold", fontsize=8)
        if i < 3:
            ax.annotate("", xy=(x_offset + 8, 22), xytext=(x_offset + 6, 22), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
            
    # Arrow out of Encoder
    ax.annotate("", xy=(58, 22), xytext=(54, 22), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    
    # 4. Encoder skip projections (Deconvolutions/Convolutions)
    # Skip 3
    ax.add_patch(patches.Rectangle((24, 34), 6, 6, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(27, 37, "Proj 3D\n(Deconv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(27, 34), xytext=(27, 30), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip 6
    ax.add_patch(patches.Rectangle((32, 34), 6, 6, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(35, 37, "Proj 3D\n(Deconv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(35, 34), xytext=(35, 30), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip 9
    ax.add_patch(patches.Rectangle((40, 34), 6, 6, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(43, 37, "Proj 3D\n(Deconv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(43, 34), xytext=(43, 30), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # 5. Decoder Blocks (Teal)
    # Dec 4 (bottleneck output reconstruction)
    ax.add_patch(patches.Rectangle((58, 17), 7, 10, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(61.5, 22, "Decoder 4\n(Up-Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    
    # Dec 3
    ax.add_patch(patches.Rectangle((68, 22), 7, 10, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(71.5, 27, "Decoder 3\n(Up-Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(68, 27), xytext=(65, 22), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2, connectionstyle="arc3,rad=-0.2"))
    
    # Dec 2
    ax.add_patch(patches.Rectangle((78, 27), 7, 10, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(81.5, 32, "Decoder 2\n(Up-Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(78, 32), xytext=(75, 27), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2, connectionstyle="arc3,rad=-0.2"))
    
    # Dec 1
    ax.add_patch(patches.Rectangle((88, 32), 7, 10, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(91.5, 37, "Decoder 1\n(Up-Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(88, 37), xytext=(85, 32), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2, connectionstyle="arc3,rad=-0.2"))
    
    # Skip connections to decoders
    # Skip 9 -> Dec 4
    ax.annotate("", xy=(58, 25), xytext=(46, 37), arrowprops=dict(arrowstyle="->", color="#7ED321", lw=1.5, ls="--", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    # Skip 6 -> Dec 3
    ax.annotate("", xy=(68, 30), xytext=(38, 37), arrowprops=dict(arrowstyle="->", color="#7ED321", lw=1.5, ls="--", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    # Skip 3 -> Dec 2
    ax.annotate("", xy=(78, 35), xytext=(30, 37), arrowprops=dict(arrowstyle="->", color="#7ED321", lw=1.5, ls="--", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    
    # Arrow from Dec 1 to output
    ax.annotate("", xy=(98, 37), xytext=(95, 37), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    
    # Output Map
    ax.add_patch(patches.Rectangle((98, 34), 1.5, 6, facecolor=c_input, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(98.7, 37, "Out\n(Mask)", ha="center", va="center", color="white", weight="bold", fontsize=6, rotation=90)
    
    # Legend
    legend_handles = [
        patches.Patch(color=c_input, label="Input/Output Tensors", alpha=0.8),
        patches.Patch(color=c_conv, label="Convolutional Layers / Projections", alpha=0.8),
        patches.Patch(color=c_vit, label="Transformer Encoder blocks (Self-Attention)", alpha=0.75),
        patches.Patch(color=c_dec, label="3D Convolutional Decoder blocks", alpha=0.8)
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=True, fontsize=9)
    
    plt.title("UNETR (Transformer-based 3D Medical Image Segmentation) Architecture", fontsize=14, pad=15)
    plt.savefig(OUTPUT_UNETR, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ UNETR Architecture diagram saved to: {OUTPUT_UNETR}")

def draw_mamba_unet():
    print("🎨 Drawing Mamba-UNet Architecture Schematic...")
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 65)
    ax.axis("off")
    
    # Colors
    c_input = "#D0021B" # Red
    c_conv = "#4A90E2"  # Blue
    c_mamba = "#F5A623" # Orange
    c_dec = "#50E3C2"   # Teal
    c_arrow = "#4A4A4A"
    
    # =========================================================================
    # PANEL A: U-Net Backbone
    # =========================================================================
    # Encoder 1 (Conv)
    ax.add_patch(patches.Rectangle((2, 45), 6, 8, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(5, 49, "Encoder 1\n(3D Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    
    # Encoder 2 (Conv)
    ax.add_patch(patches.Rectangle((12, 37), 6, 8, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(15, 41, "Encoder 2\n(3D Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(12, 41), xytext=(8, 49), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Encoder 3 (Conv)
    ax.add_patch(patches.Rectangle((22, 29), 6, 8, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(25, 33, "Encoder 3\n(3D Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(22, 33), xytext=(18, 41), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Encoder 4 (Conv)
    ax.add_patch(patches.Rectangle((32, 21), 6, 8, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(35, 25, "Encoder 4\n(3D Conv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(32, 25), xytext=(28, 33), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Bottleneck (Mamba Block)
    ax.add_patch(patches.Rectangle((42, 13), 8, 8, facecolor=c_mamba, alpha=0.8, edgecolor="black", lw=1.5))
    ax.text(46, 17, "Bottleneck\n(OmniMamba\nBlock x2)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(42, 17), xytext=(38, 25), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Decoders (Teal)
    # Decoder 4
    ax.add_patch(patches.Rectangle((54, 21), 6, 8, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(57, 25, "Decoder 4\n(3D UpConv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(54, 25), xytext=(50, 17), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Decoder 3
    ax.add_patch(patches.Rectangle((64, 29), 6, 8, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(67, 33, "Decoder 3\n(3D UpConv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(64, 33), xytext=(60, 25), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Decoder 2
    ax.add_patch(patches.Rectangle((74, 37), 6, 8, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(77, 41, "Decoder 2\n(3D UpConv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(74, 41), xytext=(70, 33), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Decoder 1
    ax.add_patch(patches.Rectangle((84, 45), 6, 8, facecolor=c_dec, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(87, 49, "Decoder 1\n(3D UpConv)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(84, 49), xytext=(80, 41), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip connections with Mamba Filters (dashed lines)
    # Skip 4 Filter
    ax.add_patch(patches.Rectangle((44, 23), 6, 4, facecolor=c_mamba, alpha=0.7, edgecolor="black", lw=1.0))
    ax.text(47, 25, "OmniMamba", ha="center", va="center", color="white", weight="bold", fontsize=7)
    ax.annotate("", xy=(44, 25), xytext=(38, 25), arrowprops=dict(arrowstyle="-", color=c_arrow, lw=1.0, ls="-."))
    ax.annotate("", xy=(54, 25), xytext=(50, 25), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip 3 Filter
    ax.add_patch(patches.Rectangle((44, 31), 6, 4, facecolor=c_mamba, alpha=0.7, edgecolor="black", lw=1.0))
    ax.text(47, 33, "OmniMamba", ha="center", va="center", color="white", weight="bold", fontsize=7)
    ax.annotate("", xy=(44, 33), xytext=(28, 33), arrowprops=dict(arrowstyle="-", color=c_arrow, lw=1.0, ls="-."))
    ax.annotate("", xy=(64, 33), xytext=(50, 33), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip 2 Filter
    ax.add_patch(patches.Rectangle((44, 39), 6, 4, facecolor=c_mamba, alpha=0.7, edgecolor="black", lw=1.0))
    ax.text(47, 41, "OmniMamba", ha="center", va="center", color="white", weight="bold", fontsize=7)
    ax.annotate("", xy=(44, 41), xytext=(18, 41), arrowprops=dict(arrowstyle="-", color=c_arrow, lw=1.0, ls="-."))
    ax.annotate("", xy=(74, 41), xytext=(50, 41), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Skip 1 (Direct to Decoder 1)
    ax.annotate("", xy=(84, 49), xytext=(8, 49), arrowprops=dict(arrowstyle="->", color="#7ED321", lw=1.2, ls="--", connectionstyle="arc3,rad=-0.1"))
    
    # Output Map
    ax.add_patch(patches.Rectangle((94, 45), 2, 8, facecolor=c_input, alpha=0.8, edgecolor="black", lw=1.2))
    ax.text(95, 49, "Out", ha="center", va="center", color="white", weight="bold", fontsize=8)
    ax.annotate("", xy=(94, 49), xytext=(90, 49), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.5))
    
    # Title for Panel A
    ax.text(5, 59, "A. 3D Mamba-UNet Architecture", weight="bold", fontsize=11, color="black")
    
    # =========================================================================
    # PANEL B: OmniMamba Layer Details
    # =========================================================================
    y_b = 3
    ax.text(5, y_b + 7, "B. OmniMamba Tri-Planar Selective Scanning Layer Detail", weight="bold", fontsize=11, color="black")
    
    # Main box for OmniMamba
    ax.add_patch(patches.Rectangle((5, y_b), 90, 6, facecolor="none", edgecolor="#4A4A4A", lw=1.5, ls="--"))
    
    # Inputs
    ax.text(12, y_b + 3, "Input (x)\n(b, c, h, w, d)", ha="center", va="center", weight="bold", fontsize=8)
    
    # Four parallel paths (drawn as arrows branching)
    ax.annotate("", xy=(24, y_b + 5), xytext=(18, y_b + 3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    ax.annotate("", xy=(24, y_b + 3.8), xytext=(18, y_b + 3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    ax.annotate("", xy=(24, y_b + 2.2), xytext=(18, y_b + 3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    ax.annotate("", xy=(24, y_b + 1), xytext=(18, y_b + 3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    
    # Blocks in parallel
    # 1. Z-axis Scan
    ax.add_patch(patches.Rectangle((24, y_b + 4.5), 18, 1.0, facecolor=c_mamba, alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(33, y_b + 5.0, "Z-Scan Mamba (S6)", ha="center", va="center", color="white", weight="bold", fontsize=7)
    
    # 2. Y-axis Scan
    ax.add_patch(patches.Rectangle((24, y_b + 3.3), 18, 1.0, facecolor=c_mamba, alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(33, y_b + 3.8, "Y-Scan Mamba (S6)", ha="center", va="center", color="white", weight="bold", fontsize=7)
    
    # 3. X-axis Scan
    ax.add_patch(patches.Rectangle((24, y_b + 1.7), 18, 1.0, facecolor=c_mamba, alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(33, y_b + 2.2, "X-Scan Mamba (S6)", ha="center", va="center", color="white", weight="bold", fontsize=7)
    
    # 4. Local Conv Branch
    ax.add_patch(patches.Rectangle((24, y_b + 0.5), 18, 1.0, facecolor=c_conv, alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(33, y_b + 1.0, "3D Local Conv (3x3x3)", ha="center", va="center", color="white", weight="bold", fontsize=7)
    
    # Addition node (drawn as circle +)
    ax.add_patch(patches.Circle((52, y_b + 3), 1.2, facecolor="#BD10E0", alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(52, y_b + 3, "+", ha="center", va="center", color="white", weight="bold", fontsize=12)
    
    # Connect parallel blocks to addition node
    ax.annotate("", xy=(50.8, y_b + 3), xytext=(42, y_b + 5.0), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0, connectionstyle="arc3,rad=-0.15"))
    ax.annotate("", xy=(50.8, y_b + 3), xytext=(42, y_b + 3.8), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0, connectionstyle="arc3,rad=-0.05"))
    ax.annotate("", xy=(50.8, y_b + 3), xytext=(42, y_b + 2.2), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0, connectionstyle="arc3,rad=0.05"))
    ax.annotate("", xy=(50.8, y_b + 3), xytext=(42, y_b + 1.0), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.0, connectionstyle="arc3,rad=0.15"))
    
    # Residual Skip Connection (bottom of Panel B)
    ax.annotate("", xy=(52, y_b + 1.8), xytext=(12, y_b + 1.5), arrowprops=dict(arrowstyle="->", color="#7ED321", lw=1.2, ls="--", connectionstyle="angle,angleA=0,angleB=90,rad=10"))
    ax.text(30, y_b + 0.15, "Residual Skip Connection", color="#7ED321", fontsize=7, weight="bold")
    
    # Output of Layer
    ax.annotate("", xy=(64, y_b + 3), xytext=(53.2, y_b + 3), arrowprops=dict(arrowstyle="->", color=c_arrow, lw=1.2))
    ax.add_patch(patches.Rectangle((64, y_b + 1.5), 18, 3, facecolor="#9B9B9B", alpha=0.8, edgecolor="black", lw=1.0))
    ax.text(73, y_b + 3.0, "Norm & Output\n(b, c, h, w, d)", ha="center", va="center", color="white", weight="bold", fontsize=8)
    
    # Legend
    legend_handles = [
        patches.Patch(color=c_conv, label="3D Convolutional Encoder Blocks / Layers", alpha=0.8),
        patches.Patch(color=c_mamba, label="OmniMamba Tri-Planar Layers (S6 state-space)", alpha=0.8),
        patches.Patch(color=c_dec, label="3D Deconvolutional Decoder Blocks", alpha=0.8)
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.15), ncol=3, frameon=True, fontsize=9)
    
    plt.suptitle("Mamba-UNet (3D Stroke Lesion Segmentation with Tri-Planar Scanning)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(OUTPUT_MAMBA, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Mamba-UNet Architecture diagram saved to: {OUTPUT_MAMBA}")

if __name__ == "__main__":
    draw_unetr()
    draw_mamba_unet()
