import os
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
GRAPHS_DIR = ANTI_OVR_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

def generate_all_graphs():
    table1_file = ANTI_OVR_DIR / "final_performance_table.csv"
    table2_file = ANTI_OVR_DIR / "baseline_vs_new_comparison.csv"
    
    if not table1_file.exists() or not table2_file.exists():
        print("Final performance tables not found. Run final_test_eval.py first.")
        return
        
    df1 = pd.read_csv(table1_file)
    df2 = pd.read_csv(table2_file)
    
    models = df1["Model"].tolist()
    
    # 1. Seen vs Unseen Accuracy Comparison
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    width = 0.25
    
    train_accs = [float(v.replace("%", "")) for v in df1["Train Acc"]]
    val_accs = [float(v.replace("%", "")) for v in df1["Validation Acc"]]
    test_accs = [float(v.replace("%", "")) for v in df1["Test Acc"]]
    
    r1 = ax.bar(x - width, train_accs, width, label="Seen / Train Acc", color="#2b5c8f")
    r2 = ax.bar(x, val_accs, width, label="Unseen / Validation Acc", color="#46a094")
    r3 = ax.bar(x + width, test_accs, width, label="Final Unseen / Test Acc", color="#d95f02")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Anti-Overfitting Models — Seen vs Unseen Data Performance", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(60, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    
    for rects in [r1, r2, r3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "seen_vs_unseen_accuracy_comparison.png", dpi=300)
    plt.close()
    
    # 2. Generalization Gap Reduction (Before vs After)
    fig, ax = plt.subplots(figsize=(9, 5))
    old_gaps = [float(v.replace(" pp", "")) for v in df2["Old Gap"]]
    new_gaps = [float(v.replace(" pp", "")) for v in df2["New Gap"]]
    
    x = np.arange(len(models))
    width = 0.35
    
    r1 = ax.bar(x - width/2, old_gaps, width, label="Old Train-Test Gap (Before)", color="#d95f02")
    r2 = ax.bar(x + width/2, new_gaps, width, label="New Train-Test Gap (After Anti-Overfitting)", color="#2b5c8f")
    
    ax.set_ylabel("Train-Test Gap (percentage points)", fontsize=12)
    ax.set_title("Drastic Reduction in Train-Test Generalization Gap", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 35)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    
    for rects in [r1, r2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f} pp",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "train_test_gap_reduction.png", dpi=300)
    plt.close()

    # 3. Confusion Matrix Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    folder_names = ["model_1_resnet_bert", "model_2_clip", "model_3_vilt"]
    
    for idx, (m_name, f_name) in enumerate(zip(models, folder_names)):
        cm_file = ANTI_OVR_DIR / f_name / "final_test_confusion_matrix.csv"
        if cm_file.exists():
            cm_df = pd.read_csv(cm_file, index_col=0)
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False,
                        annot_kws={"size": 14, "weight": "bold"})
            axes[idx].set_title(f"{m_name}\nConfusion Matrix", fontsize=12, fontweight="bold")
            axes[idx].set_xlabel("Predicted Label", fontsize=10)
            axes[idx].set_ylabel("True Label", fontsize=10)
            
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "confusion_matrices_all_models.png", dpi=300)
    plt.close()
    
    print(f"Visualizations saved to: {GRAPHS_DIR}")

if __name__ == "__main__":
    generate_all_graphs()
