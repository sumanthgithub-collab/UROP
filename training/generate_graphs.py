import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 12, "figure.autolayout": True})

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
IMPROVED_DIR = RESULTS_DIR / "improved"

GRAPHS_DIR = IMPROVED_DIR / "graphs"
CM_DIR = IMPROVED_DIR / "confusion_matrix"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

models_map = {
    "Model 1 (ResNet + BERT)": ("model_1_early_fusion", "model_1_resnet_bert"),
    "Model 2 (CLIP)": ("model_2_clip", "model_2_clip"),
    "Model 3 (ViLT)": ("model_3_vilt", "model_3_vilt")
}

def plot_confusion_matrices():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    model_names = ["Model 1 (ResNet + BERT)", "Model 2 (CLIP)", "Model 3 (ViLT)"]
    folder_names = ["model_1_resnet_bert", "model_2_clip", "model_3_vilt"]
    
    for ax, name, folder in zip(axes, model_names, folder_names):
        cm_path = IMPROVED_DIR / folder / "final_test_confusion_matrix.csv"
        if cm_path.exists():
            cm_df = pd.read_csv(cm_path, index_col=0)
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                        xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
            ax.set_title(f"{name}\nConfusion Matrix")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "improved_confusion_matrices.png", dpi=300)
    plt.close()
    print("Generated improved_confusion_matrices.png")

def plot_baseline_vs_improved():
    models = ["Model 1 (ResNet+BERT)", "Model 2 (CLIP)", "Model 3 (ViLT)"]
    
    base_acc = [70.38, 68.65, 66.73]
    impr_acc = [71.73, 70.96, 72.88]
    
    base_f1 = [71.48, 69.53, 67.17]
    impr_f1 = [72.63, 68.74, 73.45]
    
    x = np.arange(len(models))
    width = 0.35
    
    # 1. Accuracy Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, base_acc, width, label="Baseline Test Acc (%)", color="#7293CB")
    rects2 = ax.bar(x + width/2, impr_acc, width, label="Improved Test Acc (%)", color="#E1974C")
    
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final Test Accuracy Comparison (Baseline vs Improved)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(60, 80)
    ax.legend()
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()
    
    # 2. F1-Score Comparison Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    rects1 = ax.bar(x - width/2, base_f1, width, label="Baseline Test F1 (%)", color="#84BA5B")
    rects2 = ax.bar(x + width/2, impr_f1, width, label="Improved Test F1 (%)", color="#D35E60")
    
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("Final Test F1-Score Comparison (Baseline vs Improved)")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(60, 80)
    ax.legend()
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "f1_comparison.png", dpi=300)
    plt.close()
    print("Generated accuracy_comparison.png and f1_comparison.png")

def plot_cv_gap_comparison():
    models = ["Model 1 (ResNet+BERT)", "Model 2 (CLIP)", "Model 3 (ViLT)"]
    cv_gaps = [3.97, 4.83, 9.03] # 5-Fold CV Train-Val Gaps
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(models, cv_gaps, color=["#4C72B0", "#55A868", "#C44E52"], width=0.5)
    
    ax.set_ylabel("Mean Train - Val Gap (%)")
    ax.set_title("5-Fold Cross-Validation Generalization Gap (Train Acc - Val Acc)")
    ax.set_ylim(0, 15)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "generalization_gap_cv.png", dpi=300)
    plt.close()
    print("Generated generalization_gap_cv.png")

def plot_training_histories():
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    folder_names = ["model_1_resnet_bert", "model_2_clip", "model_3_vilt"]
    model_titles = ["Model 1 (ResNet+BERT)", "Model 2 (CLIP)", "Model 3 (ViLT)"]
    
    for ax, folder, title in zip(axes, folder_names, model_titles):
        hist_path = IMPROVED_DIR / folder / "final_training_history.csv"
        if hist_path.exists():
            df = pd.read_csv(hist_path)
            ax.plot(df["epoch"], df["development_accuracy"] * 100, label="Train Acc (%)", marker="o", color="#1f77b4")
            ax.plot(df["epoch"], df["val_accuracy"] * 100, label="Val Acc (%)", marker="s", linestyle="--", color="#ff7f0e")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(title)
            ax.legend()
            
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "training_curves_comparison.png", dpi=300)
    plt.close()
    print("Generated training_curves_comparison.png")

if __name__ == "__main__":
    plot_confusion_matrices()
    plot_baseline_vs_improved()
    plot_cv_gap_comparison()
    plot_training_histories()
