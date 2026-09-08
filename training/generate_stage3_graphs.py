import json
import os
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
STAGE3_DIR = RESULTS_DIR / "stage3_unseen_breakthrough"
ADVANCED_DIR = RESULTS_DIR / "advanced_generalization"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
IMPROVED_DIR = RESULTS_DIR / "improved"
GRAPHS_DIR = STAGE3_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_stage3_visualizations():
    models_info = [
        ("ResNet50 + BERT", STAGE3_DIR / "model_1_resnet_bert", ANTI_OVR_DIR / "model_1_resnet_bert", IMPROVED_DIR / "model_1_resnet_bert"),
        ("CLIP", STAGE3_DIR / "model_2_clip", ANTI_OVR_DIR / "model_2_clip", IMPROVED_DIR / "model_2_clip"),
        ("ViLT", STAGE3_DIR / "model_3_vilt", ANTI_OVR_DIR / "model_3_vilt", IMPROVED_DIR / "model_3_vilt"),
        ("Cross-Modal Transformer", STAGE3_DIR / "model_4_transformer_fusion", None, None)
    ]
    
    data = []
    for name, s3_folder, s1_folder, base_folder in models_info:
        s3_m = read_json(s3_folder / "final_test_metrics.json")
        s1_m = read_json(s1_folder / "final_test_metrics.json") if s1_folder else {}
        base_m = read_json(base_folder / "final_test_metrics.json") if base_folder else {}
        
        cv_summary = pd.read_csv(s3_folder / "stage3_cv_results.csv") if (s3_folder / "stage3_cv_results.csv").exists() else None
        hist_df = pd.read_csv(s3_folder / "final_training_history.csv") if (s3_folder / "final_training_history.csv").exists() else None
        
        data.append({
            "name": name,
            "train_acc": s3_m.get("training_accuracy", 0) * 100,
            "val_acc": s3_m.get("validation_accuracy", 0) * 100,
            "test_acc": s3_m.get("test_accuracy", 0) * 100,
            "test_f1": s3_m.get("test_f1", 0) * 100,
            "gap": s3_m.get("train_test_gap", 0) * 100,
            "s1_test_acc": s1_m.get("test_accuracy", 0) * 100,
            "base_test_acc": base_m.get("accuracy", 0) * 100,
            "hist_df": hist_df,
            "cv_df": cv_summary
        })
        
    names = [d["name"] for d in data]
    x = np.arange(len(names))
    width = 0.25
    
    # --------------------------------------------------------------------------
    # GRAPH 1: stage3_train_val_test_comparison.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width, [d["train_acc"] for d in data], width, label="Seen / Train Acc", color="#2b5c8f")
    rects2 = ax.bar(x, [d["val_acc"] for d in data], width, label="Unseen / Val Acc", color="#46a094")
    rects3 = ax.bar(x + width, [d["test_acc"] for d in data], width, label="Final Unseen / Test Acc", color="#d95f02")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Stage 3 Unseen Data Accuracy: Train vs Validation vs Test", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(60, 100)
    ax.legend(frameon=True, facecolor="white")
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "stage3_train_val_test_comparison.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 2: stage3_progression_across_stages.png
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width, [d["base_test_acc"] for d in data], width, label="Baseline Test Acc", color="#999999")
    rects2 = ax.bar(x, [d["s1_test_acc"] for d in data], width, label="Stage 1 Test Acc", color="#46a094")
    rects3 = ax.bar(x + width, [d["test_acc"] for d in data], width, label="Stage 3 Test Acc", color="#2b5c8f")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Unseen Test Accuracy Progression Across Optimization Stages", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(60, 80)
    ax.legend(frameon=True)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "stage3_progression_across_stages.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 3: stage3_training_curves.png
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, d in zip(axes, data):
        df = d["hist_df"]
        if df is not None:
            ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc (%)", marker="o", color="#2b5c8f", linewidth=2)
            ax.plot(df["epoch"], df["val_acc"] * 100, label="Val Acc (%)", marker="s", linestyle="--", color="#d95f02", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_title(d["name"], fontsize=11, fontweight="bold")
            ax.set_ylim(55, 90)
            ax.legend(frameon=True)
            
    plt.suptitle("Stage 3 Training & Validation Curves with Manifold Mixup", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "stage3_training_curves.png", dpi=300)
    plt.close()
    
    print(f"Generated Stage 3 graphs in {GRAPHS_DIR}")

if __name__ == "__main__":
    generate_stage3_visualizations()
