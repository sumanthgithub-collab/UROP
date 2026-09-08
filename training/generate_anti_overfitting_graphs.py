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
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
GRAPHS_DIR = ANTI_OVR_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_anti_overfitting_visualizations_and_report():
    models_config = [
        ("ResNet50 + BERT", ANTI_OVR_DIR / "model_1_resnet_bert"),
        ("CLIP", ANTI_OVR_DIR / "model_2_clip"),
        ("ViLT", ANTI_OVR_DIR / "model_3_vilt")
    ]
    
    data = []
    for name, folder in models_config:
        metrics = read_json(folder / "final_test_metrics.json")
        hist_p = folder / "final_training_history.csv"
        hist_df = pd.read_csv(hist_p) if hist_p.exists() else None
        
        data.append({
            "name": name,
            "train_acc": metrics["training_accuracy"] * 100,
            "val_acc": metrics["validation_accuracy"] * 100,
            "test_acc": metrics["test_accuracy"] * 100,
            "test_f1": metrics["test_f1"] * 100,
            "gap": metrics["train_test_gap"] * 100,
            "old_test_acc": metrics["old_test_accuracy"] * 100,
            "old_test_f1": metrics["old_test_f1"] * 100,
            "old_gap": metrics["old_train_test_gap"] * 100,
            "gap_reduction": metrics["gap_reduction"] * 100,
            "acc_gain": metrics["accuracy_gain"] * 100,
            "f1_gain": metrics["f1_gain"] * 100,
            "hist_df": hist_df
        })
        
    names = [d["name"] for d in data]
    
    # --------------------------------------------------------------------------
    # GRAPH 1: Train vs Validation vs Test Accuracy
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    width = 0.25
    
    rects1 = ax.bar(x - width, [d["train_acc"] for d in data], width, label="Seen / Train Acc", color="#2b5c8f")
    rects2 = ax.bar(x, [d["val_acc"] for d in data], width, label="Unseen / Val Acc", color="#46a094")
    rects3 = ax.bar(x + width, [d["test_acc"] for d in data], width, label="Final Unseen / Test Acc", color="#d95f02")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Anti-Overfitting Models: Train vs Validation vs Test Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(60, 100)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)
    
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "1_train_val_test_accuracy.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 2: Old vs New Test Accuracy
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, [d["old_test_acc"] for d in data], width, label="Old Test Acc (%)", color="#999999")
    rects2 = ax.bar(x + width/2, [d["test_acc"] for d in data], width, label="New Test Acc (%)", color="#2b5c8f")
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Final Unseen Test Accuracy: Baseline vs Anti-Overfitting", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(60, 82)
    ax.legend(frameon=True)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "2_old_vs_new_test_accuracy.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 3: Old vs New Train-Test Gap
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    
    rects1 = ax.bar(x - width/2, [d["old_gap"] for d in data], width, label="Old Train-Test Gap (pp)", color="#d95f02")
    rects2 = ax.bar(x + width/2, [d["gap"] for d in data], width, label="New Train-Test Gap (pp)", color="#46a094")
    
    ax.set_ylabel("Generalization Gap (pp)", fontsize=12)
    ax.set_title("Train-to-Test Generalization Gap Reduction (Lower = Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 35)
    ax.legend(frameon=True)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f} pp", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "3_old_vs_new_train_test_gap.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 4: Old vs New Test F1
    # --------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    
    rects1 = ax.bar(x - width/2, [d["old_test_f1"] for d in data], width, label="Old Test F1 (%)", color="#7570b3")
    rects2 = ax.bar(x + width/2, [d["test_f1"] for d in data], width, label="New Test F1 (%)", color="#1b9e77")
    
    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Final Unseen Test F1 Score: Baseline vs Anti-Overfitting", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(60, 82)
    ax.legend(frameon=True)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
                        
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "4_old_vs_new_test_f1.png", dpi=300)
    plt.close()

    # --------------------------------------------------------------------------
    # GRAPH 5: Training vs Validation Curves for Each Model
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, d in zip(axes, data):
        df = d["hist_df"]
        if df is not None:
            ax.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc (%)", marker="o", color="#2b5c8f", linewidth=2)
            ax.plot(df["epoch"], df["val_acc"] * 100, label="Val Acc (%)", marker="s", linestyle="--", color="#d95f02", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(d["name"], fontsize=12, fontweight="bold")
            ax.set_ylim(50, 95)
            ax.legend(frameon=True)
            
    plt.suptitle("Anti-Overfitting Training & Validation Accuracy Curves Across Epochs", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "5_training_vs_validation_curves.png", dpi=300)
    plt.close()
    
    print(f"Generated 5 graphs in {GRAPHS_DIR}")

    # --------------------------------------------------------------------------
    # STEP 17 TERMINAL REPORT
    # --------------------------------------------------------------------------
    best_acc_model = max(data, key=lambda x: x["test_acc"])
    best_f1_model = max(data, key=lambda x: x["test_f1"])
    smallest_gap_model = min(data, key=lambda x: x["gap"])
    
    report = f"""============================================================
ANTI-OVERFITTING FINAL RESULTS
==============================

MODEL 1 — RESNET50 + BERT

TRAIN / SEEN              : {data[0]['train_acc']:.2f}%
VALIDATION / UNSEEN       : {data[0]['val_acc']:.2f}%
FINAL TEST / UNSEEN       : {data[0]['test_acc']:.2f}%

TRAIN → TEST GAP          : {data[0]['gap']:.2f} pp
GAP REDUCTION              : {data[0]['gap_reduction']:.2f} pp

FINAL TEST F1             : {data[0]['test_f1']:.2f}%

---

MODEL 2 — CLIP

TRAIN / SEEN              : {data[1]['train_acc']:.2f}%
VALIDATION / UNSEEN       : {data[1]['val_acc']:.2f}%
FINAL TEST / UNSEEN       : {data[1]['test_acc']:.2f}%

TRAIN → TEST GAP          : {data[1]['gap']:.2f} pp
GAP REDUCTION              : {data[1]['gap_reduction']:.2f} pp

FINAL TEST F1             : {data[1]['test_f1']:.2f}%

---

MODEL 3 — VILT

TRAIN / SEEN              : {data[2]['train_acc']:.2f}%
VALIDATION / UNSEEN       : {data[2]['val_acc']:.2f}%
FINAL TEST / UNSEEN       : {data[2]['test_acc']:.2f}%

TRAIN → TEST GAP          : {data[2]['gap']:.2f} pp
GAP REDUCTION              : {data[2]['gap_reduction']:.2f} pp

FINAL TEST F1             : {data[2]['test_f1']:.2f}%

============================================================
BEST UNSEEN MODEL
=================

Best Test Accuracy : {best_acc_model['name']} ({best_acc_model['test_acc']:.2f}%)
Best Test F1       : {best_f1_model['name']} ({best_f1_model['test_f1']:.2f}%)
Smallest Gap       : {smallest_gap_model['name']} ({smallest_gap_model['gap']:.2f} pp)

============================================================
"""
    print(report)

if __name__ == "__main__":
    generate_anti_overfitting_visualizations_and_report()
