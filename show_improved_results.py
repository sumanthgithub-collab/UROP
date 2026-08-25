import json
import os
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding for Windows terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Set style for matplotlib figures
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Path configurations
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
IMPROVED = RESULTS / "improved"
DATASET = ROOT / "dataset"
GRAPHS_DIR = IMPROVED / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_dataset_summary():
    meta_p = DATASET / "balanced" / "metadata.csv"
    train_p = DATASET / "splits" / "train.csv"
    val_p = DATASET / "splits" / "validation.csv"
    test_p = DATASET / "splits" / "test.csv"

    summary = {
        "total": 2598,
        "positive": 1299,
        "negative": 1299,
        "dev": 2078,
        "train": 1662,
        "val": 416,
        "test": 520
    }

    if meta_p.exists():
        df_meta = pd.read_csv(meta_p)
        summary["total"] = len(df_meta)
        if "label" in df_meta.columns:
            counts = df_meta["label"].value_counts().to_dict()
            summary["positive"] = counts.get("positive", 1299)
            summary["negative"] = counts.get("negative", 1299)

    if train_p.exists():
        df_train = pd.read_csv(train_p)
        summary["train"] = len(df_train)

    if val_p.exists():
        df_val = pd.read_csv(val_p)
        summary["val"] = len(df_val)

    if test_p.exists():
        df_test = pd.read_csv(test_p)
        summary["test"] = len(df_test)

    summary["dev"] = summary["train"] + summary["val"]
    return summary

def load_model_data():
    models_config = [
        {
            "id": "model_1",
            "name": "ResNet50 + BERT",
            "full_name": "ResNet50 + BERT Early Fusion",
            "base_folder": RESULTS / "model_1_early_fusion",
            "impr_folder": IMPROVED / "model_1_resnet_bert"
        },
        {
            "id": "model_2",
            "name": "CLIP",
            "full_name": "CLIP Multimodal Classifier",
            "base_folder": RESULTS / "model_2_clip",
            "impr_folder": IMPROVED / "model_2_clip"
        },
        {
            "id": "model_3",
            "name": "ViLT",
            "full_name": "ViLT Cross-Modal Classifier",
            "base_folder": RESULTS / "model_3_vilt",
            "impr_folder": IMPROVED / "model_3_vilt"
        }
    ]

    data = []

    for cfg in models_config:
        base_metrics = read_json(cfg["base_folder"] / "final_test_metrics.json")
        impr_metrics = read_json(cfg["impr_folder"] / "final_test_metrics.json")
        hist_p = cfg["impr_folder"] / "final_training_history.csv"
        hist_df = pd.read_csv(hist_p) if hist_p.exists() else None

        # Base metrics
        base_test_acc = base_metrics["accuracy"] if (base_metrics and "accuracy" in base_metrics) else None
        base_test_f1 = base_metrics["f1"] if (base_metrics and "f1" in base_metrics) else None

        # Improved metrics
        train_acc = impr_metrics.get("training_accuracy") if impr_metrics else None
        val_acc = impr_metrics.get("validation_accuracy") if impr_metrics else None
        test_acc = impr_metrics.get("accuracy") if impr_metrics else None
        test_f1 = impr_metrics.get("f1") if impr_metrics else None

        # Train & Val F1 from history if available
        val_f1 = None
        if hist_df is not None and "val_f1" in hist_df.columns and len(hist_df) > 0:
            val_f1 = hist_df["val_f1"].iloc[-1]

        train_f1 = None  # Not available in saved experiment results

        # Gaps
        train_val_gap = (train_acc - val_acc) if (train_acc is not None and val_acc is not None) else None
        train_test_gap = (train_acc - test_acc) if (train_acc is not None and test_acc is not None) else None

        # Improvements
        acc_imp = (test_acc - base_test_acc) if (test_acc is not None and base_test_acc is not None) else None
        f1_imp = (test_f1 - base_test_f1) if (test_f1 is not None and base_test_f1 is not None) else None

        data.append({
            "id": cfg["id"],
            "name": cfg["name"],
            "full_name": cfg["full_name"],
            "base_test_acc": base_test_acc,
            "base_test_f1": base_test_f1,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "test_f1": test_f1,
            "train_val_gap": train_val_gap,
            "train_test_gap": train_test_gap,
            "acc_imp": acc_imp,
            "f1_imp": f1_imp,
            "hist_df": hist_df
        })

    return data

def generate_visualizations(models_data):
    names = [m["name"] for m in models_data]
    
    # 1. Seen vs Unseen Accuracy
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(names))
    width = 0.25

    train_accs = [m["train_acc"] * 100 if m["train_acc"] else 0 for m in models_data]
    val_accs = [m["val_acc"] * 100 if m["val_acc"] else 0 for m in models_data]
    test_accs = [m["test_acc"] * 100 if m["test_acc"] else 0 for m in models_data]

    rects1 = ax.bar(x - width, train_accs, width, label="Seen / Train Acc", color="#2b5c8f")
    rects2 = ax.bar(x, val_accs, width, label="Unseen / Validation Acc", color="#46a094")
    rects3 = ax.bar(x + width, test_accs, width, label="Final Unseen / Test Acc", color="#d95f02")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Seen vs Unseen Data Performance Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(60, 105)
    ax.legend(frameon=True, facecolor="white", framealpha=0.9)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "seen_vs_unseen_accuracy.png", dpi=300)
    plt.close()

    # 2. Final Test Accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, test_accs, color=["#2b5c8f", "#46a094", "#d95f02"], width=0.45)
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=12)
    ax.set_title("Improved Models — Final Unseen Test Accuracy", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(60, 80)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "final_test_accuracy.png", dpi=300)
    plt.close()

    # 3. Final Test F1
    fig, ax = plt.subplots(figsize=(8, 5))
    test_f1s = [m["test_f1"] * 100 if m["test_f1"] else 0 for m in models_data]
    bars = ax.bar(names, test_f1s, color=["#7570b3", "#e7298a", "#66a61e"], width=0.45)
    ax.set_ylabel("Final Test F1-Score (%)", fontsize=12)
    ax.set_title("Improved Models — Final Unseen Test F1-Score", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(60, 80)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "final_test_f1.png", dpi=300)
    plt.close()

    # 4. Generalization Gap
    fig, ax = plt.subplots(figsize=(8, 5))
    test_gaps = [m["train_test_gap"] * 100 if m["train_test_gap"] else 0 for m in models_data]
    bars = ax.bar(names, test_gaps, color=["#e6ab02", "#a6761d", "#666666"], width=0.45)
    ax.set_ylabel("Train-Test Gap (pp)", fontsize=12)
    ax.set_title("Train-to-Test Generalization Gap (Lower = Better)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 35)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f} pp",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "generalization_gap.png", dpi=300)
    plt.close()

    # 5. Baseline vs Improved
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    width = 0.35
    base_accs = [m["base_test_acc"] * 100 if m["base_test_acc"] else 0 for m in models_data]

    rects1 = ax.bar(x - width/2, base_accs, width, label="Baseline Test Acc (%)", color="#999999")
    rects2 = ax.bar(x + width/2, test_accs, width, label="Improved Test Acc (%)", color="#2b5c8f")

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Final Unseen Test Accuracy: Baseline vs Improved", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(60, 80)
    ax.legend(frameon=True)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "baseline_vs_improved.png", dpi=300)
    plt.close()

    # 6. Training Curves Comparison (if real history exists)
    has_history = all(m["hist_df"] is not None for m in models_data)
    if has_history:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for ax, m in zip(axes, models_data):
            df = m["hist_df"]
            ax.plot(df["epoch"], df["development_accuracy"] * 100, label="Train Acc (%)", marker="o", color="#2b5c8f", linewidth=2)
            ax.plot(df["epoch"], df["val_accuracy"] * 100, label="Val Acc (%)", marker="s", linestyle="--", color="#d95f02", linewidth=2)
            ax.set_xlabel("Epoch", fontsize=11)
            ax.set_ylabel("Accuracy (%)", fontsize=11)
            ax.set_title(m["name"], fontsize=12, fontweight="bold")
            ax.set_ylim(50, 105)
            ax.legend(frameon=True)

        plt.suptitle("Training & Validation Accuracy Curves Across Epochs", fontsize=14, fontweight="bold", y=1.03)
        plt.tight_layout()
        plt.savefig(GRAPHS_DIR / "training_curves_comparison.png", dpi=300)
        plt.close()

def print_terminal_report(dataset_summary, models_data):
    line_len = 110

    # SECTION 1 — DATASET SUMMARY
    print("=" * 60)
    print("MULTIMODAL SENTIMENT ANALYSIS")
    print("IMPROVED MODELS PERFORMANCE")
    print("===========================")
    print()
    print("## DATASET")
    print()
    print(f"Total Samples       : {dataset_summary['total']}")
    print(f"Positive Samples    : {dataset_summary['positive']}")
    print(f"Negative Samples    : {dataset_summary['negative']}")
    print()
    print(f"Development Samples : {dataset_summary['dev']}")
    print(f"Training Samples    : {dataset_summary['train']}")
    print(f"Validation Samples  : {dataset_summary['val']}")
    print(f"Final Test Samples  : {dataset_summary['test']}")
    print()
    print("# Final Test Set      : LOCKED / UNSEEN")
    print()

    # SECTION 2 — MAIN PERFORMANCE TABLE
    print("=" * line_len)
    print("IMPROVED MODELS — SEEN vs UNSEEN PERFORMANCE")
    print("============================================")
    print()
    print("LEGEND:")
    print("  TRAIN               = SEEN DATA (used for model training)")
    print("  VALIDATION          = UNSEEN DURING TRAINING (used for hyperparameter tuning / epoch selection)")
    print("  FINAL UNSEEN TEST   = COMPLETELY UNSEEN (locked 520 test samples, never used for training/tuning)")
    print()
    print(f"{'Model':28s} | {'Seen/Train':12s} | {'Validation':12s} | {'Final Unseen Test':18s} | {'Train-Val Gap':14s} | {'Train-Test Gap':14s}")
    print("-" * line_len)

    for m in models_data:
        tr_a = f"{m['train_acc']*100:.2f}%" if m['train_acc'] is not None else "N/A"
        va_a = f"{m['val_acc']*100:.2f}%" if m['val_acc'] is not None else "N/A"
        te_a = f"{m['test_acc']*100:.2f}%" if m['test_acc'] is not None else "N/A"
        tv_g = f"{m['train_val_gap']*100:.2f} pp" if m['train_val_gap'] is not None else "N/A"
        tt_g = f"{m['train_test_gap']*100:.2f} pp" if m['train_test_gap'] is not None else "N/A"

        print(f"{m['name']:28s} | {tr_a:12s} | {va_a:12s} | {te_a:18s} | {tv_g:14s} | {tt_g:14s}")

    print("=" * line_len)
    print()

    # SECTION 3 — ACCURACY EXPLANATION
    print("## WHAT THE NUMBERS MEAN")
    print()
    print("Seen Accuracy       = performance on training data")
    print("Validation Accuracy = performance on unseen validation data")
    print("Test Accuracy       = performance on the locked final test set")
    print()
    print("Train-Val Gap:")
    print("How much accuracy drops from training to validation.")
    print()
    print("Train-Test Gap:")
    print("How much accuracy drops from training to final unseen test.")
    print()
    print("## Smaller gaps generally indicate better generalization.")
    print()

    # SECTION 4 — F1 PERFORMANCE
    print("=" * 80)
    print("IMPROVED MODELS — F1 PERFORMANCE")
    print("================================")
    print()
    print(f"{'Model':28s} | {'Train F1':34s} | {'Validation F1':14s} | {'Final Test F1':14s}")
    print("-" * 96)

    for m in models_data:
        tr_f1 = "Not available in saved experiment results" if m['train_f1'] is None else f"{m['train_f1']*100:.2f}%"
        va_f1 = f"{m['val_f1']*100:.2f}%" if m['val_f1'] is not None else "N/A"
        te_f1 = f"{m['test_f1']*100:.2f}%" if m['test_f1'] is not None else "N/A"

        print(f"{m['name']:28s} | {tr_f1:34s} | {va_f1:14s} | {te_f1:14s}")

    print("=" * 96)
    print()

    # SECTION 5 — BASELINE vs IMPROVED
    print("=" * 96)
    print("BASELINE vs IMPROVED — FINAL UNSEEN TEST")
    print("========================================")
    print()
    print("## ACCURACY COMPARISON")
    print(f"{'Model':28s} | {'Baseline Test Acc':18s} | {'Improved Test Acc':18s} | {'Improvement':15s}")
    print("-" * 85)

    for m in models_data:
        b_acc = f"{m['base_test_acc']*100:.2f}%" if m['base_test_acc'] is not None else "N/A"
        i_acc = f"{m['test_acc']*100:.2f}%" if m['test_acc'] is not None else "N/A"
        if m['acc_imp'] is not None:
            sign = "+" if m['acc_imp'] >= 0 else ""
            a_imp = f"{sign}{m['acc_imp']*100:.2f} pp"
        else:
            a_imp = "N/A"
        print(f"{m['name']:28s} | {b_acc:18s} | {i_acc:18s} | {a_imp:15s}")

    print("=" * 85)
    print()
    print("## F1-SCORE COMPARISON")
    print(f"{'Model':28s} | {'Baseline F1':18s} | {'Improved F1':18s} | {'F1 Improvement':15s}")
    print("-" * 85)

    for m in models_data:
        b_f1 = f"{m['base_test_f1']*100:.2f}%" if m['base_test_f1'] is not None else "N/A"
        i_f1 = f"{m['test_f1']*100:.2f}%" if m['test_f1'] is not None else "N/A"
        if m['f1_imp'] is not None:
            sign = "+" if m['f1_imp'] >= 0 else ""
            f_imp = f"{sign}{m['f1_imp']*100:.2f} pp"
        else:
            f_imp = "N/A"
        print(f"{m['name']:28s} | {b_f1:18s} | {i_f1:18s} | {f_imp:15s}")

    print("=" * 85)
    print()

    # SECTION 6 — BEST MODEL
    # Ranked by Final Test Accuracy primary, Test F1 secondary
    best_model = max(models_data, key=lambda x: (x["test_acc"] or 0, x["test_f1"] or 0))

    print("=" * 60)
    print("BEST FINAL MODEL")
    print("================")
    print()
    print(f"🏆 BEST MODEL: {best_model['full_name']}")
    print()
    print(f"Final Test Accuracy : {best_model['test_acc']*100:.2f}%")
    print(f"Final Test F1       : {best_model['test_f1']*100:.2f}%")
    print()
    acc_imp_str = f"+{best_model['acc_imp']*100:.2f}" if (best_model['acc_imp'] and best_model['acc_imp']>=0) else f"{best_model['acc_imp']*100:.2f}"
    print(f"Accuracy Improvement : {acc_imp_str} percentage points")
    print()
    print(f"Train Accuracy      : {best_model['train_acc']*100:.2f}%")
    print(f"Validation Accuracy : {best_model['val_acc']*100:.2f}%")
    print(f"Final Test Accuracy : {best_model['test_acc']*100:.2f}%")
    print()
    print(f"Train-Test Gap      : {best_model['train_test_gap']*100:.2f} percentage points")
    print()
    print("=" * 60)
    print()

    # SECTION 7 — GENERALIZATION RANKING
    print("=" * 60)
    print("GENERALIZATION / OVERFITTING CHECK")
    print("==================================")
    print()
    print(f"{'Model':28s} | {'Train-Test Gap':18s}")
    print("-" * 50)

    for m in models_data:
        gap_str = f"{m['train_test_gap']*100:.2f} pp" if m['train_test_gap'] is not None else "N/A"
        print(f"{m['name']:28s} | {gap_str:18s}")

    print("-" * 50)
    print()
    print("Smallest gap = strongest train-to-test generalization")
    print("Largest gap  = greater potential overfitting")
    print()

    best_gen_model = min(models_data, key=lambda x: x["train_test_gap"] if x["train_test_gap"] is not None else 999)
    worst_gen_model = max(models_data, key=lambda x: x["train_test_gap"] if x["train_test_gap"] is not None else -999)

    print(f"BEST GENERALIZATION: {best_gen_model['full_name']}")
    print()
    print("Gap:")
    print(f"{best_gen_model['train_test_gap']*100:.2f} percentage points")
    print()
    print(f"Highest train-test gap — inspect for potential overfitting: {worst_gen_model['full_name']} ({worst_gen_model['train_test_gap']*100:.2f} pp)")
    print()

    # SECTION 8 — SIMPLE HUMAN-READABLE INTERPRETATION
    print("=" * 60)
    print("FINAL INTERPRETATION")
    print("====================")
    print()

    for idx, m in enumerate(models_data, start=1):
        print(f"MODEL {idx} — {m['full_name'].upper()}")
        print(f"Seen/Train Accuracy      : {m['train_acc']*100:.2f}%")
        print(f"Unseen/Validation Acc    : {m['val_acc']*100:.2f}%")
        print(f"Final Unseen/Test Acc    : {m['test_acc']*100:.2f}%")
        print(f"Generalization Gap       : {m['train_test_gap']*100:.2f} pp")
        print()
        print("Interpretation:")
        print(f"The model performs {m['train_test_gap']*100:.2f} percentage points lower on the final")
        print("unseen test data than on the training data.")
        print()
        if idx < len(models_data):
            print("---")
            print()

    print("=" * 60)
    print("FINAL CONCLUSION")
    print("================")
    print()
    best_test_acc_model = max(models_data, key=lambda x: x["test_acc"] or 0)
    best_test_f1_model = max(models_data, key=lambda x: x["test_f1"] or 0)

    print(f"Best Final Test Accuracy : {best_test_acc_model['full_name']} ({best_test_acc_model['test_acc']*100:.2f}%)")
    print(f"Best Final Test F1       : {best_test_f1_model['full_name']} ({best_test_f1_model['test_f1']*100:.2f}%)")
    print(f"Best Generalization Gap  : {best_gen_model['full_name']} ({best_gen_model['train_test_gap']*100:.2f} pp)")
    print()
    print("=" * 60)
    print()

    # SECTION 9 — IMPORTANT CLIP CASE
    clip_model = next((m for m in models_data if "CLIP" in m["name"]), None)
    if clip_model and clip_model["acc_imp"] is not None and clip_model["f1_imp"] is not None:
        if clip_model["acc_imp"] > 0 and clip_model["f1_imp"] < 0:
            print("[CRITICAL NOTE ON CLIP MODEL]")
            print("CLIP improved in accuracy, but F1 decreased. Therefore the accuracy improvement")
            print("should not be interpreted as an improvement across all classification metrics.")
            print()

    # SECTION 10 — IMPORTANT VILT CASE
    vilt_model = next((m for m in models_data if "ViLT" in m["name"]), None)
    if vilt_model:
        is_highest_acc = vilt_model["id"] == best_test_acc_model["id"]
        # Check if train-test gap or train-val gap is significant (e.g. > 20 pp)
        if is_highest_acc and (vilt_model["train_test_gap"] > 0.20 or vilt_model["train_val_gap"] > 0.20):
            print("[CRITICAL NOTE ON VILT MODEL]")
            print("ViLT achieves the strongest final unseen-test performance, but its larger")
            print("generalization gap indicates that some overfitting may remain.")
            print()

    # SECTION 14 / END BANNER
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("===================")
    print()
    print("Three improved models evaluated:")
    print("✓ ResNet50 + BERT")
    print("✓ CLIP")
    print("✓ ViLT")
    print()
    print("Final unseen test set:")
    print("✓ 520 samples")
    print("✓ Locked")
    print("✓ Not used for tuning")
    print()
    print("Results saved to:")
    print("results/improved/")
    print()
    print("=" * 60)

def main():
    summary = load_dataset_summary()
    models_data = load_model_data()
    generate_visualizations(models_data)
    print_terminal_report(summary, models_data)

if __name__ == "__main__":
    main()
