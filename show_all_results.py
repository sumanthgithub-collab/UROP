import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
IMPROVED = RESULTS / "improved"

def read_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def print_section(title):
    print("\n" + "=" * 85)
    print(title)
    print("=" * 85)

def load_cv_summary(folder_path):
    path = folder_path / "cv_summary.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip().str.lower()
    return df

def get_cv_metric(df, metric):
    if df is None:
        return None
    metric = metric.lower()
    rows = df[df.iloc[:, 0] == metric]
    if rows.empty:
        return None
    mean = float(rows.iloc[0]["mean"])
    std = float(rows.iloc[0]["std"])
    return mean, std

def print_model_comparison():
    print_section("MULTIMODAL SENTIMENT ANALYSIS - BASELINE VS IMPROVED PERFORMANCE")

    models_info = [
        ("Model 1: ResNet50 + BERT Early Fusion", RESULTS / "model_1_early_fusion", IMPROVED / "model_1_resnet_bert"),
        ("Model 2: CLIP Multimodal Classifier", RESULTS / "model_2_clip", IMPROVED / "model_2_clip"),
        ("Model 3: ViLT Cross-Modal Classifier", RESULTS / "model_3_vilt", IMPROVED / "model_3_vilt")
    ]

    print("\n" + f"{'Model':38s} | {'Base Test Acc':13s} | {'Impr Test Acc':13s} | {'Acc Gain':10s} | {'Base Test F1':12s} | {'Impr Test F1':12s} | {'F1 Gain':10s}")
    print("-" * 115)

    for name, base_folder, impr_folder in models_info:
        base_test = read_json(base_folder / "final_test_metrics.json")
        impr_test = read_json(impr_folder / "final_test_metrics.json")

        b_acc = f"{base_test['accuracy']*100:.2f}%" if base_test else "N/A"
        i_acc = f"{impr_test['accuracy']*100:.2f}%" if impr_test else "N/A"
        acc_gain = f"+{(impr_test['accuracy'] - base_test['accuracy'])*100:.2f}%" if (base_test and impr_test) else "N/A"

        b_f1 = f"{base_test['f1']*100:.2f}%" if base_test else "N/A"
        i_f1 = f"{impr_test['f1']*100:.2f}%" if impr_test else "N/A"
        f1_gain = f"+{(impr_test['f1'] - base_test['f1'])*100:.2f}%" if (base_test and impr_test) else "N/A"

        print(f"{name:38s} | {b_acc:13s} | {i_acc:13s} | {acc_gain:10s} | {b_f1:12s} | {i_f1:12s} | {f1_gain:10s}")

    print_section("5-FOLD CROSS-VALIDATION GENERALIZATION SUMMARY (DEVELOPMENT SET)")
    print(f"\n{'Model':38s} | {'CV Accuracy (%)':20s} | {'CV F1 (%)':20s} | {'CV Train-Val Gap (%)':20s}")
    print("-" * 105)

    for name, base_folder, impr_folder in models_info:
        cv_df = load_cv_summary(impr_folder)
        acc_m = get_cv_metric(cv_df, "accuracy")
        f1_m = get_cv_metric(cv_df, "f1")
        gap_m = get_cv_metric(cv_df, "train_val_gap")

        acc_str = f"{acc_m[0]*100:.2f}% ± {acc_m[1]*100:.2f}%" if acc_m else "N/A"
        f1_str = f"{f1_m[0]*100:.2f}% ± {f1_m[1]*100:.2f}%" if f1_m else "N/A"
        gap_str = f"{gap_m[0]*100:.2f}%" if gap_m else "N/A"

        print(f"{name:38s} | {acc_str:20s} | {f1_str:20s} | {gap_str:20s}")

def main():
    print_model_comparison()

if __name__ == "__main__":
    main()