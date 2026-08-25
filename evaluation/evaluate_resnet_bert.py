import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_1_DIR = BASE_DIR / "results" / "model_1_early_fusion"

def load_model_1_baseline():
    print("======================================")
    print("MODEL 1: BERT + RESNET50 BASELINE RESULTS")
    print("======================================")
    
    cv_summary_file = MODEL_1_DIR / "cv_summary.csv"
    test_metrics_file = MODEL_1_DIR / "final_test_metrics.json"
    cm_file = MODEL_1_DIR / "final_test_confusion_matrix.csv"
    
    if cv_summary_file.exists():
        cv_df = pd.read_csv(cv_summary_file)
        print("\n--- 5-Fold Cross-Validation Metrics ---")
        print(cv_df.to_string(index=False))
        
    if test_metrics_file.exists():
        with open(test_metrics_file, "r") as f:
            test_metrics = json.load(f)
        print("\n--- Final Test Set Metrics (520 samples) ---")
        print(f"Accuracy : {test_metrics['accuracy'] * 100:.2f}%")
        print(f"Precision: {test_metrics['precision'] * 100:.2f}%")
        print(f"Recall   : {test_metrics['recall'] * 100:.2f}%")
        print(f"F1 Score : {test_metrics['f1'] * 100:.2f}%")
        
    if cm_file.exists():
        cm_df = pd.read_csv(cm_file, index_col=0)
        print("\n--- Confusion Matrix ---")
        print(cm_df.to_string())

    print("\n======================================")

if __name__ == "__main__":
    load_model_1_baseline()
