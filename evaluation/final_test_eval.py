import os
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.train_anti_overfitting import (
    load_feature_map, clean_sid, FeatureDataset,
    SimplifiedResNetBERTClassifier, SimplifiedCLIPClassifier, SimplifiedViLTClassifier,
    evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
IMPROVED_DIR = RESULTS_DIR / "improved"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_final_locked_evaluation():
    print("==========================================================================", flush=True)
    print("FINAL LOCKED TEST EVALUATION (TOUCH-FREE 520 TEST SAMPLES)", flush=True)
    print("==========================================================================", flush=True)
    
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
    val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
    test_indices = [sid_to_idx[clean_sid(sid)] for sid in test_df["sample_id"]]
    
    tr_t, tr_i = text_feats[tr_indices], image_feats[tr_indices]
    tr_y = (train_df["label"] == "positive").astype(int).values
    
    val_t, val_i = text_feats[val_indices], image_feats[val_indices]
    val_y = (val_df["label"] == "positive").astype(int).values
    
    test_t, test_i = text_feats[test_indices], image_feats[test_indices]
    test_y = (test_df["label"] == "positive").astype(int).values
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=0.0), batch_size=32, shuffle=False)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=32, shuffle=False)
    test_loader = DataLoader(FeatureDataset(test_t, test_i, test_y, noise_std=0.0), batch_size=32, shuffle=False)
    
    models_config = [
        {
            "id": "model_1",
            "name": "ResNet50 + BERT",
            "class": SimplifiedResNetBERTClassifier,
            "anti_folder": ANTI_OVR_DIR / "model_1_resnet_bert",
            "impr_folder": IMPROVED_DIR / "model_1_resnet_bert"
        },
        {
            "id": "model_2",
            "name": "CLIP",
            "class": SimplifiedCLIPClassifier,
            "anti_folder": ANTI_OVR_DIR / "model_2_clip",
            "impr_folder": IMPROVED_DIR / "model_2_clip"
        },
        {
            "id": "model_3",
            "name": "ViLT",
            "class": SimplifiedViLTClassifier,
            "anti_folder": ANTI_OVR_DIR / "model_3_vilt",
            "impr_folder": IMPROVED_DIR / "model_3_vilt"
        }
    ]
    
    final_summary_rows = []
    comparative_rows = []
    
    for cfg in models_config:
        m_name = cfg["name"]
        m_class = cfg["class"]
        anti_dir = cfg["anti_folder"]
        impr_dir = cfg["impr_folder"]
        
        # Load best checkpoint
        ckpt_path = anti_dir / "best_val_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"Error: Checkpoint missing for {m_name} at {ckpt_path}")
            continue
            
        model = m_class(dropout=0.5).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        
        # Evaluate Train, Val, Test
        _, train_acc, _, _, train_f1, _, _ = evaluate(model, train_loader)
        _, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader)
        _, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader)
        
        train_test_gap = train_acc - test_acc
        train_val_gap = train_acc - val_acc
        
        # Load previous metrics for comparison
        old_metrics_file = impr_dir / "final_test_metrics.json"
        old_train_acc, old_test_acc, old_test_f1 = 0.0, 0.0, 0.0
        if old_metrics_file.exists():
            with open(old_metrics_file, "r") as f:
                old_m = json.load(f)
                old_train_acc = old_m.get("training_accuracy", 0.0)
                old_test_acc = old_m.get("accuracy", 0.0)
                old_test_f1 = old_m.get("f1", 0.0)
                
        old_gap = old_train_acc - old_test_acc
        acc_gain = test_acc - old_test_acc
        gap_reduction = old_gap - train_test_gap
        
        # Save metrics json
        new_metrics = {
            "model_name": m_name,
            "training_accuracy": float(train_acc),
            "validation_accuracy": float(val_acc),
            "test_accuracy": float(test_acc),
            "train_val_gap": float(train_val_gap),
            "train_test_gap": float(train_test_gap),
            "validation_f1": float(val_f1),
            "test_f1": float(test_f1),
            "test_precision": float(test_prec),
            "test_recall": float(test_rec),
            "old_test_accuracy": float(old_test_acc),
            "accuracy_gain": float(acc_gain),
            "old_train_test_gap": float(old_gap),
            "gap_reduction": float(gap_reduction)
        }
        
        with open(anti_dir / "final_test_metrics.json", "w") as f:
            json.dump(new_metrics, f, indent=4)
            
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])
        cm_df.to_csv(anti_dir / "final_test_confusion_matrix.csv")
        
        final_summary_rows.append({
            "Model": m_name,
            "Train Acc": f"{train_acc*100:.2f}%",
            "Validation Acc": f"{val_acc*100:.2f}%",
            "Test Acc": f"{test_acc*100:.2f}%",
            "Train-Test Gap": f"{train_test_gap*100:.2f} pp",
            "Test F1": f"{test_f1*100:.2f}%"
        })
        
        comparative_rows.append({
            "Model": m_name,
            "Old Test Acc": f"{old_test_acc*100:.2f}%",
            "New Test Acc": f"{test_acc*100:.2f}%",
            "Accuracy Gain": f"{acc_gain*100:+.2f} pp",
            "Old Gap": f"{old_gap*100:.2f} pp",
            "New Gap": f"{train_test_gap*100:.2f} pp",
            "Gap Reduction": f"{gap_reduction*100:.2f} pp"
        })

    print("\n" + "="*85, flush=True)
    print("TABLE 1: ANTI-OVERFITTING FINAL PERFORMANCE METRICS", flush=True)
    print("="*85, flush=True)
    df_table1 = pd.DataFrame(final_summary_rows)
    print(df_table1.to_string(index=False), flush=True)
    
    print("\n" + "="*95, flush=True)
    print("TABLE 2: BASELINE VS ANTI-OVERFITTING COMPARISON (BEFORE vs AFTER)", flush=True)
    print("="*95, flush=True)
    df_table2 = pd.DataFrame(comparative_rows)
    print(df_table2.to_string(index=False), flush=True)
    
    # Save overall summary files
    df_table1.to_csv(ANTI_OVR_DIR / "final_performance_table.csv", index=False)
    df_table2.to_csv(ANTI_OVR_DIR / "baseline_vs_new_comparison.csv", index=False)

if __name__ == "__main__":
    run_final_locked_evaluation()
