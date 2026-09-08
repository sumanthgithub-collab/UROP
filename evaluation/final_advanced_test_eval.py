import os
import json
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.train_advanced_generalization import (
    load_feature_map, clean_sid, FeatureDataset,
    AdvancedResNetBERTClassifier, AdvancedCLIPClassifier, AdvancedViLTClassifier,
    evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results"
ADVANCED_DIR = RESULTS_DIR / "advanced_generalization"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
IMPROVED_DIR = RESULTS_DIR / "improved"
METRICS_DIR = ADVANCED_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_final_locked_evaluation():
    print("==========================================================================", flush=True)
    print("STAGE 2: FINAL LOCKED TEST EVALUATION (TOUCH-FREE 520 TEST SAMPLES)", flush=True)
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
    
    cfg_file = ADVANCED_DIR / "best_model_configs.json"
    if not cfg_file.exists():
        print(f"Error: Config file missing at {cfg_file}", flush=True)
        return
        
    with open(cfg_file, "r", encoding="utf-8") as f:
        best_configs = json.load(f)
        
    models_config = [
        {
            "id": "model_1",
            "key": "ResNet50+BERT",
            "name": "ResNet50 + BERT",
            "class": AdvancedResNetBERTClassifier,
            "adv_folder": ADVANCED_DIR / "model_1_resnet_bert",
            "stage1_folder": ANTI_OVR_DIR / "model_1_resnet_bert"
        },
        {
            "id": "model_2",
            "key": "CLIP",
            "name": "CLIP",
            "class": AdvancedCLIPClassifier,
            "adv_folder": ADVANCED_DIR / "model_2_clip",
            "stage1_folder": ANTI_OVR_DIR / "model_2_clip"
        },
        {
            "id": "model_3",
            "key": "ViLT",
            "name": "ViLT",
            "class": AdvancedViLTClassifier,
            "adv_folder": ADVANCED_DIR / "model_3_vilt",
            "stage1_folder": ANTI_OVR_DIR / "model_3_vilt"
        }
    ]
    
    final_summary_rows = []
    comparative_rows = []
    
    model_test_probs = []
    
    for cfg in models_config:
        m_key = cfg["key"]
        m_name = cfg["name"]
        m_class = cfg["class"]
        adv_dir = cfg["adv_folder"]
        stage1_dir = cfg["stage1_folder"]
        
        hparams = best_configs[m_key]["hparams"]
        proj_dim = hparams.get("proj_dim", 128)
        hidden_dim = hparams.get("hidden_dim", 96)
        dropout = hparams.get("dropout", 0.45)
        fusion_mode = hparams.get("fusion_mode", "full_interaction")
        use_diff = hparams.get("use_diff", True)
        num_heads = hparams.get("num_heads", 4)
        
        ckpt_path = adv_dir / "best_val_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"Error: Checkpoint missing for {m_name} at {ckpt_path}", flush=True)
            continue
            
        kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
        if m_class == AdvancedResNetBERTClassifier:
            kwargs["fusion_mode"] = fusion_mode
        elif m_class == AdvancedCLIPClassifier:
            kwargs["use_diff"] = use_diff
        elif m_class == AdvancedViLTClassifier:
            kwargs["num_heads"] = num_heads
            
        model = m_class(**kwargs).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        
        # Evaluate Train, Val, Test
        _, train_acc, _, _, train_f1, _, _ = evaluate(model, train_loader)
        _, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader)
        _, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader)
        
        probs_list = []
        with torch.no_grad():
            for t_f, i_f, _ in test_loader:
                t_f, i_f = t_f.to(device), i_f.to(device)
                l = model(t_f, i_f)
                p = torch.softmax(l, dim=1).cpu().numpy()
                probs_list.append(p)
        model_test_probs.append(np.vstack(probs_list))
        
        train_test_gap = train_acc - test_acc
        train_val_gap = train_acc - val_acc
        
        # Load Stage 1 metrics for comparison
        stage1_file = stage1_dir / "final_test_metrics.json"
        stage1_test_acc, stage1_test_f1, stage1_gap = 0.0, 0.0, 0.0
        if stage1_file.exists():
            with open(stage1_file, "r", encoding="utf-8") as f:
                s1_m = json.load(f)
                stage1_test_acc = s1_m.get("test_accuracy", 0.0)
                stage1_test_f1 = s1_m.get("test_f1", 0.0)
                stage1_gap = s1_m.get("train_test_gap", 0.0)
                
        # Load Baseline metrics for comparison
        base_file = IMPROVED_DIR / cfg["stage1_folder"].name / "final_test_metrics.json"
        base_test_acc, base_test_f1, base_gap = 0.0, 0.0, 0.0
        if base_file.exists():
            with open(base_file, "r", encoding="utf-8") as f:
                b_m = json.load(f)
                base_test_acc = b_m.get("accuracy", 0.0)
                base_test_f1 = b_m.get("f1", 0.0)
                base_gap = b_m.get("training_accuracy", 0.0) - base_test_acc
                
        acc_gain_vs_stage1 = test_acc - stage1_test_acc
        acc_gain_vs_baseline = test_acc - base_test_acc
        gap_reduction_vs_stage1 = stage1_gap - train_test_gap
        gap_reduction_vs_baseline = base_gap - train_test_gap
        f1_gain_vs_stage1 = test_f1 - stage1_test_f1
        
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
            "stage1_test_accuracy": float(stage1_test_acc),
            "stage1_test_f1": float(stage1_test_f1),
            "stage1_train_test_gap": float(stage1_gap),
            "baseline_test_accuracy": float(base_test_acc),
            "baseline_train_test_gap": float(base_gap),
            "acc_gain_vs_stage1": float(acc_gain_vs_stage1),
            "acc_gain_vs_baseline": float(acc_gain_vs_baseline),
            "gap_reduction_vs_stage1": float(gap_reduction_vs_stage1),
            "gap_reduction_vs_baseline": float(gap_reduction_vs_baseline)
        }
        
        with open(adv_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(new_metrics, f, indent=4)
            
        cm = confusion_matrix(test_labels, test_preds)
        cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])
        cm_df.to_csv(adv_dir / "final_test_confusion_matrix.csv")
        
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
            "Baseline Test Acc": f"{base_test_acc*100:.2f}%",
            "Stage 1 Test Acc": f"{stage1_test_acc*100:.2f}%",
            "Stage 2 Test Acc": f"{test_acc*100:.2f}%",
            "Stage 2 Acc Gain": f"{acc_gain_vs_stage1*100:+.2f} pp",
            "Baseline Gap": f"{base_gap*100:.2f} pp",
            "Stage 1 Gap": f"{stage1_gap*100:.2f} pp",
            "Stage 2 Gap": f"{train_test_gap*100:.2f} pp",
            "Total Gap Reduction": f"{gap_reduction_vs_baseline*100:.2f} pp"
        })

    ens_probs = np.mean(model_test_probs, axis=0)
    ens_preds = np.argmax(ens_probs, axis=1)
    ens_acc = accuracy_score(test_y, ens_preds)
    ens_prec, ens_rec, ens_f1, _ = precision_recall_fscore_support(test_y, ens_preds, average="binary", zero_division=0)
    
    ens_dir = ADVANCED_DIR / "model_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    
    ens_metrics = {
        "model_name": "Multi-Model Ensemble",
        "test_accuracy": float(ens_acc),
        "test_f1": float(ens_f1),
        "test_precision": float(ens_prec),
        "test_recall": float(ens_rec)
    }
    with open(ens_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(ens_metrics, f, indent=4)
        
    final_summary_rows.append({
        "Model": "Multi-Model Ensemble",
        "Train Acc": "N/A",
        "Validation Acc": "N/A",
        "Test Acc": f"{ens_acc*100:.2f}%",
        "Train-Test Gap": "N/A",
        "Test F1": f"{ens_f1*100:.2f}%"
    })

    print("\n" + "="*85, flush=True)
    print("TABLE 1: STAGE 2 ADVANCED GENERALIZATION FINAL PERFORMANCE METRICS", flush=True)
    print("="*85, flush=True)
    df_table1 = pd.DataFrame(final_summary_rows)
    print(df_table1.to_string(index=False), flush=True)
    
    print("\n" + "="*115, flush=True)
    print("TABLE 2: MULTI-STAGE PROGRESSION COMPARISON (BASELINE -> STAGE 1 -> STAGE 2)", flush=True)
    print("="*115, flush=True)
    df_table2 = pd.DataFrame(comparative_rows)
    print(df_table2.to_string(index=False), flush=True)
    
    df_table1.to_csv(METRICS_DIR / "final_performance_table.csv", index=False)
    df_table2.to_csv(METRICS_DIR / "baseline_vs_current_vs_new_comparison.csv", index=False)

if __name__ == "__main__":
    run_final_locked_evaluation()
