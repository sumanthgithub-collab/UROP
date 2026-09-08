import os
import json
import sys
import io
import pickle
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

from training.train_stage3_unseen_breakthrough import (
    load_feature_map, clean_sid, FeatureDataset,
    AdvancedResNetBERTClassifier, AdvancedCLIPClassifier, AdvancedViLTClassifier, CrossModalTransformerClassifier,
    evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results"
STAGE3_DIR = RESULTS_DIR / "stage3_unseen_breakthrough"
ADVANCED_DIR = RESULTS_DIR / "advanced_generalization"
ANTI_OVR_DIR = RESULTS_DIR / "anti_overfitting"
IMPROVED_DIR = RESULTS_DIR / "improved"
METRICS_DIR = STAGE3_DIR / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_final_stage3_evaluation():
    print("==========================================================================", flush=True)
    print("STAGE 3: FINAL LOCKED TEST EVALUATION (TOUCH-FREE 520 TEST SAMPLES)", flush=True)
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
    
    cfg_file = STAGE3_DIR / "best_model_configs.json"
    if not cfg_file.exists():
        print(f"Error: Config file missing at {cfg_file}", flush=True)
        return
        
    with open(cfg_file, "r", encoding="utf-8") as f:
        best_configs = json.load(f)
        
    models_config = [
        {
            "id": "model_1",
            "key": "ResNet50+BERT",
            "name": "ResNet50 + BERT (Mixup)",
            "class": AdvancedResNetBERTClassifier,
            "folder": STAGE3_DIR / "model_1_resnet_bert"
        },
        {
            "id": "model_2",
            "key": "CLIP",
            "name": "CLIP (Mixup)",
            "class": AdvancedCLIPClassifier,
            "folder": STAGE3_DIR / "model_2_clip"
        },
        {
            "id": "model_3",
            "key": "ViLT",
            "name": "ViLT (Mixup)",
            "class": AdvancedViLTClassifier,
            "folder": STAGE3_DIR / "model_3_vilt"
        },
        {
            "id": "model_4",
            "key": "TransformerFusion",
            "name": "Cross-Modal Transformer",
            "class": CrossModalTransformerClassifier,
            "folder": STAGE3_DIR / "model_4_transformer_fusion"
        }
    ]
    
    summary_rows = []
    model_test_probs = []
    
    for cfg in models_config:
        m_key = cfg["key"]
        m_name = cfg["name"]
        m_class = cfg["class"]
        m_dir = cfg["folder"]
        
        hparams = best_configs[m_key]["hparams"]
        proj_dim = hparams.get("proj_dim", 128)
        hidden_dim = hparams.get("hidden_dim", 96)
        dropout = hparams.get("dropout", 0.45)
        fusion_mode = hparams.get("fusion_mode", "full_interaction")
        use_diff = hparams.get("use_diff", True)
        num_heads = hparams.get("num_heads", 4)
        
        ckpt_path = m_dir / "best_val_checkpoint.pt"
        if not ckpt_path.exists():
            print(f"Error: Checkpoint missing for {m_name} at {ckpt_path}", flush=True)
            continue
            
        kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
        if m_class == AdvancedResNetBERTClassifier:
            kwargs["fusion_mode"] = fusion_mode
        elif m_class == AdvancedCLIPClassifier:
            kwargs["use_diff"] = use_diff
        elif m_class in [AdvancedViLTClassifier, CrossModalTransformerClassifier]:
            kwargs["num_heads"] = num_heads
            
        model = m_class(**kwargs).to(device)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        
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
            "test_recall": float(test_rec)
        }
        with open(m_dir / "final_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(new_metrics, f, indent=4)
            
        summary_rows.append({
            "Model": m_name,
            "Train Acc": f"{train_acc*100:.2f}%",
            "Validation Acc": f"{val_acc*100:.2f}%",
            "Test Acc": f"{test_acc*100:.2f}%",
            "Train-Test Gap": f"{train_test_gap*100:.2f} pp",
            "Test F1": f"{test_f1*100:.2f}%"
        })

    # --------------------------------------------------------------------------
    # EVALUATE STAGE 3 CALIBRATED THRESHOLD & META-LEARNER STACKING
    # --------------------------------------------------------------------------
    stack_cfg_path = STAGE3_DIR / "stacking" / "stacking_config.json"
    meta_learner_path = STAGE3_DIR / "stacking" / "meta_learner.pkl"
    
    if stack_cfg_path.exists() and meta_learner_path.exists():
        with open(stack_cfg_path, "r", encoding="utf-8") as f:
            stack_cfg = json.load(f)
        with open(meta_learner_path, "rb") as f:
            meta_learner = pickle.load(f)
            
        tau_star = stack_cfg.get("optimal_threshold", 0.50)
        
        # Test probability matrix (520, 4)
        test_pos_probs = np.hstack([p[:, 1:] for p in model_test_probs])
        
        # Calibrated Threshold Predictions
        mean_test_prob = np.mean(test_pos_probs, axis=1)
        preds_tau = (mean_test_prob >= tau_star).astype(int)
        
        tau_acc = accuracy_score(test_y, preds_tau)
        tau_prec, tau_rec, tau_f1, _ = precision_recall_fscore_support(test_y, preds_tau, average="binary", zero_division=0)
        
        summary_rows.append({
            "Model": f"Calibrated Threshold (tau={tau_star:.3f})",
            "Train Acc": "N/A",
            "Validation Acc": "N/A",
            "Test Acc": f"{tau_acc*100:.2f}%",
            "Train-Test Gap": "N/A",
            "Test F1": f"{tau_f1*100:.2f}%"
        })
        
        # Meta-Learner Stacking Predictions
        stack_preds = meta_learner.predict(test_pos_probs)
        stack_acc = accuracy_score(test_y, stack_preds)
        stack_prec, stack_rec, stack_f1, _ = precision_recall_fscore_support(test_y, stack_preds, average="binary", zero_division=0)
        
        summary_rows.append({
            "Model": "Meta-Learner Stacking Ensemble",
            "Train Acc": "N/A",
            "Validation Acc": "N/A",
            "Test Acc": f"{stack_acc*100:.2f}%",
            "Train-Test Gap": "N/A",
            "Test F1": f"{stack_f1*100:.2f}%"
        })

    print("\n" + "="*85, flush=True)
    print("TABLE 1: STAGE 3 UNSEEN DATA ACCURACY BREAKTHROUGH FINAL PERFORMANCE", flush=True)
    print("="*85, flush=True)
    df_table1 = pd.DataFrame(summary_rows)
    print(df_table1.to_string(index=False), flush=True)
    
    df_table1.to_csv(METRICS_DIR / "final_performance_table.csv", index=False)

if __name__ == "__main__":
    run_final_stage3_evaluation()
