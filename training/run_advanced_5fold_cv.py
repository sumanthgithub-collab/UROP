import os
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.train_advanced_generalization import (
    load_feature_map, clean_sid, FeatureDataset,
    AdvancedResNetBERTClassifier, AdvancedCLIPClassifier, AdvancedViLTClassifier,
    train_one_epoch, evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
KFOLD_DIR = DATASET_DIR / "kfold"
RESULTS_DIR = BASE_DIR / "results" / "advanced_generalization"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_probabilities(model, loader):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for t_feat, i_feat, _ in loader:
            t_feat, i_feat = t_feat.to(device), i_feat.to(device)
            logits = model(t_feat, i_feat)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    return np.vstack(all_probs)

def run_advanced_5fold_cv():
    print("==========================================================================", flush=True)
    print("STAGE 2: 5-FOLD CROSS VALIDATION & ENSEMBLE EVALUATION (DEV SET ONLY)", flush=True)
    print("==========================================================================", flush=True)
    
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    cfg_file = RESULTS_DIR / "best_model_configs.json"
    if not cfg_file.exists():
        print("Best model configs not found. Run train_advanced_generalization.py first.", flush=True)
        return
        
    with open(cfg_file, "r") as f:
        best_configs = json.load(f)
        
    models_info = [
        ("ResNet50+BERT", AdvancedResNetBERTClassifier, "model_1_resnet_bert"),
        ("CLIP", AdvancedCLIPClassifier, "model_2_clip"),
        ("ViLT", AdvancedViLTClassifier, "model_3_vilt")
    ]
    
    # Store per-fold predictions for ensemble evaluation
    fold_probs_dev = {1: [], 2: [], 3: [], 4: [], 5: []}
    fold_probs_val = {1: [], 2: [], 3: [], 4: [], 5: []}
    fold_labels_dev = {}
    fold_labels_val = {}
    
    for m_name, m_class, folder in models_info:
        print(f"\n--- 5-Fold Cross Validation: {m_name} ---", flush=True)
        out_dir = RESULTS_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        
        hparams = best_configs[m_name]["hparams"]
        lr = hparams.get("lr", 1e-4)
        weight_decay = hparams.get("weight_decay", 0.05)
        dropout = hparams.get("dropout", 0.45)
        proj_dim = hparams.get("proj_dim", 128)
        hidden_dim = hparams.get("hidden_dim", 96)
        noise_std = hparams.get("noise_std", 0.0)
        label_smooth = hparams.get("label_smooth", 0.0)
        fusion_mode = hparams.get("fusion_mode", "full_interaction")
        use_diff = hparams.get("use_diff", True)
        num_heads = hparams.get("num_heads", 4)
        batch_size = hparams.get("batch_size", 32)
        max_epochs = hparams.get("epochs", 35)
        patience = hparams.get("patience", 4)
        
        cv_metrics = []
        
        for fold in range(1, 6):
            fold_dir = KFOLD_DIR / f"fold_{fold}"
            train_df = pd.read_csv(fold_dir / "train.csv")
            val_df = pd.read_csv(fold_dir / "validation.csv")
            
            tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
            val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
            
            tr_t, tr_i = text_feats[tr_indices], image_feats[tr_indices]
            tr_y = (train_df["label"] == "positive").astype(int).values
            val_t, val_i = text_feats[val_indices], image_feats[val_indices]
            val_y = (val_df["label"] == "positive").astype(int).values
            
            fold_labels_dev[fold] = tr_y
            fold_labels_val[fold] = val_y
            
            train_loader_eval = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=0.0), batch_size=batch_size, shuffle=False)
            train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=batch_size, shuffle=False)
            
            kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
            if m_class == AdvancedResNetBERTClassifier:
                kwargs["fusion_mode"] = fusion_mode
            elif m_class == AdvancedCLIPClassifier:
                kwargs["use_diff"] = use_diff
            elif m_class == AdvancedViLTClassifier:
                kwargs["num_heads"] = num_heads
                
            model = m_class(**kwargs).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
            
            best_val_loss = float("inf")
            best_metrics = None
            best_state = None
            no_improve = 0
            
            for epoch in range(1, max_epochs + 1):
                tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_metrics = {
                        "fold": f"fold_{fold}",
                        "best_epoch": epoch,
                        "train_accuracy": tr_acc,
                        "accuracy": val_acc,
                        "precision": val_prec,
                        "recall": val_rec,
                        "f1": val_f1,
                        "train_val_gap": tr_acc - val_acc
                    }
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                        
            # Reload best model state to collect probabilities
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            
            tr_p = get_model_probabilities(model, train_loader_eval)
            val_p = get_model_probabilities(model, val_loader)
            
            fold_probs_dev[fold].append(tr_p)
            fold_probs_val[fold].append(val_p)
            
            cv_metrics.append(best_metrics)
            print(f"Fold {fold} (Epoch {best_metrics['best_epoch']:2d}) - Train Acc: {best_metrics['train_accuracy']*100:.2f}%, Val Acc: {best_metrics['accuracy']*100:.2f}%, Val F1: {best_metrics['f1']*100:.2f}%, Gap: {best_metrics['train_val_gap']*100:.2f} pp", flush=True)

        cv_df = pd.DataFrame(cv_metrics)
        cv_df.to_csv(out_dir / "cv_results.csv", index=False)
        
        summary_data = [
            {"metric": "accuracy", "mean": cv_df["accuracy"].mean(), "std": cv_df["accuracy"].std()},
            {"metric": "precision", "mean": cv_df["precision"].mean(), "std": cv_df["precision"].std()},
            {"metric": "recall", "mean": cv_df["recall"].mean(), "std": cv_df["recall"].std()},
            {"metric": "f1", "mean": cv_df["f1"].mean(), "std": cv_df["f1"].std()},
            {"metric": "train_accuracy", "mean": cv_df["train_accuracy"].mean(), "std": cv_df["train_accuracy"].std()},
            {"metric": "train_val_gap", "mean": cv_df["train_val_gap"].mean(), "std": cv_df["train_val_gap"].std()},
        ]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(out_dir / "cv_summary.csv", index=False)
        
        print(f"\n--- {m_name} Stage 2 5-Fold CV Summary ---", flush=True)
        print(summary_df.to_string(index=False), flush=True)

    # --------------------------------------------------------------------------
    # MULTI-MODEL ENSEMBLE EVALUATION (DEV SET ONLY)
    # --------------------------------------------------------------------------
    print("\n" + "="*80, flush=True)
    print("STEP 10: MULTI-MODEL ENSEMBLE EVALUATION ON DEV CV SET", flush=True)
    print("="*80, flush=True)
    
    ensemble_cv_metrics = []
    for fold in range(1, 6):
        # Average predictions from the 3 models for this fold
        val_probs_arr = np.mean(fold_probs_val[fold], axis=0) # (N_val, 2)
        dev_probs_arr = np.mean(fold_probs_dev[fold], axis=0) # (N_dev, 2)
        
        val_preds = np.argmax(val_probs_arr, axis=1)
        dev_preds = np.argmax(dev_probs_arr, axis=1)
        
        val_y = fold_labels_val[fold]
        dev_y = fold_labels_dev[fold]
        
        tr_acc = accuracy_score(dev_y, dev_preds)
        val_acc = accuracy_score(val_y, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(val_y, val_preds, average="binary", zero_division=0)
        
        m_row = {
            "fold": f"fold_{fold}",
            "train_accuracy": tr_acc,
            "accuracy": val_acc,
            "precision": val_prec,
            "recall": val_rec,
            "f1": val_f1,
            "train_val_gap": tr_acc - val_acc
        }
        ensemble_cv_metrics.append(m_row)
        print(f"Ensemble Fold {fold} - Train Acc: {tr_acc*100:.2f}%, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1*100:.2f}%, Gap: {(tr_acc-val_acc)*100:.2f} pp", flush=True)

    ens_df = pd.DataFrame(ensemble_cv_metrics)
    ens_dir = RESULTS_DIR / "model_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)
    ens_df.to_csv(ens_dir / "cv_results.csv", index=False)
    
    ens_summary = [
        {"metric": "accuracy", "mean": ens_df["accuracy"].mean(), "std": ens_df["accuracy"].std()},
        {"metric": "precision", "mean": ens_df["precision"].mean(), "std": ens_df["precision"].std()},
        {"metric": "recall", "mean": ens_df["recall"].mean(), "std": ens_df["recall"].std()},
        {"metric": "f1", "mean": ens_df["f1"].mean(), "std": ens_df["f1"].std()},
        {"metric": "train_accuracy", "mean": ens_df["train_accuracy"].mean(), "std": ens_df["train_accuracy"].std()},
        {"metric": "train_val_gap", "mean": ens_df["train_val_gap"].mean(), "std": ens_df["train_val_gap"].std()},
    ]
    ens_summary_df = pd.DataFrame(ens_summary)
    ens_summary_df.to_csv(ens_dir / "cv_summary.csv", index=False)
    
    print("\n--- Multi-Model Ensemble 5-Fold CV Summary ---", flush=True)
    print(ens_summary_df.to_string(index=False), flush=True)

if __name__ == "__main__":
    run_advanced_5fold_cv()
