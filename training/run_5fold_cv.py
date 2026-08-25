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

from training.train_anti_overfitting import (
    load_feature_map, clean_sid, FeatureDataset,
    SimplifiedResNetBERTClassifier, SimplifiedCLIPClassifier, SimplifiedViLTClassifier,
    train_one_epoch, evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
KFOLD_DIR = DATASET_DIR / "kfold"
RESULTS_DIR = BASE_DIR / "results" / "anti_overfitting"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_5fold_cv_for_model(model_name, model_class, folder_name, hparams, text_feats, image_feats, sid_to_idx):
    print(f"\n==========================================================================", flush=True)
    print(f"RUNNING 5-FOLD CROSS VALIDATION: {model_name}", flush=True)
    print(f"==========================================================================", flush=True)
    
    out_dir = RESULTS_DIR / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lr = hparams.get("lr", 1e-4)
    weight_decay = hparams.get("weight_decay", 0.05)
    dropout = hparams.get("dropout", 0.5)
    noise_std = hparams.get("noise_std", 0.01)
    label_smooth = hparams.get("label_smooth", 0.05)
    batch_size = hparams.get("batch_size", 32)
    max_epochs = hparams.get("epochs", 40)
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
        
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=batch_size, shuffle=False)
        
        model = model_class(dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        best_val_loss = float("inf")
        best_metrics = None
        no_improve = 0
        
        for epoch in range(1, max_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
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
                    
        cv_metrics.append(best_metrics)
        print(f"Fold {fold} (Epoch {best_metrics['best_epoch']:2d}) - Train Acc: {best_metrics['train_accuracy']*100:.2f}%, Val Acc: {best_metrics['accuracy']*100:.2f}%, Val F1: {best_metrics['f1']*100:.2f}%, Gap: {best_metrics['train_val_gap']*100:.2f}%", flush=True)

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
    
    print(f"\n--- {model_name} Anti-Overfitting 5-Fold CV Summary ---", flush=True)
    print(summary_df.to_string(index=False), flush=True)

def main():
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    cfg_file = RESULTS_DIR / "best_model_configs.json"
    if not cfg_file.exists():
        print("Best model configs not found. Run train_anti_overfitting.py first.")
        return
        
    with open(cfg_file, "r") as f:
        best_configs = json.load(f)
        
    models_map = [
        ("ResNet50+BERT", SimplifiedResNetBERTClassifier, "model_1_resnet_bert"),
        ("CLIP", SimplifiedCLIPClassifier, "model_2_clip"),
        ("ViLT", SimplifiedViLTClassifier, "model_3_vilt")
    ]
    
    for m_name, m_class, folder in models_map:
        hparams = best_configs[m_name]["hparams"]
        run_5fold_cv_for_model(m_name, m_class, folder, hparams, text_feats, image_feats, sid_to_idx)

if __name__ == "__main__":
    main()
