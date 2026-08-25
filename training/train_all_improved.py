import os
import json
import time
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.resnet_bert import BERTResNet50FeatureClassifier
from models.clip import CLIPMultimodalImprovedClassifier
from models.vilt import ViLTImprovedCrossModalClassifier

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
KFOLD_DIR = DATASET_DIR / "kfold"
SPLITS_DIR = DATASET_DIR / "splits"
IMPROVED_RESULTS_DIR = BASE_DIR / "results" / "improved"
IMPROVED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sid(sid):
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

class FeatureDataset(Dataset):
    def __init__(self, text_feats, image_feats, labels, jitter_noise=0.0):
        self.text_feats = torch.as_tensor(text_feats.copy() if isinstance(text_feats, np.ndarray) else text_feats, dtype=torch.float32)
        self.image_feats = torch.as_tensor(image_feats.copy() if isinstance(image_feats, np.ndarray) else image_feats, dtype=torch.float32)
        self.labels = torch.as_tensor(labels.copy() if isinstance(labels, np.ndarray) else labels, dtype=torch.long)
        self.jitter_noise = jitter_noise

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t_f = self.text_feats[idx]
        i_f = self.image_feats[idx]
        if self.jitter_noise > 0.0:
            t_f = t_f + torch.randn_like(t_f) * self.jitter_noise
            i_f = i_f + torch.randn_like(i_f) * self.jitter_noise
        return t_f, i_f, self.labels[idx]

def load_feature_map():
    text_data = torch.load(FEATURES_DIR / "text_features.pt", weights_only=False)
    image_data = torch.load(FEATURES_DIR / "image_features.pt", weights_only=False)

    sids = [clean_sid(sid) for sid in text_data["sample_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    if "5995" not in sid_to_idx and "6708" in sid_to_idx:
        sid_to_idx["5995"] = sid_to_idx["6708"]
        
    return text_data["features"], image_data["features"], sid_to_idx

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for t_feat, i_feat, labels in loader:
        t_feat, i_feat, labels = t_feat.to(device), i_feat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(t_feat, i_feat)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, acc

def evaluate(model, loader, criterion=None):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for t_feat, i_feat, labels in loader:
            t_feat, i_feat, labels_dev = t_feat.to(device), i_feat.to(device), labels.to(device)
            logits = model(t_feat, i_feat)
            if criterion:
                loss = criterion(logits, labels_dev)
                total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    avg_loss = total_loss / len(loader.dataset) if criterion else 0.0
    return avg_loss, acc, prec, rec, f1, all_preds, all_labels

def run_model_pipeline(model_name, model_class, text_feats, image_feats, sid_to_idx, save_folder_name, hyperparams):
    print("\n" + "="*80, flush=True)
    print(f"RUNNING IMPROVED PIPELINE: {model_name}", flush=True)
    print("="*80, flush=True)
    
    out_dir = IMPROVED_RESULTS_DIR / save_folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    lr = hyperparams.get("lr", 3e-4)
    weight_decay = hyperparams.get("weight_decay", 0.02)
    dropout = hyperparams.get("dropout", 0.4)
    max_epochs = hyperparams.get("epochs", 30)
    patience = hyperparams.get("patience", 4)
    jitter = hyperparams.get("jitter", 0.01)
    batch_size = hyperparams.get("batch_size", 32)
    label_smooth = hyperparams.get("label_smooth", 0.05)
    
    # ----------------------------------------------------
    # 1. 5-FOLD CROSS VALIDATION
    # ----------------------------------------------------
    print(f"--- 5-Fold Cross-Validation on Development Set ({model_name}) ---", flush=True)
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
        
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, jitter_noise=jitter), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, jitter_noise=0.0), batch_size=batch_size, shuffle=False)
        
        model = model_class(dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        best_val_loss = float("inf")
        best_metrics = None
        best_state = None
        no_improve_epochs = 0
        
        for epoch in range(1, max_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step()
            
            # Monitor validation loss with minimum 2 epochs requirement
            if epoch >= 2 and val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = model.state_dict().copy()
                no_improve_epochs = 0
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
            elif epoch >= 2:
                no_improve_epochs += 1
                if no_improve_epochs >= patience:
                    break
                    
        if best_metrics is None:
            best_metrics = {
                "fold": f"fold_{fold}",
                "best_epoch": 1,
                "train_accuracy": tr_acc,
                "accuracy": val_acc,
                "precision": val_prec,
                "recall": val_rec,
                "f1": val_f1,
                "train_val_gap": tr_acc - val_acc
            }
            
        cv_metrics.append(best_metrics)
        print(f"Fold {fold} (Best Epoch {best_metrics['best_epoch']}) - Train Acc: {best_metrics['train_accuracy']*100:.2f}%, Val Acc: {best_metrics['accuracy']*100:.2f}%, Val F1: {best_metrics['f1']*100:.2f}%, Gap: {best_metrics['train_val_gap']*100:.2f}%", flush=True)

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
    
    print(f"\n--- {model_name} 5-Fold CV Summary ---", flush=True)
    print(summary_df.to_string(index=False), flush=True)
    
    # ----------------------------------------------------
    # 2. FINAL TRAINING & TOUCH-FREE TEST EVALUATION
    # ----------------------------------------------------
    print(f"\n--- Final Model Retraining & Evaluation ({model_name}) ---", flush=True)
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
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
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, jitter_noise=jitter), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, jitter_noise=0.0), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(FeatureDataset(test_t, test_i, test_y, jitter_noise=0.0), batch_size=batch_size, shuffle=False)
    
    final_model = model_class(dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = []
    no_improve = 0
    
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(final_model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(final_model, val_loader, criterion)
        scheduler.step()
        
        history.append({
            "epoch": epoch,
            "development_loss": tr_loss,
            "development_accuracy": tr_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1
        })
        
        if epoch >= 2 and val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = final_model.state_dict().copy()
            best_epoch = epoch
            no_improve = 0
        elif epoch >= 2:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch}.", flush=True)
                break
                
    if best_state is not None:
        final_model.load_state_dict(best_state)
    elif best_epoch == 0:
        best_epoch = 1
        
    torch.save(final_model.state_dict(), out_dir / "final_model.pt")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / "final_training_history.csv", index=False)
    
    # Evaluate final model on train, val, and locked test
    tr_loss, final_tr_acc, _, _, final_tr_f1, _, _ = evaluate(final_model, train_loader)
    val_loss, final_val_acc, _, _, final_val_f1, _, _ = evaluate(final_model, val_loader)
    test_loss, final_ts_acc, final_ts_prec, final_ts_rec, final_ts_f1, test_preds, test_trues = evaluate(final_model, test_loader)
    
    metrics = {
        "model": model_name,
        "development_samples": len(dev_df),
        "final_test_samples": len(test_df),
        "best_epoch": int(best_epoch),
        "training_accuracy": float(final_tr_acc),
        "validation_accuracy": float(final_val_acc),
        "accuracy": float(final_ts_acc),
        "precision": float(final_ts_prec),
        "recall": float(final_ts_rec),
        "f1": float(final_ts_f1),
        "train_val_gap": float(final_tr_acc - final_val_acc),
        "train_test_gap": float(final_tr_acc - final_ts_acc)
    }
    
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    cm = confusion_matrix(test_trues, test_preds)
    cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])
    cm_df.to_csv(out_dir / "final_test_confusion_matrix.csv")
    
    pred_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "true_label": test_trues,
        "predicted_label": test_preds
    })
    pred_df.to_csv(out_dir / "final_test_predictions.csv", index=False)
    
    print(f"\n[{model_name} Improved Results Summary]", flush=True)
    print(f"Best Epoch       : {best_epoch}", flush=True)
    print(f"Training Accuracy: {final_tr_acc * 100:.2f}%", flush=True)
    print(f"Val Accuracy     : {final_val_acc * 100:.2f}%", flush=True)
    print(f"Final Test Acc   : {final_ts_acc * 100:.2f}%", flush=True)
    print(f"Final Test F1    : {final_ts_f1 * 100:.2f}%", flush=True)
    print(f"Train-Val Gap    : {metrics['train_val_gap'] * 100:.2f}%", flush=True)
    print(f"Train-Test Gap   : {metrics['train_test_gap'] * 100:.2f}%", flush=True)
    print("\nConfusion Matrix:", flush=True)
    print(cm_df.to_string(), flush=True)
    
    return metrics

def main():
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    # Model 1 (BERT + ResNet50 Early Fusion Improved)
    m1_hp = {"lr": 3e-4, "weight_decay": 0.02, "dropout": 0.4, "epochs": 30, "patience": 4, "jitter": 0.01, "label_smooth": 0.05}
    m1_res = run_model_pipeline(
        "Model 1 (BERT + ResNet50 Early Fusion)",
        BERTResNet50FeatureClassifier,
        text_feats, image_feats, sid_to_idx,
        "model_1_resnet_bert",
        m1_hp
    )
    
    # Model 2 (CLIP Multimodal Improved)
    m2_hp = {"lr": 3e-4, "weight_decay": 0.02, "dropout": 0.35, "epochs": 30, "patience": 4, "jitter": 0.01, "label_smooth": 0.05}
    m2_res = run_model_pipeline(
        "Model 2 (CLIP Multimodal Classifier)",
        CLIPMultimodalImprovedClassifier,
        text_feats, image_feats, sid_to_idx,
        "model_2_clip",
        m2_hp
    )
    
    # Model 3 (ViLT Cross-Attention Improved)
    m3_hp = {"lr": 2e-4, "weight_decay": 0.02, "dropout": 0.4, "epochs": 30, "patience": 4, "jitter": 0.01, "label_smooth": 0.05}
    m3_res = run_model_pipeline(
        "Model 3 (ViLT Cross-Modal Classifier)",
        ViLTImprovedCrossModalClassifier,
        text_feats, image_feats, sid_to_idx,
        "model_3_vilt",
        m3_hp
    )

if __name__ == "__main__":
    main()
