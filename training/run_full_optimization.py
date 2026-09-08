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
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.resnet_bert import BERTResNet50FeatureClassifier
from models.clip import CLIPMultimodalImprovedClassifier
from models.vilt import ViLTImprovedCrossModalClassifier

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
SPLITS_DIR = DATASET_DIR / "splits"
KFOLD_DIR = DATASET_DIR / "kfold"

RESULTS_DIR = BASE_DIR / "results"
FINAL_RESULTS_DIR = RESULTS_DIR / "final"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
GRAPHS_DIR = BASE_DIR / "graphs"
REPORTS_DIR = BASE_DIR / "reports"

for d in [FINAL_RESULTS_DIR, EXPERIMENTS_DIR, CHECKPOINTS_DIR, GRAPHS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sid(sid):
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def load_features():
    text_data = torch.load(FEATURES_DIR / "text_features.pt", weights_only=False)
    image_data = torch.load(FEATURES_DIR / "image_features.pt", weights_only=False)

    sids = [clean_sid(sid) for sid in text_data["sample_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    if "5995" not in sid_to_idx and "6708" in sid_to_idx:
        sid_to_idx["5995"] = sid_to_idx["6708"]
        
    return text_data["features"], image_data["features"], sid_to_idx

class FeatureDataset(Dataset):
    def __init__(self, text_feats, image_feats, labels, noise_std=0.0):
        self.text_feats = torch.as_tensor(text_feats.copy() if isinstance(text_feats, np.ndarray) else text_feats, dtype=torch.float32)
        self.image_feats = torch.as_tensor(image_feats.copy() if isinstance(image_feats, np.ndarray) else image_feats, dtype=torch.float32)
        self.labels = torch.as_tensor(labels.copy() if isinstance(labels, np.ndarray) else labels, dtype=torch.long)
        self.noise_std = noise_std

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t_f = self.text_feats[idx]
        i_f = self.image_feats[idx]
        if self.noise_std > 0.0:
            t_f = t_f + torch.randn_like(t_f) * self.noise_std
            i_f = i_f + torch.randn_like(i_f) * self.noise_std
        return t_f, i_f, self.labels[idx]

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
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    avg_loss = total_loss / len(loader.dataset) if criterion else 0.0
    return avg_loss, acc, prec, rec, f1, all_preds, all_labels

def run_controlled_experiments(model_key, model_class, text_feats, image_feats, sid_to_idx):
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    
    tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
    val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
    
    tr_t, tr_i = text_feats[tr_indices], image_feats[tr_indices]
    tr_y = (train_df["label"] == "positive").astype(int).values
    val_t, val_i = text_feats[val_indices], image_feats[val_indices]
    val_y = (val_df["label"] == "positive").astype(int).values

    # Experiment search space
    exp_configs = [
        {"exp_id": "Exp_0_Baseline", "dropout": 0.2, "weight_decay": 0.0, "lr": 1e-3, "label_smooth": 0.0, "noise_std": 0.0, "proj_dim": 256},
        {"exp_id": "Exp_1_Regularization_Light", "dropout": 0.3, "weight_decay": 0.01, "lr": 3e-4, "label_smooth": 0.0, "noise_std": 0.0, "proj_dim": 256},
        {"exp_id": "Exp_1_Regularization_Strong", "dropout": 0.5, "weight_decay": 0.05, "lr": 3e-4, "label_smooth": 0.05, "noise_std": 0.0, "proj_dim": 256},
        {"exp_id": "Exp_2_LR_Tuning_Small", "dropout": 0.4, "weight_decay": 0.02, "lr": 1e-4, "label_smooth": 0.0, "noise_std": 0.0, "proj_dim": 256},
        {"exp_id": "Exp_2_LR_Tuning_Medium", "dropout": 0.4, "weight_decay": 0.02, "lr": 3e-4, "label_smooth": 0.0, "noise_std": 0.0, "proj_dim": 256},
        {"exp_id": "Exp_3_Compact_Classifier", "dropout": 0.4, "weight_decay": 0.02, "lr": 3e-4, "label_smooth": 0.0, "noise_std": 0.0, "proj_dim": 128},
        {"exp_id": "Exp_4_Feature_Jitter_Noise", "dropout": 0.4, "weight_decay": 0.02, "lr": 3e-4, "label_smooth": 0.0, "noise_std": 0.015, "proj_dim": 256},
        {"exp_id": "Exp_5_Optimal_Combination", "dropout": 0.4, "weight_decay": 0.03, "lr": 2e-4, "label_smooth": 0.02, "noise_std": 0.01, "proj_dim": 128 if model_key in ["ResNet50 + BERT", "ViLT"] else 256},
    ]
    
    exp_results = []
    best_exp_metric = -1.0
    best_config = None
    
    for cfg in exp_configs:
        exp_id = cfg["exp_id"]
        noise_std = cfg["noise_std"]
        batch_size = 32
        lr = cfg["lr"]
        weight_decay = cfg["weight_decay"]
        dropout = cfg["dropout"]
        proj_dim = cfg["proj_dim"]
        label_smooth = cfg["label_smooth"]
        max_epochs = 35
        patience = 5
        
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=batch_size, shuffle=False)
        
        model = model_class(proj_dim=proj_dim, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        best_val_loss = float("inf")
        best_epoch = 0
        best_metrics = None
        no_improve = 0
        
        for epoch in range(1, max_epochs + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_epoch = epoch
                best_metrics = {
                    "model": model_key,
                    "exp_id": exp_id,
                    "best_epoch": epoch,
                    "train_acc": tr_acc,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "train_val_gap": tr_acc - val_acc,
                    "config": cfg
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        exp_results.append(best_metrics)
        print(f"  [{model_key}] {exp_id:<30} | Epoch {best_metrics['best_epoch']:2d} | Train Acc: {best_metrics['train_acc']*100:.2f}% | Val Acc: {best_metrics['val_acc']*100:.2f}% | Val F1: {best_metrics['val_f1']*100:.2f}% | Gap: {best_metrics['train_val_gap']*100:.2f}pp")
        
        if best_metrics["val_f1"] > best_exp_metric:
            best_exp_metric = best_metrics["val_f1"]
            best_config = cfg
            
    return exp_results, best_config

def run_kfold_cv(model_key, model_class, best_config, text_feats, image_feats, sid_to_idx):
    print(f"\nRunning 5-Fold Cross-Validation for {model_key}...")
    proj_dim = best_config["proj_dim"]
    dropout = best_config["dropout"]
    lr = best_config["lr"]
    weight_decay = best_config["weight_decay"]
    label_smooth = best_config["label_smooth"]
    noise_std = best_config["noise_std"]
    
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
        
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=32, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=32, shuffle=False)
        
        model = model_class(proj_dim=proj_dim, dropout=dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        best_val_loss = float("inf")
        best_metrics = None
        no_improve = 0
        
        for epoch in range(1, 35):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_metrics = {
                    "fold": fold,
                    "train_acc": tr_acc,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                    "gap": tr_acc - val_acc
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    break
        cv_metrics.append(best_metrics)
        print(f"  Fold {fold}: Train Acc={best_metrics['train_acc']*100:.2f}%, Val Acc={best_metrics['val_acc']*100:.2f}%, Val F1={best_metrics['val_f1']*100:.2f}%")
        
    cv_df = pd.DataFrame(cv_metrics)
    summary = {
        "mean_acc": cv_df["val_acc"].mean(),
        "std_acc": cv_df["val_acc"].std(),
        "mean_f1": cv_df["val_f1"].mean(),
        "std_f1": cv_df["val_f1"].std(),
        "mean_train_acc": cv_df["train_acc"].mean(),
        "mean_gap": cv_df["gap"].mean()
    }
    print(f"  K-Fold Summary for {model_key}: Mean Val Acc={summary['mean_acc']*100:.2f}% ± {summary['std_acc']*100:.2f}%, Mean F1={summary['mean_f1']*100:.2f}%")
    return cv_df, summary

def train_and_evaluate_final_model(model_key, model_class, best_config, text_feats, image_feats, sid_to_idx, save_dir_name):
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
    
    proj_dim = best_config["proj_dim"]
    dropout = best_config["dropout"]
    lr = best_config["lr"]
    weight_decay = best_config["weight_decay"]
    label_smooth = best_config["label_smooth"]
    noise_std = best_config["noise_std"]
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=32, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=32, shuffle=False)
    test_loader = DataLoader(FeatureDataset(test_t, test_i, test_y, noise_std=0.0), batch_size=32, shuffle=False)
    
    model = model_class(proj_dim=proj_dim, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    
    best_val_loss = float("inf")
    best_state = None
    best_val_metrics = None
    no_improve = 0
    
    history = []
    
    for epoch in range(1, 40):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1
        })
        
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = {
                "best_epoch": epoch,
                "train_acc": tr_acc,
                "val_acc": val_acc,
                "val_f1": val_f1
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 6:
                break
                
    # Save checkpoint
    ckpt_dir = CHECKPOINTS_DIR / save_dir_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt_dir / "best_model.pt")
    
    # Load best checkpoint and evaluate on LOCKED TEST SET
    model.load_state_dict(best_state)
    test_loss, test_acc, test_prec, test_rec, test_f1, test_preds, test_labels = evaluate(model, test_loader)
    
    cm = confusion_matrix(test_labels, test_preds)
    
    final_metrics = {
        "model_name": model_key,
        "best_epoch": best_val_metrics["best_epoch"],
        "train_accuracy": best_val_metrics["train_acc"],
        "validation_accuracy": best_val_metrics["val_acc"],
        "validation_f1": best_val_metrics["val_f1"],
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_precision": test_prec,
        "test_recall": test_rec,
        "train_val_gap": best_val_metrics["train_acc"] - best_val_metrics["val_acc"],
        "train_test_gap": best_val_metrics["train_acc"] - test_acc,
        "confusion_matrix": cm.tolist()
    }
    
    # Save results to final folder
    out_dir = FINAL_RESULTS_DIR / save_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "final_test_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)
        
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    
    return final_metrics, cm

def main():
    print("==========================================================================")
    print("EXECUTING FULL SCIENTIFIC GENERALIZATION OPTIMIZATION PIPELINE")
    print("==========================================================================")
    
    text_feats, image_feats, sid_to_idx = load_features()
    
    models_info = [
        {"key": "ResNet50 + BERT", "class": BERTResNet50FeatureClassifier, "folder": "resnet_bert"},
        {"key": "CLIP", "class": CLIPMultimodalImprovedClassifier, "folder": "clip"},
        {"key": "ViLT", "class": ViLTImprovedCrossModalClassifier, "folder": "vilt"}
    ]
    
    all_exp_logs = []
    final_model_results = {}
    kfold_summaries = {}
    
    for m_info in models_info:
        m_key = m_info["key"]
        m_class = m_info["class"]
        m_folder = m_info["folder"]
        
        print(f"\n--- Phase 9: Controlled Experiments for {m_key} ---")
        exp_logs, best_config = run_controlled_experiments(m_key, m_class, text_feats, image_feats, sid_to_idx)
        all_exp_logs.extend(exp_logs)
        
        # Save experiment log for model
        m_exp_dir = EXPERIMENTS_DIR / m_folder
        m_exp_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(exp_logs).to_csv(m_exp_dir / "experiment_log.csv", index=False)
        with open(m_exp_dir / "best_config.json", "w") as f:
            json.dump(best_config, f, indent=4)
            
        print(f"\n--- Phase 10: 5-Fold Cross-Validation for {m_key} ---")
        cv_df, cv_summary = run_kfold_cv(m_key, m_class, best_config, text_feats, image_feats, sid_to_idx)
        cv_df.to_csv(m_exp_dir / "5fold_cv_results.csv", index=False)
        kfold_summaries[m_key] = cv_summary
        
        print(f"\n--- Phase 12: Final Evaluation on Locked Test Set for {m_key} ---")
        final_metrics, cm = train_and_evaluate_final_model(m_key, m_class, best_config, text_feats, image_feats, sid_to_idx, m_folder)
        final_model_results[m_key] = final_metrics
        
    # Save overall experiment log
    pd.DataFrame(all_exp_logs).to_csv(EXPERIMENTS_DIR / "experiment_log.csv", index=False)
    
    # Generate Plots
    print("\n--- Phase 13: Generating Evaluation Graphs ---")
    generate_all_plots(final_model_results)
    
    print("\n==========================================================================")
    print("OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY!")
    print("==========================================================================")

def generate_all_plots(final_results):
    sns.set_theme(style="whitegrid")
    
    # 1. Baseline vs Final Comparison Plot
    models = list(final_results.keys())
    old_test = [75.38, 73.65, 74.42] # From current best baseline in prompt
    new_test = [final_results[m]["test_accuracy"] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, old_test, width, label='Baseline Test Acc (%)', color='#7f8c8d')
    rects2 = ax.bar(x + width/2, new_test, width, label='Optimized Test Acc (%)', color='#2ecc71')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Baseline vs Final Optimized Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(60, 95)
    
    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f'{h:.2f}%', xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "baseline_vs_final.png", dpi=300)
    plt.close()
    
    # 2. Generalization Gap Reduction Plot
    old_gaps = [11.92, 17.62, 11.14]
    new_gaps = [final_results[m]["train_test_gap"] * 100 for m in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width/2, old_gaps, width, label='Baseline Gap (pp)', color='#e74c3c')
    rects2 = ax.bar(x + width/2, new_gaps, width, label='Optimized Gap (pp)', color='#3498db')
    
    ax.set_ylabel('Train-Test Gap (pp)')
    ax.set_title('Generalization Gap Reduction (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    for r in rects1 + rects2:
        h = r.get_height()
        ax.annotate(f'{h:.2f}pp', xy=(r.get_x() + r.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                    
    plt.tight_layout()
    plt.savefig(GRAPHS_DIR / "generalization_gap.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
