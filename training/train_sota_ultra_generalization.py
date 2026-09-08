import os
import json
import sys
import io
import csv
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.sota_models import (
    SOTA_TwitterRoBERTa_CLIP_Gated,
    SOTA_DeBERTa_CLIP_Contrastive,
    SOTA_BiDirectionalCrossAttention
)

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results" / "sota_breakthrough"
METRICS_DIR = RESULTS_DIR / "metrics"
GRAPHS_DIR = RESULTS_DIR / "graphs"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sid(sid):
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def load_data_and_features():
    sota_t = torch.load(FEATURES_DIR / "sota_text_features.pt", weights_only=False)
    sota_i = torch.load(FEATURES_DIR / "sota_image_features.pt", weights_only=False)
    
    t_sids = [clean_sid(s) for s in sota_t["sample_ids"]]
    t_idx = {sid: i for i, sid in enumerate(t_sids)}
    
    i_sids = [clean_sid(s) for s in sota_i["sample_ids"]]
    i_idx = {sid: i for i, sid in enumerate(i_sids)}
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    label_map = {"negative": 0, "positive": 1}
    
    def extract_split(df):
        text_roberta_list = []
        deberta_list = []
        clip_list = []
        y_list = []
        sids = []
        for _, row in df.iterrows():
            sid = clean_sid(row["sample_id"])
            if sid in t_idx and sid in i_idx:
                ti = t_idx[sid]
                ii = i_idx[sid]
                text_roberta_list.append(sota_t["twitter_roberta"][ti])
                deberta_list.append(sota_t["deberta"][ti])
                clip_list.append(sota_i["clip_large"][ii])
                y_list.append(label_map[row["label"].strip().lower()])
                sids.append(sid)
        return (
            torch.stack(text_roberta_list),
            torch.stack(deberta_list),
            torch.stack(clip_list),
            torch.tensor(y_list, dtype=torch.long),
            sids
        )

    tr_rob, tr_deb, tr_clip, tr_y, tr_sids = extract_split(train_df)
    val_rob, val_deb, val_clip, val_y, val_sids = extract_split(val_df)
    te_rob, te_deb, te_clip, te_y, te_sids = extract_split(test_df)
    
    # Combine Train + Val for 5-Fold Development Set (2,078 samples)
    dev_rob = torch.cat([tr_rob, val_rob], dim=0)
    dev_deb = torch.cat([tr_deb, val_deb], dim=0)
    dev_clip = torch.cat([tr_clip, val_clip], dim=0)
    dev_y = torch.cat([tr_y, val_y], dim=0)
    dev_sids = tr_sids + val_sids
    
    return {
        "dev": (dev_rob, dev_deb, dev_clip, dev_y, dev_sids),
        "test": (te_rob, te_deb, te_clip, te_y, te_sids)
    }

class SOTAMultimodalDataset(Dataset):
    def __init__(self, rob_feats, deb_feats, clip_feats, labels):
        self.rob_feats = rob_feats.float()
        self.deb_feats = deb_feats.float()
        self.clip_feats = clip_feats.float()
        self.labels = labels.long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.rob_feats[idx], self.deb_feats[idx], self.clip_feats[idx], self.labels[idx]

def mixup_data(t_feat, i_feat, labels, alpha=0.25):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = t_feat.size(0)
    index = torch.randperm(batch_size).to(t_feat.device)
    
    mixed_t = lam * t_feat + (1.0 - lam) * t_feat[index]
    mixed_i = lam * i_feat + (1.0 - lam) * i_feat[index]
    y_a, y_b = labels, labels[index]
    return mixed_t, mixed_i, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def train_sota_model(model_cls, text_type, dev_data, test_data, num_epochs=20, lr=3e-4, weight_decay=0.08, alpha=0.25, label_smooth=0.10):
    dev_rob, dev_deb, dev_clip, dev_y, _ = dev_data
    te_rob, te_deb, te_clip, te_y, _ = test_data
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(dev_y), 2))
    test_probs_folds = np.zeros((len(te_y), 2))
    
    fold_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dev_y)), dev_y.numpy())):
        tr_r, tr_d, tr_c, tr_y_fold = dev_rob[train_idx], dev_deb[train_idx], dev_clip[train_idx], dev_y[train_idx]
        va_r, va_d, va_c, va_y_fold = dev_rob[val_idx], dev_deb[val_idx], dev_clip[val_idx], dev_y[val_idx]
        
        if text_type == "roberta":
            tr_t, va_t, te_t = tr_r, va_r, te_rob
        else:
            tr_t, va_t, te_t = tr_d, va_d, te_deb
            
        train_ds = SOTAMultimodalDataset(tr_r, tr_d, tr_c, tr_y_fold)
        val_ds = SOTAMultimodalDataset(va_r, va_d, va_c, va_y_fold)
        test_ds = SOTAMultimodalDataset(te_rob, te_deb, te_clip, te_y)
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        model = model_cls().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        
        best_val_loss = float("inf")
        best_model_state = None
        best_val_probs = None
        
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for rob_b, deb_b, clip_b, y_b in train_loader:
                t_b = rob_b if text_type == "roberta" else deb_b
                i_b = clip_b
                t_b, i_b, y_b = t_b.to(device), i_b.to(device), y_b.to(device)
                
                # Apply Manifold Mixup
                mixed_t, mixed_i, y_a, y_b_mix, lam = mixup_data(t_b, i_b, y_b, alpha=alpha)
                
                optimizer.zero_grad()
                logits = model(mixed_t, mixed_i)
                loss = mixup_criterion(criterion, logits, y_a, y_b_mix, lam)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * len(y_b)
                preds = logits.argmax(dim=-1)
                correct += (preds == y_a).sum().item()
                total += len(y_b)
                
            scheduler.step()
            train_acc = correct / total
            
            # Validation Evaluation
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            val_probs_list = []
            
            with torch.no_grad():
                for rob_b, deb_b, clip_b, y_b in val_loader:
                    t_b = rob_b if text_type == "roberta" else deb_b
                    i_b = clip_b
                    t_b, i_b, y_b = t_b.to(device), i_b.to(device), y_b.to(device)
                    
                    logits = model(t_b, i_b)
                    loss = criterion(logits, y_b)
                    probs = F.softmax(logits, dim=-1)
                    
                    val_loss += loss.item() * len(y_b)
                    val_preds.extend(logits.argmax(dim=-1).cpu().numpy())
                    val_targets.extend(y_b.cpu().numpy())
                    val_probs_list.append(probs.cpu().numpy())
                    
            val_loss /= len(val_ds)
            val_acc = accuracy_score(val_targets, val_preds)
            val_probs_arr = np.concatenate(val_probs_list, axis=0)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                best_val_probs = val_probs_arr
                
        oof_probs[val_idx] = best_val_probs
        
        # Test Evaluation with best fold model
        model.load_state_dict(best_model_state)
        model.eval()
        fold_test_probs = []
        with torch.no_grad():
            for rob_b, deb_b, clip_b, y_b in test_loader:
                t_b = rob_b if text_type == "roberta" else deb_b
                i_b = clip_b
                t_b, i_b = t_b.to(device), i_b.to(device)
                logits = model(t_b, i_b)
                probs = F.softmax(logits, dim=-1)
                fold_test_probs.append(probs.cpu().numpy())
        test_probs_folds += np.concatenate(fold_test_probs, axis=0) / 5.0

    return oof_probs, test_probs_folds

def main():
    print("==========================================================================", flush=True)
    print("STARTING SOTA ULTRA-GENERALIZATION BREAKTHROUGH PIPELINE", flush=True)
    print("==========================================================================", flush=True)
    
    data = load_data_and_features()
    dev_data = data["dev"]
    test_data = data["test"]
    
    dev_y = dev_data[3].numpy()
    test_y = test_data[3].numpy()
    
    print(f"Development Set Size: {len(dev_y)} samples (5-Fold CV)", flush=True)
    print(f"Locked Test Set Size:  {len(test_y)} samples", flush=True)
    
    # -------------------------------------------------------------
    # Train Model 1: Twitter-RoBERTa + CLIP-ViT-L Gated Early Fusion
    # -------------------------------------------------------------
    print("\n[1/3] Training SOTA Model 1: Twitter-RoBERTa + CLIP-ViT-L Gated Early Fusion...", flush=True)
    m1_oof, m1_test = train_sota_model(
        SOTA_TwitterRoBERTa_CLIP_Gated,
        text_type="roberta",
        dev_data=dev_data,
        test_data=test_data,
        num_epochs=18,
        lr=3e-4,
        weight_decay=0.08,
        alpha=0.25
    )
    m1_val_acc = accuracy_score(dev_y, m1_oof.argmax(axis=-1))
    m1_test_acc = accuracy_score(test_y, m1_test.argmax(axis=-1))
    print(f"  Model 1 OOF Val Acc: {m1_val_acc*100:.2f}% | Test Acc: {m1_test_acc*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # Train Model 2: DeBERTa-v3 + CLIP-ViT-L Contrastive Classifier
    # -------------------------------------------------------------
    print("\n[2/3] Training SOTA Model 2: DeBERTa-v3 + CLIP-ViT-L Contrastive Interaction...", flush=True)
    m2_oof, m2_test = train_sota_model(
        SOTA_DeBERTa_CLIP_Contrastive,
        text_type="deberta",
        dev_data=dev_data,
        test_data=test_data,
        num_epochs=18,
        lr=3e-4,
        weight_decay=0.08,
        alpha=0.25
    )
    m2_val_acc = accuracy_score(dev_y, m2_oof.argmax(axis=-1))
    m2_test_acc = accuracy_score(test_y, m2_test.argmax(axis=-1))
    print(f"  Model 2 OOF Val Acc: {m2_val_acc*100:.2f}% | Test Acc: {m2_test_acc*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # Train Model 3: Bi-directional Cross-Attention Transformer
    # -------------------------------------------------------------
    print("\n[3/3] Training SOTA Model 3: Bi-directional Cross-Attention Transformer...", flush=True)
    m3_oof, m3_test = train_sota_model(
        SOTA_BiDirectionalCrossAttention,
        text_type="roberta",
        dev_data=dev_data,
        test_data=test_data,
        num_epochs=18,
        lr=3e-4,
        weight_decay=0.08,
        alpha=0.25
    )
    m3_val_acc = accuracy_score(dev_y, m3_oof.argmax(axis=-1))
    m3_test_acc = accuracy_score(test_y, m3_test.argmax(axis=-1))
    print(f"  Model 3 OOF Val Acc: {m3_val_acc*100:.2f}% | Test Acc: {m3_test_acc*100:.2f}%", flush=True)

    # -------------------------------------------------------------
    # Meta-Learner Stacking Ensemble & Calibrated Threshold
    # -------------------------------------------------------------
    print("\n[Stacking Ensemble] Training Meta-Learner on Out-Of-Fold Predictions...", flush=True)
    X_oof = np.hstack([m1_oof, m2_oof, m3_oof])
    X_test = np.hstack([m1_test, m2_test, m3_test])
    
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta_learner.fit(X_oof, dev_y)
    
    meta_oof_probs = meta_learner.predict_proba(X_oof)[:, 1]
    meta_test_probs = meta_learner.predict_proba(X_test)[:, 1]
    
    # Calibrate probability threshold tau on OOF predictions
    best_tau = 0.5
    best_val_score = 0.0
    for tau in np.arange(0.35, 0.65, 0.01):
        preds = (meta_oof_probs >= tau).astype(int)
        score = accuracy_score(dev_y, preds)
        if score > best_val_score:
            best_val_score = score
            best_tau = tau
            
    print(f"Optimal Probability Threshold tau: {best_tau:.3f} (OOF Val Acc: {best_val_score*100:.2f}%)", flush=True)
    
    final_test_preds = (meta_test_probs >= best_tau).astype(int)
    final_test_acc = accuracy_score(test_y, final_test_preds)
    p, r, f1, _ = precision_recall_fscore_support(test_y, final_test_preds, average="macro")
    
    print(f"\n🏆 FINAL SOTA STACKING ENSEMBLE TEST ACCURACY: {final_test_acc*100:.2f}% (F1: {f1*100:.2f}%)", flush=True)

    # Save detailed CSV results
    results_summary = [
        {
            "Model": "Twitter-RoBERTa + CLIP-ViT-L Gated",
            "Validation Acc": f"{m1_val_acc*100:.2f}%",
            "Test Acc": f"{m1_test_acc*100:.2f}%",
            "Test F1": f"{precision_recall_fscore_support(test_y, m1_test.argmax(axis=-1), average='macro')[2]*100:.2f}%"
        },
        {
            "Model": "DeBERTa-v3 + CLIP-ViT-L Contrastive",
            "Validation Acc": f"{m2_val_acc*100:.2f}%",
            "Test Acc": f"{m2_test_acc*100:.2f}%",
            "Test F1": f"{precision_recall_fscore_support(test_y, m2_test.argmax(axis=-1), average='macro')[2]*100:.2f}%"
        },
        {
            "Model": "Bi-directional Cross-Attention",
            "Validation Acc": f"{m3_val_acc*100:.2f}%",
            "Test Acc": f"{m3_test_acc*100:.2f}%",
            "Test F1": f"{precision_recall_fscore_support(test_y, m3_test.argmax(axis=-1), average='macro')[2]*100:.2f}%"
        },
        {
            "Model": "🏆 SOTA Meta-Learner Stacking Ensemble",
            "Validation Acc": f"{best_val_score*100:.2f}%",
            "Test Acc": f"{final_test_acc*100:.2f}%",
            "Test F1": f"{f1*100:.2f}%"
        }
    ]
    
    df_res = pd.DataFrame(results_summary)
    csv_file = METRICS_DIR / "sota_performance_summary.csv"
    df_res.to_csv(csv_file, index=False)
    print(f"\nResults saved to {csv_file}")

if __name__ == "__main__":
    main()
