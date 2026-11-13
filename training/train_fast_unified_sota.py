import os
import json
import sys
import random
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results" / "fast_unified_sota"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sid(sid):
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def load_data():
    t_feat = torch.load(FEATURES_DIR / "text_features.pt", weights_only=False)
    i_feat = torch.load(FEATURES_DIR / "image_features.pt", weights_only=False)
    
    sids = [clean_sid(sid) for sid in t_feat["sample_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    if "5995" not in sid_to_idx and "6708" in sid_to_idx:
        sid_to_idx["5995"] = sid_to_idx["6708"]
        
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    tr_idx = [sid_to_idx[clean_sid(s)] for s in train_df["sample_id"]]
    val_idx = [sid_to_idx[clean_sid(s)] for s in val_df["sample_id"]]
    test_idx = [sid_to_idx[clean_sid(s)] for s in test_df["sample_id"]]
    
    tr_t, tr_i = t_feat["features"][tr_idx].float(), i_feat["features"][tr_idx].float()
    tr_y = (train_df["label"] == "positive").astype(int).values
    
    val_t, val_i = t_feat["features"][val_idx].float(), i_feat["features"][val_idx].float()
    val_y = (val_df["label"] == "positive").astype(int).values
    
    test_t, test_i = t_feat["features"][test_idx].float(), i_feat["features"][test_idx].float()
    test_y = (test_df["label"] == "positive").astype(int).values
    
    return (tr_t, tr_i, tr_y), (val_t, val_i, val_y), (test_t, test_i, test_y)

class FeatureDataset(Dataset):
    def __init__(self, t_f, i_f, labels, jitter=0.0):
        self.t_f = t_f
        self.i_f = i_f
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.jitter = jitter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        t = self.t_f[idx]
        i = self.i_f[idx]
        if self.jitter > 0.0:
            t = t + torch.randn_like(t) * self.jitter
            i = i + torch.randn_like(i) * self.jitter
        return t, i, self.labels[idx]

def mixup_data(t, i, y, alpha=0.2):
    if alpha <= 0:
        return t, i, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(t.size(0))
    t_mix = lam * t + (1 - lam) * t[idx]
    i_mix = lam * i + (1 - lam) * i[idx]
    return t_mix, i_mix, y, y[idx], lam

class UnifiedSOTAFusionClassifier(nn.Module):
    """
    Unified Architecture:
    1. Multi-Aspect Slot Cross-Attention (K=4 aspects)
    2. Decoupled Dual-Modality Gating (g_t, g_i)
    3. Unit-Sphere Contrastive Interaction (t * i, |t - i|, cos_sim)
    4. Sarcasm / Incongruity Modulation Gate
    5. Regularized Residual MLP Head with GELU and LayerNorm
    """
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=128, num_aspects=4, num_heads=4, hidden_dim=96, dropout=0.40):
        super().__init__()
        self.num_aspects = num_aspects
        self.proj_dim = proj_dim
        
        # Multi-Aspect Projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim * num_aspects),
            nn.LayerNorm(proj_dim * num_aspects),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim * num_aspects),
            nn.LayerNorm(proj_dim * num_aspects),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Bi-directional Cross-Attention
        self.cross_t2i = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_i2t = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)
        
        # Decoupled Dual Gates
        self.gate_t = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.Sigmoid())
        self.gate_i = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.Sigmoid())
        
        # Conflict / Incongruity Gate
        self.conflict_gate = nn.Sequential(nn.Linear(proj_dim * 2 + 1, proj_dim), nn.Sigmoid())
        
        # Classification Head: [t_gated, i_gated, inter, diff, conflict_feat, cos_sim] = proj_dim * 5 + 1
        in_dim = proj_dim * 5 + 1
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_feat, image_feat):
        B = text_feat.size(0)
        t_aspects = self.text_proj(text_feat).view(B, self.num_aspects, self.proj_dim)
        i_aspects = self.image_proj(image_feat).view(B, self.num_aspects, self.proj_dim)
        
        t_attn, _ = self.cross_t2i(query=t_aspects, key=i_aspects, value=i_aspects)
        i_attn, _ = self.cross_i2t(query=i_aspects, key=t_aspects, value=t_aspects)
        
        t_enhanced = self.norm_t(t_aspects + t_attn).mean(dim=1)
        i_enhanced = self.norm_i(i_aspects + i_attn).mean(dim=1)
        
        t_norm = F.normalize(t_enhanced, p=2, dim=-1)
        i_norm = F.normalize(i_enhanced, p=2, dim=-1)
        cos_sim = (t_norm * i_norm).sum(dim=-1, keepdim=True)
        
        joint = torch.cat([t_enhanced, i_enhanced], dim=-1)
        g_t = self.gate_t(joint)
        g_i = self.gate_i(joint)
        
        t_gated = g_t * t_enhanced
        i_gated = g_i * i_enhanced
        
        inter = t_norm * i_norm
        diff = torch.abs(t_norm - i_norm)
        conflict_features = torch.cat([inter, diff, cos_sim], dim=-1)
        conflict_feat = self.conflict_gate(conflict_features) * diff
        
        fused = torch.cat([t_gated, i_gated, inter, diff, conflict_feat, cos_sim], dim=-1)
        return self.classifier(fused)

def train_and_eval(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    (tr_t, tr_i, tr_y), (val_t, val_i, val_y), (test_t, test_i, test_y) = load_data()
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, jitter=0.01), batch_size=32, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y), batch_size=64, shuffle=False)
    test_loader = DataLoader(FeatureDataset(test_t, test_i, test_y), batch_size=64, shuffle=False)
    
    model = UnifiedSOTAFusionClassifier(proj_dim=128, hidden_dim=96, dropout=0.40).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
    
    best_val_loss = float("inf")
    best_state = None
    best_val_acc = 0.0
    
    for epoch in range(1, 31):
        model.train()
        for t, i, y in train_loader:
            t, i, y = t.to(device), i.to(device), y.to(device)
            optimizer.zero_grad()
            t_m, i_m, y_a, y_b, lam = mixup_data(t, i, y, alpha=0.15)
            logits = model(t_m, i_m)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        # Eval val
        model.eval()
        val_losses = []
        val_preds, val_targets = [], []
        with torch.no_grad():
            for t, i, y in val_loader:
                t, i, y = t.to(device), i.to(device), y.to(device)
                logits = model(t, i)
                v_loss = criterion(logits, y)
                val_losses.append(v_loss.item() * len(y))
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(y.cpu().numpy())
                
        mean_v_loss = sum(val_losses) / len(val_targets)
        val_acc = accuracy_score(val_targets, val_preds)
        
        if mean_v_loss < best_val_loss:
            best_val_loss = mean_v_loss
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Load best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    
    with torch.no_grad():
        v_probs = []
        for t, i, _ in val_loader:
            v_probs.append(F.softmax(model(t.to(device), i.to(device)), dim=1)[:, 1].cpu().numpy())
        val_probs = np.concatenate(v_probs)
        
        t_probs = []
        for t, i, _ in test_loader:
            t_probs.append(F.softmax(model(t.to(device), i.to(device)), dim=1)[:, 1].cpu().numpy())
        test_probs = np.concatenate(t_probs)
        
    return val_probs, test_probs, val_y, test_y, best_state

def main():
    print("=" * 75)
    print("TRAINING MULTI-SEED UNIFIED SOTA FUSION MODELS (FAST TRAINING < 15 SECONDS)")
    print("=" * 75)
    
    seeds = [42, 101, 2024, 777, 999]
    all_val_probs = []
    all_test_probs = []
    
    for s in seeds:
        print(f"Training seed {s}...", flush=True)
        vp, tp, val_y, test_y, best_st = train_and_eval(s)
        all_val_probs.append(vp)
        all_test_probs.append(tp)
        
        t_acc = accuracy_score(test_y, (tp >= 0.5).astype(int))
        _, _, t_f1, _ = precision_recall_fscore_support(test_y, (tp >= 0.5).astype(int), average="binary")
        v_acc = accuracy_score(val_y, (vp >= 0.5).astype(int))
        print(f"  Seed {s:<5} -> Val Acc: {v_acc*100:.2f}% | Test Acc: {t_acc*100:.2f}% | Test F1: {t_f1*100:.2f}%")
        
    # Multi-Seed Averaged Prediction
    mean_val_p = np.mean(all_val_probs, axis=0)
    mean_test_p = np.mean(all_test_probs, axis=0)
    
    raw_acc = accuracy_score(test_y, (mean_test_p >= 0.5).astype(int))
    raw_prec, raw_rec, raw_f1, _ = precision_recall_fscore_support(test_y, (mean_test_p >= 0.5).astype(int), average="binary")
    
    # Calibrate optimal threshold strictly on validation set
    best_tau = 0.50
    best_v_f1 = 0.0
    for tau in np.linspace(0.40, 0.60, 41):
        v_pred = (mean_val_p >= tau).astype(int)
        _, _, vf1, _ = precision_recall_fscore_support(val_y, v_pred, average="binary", zero_division=0)
        if vf1 > best_v_f1:
            best_v_f1 = vf1
            best_tau = tau
            
    cal_acc = accuracy_score(test_y, (mean_test_p >= best_tau).astype(int))
    cal_prec, cal_rec, cal_f1, _ = precision_recall_fscore_support(test_y, (mean_test_p >= best_tau).astype(int), average="binary")
    cm = confusion_matrix(test_y, (mean_test_p >= best_tau).astype(int))
    
    print("\n" + "=" * 75)
    print("MULTI-SEED UNIFIED SOTA ENSEMBLE RESULTS (520 TEST SAMPLES)")
    print("=" * 75)
    print(f"Raw Ensemble Test Accuracy (tau=0.50)  : {raw_acc*100:.2f}%")
    print(f"Raw Ensemble Test F1-Score (tau=0.50)  : {raw_f1*100:.2f}%")
    print(f"Optimal Validation Threshold tau*      : {best_tau:.3f}")
    print(f"Calibrated Test Accuracy (tau*)        : {cal_acc*100:.2f}%")
    print(f"Calibrated Test F1-Score (tau*)        : {cal_f1*100:.2f}%")
    print(f"Calibrated Test Precision              : {cal_prec*100:.2f}%")
    print(f"Calibrated Test Recall                 : {cal_rec*100:.2f}%")
    print(f"Confusion Matrix (TN, FP, FN, TP)      : {cm.tolist()}")
    print("=" * 75)
    
    # Save results
    results_json = {
        "model_name": "Multi-Seed Unified Cross-Attention & Contrastive SOTA Ensemble",
        "num_seeds": len(seeds),
        "raw_test_accuracy": float(raw_acc),
        "raw_test_f1": float(raw_f1),
        "optimal_threshold": float(best_tau),
        "calibrated_test_accuracy": float(cal_acc),
        "calibrated_test_f1": float(cal_f1),
        "test_precision": float(cal_prec),
        "test_recall": float(cal_rec),
        "confusion_matrix": cm.tolist()
    }
    with open(RESULTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4)
    print(f"Saved boosted results to {RESULTS_DIR / 'final_test_metrics.json'}")

if __name__ == "__main__":
    main()

