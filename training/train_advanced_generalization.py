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

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
SPLITS_DIR = DATASET_DIR / "splits"
KFOLD_DIR = DATASET_DIR / "kfold"
RESULTS_DIR = BASE_DIR / "results" / "advanced_generalization"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_sid(sid):
    s = str(sid).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def load_feature_map():
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

# ==============================================================================
# ADVANCED STAGE 2 ARCHITECTURES
# ==============================================================================

class AdvancedResNetBERTClassifier(nn.Module):
    """
    Model 1: Advanced ResNet50 + BERT Classifier
    Supports fusion_mode: 'gated', 'concat', 'interaction', 'full_interaction'
    """
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=128, hidden_dim=96, dropout=0.45, fusion_mode='full_interaction'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        if fusion_mode == 'gated':
            self.gate = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.Sigmoid())
            in_dim = proj_dim * 3
        elif fusion_mode == 'concat':
            in_dim = proj_dim * 2
        elif fusion_mode == 'interaction':
            in_dim = proj_dim * 3
        else: # full_interaction
            in_dim = proj_dim * 4
            
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        
        if self.fusion_mode == 'gated':
            g = self.gate(torch.cat([t, i], dim=1))
            gated = g * t + (1.0 - g) * i
            fused = torch.cat([gated, t, i], dim=1)
        elif self.fusion_mode == 'concat':
            fused = torch.cat([t, i], dim=1)
        elif self.fusion_mode == 'interaction':
            inter = t * i
            fused = torch.cat([t, i, inter], dim=1)
        else: # full_interaction
            inter = t * i
            diff = torch.abs(t - i)
            fused = torch.cat([t, i, inter, diff], dim=1)
            
        return self.classifier(fused)

class AdvancedCLIPClassifier(nn.Module):
    """
    Model 2: Advanced CLIP Classifier with Contrastive Alignment & Multi-Feature Interaction
    """
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=96, hidden_dim=64, dropout=0.45, use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        in_dim = (proj_dim * 4 + 1) if use_diff else (proj_dim * 3 + 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        
        t_norm = t / (t.norm(dim=-1, keepdim=True) + 1e-8)
        i_norm = i / (i.norm(dim=-1, keepdim=True) + 1e-8)
        
        inter = t_norm * i_norm
        cos_sim = (t_norm * i_norm).sum(dim=-1, keepdim=True)
        
        if self.use_diff:
            diff = torch.abs(t_norm - i_norm)
            fused = torch.cat([t_norm, i_norm, inter, diff, cos_sim], dim=1)
        else:
            fused = torch.cat([t_norm, i_norm, inter, cos_sim], dim=1)
            
        return self.classifier(fused)

class AdvancedViLTClassifier(nn.Module):
    """
    Model 3: Advanced ViLT Bi-Directional Cross-Attention Classifier
    """
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=128, hidden_dim=96, num_heads=4, dropout=0.45):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.cross_attn_text = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.cross_attn_img = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, text_feat, image_feat):
        t_p = self.text_proj(text_feat).unsqueeze(1)
        i_p = self.image_proj(image_feat).unsqueeze(1)
        
        t_attn, _ = self.cross_attn_text(query=t_p, key=i_p, value=i_p)
        i_attn, _ = self.cross_attn_img(query=i_p, key=t_p, value=t_p)
        
        t_out = self.norm_t(t_p + t_attn).squeeze(1)
        i_out = self.norm_i(i_p + i_attn).squeeze(1)
        
        fused = torch.cat([t_out, i_out], dim=1)
        return self.classifier(fused)

# ==============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ==============================================================================

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

def run_experiment_on_dev(model_class, text_feats, image_feats, sid_to_idx, hparams):
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    
    tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
    val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
    
    tr_t, tr_i = text_feats[tr_indices], image_feats[tr_indices]
    tr_y = (train_df["label"] == "positive").astype(int).values
    val_t, val_i = text_feats[val_indices], image_feats[val_indices]
    val_y = (val_df["label"] == "positive").astype(int).values
    
    noise_std = hparams.get("noise_std", 0.0)
    batch_size = hparams.get("batch_size", 32)
    lr = hparams.get("lr", 1e-4)
    weight_decay = hparams.get("weight_decay", 0.05)
    dropout = hparams.get("dropout", 0.45)
    proj_dim = hparams.get("proj_dim", 128)
    hidden_dim = hparams.get("hidden_dim", 96)
    fusion_mode = hparams.get("fusion_mode", "full_interaction")
    use_diff = hparams.get("use_diff", True)
    num_heads = hparams.get("num_heads", 4)
    label_smooth = hparams.get("label_smooth", 0.0)
    max_epochs = hparams.get("epochs", 35)
    patience = hparams.get("patience", 4)
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=noise_std), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=batch_size, shuffle=False)
    
    # Instantiate model with specific kwargs based on class
    kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
    if model_class == AdvancedResNetBERTClassifier:
        kwargs["fusion_mode"] = fusion_mode
    elif model_class == AdvancedCLIPClassifier:
        kwargs["use_diff"] = use_diff
    elif model_class == AdvancedViLTClassifier:
        kwargs["num_heads"] = num_heads
        
    model = model_class(**kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_metrics = None
    no_improve = 0
    history = []
    
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "gap": tr_acc - val_acc
        })
        
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = {
                "best_epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "train_val_gap": tr_acc - val_acc
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
                
    return best_metrics, history, best_state

def execute_stage2_experiments():
    print("==========================================================================", flush=True)
    print("STAGE 2: ADVANCED GENERALIZATION CONTROLLED EXPERIMENTS (DEV SET ONLY)", flush=True)
    print("==========================================================================", flush=True)
    
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    models = [
        ("ResNet50+BERT", AdvancedResNetBERTClassifier, "model_1_resnet_bert"),
        ("CLIP", AdvancedCLIPClassifier, "model_2_clip"),
        ("ViLT", AdvancedViLTCrossModalClassifier if 'AdvancedViLTCrossModalClassifier' in globals() else AdvancedViLTClassifier, "model_3_vilt")
    ]
    
    # Stage 2 Model-Specific Experiment Grids
    resnet_bert_exps = [
        ("Exp 0: Stage 1 Winner Baseline", {"lr": 1.5e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 256, "hidden_dim": 128, "fusion_mode": "gated", "noise_std": 0.0, "label_smooth": 0.0}),
        ("Exp 1: Full Interaction Fusion", {"lr": 1.2e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "fusion_mode": "full_interaction", "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 2: Concat + Interaction Fusion", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "fusion_mode": "interaction", "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 3: Low LR (8e-5) + Weight Decay 0.08", {"lr": 8e-5, "dropout": 0.45, "weight_decay": 0.08, "proj_dim": 128, "hidden_dim": 96, "fusion_mode": "full_interaction", "noise_std": 0.0, "label_smooth": 0.05}),
        ("Exp 4: Compact Head (96) + Dropout 0.50", {"lr": 8e-5, "dropout": 0.50, "weight_decay": 0.08, "proj_dim": 96, "hidden_dim": 64, "fusion_mode": "full_interaction", "noise_std": 0.0, "label_smooth": 0.05}),
        ("Exp 5: Train Feature Jitter (0.01)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "fusion_mode": "full_interaction", "noise_std": 0.01, "label_smooth": 0.03}),
        ("Exp 6: Stage 2 Optimal Combined", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.06, "proj_dim": 128, "hidden_dim": 96, "fusion_mode": "full_interaction", "noise_std": 0.005, "label_smooth": 0.04})
    ]
    
    clip_exps = [
        ("Exp 0: Stage 1 Winner Baseline", {"lr": 1.5e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 256, "hidden_dim": 128, "use_diff": False, "noise_std": 0.0, "label_smooth": 0.0}),
        ("Exp 1: Full Contrastive Interaction", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "use_diff": True, "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 2: Compact Head (96) + WD 0.08", {"lr": 8e-5, "dropout": 0.45, "weight_decay": 0.08, "proj_dim": 96, "hidden_dim": 64, "use_diff": True, "noise_std": 0.0, "label_smooth": 0.05}),
        ("Exp 3: High WD (0.10) + Dropout 0.50", {"lr": 7e-5, "dropout": 0.50, "weight_decay": 0.10, "proj_dim": 96, "hidden_dim": 64, "use_diff": True, "noise_std": 0.0, "label_smooth": 0.05}),
        ("Exp 4: Low LR (6e-5) + Moderate WD", {"lr": 6e-5, "dropout": 0.45, "weight_decay": 0.06, "proj_dim": 128, "hidden_dim": 96, "use_diff": True, "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 5: Train Feature Jitter (0.01)", {"lr": 8e-5, "dropout": 0.45, "weight_decay": 0.08, "proj_dim": 96, "hidden_dim": 64, "use_diff": True, "noise_std": 0.01, "label_smooth": 0.04}),
        ("Exp 6: Stage 2 Optimal Combined", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.07, "proj_dim": 128, "hidden_dim": 96, "use_diff": True, "noise_std": 0.005, "label_smooth": 0.04})
    ]
    
    vilt_exps = [
        ("Exp 0: Stage 1 Winner Baseline", {"lr": 2e-4, "dropout": 0.45, "weight_decay": 0.01, "proj_dim": 256, "hidden_dim": 256, "num_heads": 4, "noise_std": 0.0, "label_smooth": 0.0}),
        ("Exp 1: Bi-Cross Attention (128-dim)", {"lr": 1.2e-4, "dropout": 0.45, "weight_decay": 0.03, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 2: Bi-Cross Attention (2 Heads)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.04, "proj_dim": 128, "hidden_dim": 96, "num_heads": 2, "noise_std": 0.0, "label_smooth": 0.03}),
        ("Exp 3: Low LR (8e-5) + Weight Decay 0.05", {"lr": 8e-5, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "noise_std": 0.0, "label_smooth": 0.04}),
        ("Exp 4: Compact Head (96) + Dropout 0.50", {"lr": 8e-5, "dropout": 0.50, "weight_decay": 0.05, "proj_dim": 96, "hidden_dim": 64, "num_heads": 4, "noise_std": 0.0, "label_smooth": 0.05}),
        ("Exp 5: Mild Feature Jitter (0.008)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.04, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "noise_std": 0.008, "label_smooth": 0.03}),
        ("Exp 6: Stage 2 Optimal Combined", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.04, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "noise_std": 0.005, "label_smooth": 0.03})
    ]
    
    model_exps_map = {
        "ResNet50+BERT": resnet_bert_exps,
        "CLIP": clip_exps,
        "ViLT": vilt_exps
    }
    
    best_model_configs = {}
    
    for m_name, m_class, folder_name in models:
        print(f"\n" + "="*80, flush=True)
        print(f"STAGE 2 EXPERIMENTS FOR MODEL: {m_name}", flush=True)
        print("="*80, flush=True)
        
        m_results_dir = RESULTS_DIR / folder_name
        m_results_dir.mkdir(parents=True, exist_ok=True)
        
        exp_records = []
        best_exp_val_loss = float("inf")
        best_exp_info = None
        best_state_to_save = None
        best_history_to_save = None
        
        exp_list = model_exps_map[m_name]
        
        for exp_name, hparams in exp_list:
            metrics, history, state = run_experiment_on_dev(m_class, text_feats, image_feats, sid_to_idx, hparams)
            metrics["experiment"] = exp_name
            exp_records.append(metrics)
            
            print(f"[{exp_name:44s}] Epoch {metrics['best_epoch']:2d} | Train Acc: {metrics['train_acc']*100:.2f}% | Val Acc: {metrics['val_acc']*100:.2f}% | Val F1: {metrics['val_f1']*100:.2f}% | Gap: {metrics['train_val_gap']*100:.2f} pp | Val Loss: {metrics['val_loss']:.4f}", flush=True)
            
            if metrics["val_loss"] < best_exp_val_loss:
                best_exp_val_loss = metrics["val_loss"]
                best_exp_info = (exp_name, hparams, metrics)
                best_state_to_save = state
                best_history_to_save = history
                
        df_exps = pd.DataFrame(exp_records)
        df_exps.to_csv(m_results_dir / "experiment_grid_results.csv", index=False)
        
        torch.save(best_state_to_save, m_results_dir / "best_val_checkpoint.pt")
        df_hist = pd.DataFrame(best_history_to_save)
        df_hist.to_csv(m_results_dir / "final_training_history.csv", index=False)
        
        best_model_configs[m_name] = {
            "hparams": best_exp_info[1],
            "metrics": best_exp_info[2],
            "exp_name": best_exp_info[0]
        }
        
        print(f"\n>>> STAGE 2 WINNING CONFIGURATION FOR {m_name}: {best_exp_info[0]}", flush=True)
        print(f"    Best Val Loss: {best_exp_info[2]['val_loss']:.4f} | Val Acc: {best_exp_info[2]['val_acc']*100:.2f}% | Val F1: {best_exp_info[2]['val_f1']*100:.2f}% | Train-Val Gap: {best_exp_info[2]['train_val_gap']*100:.2f} pp", flush=True)

    with open(RESULTS_DIR / "best_model_configs.json", "w") as f:
        json.dump(best_model_configs, f, indent=4)

if __name__ == "__main__":
    execute_stage2_experiments()
