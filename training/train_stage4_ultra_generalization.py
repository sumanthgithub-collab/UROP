import os
import json
import sys
import io
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results" / "stage4_ultra_generalization"
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
    def __init__(self, text_feats, image_feats, labels):
        self.text_feats = torch.as_tensor(text_feats.copy() if isinstance(text_feats, np.ndarray) else text_feats, dtype=torch.float32)
        self.image_feats = torch.as_tensor(image_feats.copy() if isinstance(image_feats, np.ndarray) else image_feats, dtype=torch.float32)
        self.labels = torch.as_tensor(labels.copy() if isinstance(labels, np.ndarray) else labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_feats[idx], self.image_feats[idx], self.labels[idx]

def mixup_data(t_feat, i_feat, labels, alpha=0.2):
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

class Stage4ResNetBERT(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=128, hidden_dim=96, dropout=0.45):
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
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        inter = t * i
        diff = torch.abs(t - i)
        fused = torch.cat([t, i, inter, diff], dim=1)
        h = self.inter_proj(fused)
        return self.classifier(h)

class Stage4CLIP(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, proj_dim=128, hidden_dim=96, dropout=0.45):
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
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 4 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        t_norm = F.normalize(t, p=2, dim=-1)
        i_norm = F.normalize(i, p=2, dim=-1)
        inter = t_norm * i_norm
        diff = torch.abs(t_norm - i_norm)
        cos_sim = (t_norm * i_norm).sum(dim=-1, keepdim=True)
        fused = torch.cat([t_norm, i_norm, inter, diff, cos_sim], dim=1)
        h = self.inter_proj(fused)
        return self.classifier(h)

class Stage4ViLT(nn.Module):
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
        self.attn_text = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_img = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t_p = self.text_proj(text_feat).unsqueeze(1)
        i_p = self.image_proj(image_feat).unsqueeze(1)
        t_attn, _ = self.attn_text(query=t_p, key=i_p, value=i_p)
        i_attn, _ = self.attn_img(query=i_p, key=t_p, value=t_p)
        t_out = self.norm_t(t_p + t_attn).squeeze(1)
        i_out = self.norm_i(i_p + i_attn).squeeze(1)
        fused = torch.cat([t_out, i_out], dim=1)
        h = self.inter_proj(fused)
        return self.classifier(h)

class Stage4TransformerFusion(nn.Module):
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
        self.attn_text = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_img = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.norm_t = nn.LayerNorm(proj_dim)
        self.norm_i = nn.LayerNorm(proj_dim)
        self.inter_proj = nn.Sequential(
            nn.Linear(proj_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, text_feat, image_feat):
        t_p = self.text_proj(text_feat).unsqueeze(1)
        i_p = self.image_proj(image_feat).unsqueeze(1)
        t_attn, _ = self.attn_text(query=t_p, key=i_p, value=i_p)
        i_attn, _ = self.attn_img(query=i_p, key=t_p, value=t_p)
        t_out = self.norm_t(t_p + t_attn).squeeze(1)
        i_out = self.norm_i(i_p + i_attn).squeeze(1)
        inter = t_out * i_out
        diff = torch.abs(t_out - i_out)
        fused = torch.cat([t_out, i_out, inter, diff], dim=1)
        h = self.inter_proj(fused)
        return self.classifier(h)

def train_one_epoch_stage4(model, loader, optimizer, criterion, alpha=0.2):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for t_feat, i_feat, labels in loader:
        t_feat, i_feat, labels = t_feat.to(device), i_feat.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if alpha > 0.0:
            mixed_t, mixed_i, y_a, y_b, lam = mixup_data(t_feat, i_feat, labels, alpha=alpha)
            logits = model(mixed_t, mixed_i)
            loss = lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
        else:
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

def evaluate_stage4(model, loader, criterion=None):
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

def run_experiment_stage4(model_class, text_feats, image_feats, sid_to_idx, hparams):
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    
    tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
    val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
    
    tr_t, tr_i = text_feats[tr_indices], image_feats[tr_indices]
    tr_y = (train_df["label"] == "positive").astype(int).values
    val_t, val_i = text_feats[val_indices], image_feats[val_indices]
    val_y = (val_df["label"] == "positive").astype(int).values
    
    batch_size = hparams.get("batch_size", 32)
    lr = hparams.get("lr", 1e-4)
    weight_decay = hparams.get("weight_decay", 0.05)
    dropout = hparams.get("dropout", 0.45)
    proj_dim = hparams.get("proj_dim", 128)
    hidden_dim = hparams.get("hidden_dim", 96)
    num_heads = hparams.get("num_heads", 4)
    alpha = hparams.get("mixup_alpha", 0.2)
    label_smooth = hparams.get("label_smooth", 0.03)
    max_epochs = hparams.get("epochs", 35)
    patience = hparams.get("patience", 4)
    
    train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y), batch_size=batch_size, shuffle=False)
    
    kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
    if model_class in [Stage4ViLT, Stage4TransformerFusion]:
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
        tr_loss, tr_acc = train_one_epoch_stage4(model, train_loader, optimizer, criterion, alpha=alpha)
        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate_stage4(model, val_loader, criterion)
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

def execute_stage4_experiments():
    print("==========================================================================", flush=True)
    print("STAGE 4: ULTRA-HIGH GENERALIZATION CONTROLLED EXPERIMENTS (DEV SET ONLY)", flush=True)
    print("==========================================================================", flush=True)
    
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    models = [
        ("ResNet50+BERT", Stage4ResNetBERT, "model_1_resnet_bert"),
        ("CLIP", Stage4CLIP, "model_2_clip"),
        ("ViLT", Stage4ViLT, "model_3_vilt"),
        ("TransformerFusion", Stage4TransformerFusion, "model_4_transformer_fusion")
    ]
    
    stage4_exps = {
        "ResNet50+BERT": [
            ("Stage 4 Exp 1: Manifold Mixup (alpha=0.2)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "mixup_alpha": 0.2, "label_smooth": 0.03}),
            ("Stage 4 Exp 2: Mixup (0.15) + WD 0.06", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.06, "proj_dim": 128, "hidden_dim": 96, "mixup_alpha": 0.15, "label_smooth": 0.03})
        ],
        "CLIP": [
            ("Stage 4 Exp 1: Contrastive Interaction + Mixup (0.2)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "mixup_alpha": 0.2, "label_smooth": 0.03}),
            ("Stage 4 Exp 2: Contrastive Interaction + Mixup (0.15)", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.06, "proj_dim": 128, "hidden_dim": 96, "mixup_alpha": 0.15, "label_smooth": 0.03})
        ],
        "ViLT": [
            ("Stage 4 Exp 1: Bi-Cross Attention + Mixup (0.2)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.04, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "mixup_alpha": 0.2, "label_smooth": 0.03}),
            ("Stage 4 Exp 2: Bi-Cross Attention + Mixup (0.15)", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "mixup_alpha": 0.15, "label_smooth": 0.03})
        ],
        "TransformerFusion": [
            ("Stage 4 Exp 1: Cross-Modal Transformer + Mixup (0.2)", {"lr": 1e-4, "dropout": 0.45, "weight_decay": 0.04, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "mixup_alpha": 0.2, "label_smooth": 0.03}),
            ("Stage 4 Exp 2: Cross-Modal Transformer + Mixup (0.15)", {"lr": 9e-5, "dropout": 0.45, "weight_decay": 0.05, "proj_dim": 128, "hidden_dim": 96, "num_heads": 4, "mixup_alpha": 0.15, "label_smooth": 0.03})
        ]
    }
    
    best_model_configs = {}
    
    for m_name, m_class, folder_name in models:
        print(f"\n" + "="*80, flush=True)
        print(f"STAGE 4 EXPERIMENTS FOR MODEL: {m_name}", flush=True)
        print("="*80, flush=True)
        
        m_results_dir = RESULTS_DIR / folder_name
        m_results_dir.mkdir(parents=True, exist_ok=True)
        
        exp_records = []
        best_exp_val_loss = float("inf")
        best_exp_info = None
        best_state_to_save = None
        best_history_to_save = None
        
        for exp_name, hparams in stage4_exps[m_name]:
            metrics, history, state = run_experiment_stage4(m_class, text_feats, image_feats, sid_to_idx, hparams)
            metrics["experiment"] = exp_name
            exp_records.append(metrics)
            
            print(f"[{exp_name:52s}] Epoch {metrics['best_epoch']:2d} | Train Acc: {metrics['train_acc']*100:.2f}% | Val Acc: {metrics['val_acc']*100:.2f}% | Val F1: {metrics['val_f1']*100:.2f}% | Gap: {metrics['train_val_gap']*100:.2f} pp | Val Loss: {metrics['val_loss']:.4f}", flush=True)
            
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
        
        print(f"\n>>> STAGE 4 WINNING CONFIGURATION FOR {m_name}: {best_exp_info[0]}", flush=True)
        print(f"    Best Val Loss: {best_exp_info[2]['val_loss']:.4f} | Val Acc: {best_exp_info[2]['val_acc']*100:.2f}% | Val F1: {best_exp_info[2]['val_f1']*100:.2f}% | Train-Val Gap: {best_exp_info[2]['train_val_gap']*100:.2f} pp", flush=True)

    with open(RESULTS_DIR / "best_model_configs.json", "w", encoding="utf-8") as f:
        json.dump(best_model_configs, f, indent=4)

if __name__ == "__main__":
    execute_stage4_experiments()
