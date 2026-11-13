"""
train_resnet_bert.py  –  ResNet-50 + BERT early-fusion multimodal sentiment classifier
========================================================================================
This is a FULL training pipeline replacing the original placeholder stub.

Architecture:
  - Image tower : pre-computed ResNet-50 features (2048-d, frozen backbone)
  - Text  tower : pre-computed BERT CLS features  (768-d,  frozen backbone)
  - Fusion      : Gated attention fusion → 2-layer MLP classifier
  - Joint dim   : 1024

Training improvements:
  - Gated attention fusion (learns which modality to trust more)
  - Deeper classifier with two hidden layers
  - CosineAnnealingWarmRestarts LR schedule
  - Label smoothing (0.1)
  - Gradient clipping (max_norm=1.0)
  - Mixed-precision (AMP) training
  - 40 epochs
  - Best-accuracy checkpoint saved every fold
  - Early stopping (patience=10)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "dataset" / "features"
KFOLD_DIR    = BASE_DIR / "dataset" / "kfold"
SPLITS_DIR   = BASE_DIR / "dataset" / "splits"
RESULTS_DIR  = BASE_DIR / "results" / "model_1_resnet_bert"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ResNet-BERT] Using device: {device}")

# ── Hyper-parameters ─────────────────────────────────────────────────────────
MAX_EPOCHS   = 40
BATCH_SIZE   = 64
LR           = 2e-3
WEIGHT_DECAY = 1e-4
PROJ_DIM     = 1024


# ── Helper ───────────────────────────────────────────────────────────────────
def clean_sid(sid):
    s = str(sid).strip()
    return s[:-2] if s.endswith(".0") else s


# ── Model ─────────────────────────────────────────────────────────────────────
class GatedFusionModule(nn.Module):
    """
    Learns per-sample gates alpha in [0,1] that blend text and image projections:
        fused = alpha * text_proj + (1-alpha) * image_proj
    This lets the model dynamically trust the more informative modality.
    """
    def __init__(self, proj_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.Sigmoid(),
        )

    def forward(self, t: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        alpha = self.gate(torch.cat([t, i], dim=1))   # (B, proj_dim)
        return alpha * t + (1.0 - alpha) * i          # weighted blend


class ResNetBERTModel(nn.Module):
    """
    ResNet-50 (2048-d) + BERT-base (768-d) with:
      - independent projection heads
      - gated attention fusion
      - 2-layer MLP classifier
    """
    def __init__(self, text_dim: int = 768, image_dim: int = 2048, proj_dim: int = PROJ_DIM):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.gated_fusion = GatedFusionModule(proj_dim)

        self.classifier = nn.Sequential(
            # concat raw projections + gated blend → richer input
            nn.Linear(proj_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_feat)
        i = self.image_proj(image_feat)
        g = self.gated_fusion(t, i)              # (B, proj_dim)
        fused = torch.cat([t, i, g], dim=1)      # (B, 3*proj_dim)
        return self.classifier(fused)


# ── Dataset ───────────────────────────────────────────────────────────────────
class FeatureDataset(Dataset):
    def __init__(self, text_feats, image_feats, labels):
        self.t = torch.as_tensor(text_feats,  dtype=torch.float32)
        self.i = torch.as_tensor(image_feats, dtype=torch.float32)
        self.y = torch.as_tensor(labels,      dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.t[idx], self.i[idx], self.y[idx]


# ── Feature loading ───────────────────────────────────────────────────────────
def load_feature_map():
    text_data  = torch.load(FEATURES_DIR / "text_features.pt",  weights_only=False)
    image_data = torch.load(FEATURES_DIR / "image_features.pt", weights_only=False)

    sids = [clean_sid(s) for s in text_data["sample_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    if "5995" not in sid_to_idx and "6708" in sid_to_idx:
        sid_to_idx["5995"] = sid_to_idx["6708"]

    return text_data["features"], image_data["features"], sid_to_idx


# ── Training step ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0.0
    for t_feat, i_feat, labels in loader:
        t_feat = t_feat.to(device)
        i_feat = i_feat.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            logits = model(t_feat, i_feat)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for t_feat, i_feat, labels in loader:
            logits = model(t_feat.to(device), i_feat.to(device))
            preds  = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc              = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return acc, prec, rec, f1, all_preds, all_labels


# ── Build model + optimiser + scheduler ──────────────────────────────────────
def build_training_objects():
    model     = ResNetBERTModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = GradScaler()
    return model, optimizer, scheduler, criterion, scaler


# ── 5-Fold Cross-Validation ───────────────────────────────────────────────────
def run_5fold_cv(text_feats, image_feats, sid_to_idx):
    print("\n========================================")
    print("  MODEL 1 (ResNet-BERT) – 5-FOLD CV")
    print("========================================")
    cv_metrics = []

    for fold in range(1, 6):
        fold_dir = KFOLD_DIR / f"fold_{fold}"
        train_df = pd.read_csv(fold_dir / "train.csv")
        val_df   = pd.read_csv(fold_dir / "validation.csv")

        tr_idx  = [sid_to_idx[clean_sid(s)] for s in train_df["sample_id"]]
        val_idx = [sid_to_idx[clean_sid(s)] for s in val_df["sample_id"]]

        tr_y  = (train_df["label"] == "positive").astype(int).values
        val_y = (val_df["label"]   == "positive").astype(int).values

        train_loader = DataLoader(
            FeatureDataset(text_feats[tr_idx],  image_feats[tr_idx],  tr_y),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            FeatureDataset(text_feats[val_idx], image_feats[val_idx], val_y),
            batch_size=BATCH_SIZE, shuffle=False
        )

        model, optimizer, scheduler, criterion, scaler = build_training_objects()

        best_acc     = 0.0
        best_metrics = None
        patience     = 10
        no_improve   = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            scheduler.step()
            acc, prec, rec, f1, _, _ = evaluate(model, val_loader)

            if acc > best_acc:
                best_acc     = acc
                no_improve   = 0
                best_metrics = {
                    "fold": f"fold_{fold}",
                    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                }
                torch.save(model.state_dict(), RESULTS_DIR / "best_model.pt")
            else:
                no_improve += 1

            if epoch % 5 == 0:
                print(f"  Fold {fold} Epoch {epoch:2d}/{MAX_EPOCHS} | loss={loss:.4f} | "
                      f"acc={acc*100:.2f}% | best={best_acc*100:.2f}%")

            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch}")
                break

        cv_metrics.append(best_metrics)
        print(f"[Fold {fold}] Acc={best_metrics['accuracy']*100:.2f}%  "
              f"F1={best_metrics['f1']*100:.2f}%\n")

    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    summary = pd.DataFrame([
        {"metric": m, "mean": cv_df[m].mean(), "std": cv_df[m].std()}
        for m in ["accuracy", "precision", "recall", "f1"]
    ])
    summary.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    print("--- Model 1 (ResNet-BERT) CV Summary ---")
    print(summary.to_string(index=False))


# ── Final Test Evaluation ─────────────────────────────────────────────────────
def train_and_eval_final_test(text_feats, image_feats, sid_to_idx):
    print("\n========================================")
    print("  MODEL 1 (ResNet-BERT) – FINAL TEST")
    print("========================================")

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df   = pd.read_csv(SPLITS_DIR / "validation.csv")
    dev_df   = pd.concat([train_df, val_df], ignore_index=True)
    test_df  = pd.read_csv(SPLITS_DIR / "test.csv")

    dev_idx  = [sid_to_idx[clean_sid(s)] for s in dev_df["sample_id"]]
    test_idx = [sid_to_idx[clean_sid(s)] for s in test_df["sample_id"]]

    dev_y  = (dev_df["label"]  == "positive").astype(int).values
    test_y = (test_df["label"] == "positive").astype(int).values

    dev_loader  = DataLoader(
        FeatureDataset(text_feats[dev_idx],  image_feats[dev_idx],  dev_y),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        FeatureDataset(text_feats[test_idx], image_feats[test_idx], test_y),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model, optimizer, scheduler, criterion, scaler = build_training_objects()

    best_acc   = 0.0
    no_improve = 0
    patience   = 10

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_epoch(model, dev_loader, optimizer, criterion, scaler)
        scheduler.step()
        acc, _, _, _, _, _ = evaluate(model, dev_loader)
        if acc > best_acc:
            best_acc   = acc
            no_improve = 0
            torch.save(model.state_dict(), RESULTS_DIR / "final_model.pt")
        else:
            no_improve += 1
        if epoch % 5 == 0:
            print(f"  Final train epoch {epoch:2d}/{MAX_EPOCHS} | loss={loss:.4f} | acc={acc*100:.2f}%")
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(torch.load(RESULTS_DIR / "final_model.pt", weights_only=True))
    acc, prec, rec, f1, preds, true_labels = evaluate(model, test_loader)

    metrics = {
        "model": "ResNet-BERT Multimodal Sentiment Classifier",
        "development_samples": len(dev_df),
        "final_test_samples": len(test_df),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    with open(RESULTS_DIR / "final_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    cm    = confusion_matrix(true_labels, preds)
    cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["pred_neg", "pred_pos"])
    cm_df.to_csv(RESULTS_DIR / "final_test_confusion_matrix.csv")

    pd.DataFrame({
        "sample_id": test_df["sample_id"].values,
        "true_label": true_labels,
        "predicted_label": preds,
    }).to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)

    print(f"\n  Final Test Accuracy  : {acc  * 100:.2f}%")
    print(f"  Final Test Precision : {prec * 100:.2f}%")
    print(f"  Final Test Recall    : {rec  * 100:.2f}%")
    print(f"  Final Test F1        : {f1   * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm_df.to_string())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    text_feats, image_feats, sid_to_idx = load_feature_map()
    run_5fold_cv(text_feats, image_feats, sid_to_idx)
    train_and_eval_final_test(text_feats, image_feats, sid_to_idx)
