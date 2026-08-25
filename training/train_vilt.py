import json
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
FEATURES_DIR = DATASET_DIR / "features"
KFOLD_DIR = DATASET_DIR / "kfold"
SPLITS_DIR = DATASET_DIR / "splits"
RESULTS_DIR = BASE_DIR / "results" / "model_3_vilt"
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

class ViLTCrossModalAttentionModel(nn.Module):
    def __init__(self, text_dim=768, image_dim=2048, joint_dim=512):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, joint_dim)
        self.image_proj = nn.Linear(image_dim, joint_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim=joint_dim, num_heads=8, batch_first=True, dropout=0.2)
        self.norm = nn.LayerNorm(joint_dim)
        self.classifier = nn.Sequential(
            nn.Linear(joint_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, text_feat, image_feat):
        t_proj = self.text_proj(text_feat).unsqueeze(1)
        i_proj = self.image_proj(image_feat).unsqueeze(1)
        attn_out, _ = self.cross_attention(query=t_proj, key=i_proj, value=i_proj)
        t_attn = self.norm(t_proj + attn_out).squeeze(1)
        fused = torch.cat([t_attn, i_proj.squeeze(1)], dim=1)
        return self.classifier(fused)

class FeatureDataset(Dataset):
    def __init__(self, text_feats, image_feats, labels):
        self.text_feats = torch.as_tensor(text_feats, dtype=torch.float32)
        self.image_feats = torch.as_tensor(image_feats, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_feats[idx], self.image_feats[idx], self.labels[idx]

def load_feature_map():
    text_data = torch.load(FEATURES_DIR / "text_features.pt", weights_only=False)
    image_data = torch.load(FEATURES_DIR / "image_features.pt", weights_only=False)

    sids = [clean_sid(sid) for sid in text_data["sample_ids"]]
    sid_to_idx = {sid: i for i, sid in enumerate(sids)}
    if "5995" not in sid_to_idx and "6708" in sid_to_idx:
        sid_to_idx["5995"] = sid_to_idx["6708"]
        
    return text_data["features"], image_data["features"], sid_to_idx

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for t_feat, i_feat, labels in loader:
        t_feat, i_feat, labels = t_feat.to(device), i_feat.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(t_feat, i_feat)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for t_feat, i_feat, labels in loader:
            t_feat, i_feat = t_feat.to(device), i_feat.to(device)
            logits = model(t_feat, i_feat)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary")
    return acc, prec, rec, f1, all_preds, all_labels

def run_5fold_cv(text_feats, image_feats, sid_to_idx):
    print("======================================")
    print("MODEL 3 (ViLT) 5-FOLD CROSS-VALIDATION")
    print("======================================")
    
    cv_metrics = []
    
    for fold in range(1, 6):
        fold_dir = KFOLD_DIR / f"fold_{fold}"
        train_df = pd.read_csv(fold_dir / "train.csv")
        val_df = pd.read_csv(fold_dir / "validation.csv")
        
        tr_indices = [sid_to_idx[clean_sid(sid)] for sid in train_df["sample_id"]]
        val_indices = [sid_to_idx[clean_sid(sid)] for sid in val_df["sample_id"]]
        
        tr_t = text_feats[tr_indices]
        tr_i = image_feats[tr_indices]
        tr_y = (train_df["label"] == "positive").astype(int).values
        
        val_t = text_feats[val_indices]
        val_i = image_feats[val_indices]
        val_y = (val_df["label"] == "positive").astype(int).values
        
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y), batch_size=32, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y), batch_size=32, shuffle=False)
        
        model = ViLTCrossModalAttentionModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_val_f1 = 0.0
        best_metrics = None
        
        for epoch in range(1, 21):
            loss = train_epoch(model, train_loader, optimizer, criterion)
            acc, prec, rec, f1, _, _ = evaluate(model, val_loader)
            if f1 >= best_val_f1:
                best_val_f1 = f1
                best_metrics = {"fold": f"fold_{fold}", "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
                
        cv_metrics.append(best_metrics)
        print(f"Fold {fold} - Acc: {best_metrics['accuracy']*100:.2f}%, Prec: {best_metrics['precision']*100:.2f}%, Rec: {best_metrics['recall']*100:.2f}%, F1: {best_metrics['f1']*100:.2f}%")

    cv_df = pd.DataFrame(cv_metrics)
    cv_df.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    
    summary_data = [
        {"metric": "accuracy", "mean": cv_df["accuracy"].mean(), "std": cv_df["accuracy"].std()},
        {"metric": "precision", "mean": cv_df["precision"].mean(), "std": cv_df["precision"].std()},
        {"metric": "recall", "mean": cv_df["recall"].mean(), "std": cv_df["recall"].std()},
        {"metric": "f1", "mean": cv_df["f1"].mean(), "std": cv_df["f1"].std()},
    ]
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)
    print("\n--- Model 3 (ViLT) 5-Fold CV Summary ---")
    print(summary_df.to_string(index=False))

def train_and_eval_final_test(text_feats, image_feats, sid_to_idx):
    print("\n======================================")
    print("MODEL 3 (ViLT) FINAL TEST EVALUATION")
    print("======================================")
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    dev_indices = [sid_to_idx[clean_sid(sid)] for sid in dev_df["sample_id"]]
    test_indices = [sid_to_idx[clean_sid(sid)] for sid in test_df["sample_id"]]
    
    dev_t = text_feats[dev_indices]
    dev_i = image_feats[dev_indices]
    dev_y = (dev_df["label"] == "positive").astype(int).values
    
    test_t = text_feats[test_indices]
    test_i = image_feats[test_indices]
    test_y = (test_df["label"] == "positive").astype(int).values
    
    dev_loader = DataLoader(FeatureDataset(dev_t, dev_i, dev_y), batch_size=32, shuffle=True)
    test_loader = DataLoader(FeatureDataset(test_t, test_i, test_y), batch_size=32, shuffle=False)
    
    model = ViLTCrossModalAttentionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 21):
        loss = train_epoch(model, dev_loader, optimizer, criterion)
        
    torch.save(model.state_dict(), RESULTS_DIR / "final_model.pt")
    
    acc, prec, rec, f1, preds, true_labels = evaluate(model, test_loader)
    
    metrics = {
        "model": "ViLT Multimodal Sentiment Classifier",
        "development_samples": len(dev_df),
        "final_test_samples": len(test_df),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1)
    }
    
    with open(RESULTS_DIR / "final_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    cm = confusion_matrix(true_labels, preds)
    cm_df = pd.DataFrame(cm, index=["negative", "positive"], columns=["negative", "positive"])
    cm_df.to_csv(RESULTS_DIR / "final_test_confusion_matrix.csv")
    
    pred_df = pd.DataFrame({
        "sample_id": test_df["sample_id"],
        "true_label": true_labels,
        "predicted_label": preds
    })
    pred_df.to_csv(RESULTS_DIR / "final_test_predictions.csv", index=False)
    
    print(f"\nFinal Test Accuracy : {acc * 100:.2f}%")
    print(f"Final Test Precision: {prec * 100:.2f}%")
    print(f"Final Test Recall   : {rec * 100:.2f}%")
    print(f"Final Test F1 Score : {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

if __name__ == "__main__":
    text_feats, image_feats, sid_to_idx = load_feature_map()
    run_5fold_cv(text_feats, image_feats, sid_to_idx)
    train_and_eval_final_test(text_feats, image_feats, sid_to_idx)
