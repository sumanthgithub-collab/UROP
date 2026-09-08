import os
import json
import sys
import io
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.train_stage3_unseen_breakthrough import (
    load_feature_map, clean_sid, FeatureDataset, mixup_data, mixup_criterion,
    AdvancedResNetBERTClassifier, AdvancedCLIPClassifier, AdvancedViLTClassifier, CrossModalTransformerClassifier,
    train_one_epoch_mixup, evaluate
)

DATASET_DIR = BASE_DIR / "dataset"
KFOLD_DIR = DATASET_DIR / "kfold"
RESULTS_DIR = BASE_DIR / "results" / "stage3_unseen_breakthrough"
STACKING_DIR = RESULTS_DIR / "stacking"
STACKING_DIR.mkdir(parents=True, exist_ok=True)

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

def run_stage3_oof_and_stacking():
    print("==========================================================================", flush=True)
    print("STAGE 3: 5-FOLD OOF EXTRACTION, THRESHOLD CALIBRATION & META-LEARNER STACKING", flush=True)
    print("==========================================================================", flush=True)
    
    text_feats, image_feats, sid_to_idx = load_feature_map()
    
    cfg_file = RESULTS_DIR / "best_model_configs.json"
    if not cfg_file.exists():
        print("Best model configs missing.", flush=True)
        return
        
    with open(cfg_file, "r", encoding="utf-8") as f:
        best_configs = json.load(f)
        
    models_info = [
        ("ResNet50+BERT", AdvancedResNetBERTClassifier, "model_1_resnet_bert"),
        ("CLIP", AdvancedCLIPClassifier, "model_2_clip"),
        ("ViLT", AdvancedViLTClassifier, "model_3_vilt"),
        ("TransformerFusion", CrossModalTransformerClassifier, "model_4_transformer_fusion")
    ]
    
    # Store OOF probabilities for dev set (2078 samples)
    oof_predictions = {m[0]: [] for m in models_info}
    oof_true_labels = []
    
    # Store per-fold test predictions
    fold_cv_results = {m[0]: [] for m in models_info}
    
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
        
        if len(oof_true_labels) < fold * len(val_y):
            oof_true_labels.extend(val_y)
            
        train_loader = DataLoader(FeatureDataset(tr_t, tr_i, tr_y, noise_std=0.0), batch_size=32, shuffle=True)
        val_loader = DataLoader(FeatureDataset(val_t, val_i, val_y, noise_std=0.0), batch_size=32, shuffle=False)
        
        for m_name, m_class, folder in models_info:
            hparams = best_configs[m_name]["hparams"]
            lr = hparams.get("lr", 1e-4)
            weight_decay = hparams.get("weight_decay", 0.05)
            dropout = hparams.get("dropout", 0.45)
            proj_dim = hparams.get("proj_dim", 128)
            hidden_dim = hparams.get("hidden_dim", 96)
            fusion_mode = hparams.get("fusion_mode", "full_interaction")
            use_diff = hparams.get("use_diff", True)
            num_heads = hparams.get("num_heads", 4)
            alpha = hparams.get("mixup_alpha", 0.2)
            label_smooth = hparams.get("label_smooth", 0.03)
            max_epochs = hparams.get("epochs", 35)
            patience = hparams.get("patience", 4)
            
            kwargs = {"proj_dim": proj_dim, "hidden_dim": hidden_dim, "dropout": dropout}
            if m_class == AdvancedResNetBERTClassifier:
                kwargs["fusion_mode"] = fusion_mode
            elif m_class == AdvancedCLIPClassifier:
                kwargs["use_diff"] = use_diff
            elif m_class in [AdvancedViLTClassifier, CrossModalTransformerClassifier]:
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
                tr_loss, tr_acc = train_one_epoch_mixup(model, train_loader, optimizer, criterion, alpha=alpha)
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
                        
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            
            val_p = get_model_probabilities(model, val_loader)
            oof_predictions[m_name].append(val_p)
            fold_cv_results[m_name].append(best_metrics)
            
            print(f"[{m_name:18s}] Fold {fold} (Epoch {best_metrics['best_epoch']:2d}) - Val Acc: {best_metrics['accuracy']*100:.2f}%, Val F1: {best_metrics['f1']*100:.2f}%, Gap: {best_metrics['train_val_gap']*100:.2f} pp", flush=True)

    # Concatenate OOF probability vectors across all 5 folds
    oof_y = np.array(oof_true_labels)
    oof_matrix_list = []
    
    print("\n--- Stage 3 5-Fold Cross Validation Summaries ---", flush=True)
    for m_name, _, folder in models_info:
        m_oof_probs = np.vstack(oof_predictions[m_name]) # (2078, 2)
        oof_matrix_list.append(m_oof_probs[:, 1:]) # Positive class prob (2078, 1)
        
        cv_df = pd.DataFrame(fold_cv_results[m_name])
        cv_df.to_csv(RESULTS_DIR / folder / "stage3_cv_results.csv", index=False)
        print(f"{m_name:18s}: Mean Val Acc: {cv_df['accuracy'].mean()*100:.2f}% | Mean Val F1: {cv_df['f1'].mean()*100:.2f}% | Mean Gap: {cv_df['train_val_gap'].mean()*100:.2f} pp", flush=True)

    oof_X = np.hstack(oof_matrix_list) # (2078, 4)
    
    # --------------------------------------------------------------------------
    # THRESHOLD CALIBRATION ON OOF PREDICTIONS
    # --------------------------------------------------------------------------
    print("\n--- Calibrating Optimal Decision Threshold on Dev OOF Predictions ---", flush=True)
    mean_oof_prob = np.mean(oof_X, axis=1)
    best_tau = 0.50
    best_tau_f1 = 0.0
    
    for tau in np.linspace(0.35, 0.65, 31):
        preds_tau = (mean_oof_prob >= tau).astype(int)
        _, _, f1_tau, _ = precision_recall_fscore_support(oof_y, preds_tau, average="binary", zero_division=0)
        if f1_tau > best_tau_f1:
            best_tau_f1 = f1_tau
            best_tau = tau
            
    print(f"Optimal Calibrated Threshold: tau* = {best_tau:.3f} | OOF F1 Score: {best_tau_f1*100:.2f}%", flush=True)
    
    # --------------------------------------------------------------------------
    # TRAIN META-LEARNER STACKING ENSEMBLE ON OOF MATRIX
    # --------------------------------------------------------------------------
    meta_learner = LogisticRegression(C=1.0, random_state=SEED)
    meta_learner.fit(oof_X, oof_y)
    
    stacking_oof_preds = meta_learner.predict(oof_X)
    stacking_acc = accuracy_score(oof_y, stacking_oof_preds)
    _, _, stacking_f1, _ = precision_recall_fscore_support(oof_y, stacking_oof_preds, average="binary", zero_division=0)
    
    print(f"Meta-Learner Stacking OOF Performance: Accuracy = {stacking_acc*100:.2f}% | F1 Score = {stacking_f1*100:.2f}%", flush=True)
    
    # Save meta-learner and threshold config
    with open(STACKING_DIR / "meta_learner.pkl", "wb") as f:
        pickle.dump(meta_learner, f)
        
    stacking_config = {
        "optimal_threshold": float(best_tau),
        "oof_accuracy": float(stacking_acc),
        "oof_f1": float(stacking_f1),
        "meta_learner_coefs": meta_learner.coef_.tolist()
    }
    with open(STACKING_DIR / "stacking_config.json", "w", encoding="utf-8") as f:
        json.dump(stacking_config, f, indent=4)

if __name__ == "__main__":
    run_stage3_oof_and_stacking()
