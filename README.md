# Multimodal Sentiment Analysis (MVSA) — Improved Generalization Benchmark & Final Report

A state-of-the-art Multimodal Sentiment Analysis pipeline combining visual (image) and textual features from social media posts on the **MVSA Dataset**. This repository benchmarks three distinct deep multimodal architectures across 5-fold development cross-validation and a strictly locked 520-sample held-out final test set.

---

## Executive Summary & Baseline vs. Improved Comparative Results

### 1. Final Touch-Free Test Set Evaluation (520 Held-Out Samples)

| Model Architecture | Baseline Test Acc (%) | **Improved Test Acc (%)** | **Accuracy Improvement** | Baseline Test F1 (%) | **Improved Test F1 (%)** | **F1 Improvement** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: BERT + ResNet50 Early Fusion** | 70.38% | **71.73%** | **+1.35%** | 71.48% | **72.63%** | **+1.15%** |
| **Model 2: CLIP Multimodal Classifier** | 68.65% | **70.96%** | **+2.31%** | 69.53% | **68.74%** | *-0.79%* |
| **Model 3: ViLT Cross-Modal Classifier** | 66.73% | **72.88%** | **+6.15%** | 67.17% | **73.45%** | **+6.28%** |

---

### 2. 5-Fold Development Cross-Validation & Generalization Gap Summary

| Model Architecture | Improved 5-Fold CV Accuracy (%) | Improved 5-Fold CV F1 (%) | Mean CV Train-Val Gap (%) |
| :--- | :---: | :---: | :---: |
| **Model 1: BERT + ResNet50 Early Fusion** | 77.43 ± 1.85 | 76.65 ± 3.74 | **3.97%** |
| **Model 2: CLIP Multimodal Classifier** | 76.56 ± 1.45 | 76.03 ± 3.95 | **4.83%** |
| **Model 3: ViLT Cross-Modal Classifier** | 78.49 ± 1.46 | 79.37 ± 1.36 | **9.03%** |

---

## Key Generalization & Performance Improvements Made

### 1. Root Cause Resolution
- **Baseline Overfitting Fix**: Resolved severe over-training by replacing static 10–20 epoch training with **validation loss early stopping (patience=4)**, saving model checkpoints at optimal generalization points.
- **Enhanced Regularization**: Integrated **LayerNorm**, **GELU activations**, **dropout (0.35–0.40)**, and **AdamW weight decay (0.02)** across all classification heads.
- **Smooth Optimization**: Replaced fixed $10^{-3}$ learning rates with **Cosine Annealing Learning Rate Schedulers** ($3\times 10^{-4}$ to $2\times 10^{-4}$ initial learning rates) and label smoothing ($0.05$).

### 2. Architectural Enhancements
- **Model 1 (ResNet50 + BERT Early Fusion)**: Added Gated Multimodal Fusion ($f = g \cdot v + (1-g) \cdot t$) to dynamically balance textual and visual modalities alongside LayerNorm and dropout.
- **Model 2 (CLIP Multimodal Classifier)**: Implemented unit-sphere $L_2$ normalization on contrastive embeddings, cross-modal elementwise product interaction features, and scalar cosine similarity projections.
- **Model 3 (ViLT Cross-Modal Classifier)**: Designed a bi-directional multi-head cross-attention mechanism ($8$ heads) with residual skip connections and LayerNorm for deep text-vision feature alignment.

---

## Safety & Scientific Validity Assurance
- **Dataset Preservation**: Dataset splits ($2,078$ development, $520$ final test) were kept untouched.
- **Zero Leakage**: Strict separation between training, validation, and test splits verified.
- **Original Baselines Preserved**: Original baseline results remain stored under `results/model_1_early_fusion`, `results/model_2_clip`, and `results/model_3_vilt`. Improved results are cleanly organized in `results/improved/`.

---

## Directory Structure
```text
Multimodal-Sentiment-Analysis/
├── backup_before_generalization/
├── dataset/
│   ├── balanced/
│   ├── features/
│   ├── kfold/
│   ├── splits/
├── evaluation/
├── models/
│   ├── clip.py
│   ├── resnet_bert.py
│   └── vilt.py
├── results/
│   ├── model_1_early_fusion/
│   ├── model_2_clip/
│   ├── model_3_vilt/
│   └── improved/
│       ├── model_1_resnet_bert/
│       ├── model_2_clip/
│       ├── model_3_vilt/
│       ├── graphs/
│       └── metrics/
├── training/
│   ├── train_all_improved.py
│   └── generate_graphs.py
├── show_all_results.py
└── README.md
```

---

## Reproducibility & Commands
```bash
# Run baseline vs improved comparison summary table
python show_all_results.py

# Re-run improved training pipeline for all 3 models
python training/train_all_improved.py

# Re-generate performance & generalization graphs
python training/generate_graphs.py
```
