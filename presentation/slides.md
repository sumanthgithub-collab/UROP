# Multimodal Sentiment Analysis — Final Project Presentation

---

## Slide 1: Title
### Multimodal Sentiment Analysis Using Image and Text
- **Dataset**: MVSA Balanced Dataset (2,598 Samples)
- **Architectures**: BERT + ResNet50, CLIP, ViLT
- **Methodology**: 5-Fold Cross-Validation & Locked Test Set

---

## Slide 2: Introduction & Motivation
- Social media content is inherently multimodal (text captions + digital images).
- Text alone can be ambiguous, sarcastic, or incomplete.
- Images provide vital visual context to disambiguate sentiment.

---

## Slide 3: Problem Statement
- Predict binary sentiment: **Positive (1)** vs **Negative (0)**.
- Challenge: Fuse heterogeneous feature representations (768-dim text vs 2048-dim vision).
- Requirement: Robust evaluation with zero test set leakage.

---

## Slide 4: Project Objectives
- Establish an untouched baseline model (**BERT + ResNet50 Early Fusion**).
- Implement Contrastive Multimodal Sentiment Classifier (**CLIP**).
- Implement Vision-and-Language Transformer (**ViLT**).
- Benchmark models across identical 5-fold CV and locked 520 test samples.

---

## Slide 5: Dataset Overview
- **Total Working Dataset**: 2,598 paired text-image samples
- **Positive Sentiment**: 1,299 samples (50.0%)
- **Negative Sentiment**: 1,299 samples (50.0%)
- **Prior Distribution**: Perfectly balanced (1:1 ratio)

---

## Slide 6: Dataset Splits
- **Development Set (80%)**: 2,078 samples (used exclusively for 5-Fold CV)
- **Final Test Set (20%)**: 520 samples (locked until final evaluation)
- **Stratification**: Class balance preserved across all splits and folds.

---

## Slide 7: Preprocessing Pipeline
- **Text**: HTML decoding, URL removal, @mention stripping, whitespace normalization.
- **Text Tokenization**: `bert-base-uncased` WordPiece tokenizer (`max_seq_len=128`).
- **Image**: RGB conversion, Resizing (256x256), Center Crop (224x224), ImageNet normalization.

---

## Slide 8: K-Fold Cross Validation Methodology
- **5 Stratified Folds** on 2,078 development samples.
- **Verification Passed**:
  - Development / Test Leakage = 0
  - Train / Validation Overlap per fold = 0
  - Every dev sample validated exactly 1 time, trained 4 times.

---

## Slide 9: Model 1 — BERT + ResNet50 Early Fusion (Baseline)
- Text Backbone: `bert-base-uncased` (768-dim `[CLS]`).
- Image Backbone: `ResNet50` (2048-dim feature map).
- Projection & Fusion: Linear(768->256) + Linear(2048->256) -> Concatenate (512-dim) -> MLP Classifier.

---

## Slide 10: Model 1 V2 (Development Variation)
- Architectural refinement with batch normalization and adjusted dropout.
- Evaluated strictly on 5-fold cross-validation development set.
- *Final test set evaluation was not performed for V2.*

---

## Slide 11: Model 2 — CLIP Multimodal Classifier
- Utilizes contrastive text and vision projection heads.
- Normalizes text and image embeddings in a shared 512-dim multimodal space.
- Fuses normalized embeddings through a classification head.

---

## Slide 12: Model 3 — ViLT Transformer
- Vision-and-Language Transformer cross-attention interaction module.
- Multi-head attention allows text tokens to dynamically query visual feature maps.
- Fuses cross-attended text and image features into a joint classification layer.

---

## Slide 13: Evaluation Metrics
- **Accuracy**: Overall classification correctness.
- **Precision**: Proportion of true positive predictions.
- **Recall**: Proportion of actual positive samples correctly identified.
- **F1 Score**: Harmonic mean of Precision and Recall.

---

## Slide 14: 5-Fold Cross-Validation Results
| Model | CV Accuracy (%) | CV Precision (%) | CV Recall (%) | CV F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **BERT + ResNet50** | 77.81 ± 2.36 | 74.38 ± 4.46 | 85.76 ± 4.17 | 79.48 ± 1.15 |
| **BERT + ResNet50 V2** | 78.34 ± 0.93 | 77.17 ± 3.25 | 80.94 ± 3.86 | 78.89 ± 0.48 |
| **CLIP** | **78.58 ± 1.60** | **75.44 ± 3.77** | **85.37 ± 3.78** | **79.96 ± 0.70** |
| **ViLT** | 76.42 ± 2.87 | 73.13 ± 4.75 | 84.51 ± 3.60 | 78.24 ± 1.56 |

---

## Slide 15: Final Test Set Results (520 Locked Samples)
| Model | Test Accuracy (%) | Test Precision (%) | Test Recall (%) | Test F1 (%) |
| :--- | :---: | :---: | :---: | :---: |
| **BERT + ResNet50** | **70.38** | **68.93** | **74.23** | **71.48** |
| **CLIP** | 68.65 | 67.64 | 71.54 | 69.53 |
| **ViLT** | 66.73 | 66.29 | 68.08 | 67.17 |
| **BERT + ResNet50 V2** | *N/A* | *N/A* | *N/A* | *N/A* |

---

## Slide 16: Final Test Confusion Matrices
- **BERT + ResNet50**: `[[173, 87], [67, 193]]` (Accuracy: 70.38%)
- **CLIP**: `[[171, 89], [74, 186]]` (Accuracy: 68.65%)
- **ViLT**: `[[170, 90], [83, 177]]` (Accuracy: 66.73%)

---

## Slide 17: Model Ranking Comparison
- **Best 5-Fold CV Accuracy**: CLIP (**78.58%**)
- **Best 5-Fold CV F1**: CLIP (**79.96%**)
- **Best Final Test Accuracy**: BERT + ResNet50 (**70.38%**)
- **Best Final Test F1**: BERT + ResNet50 (**71.48%**)

---

## Slide 18: Academic Discussion & Key Insights
- CLIP achieved the highest cross-validation score during development.
- BERT + ResNet50 Early Fusion demonstrated superior generalization on the unseen 520 test set.
- Direct feature concatenation early-fusion retains raw feature diversity better on held-out test data.

---

## Slide 19: Conclusion
- Successfully benchmarked three multimodal architectures on MVSA.
- Baseline Early Fusion remains the top-performing model on locked test data (**70.38% Acc / 71.48% F1**).
- All experimental outputs, confusion matrices, and metrics have been systematically documented.

---

## Slide 20: Future Work
- End-to-end GPU fine-tuning of vision-language transformer backbones.
- Exploration of cross-attention fusion with larger multimodal pre-training corpora.
- Expanding dataset size and addressing informal text/emoji noise.
