# Multimodal Sentiment Analysis Using Image and Text

**Author**: Empirical Research Group  
**Dataset**: MVSA Balanced Dataset (2,598 Samples)  
**Date**: August 2026  

---

## 1. TITLE
**Multimodal Sentiment Analysis Using Image and Text**

---

## 2. ABSTRACT
Multimodal sentiment analysis aims to integrate textual and visual signals to predict the sentiment expressed in user-generated social media content. In this project, we benchmarked three distinct deep multimodal architectures: **BERT + ResNet50 Early Fusion**, **Contrastive Language-Image Pre-training (CLIP)**, and **Vision-and-Language Transformer (ViLT)**. Experiments were conducted on a balanced subset of 2,598 samples from the MVSA dataset (1,299 positive, 1,299 negative). The development set (2,078 samples) was evaluated using rigorous 5-fold cross-validation with zero data leakage. Models selected via 5-fold CV were subsequently evaluated **once** on a strictly locked 520-sample held-out final test set. On 5-fold cross-validation, **CLIP** achieved the highest development performance with an accuracy of **78.58% ± 1.60%** and an F1 score of **79.96% ± 0.70%**. On the held-out final test set, **BERT + ResNet50 Early Fusion** demonstrated superior generalization, achieving a test accuracy of **70.38%** and an F1 score of **71.48%**.

---

## 3. INTRODUCTION
Social media platforms contain high volumes of user content consisting of paired text captions and digital images. Single-modality models often fail when text is ambiguous or sarcasm is present, whereas visual features provide critical contextual alignment. Multimodal sentiment analysis addresses this problem by fusing feature representations from text and vision backbones into a joint decision-making classifier.

---

## 4. PROBLEM STATEMENT
Given a paired text message \( T \) and image \( I \), the objective is to predict a binary sentiment label \( y \in \{\text{positive}, \text{negative}\} \). The primary challenge lies in effectively aligning heterogeneous vector representations (e.g., 768-dimensional textual embeddings vs 2048-dimensional visual embeddings) while preventing over-fitting to noisy social media data.

---

## 5. DATASET
The benchmark dataset consists of 2,598 paired text-image samples extracted and filtered from the Multi-View Sentiment Analysis (MVSA) dataset.

- **Total Working Dataset**: 2,598 paired text-image samples
- **Positive Class**: 1,299 samples (50.0%)
- **Negative Class**: 1,299 samples (50.0%)
- **Development Subset**: 2,078 samples (80% of total)
- **Final Test Subset**: 520 samples (20% of total)

---

## 6. DATASET BALANCING
The raw MVSA dataset suffers from severe class imbalance and noise in manual annotator labels. Majority voting across annotator labels was performed to identify unanimous sentiment consensus. The working dataset was strictly balanced by sampling exactly 1,299 positive and 1,299 negative pairs, establishing a 1:1 prior class distribution.

---

## 7. PREPROCESSING & FEATURE EXTRACTION
1. **Text Preprocessing**: Decoding HTML entities (`&amp;` -> `&`), removing URLs (`http://...`), stripping Twitter handles (`@user`), normalizing whitespace, and converting text to lower case.
2. **BERT Tokenization**: Input sequences tokenized using `bert-base-uncased` with `[CLS]` and `[SEP]` tokens (`max_seq_len=128`).
3. **Image Preprocessing**: Images resized to \(256 \times 256\), center-cropped to \(224 \times 224\), converted to RGB, and normalized using standard ImageNet mean (`[0.485, 0.456, 0.406]`) and std (`[0.229, 0.224, 0.225]`).
4. **Feature Backbones**:
   - Text Backbone: Pretrained `bert-base-uncased` extracting 768-dimensional `[CLS]` embeddings.
   - Vision Backbone: Pretrained `ResNet50` extracting 2048-dimensional feature vectors.

---

## 8. K-FOLD VALIDATION METHODOLOGY
To guarantee statistically sound evaluation without data leakage:
- Stratified 5-Fold Cross-Validation was performed across the 2,078 development samples.
- **Overlap & Leakage Verification**:
  - Development / Test Leakage = 0
  - Train / Validation Overlap per fold = 0
  - Every development sample appeared exactly **once** in validation and **four** times in training.
- The 520 final test samples were locked and evaluated **only once** after model selection.

---

## 9. MODEL 1 — BERT + RESNET50 EARLY FUSION (BASELINE)
Model 1 maps 768-dim BERT `[CLS]` features and 2048-dim ResNet50 features through linear projection layers (256-dim each), concatenates them into a 512-dim joint representation, and classifies via a 2-layer MLP head with ReLU activations and Dropout (0.3).

- **5-Fold Cross-Validation Results**:
  - Accuracy: **77.81% ± 2.36%**
  - Precision: **74.38% ± 4.46%**
  - Recall: **85.76% ± 4.17%**
  - F1 Score: **79.48% ± 1.15%**
- **Final Test Set Evaluation (520 samples)**:
  - Accuracy: **70.38%**
  - Precision: **68.93%**
  - Recall: **74.23%**
  - F1 Score: **71.48%**
- **Final Test Confusion Matrix**: `[[173, 87], [67, 193]]`

---

## 10. MODEL 1 V2 — EXPERIMENTAL EARLY FUSION
Model 1 V2 incorporated batch normalization and lower learning rate decay on development folds.
- **5-Fold Cross-Validation Results**:
  - Accuracy: **78.34% ± 0.93%**
  - Precision: **77.17% ± 3.25%**
  - Recall: **80.94% ± 3.86%**
  - F1 Score: **78.89% ± 0.48%**
- *Note: Model 1 V2 was an experimental development variation and was not evaluated on the final test set.*

---

## 11. MODEL 2 — CLIP MULTIMODAL CLASSIFIER
Model 2 utilizes contrastive embedding projections inspired by OpenAI CLIP. Text and image features are mapped into a shared 512-dimensional multimodal embedding space with L2 normalization before joint classification.

- **5-Fold Cross-Validation Results**:
  - Accuracy: **78.58% ± 1.60%** (Highest CV Accuracy)
  - Precision: **75.44% ± 3.77%**
  - Recall: **85.37% ± 3.78%**
  - F1 Score: **79.96% ± 0.70%** (Highest CV F1)
- **Final Test Set Evaluation (520 samples)**:
  - Accuracy: **68.65%**
  - Precision: **67.64%**
  - Recall: **71.54%**
  - F1 Score: **69.53%**
- **Final Test Confusion Matrix**: `[[171, 89], [74, 186]]`

---

## 12. MODEL 3 — ViLT TRANSFORMER
Model 3 incorporates a Vision-and-Language Transformer cross-attention module, allowing textual tokens to dynamically query visual representations via multi-head attention.

- **5-Fold Cross-Validation Results**:
  - Accuracy: **76.42% ± 2.87%**
  - Precision: **73.13% ± 4.75%**
  - Recall: **84.51% ± 3.60%**
  - F1 Score: **78.24% ± 1.56%**
- **Final Test Set Evaluation (520 samples)**:
  - Accuracy: **66.73%**
  - Precision: **66.29%**
  - Recall: **68.08%**
  - F1 Score: **67.17%**
- **Final Test Confusion Matrix**: `[[170, 90], [83, 177]]`

---

## 13. EVALUATION METRICS
Standard binary classification metrics were computed:
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
$$\text{Precision} = \frac{TP}{TP + FP}$$
$$\text{Recall} = \frac{TP}{TP + FN}$$
$$\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 14. RESULTS SUMMARY TABLE

| Model | CV Acc (%) | CV Prec (%) | CV Rec (%) | CV F1 (%) | Test Acc (%) | Test Prec (%) | Test Rec (%) | Test F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BERT + ResNet50** | 77.81 ± 2.36 | 74.38 ± 4.46 | 85.76 ± 4.17 | 79.48 ± 1.15 | **70.38** | **68.93** | **74.23** | **71.48** |
| **BERT + ResNet50 V2** | 78.34 ± 0.93 | 77.17 ± 3.25 | 80.94 ± 3.86 | 78.89 ± 0.48 | *N/A* | *N/A* | *N/A* | *N/A* |
| **CLIP** | **78.58 ± 1.60** | **75.44 ± 3.77** | **85.37 ± 3.78** | **79.96 ± 0.70** | 68.65 | 67.64 | 71.54 | 69.53 |
| **ViLT** | 76.42 ± 2.87 | 73.13 ± 4.75 | 84.51 ± 3.60 | 78.24 ± 1.56 | 66.73 | 66.29 | 68.08 | 67.17 |

---

## 15. CONFUSION MATRICES SUMMARY

| Model | True Negative (TN) | False Positive (FP) | False Negative (FN) | True Positive (TP) | Total Test Samples |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BERT + ResNet50** | 173 | 87 | 67 | 193 | 520 |
| **CLIP** | 171 | 89 | 74 | 186 | 520 |
| **ViLT** | 170 | 90 | 83 | 177 | 520 |

---

## 16. DISCUSSION & ACADEMIC ANALYSIS
1. **CV vs Held-Out Test Discrepancy**: While CLIP achieved the best 5-fold cross-validation accuracy (**78.58%**), BERT + ResNet50 Early Fusion yielded the highest final test accuracy (**70.38%**). This observation suggests that direct concatenation early-fusion retains broader raw feature diversity when encountering out-of-distribution test samples, whereas contrastive projection heads may suffer slight domain shifts when trained on small development sets.
2. **Recall Bias**: All three models demonstrated higher recall (71% - 85%) than precision (66% - 75%), reflecting a slight sensitivity preference toward detecting positive sentiment cues in multimodal social media posts.

---

## 17. LIMITATIONS
- **CPU-Only Hardware Environment**: Pretrained backbones were evaluated in feature-frozen or lightweight head fine-tuning modes due to local CPU execution constraints.
- **Dataset Size**: 2,598 samples represent a moderate dataset size; scaling to larger datasets would benefit complex vision-language transformers like ViLT.
- **Noisy Social Data**: Text captions on social media often contain heavy informal slang and ambiguous emojis.

---

## 18. CONCLUSION
We successfully benchmarked three multimodal architectures on the MVSA dataset. Early Fusion (BERT + ResNet50) remains the most robust baseline on held-out test data (**70.38% Accuracy / 71.48% F1**), while CLIP provides superior cross-validation performance (**78.58% Accuracy / 79.96% F1**).

---

## 19. REFERENCES
1. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021 (CLIP).
2. Kim et al. "ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision." ICML 2021.
3. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.
4. He et al. "Deep Residual Learning for Image Recognition." CVPR 2016 (ResNet).
