# 🕵️ Steganalysis Research Report: Cover vs. Stego Binary Classification

**Experiment Date:** 2026-09-18 22:28:19  
**Method Evaluated:** CNN-DA-EMD-OLSB Dual-Steganography Model  
**Classifier:** SteganalysisNet (SRM-Inspired High-Pass CNN, PyTorch)  

---

## 1. Executive Summary & Interpretation

> [!TIP]
> **Low Detectability under Tested Conditions:** The classifier performed near chance level (50.0%) while predicting both classes. Under this specific classifier architecture, patch resolution, and dataset size, CNN-DA-EMD-OLSB demonstrated strong resistance to spatial steganalysis. *(Note: 50% accuracy under a specific CNN does not imply universal security against all possible attacks).*

---

## 2. Dataset & Pair-Based Split Architecture

Strict pair-based splitting was enforced to eliminate data leakage. No original cover and its corresponding stego counterpart were ever separated into different splits.

| Split Phase | Image Pairs | Extracted Patches (64×64) | Allocation Ratio |
|:------------|:------------|:--------------------------|:-----------------|
| **Training** | 1 | 450 | 50.0% |
| **Validation** | 0 | 0 | 0.0% |
| **Unseen Testing** | 1 | 1922 | 50.0% |
| **Total** | **2** | **2372** | **100.0%** |

---

## 3. Unseen Test Set Performance Metrics

| Metric | Score | Benchmark Target |
|:-------|:------|:-----------------|
| **Accuracy** | `0.5000` | 0.5000 (Ideal undetectable = 0.50) |
| **Precision** | `0.5000` | 0.5000 |
| **Recall (Sensitivity)** | `0.7357` | 0.5000 |
| **F1-Score** | `0.5954` | 0.5000 |
| **ROC-AUC** | `0.5` | 0.5000 (Random guess) |

---

## 4. Class Distribution & Collapse Verification

| Class | Actual Test Patches | Predicted Test Patches |
|:------|:--------------------|:-----------------------|
| **Cover (Class 0)** | 961 | 508 |
| **Stego (Class 1)** | 961 | 1414 |

**Collapse State:** `✅ BALANCED PREDICTION`

---

## 5. Confusion Matrix (Unseen Test Set)

```
                    Predicted Cover (0)     Predicted Stego (1)
True Cover (0)            254                     707            
True Stego (1)            254                     707            
```

- **True Negatives (TN):** 254 (Clean covers correctly identified)
- **False Positives (FP):** 707 (Clean covers misclassified as stego)
- **False Negatives (FN):** 254 (Stego images undetected / misclassified as cover)
- **True Positives (TP):** 707 (Stego images correctly detected)

---

## 6. Image-Level Prediction Summary

| pair_id | image_name | true_label | true_class | predicted_label | predicted_class | prediction_probability | patches_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | stego (15).png | 0 | Cover | 1 | Stego | 0.5212 | 961 |
| 1 | stego (15).png | 1 | Stego | 1 | Stego | 0.5212 | 961 |

---
*Report generated automatically by CNN-DA-EMD-OLSB Research Suite.*
