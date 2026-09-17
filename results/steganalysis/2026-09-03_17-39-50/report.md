# 🕵️ Steganalysis Research Report: Cover vs. Stego Binary Classification

**Experiment Date:** 2026-09-03 17:39:50  
**Method Evaluated:** CNN-DA-EMD-OLSB Dual-Steganography Model  
**Classifier:** SteganalysisNet (SRM-Inspired High-Pass CNN, PyTorch)  

---

## 1. Executive Summary & Interpretation

> [!WARNING]
> **Classifier Collapse Detected:** The network predicted all test samples as a single class. This indicates that the classifier failed to find separable statistical boundaries between cover and stego patches under the current hyperparameters or sample size. This outcome does NOT prove undetectability; rather, it reflects an inconclusive or collapsed optimization state.

---

## 2. Dataset & Pair-Based Split Architecture

Strict pair-based splitting was enforced to eliminate data leakage. No original cover and its corresponding stego counterpart were ever separated into different splits.

| Split Phase | Image Pairs | Extracted Patches (64×64) | Allocation Ratio |
|:------------|:------------|:--------------------------|:-----------------|
| **Training** | 2 | 36 | 66.7% |
| **Validation** | 0 | 0 | 0.0% |
| **Unseen Testing** | 1 | 18 | 33.3% |
| **Total** | **3** | **54** | **100.0%** |

---

## 3. Unseen Test Set Performance Metrics

| Metric | Score | Benchmark Target |
|:-------|:------|:-----------------|
| **Accuracy** | `0.5000` | 0.5000 (Ideal undetectable = 0.50) |
| **Precision** | `0.5000` | 0.5000 |
| **Recall (Sensitivity)** | `1.0000` | 0.5000 |
| **F1-Score** | `0.6667` | 0.5000 |
| **ROC-AUC** | `0.5` | 0.5000 (Random guess) |

---

## 4. Class Distribution & Collapse Verification

| Class | Actual Test Patches | Predicted Test Patches |
|:------|:--------------------|:-----------------------|
| **Cover (Class 0)** | 9 | 0 |
| **Stego (Class 1)** | 9 | 18 |

**Collapse State:** `⚠️ COLLAPSED`

---

## 5. Confusion Matrix (Unseen Test Set)

```
                    Predicted Cover (0)     Predicted Stego (1)
True Cover (0)            0                       9              
True Stego (1)            0                       9              
```

- **True Negatives (TN):** 0 (Clean covers correctly identified)
- **False Positives (FP):** 9 (Clean covers misclassified as stego)
- **False Negatives (FN):** 0 (Stego images undetected / misclassified as cover)
- **True Positives (TP):** 9 (Stego images correctly detected)

---

## 6. Image-Level Prediction Summary

| pair_id | image_name | true_label | true_class | predicted_label | predicted_class | prediction_probability | patches_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | img_003.png | 0 | Cover | 1 | Stego | 0.5075 | 9 |
| 3 | img_003_stego.png | 1 | Stego | 1 | Stego | 0.5075 | 9 |

---
*Report generated automatically by CNN-DA-EMD-OLSB Research Suite.*
