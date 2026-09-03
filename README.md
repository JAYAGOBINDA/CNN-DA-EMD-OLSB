# CNN-Guided Distortion-Aware Adaptive EMD-OLSB Framework for Reversible Data Hiding in RGB Images

A Python research project benchmarking **5 literature-derived RDH baselines** against a **proposed CNN-DA-EMD-OLSB** system for reversible data hiding in RGB images.

---

## 🌟 Key Features

1. **6-Model Comparative Benchmark Framework**:
   - **Model 1 — MPEH-RDH**: Multidirectional Prediction Error Histogram RDH
   - **Model 2 — MCSH-RDH**: Multi-Channel Synchronized Histogram RDH
   - **Model 3 — CNN-RDH Predictor**: PyTorch CNN Prediction Difference Histogram RDH
   - **Model 4 — SRDNN-Stego**: Super-Resolution Deep Neural Network Steganography (non-RDH)
   - **Model 5 — EMD-OLSB RDH**: Dual-Image Exploiting Modification Direction + OLSB
   - **Model 6 — CNN-DA-EMD-OLSB (Proposed)**: CNN-Guided Distortion-Aware Adaptive EMD-OLSB

2. **Proposed System: CNN-DA-EMD-OLSB**:
   - **DistortionCNN**: Multi-scale 3-branch CNN computes per-channel distortion sensitivity maps D_r, D_g, D_b
   - **Adaptive Routing**: Pixels classified into Class 0/1 (EMD) or Class 2 (OLSB) based on distortion tolerance
   - **EMD Embedding**: R-G channel pairs use mod-5 extraction function f(p1,p2) = (p1 + 2·p2) mod 5
   - **OLSB Embedding**: Blue channel uses adaptive 3-bit LSB substitution for high-texture regions
   - **Dual-Stego Output**: Two stego images (S1, S2) produced for exact cover recovery via `round((S1+S2)/2)`
   - **AES-256-GCM**: Authenticated encryption with PBKDF2 key derivation

3. **Interactive Streamlit Web Dashboard**:
   - **Home**: Architecture overview, model descriptions, 6-model protocol comparison table
   - **Embed Payload**: Upload cover, embed secret data, download dual stego images S1 & S2
   - **Extract Payload**: Upload S1 & S2, decrypt and extract hidden payload, recover cover
   - **Research Benchmark**: Run all 6 models on same cover/payload, compare PSNR/SSIM/BER/Carrier Recovery
   - **Robustness Attacks**: Test stego resilience under JPEG compression, noise, cropping, resizing

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Streamlit Web Application
```bash
streamlit run app.py
```

### 3. Run Automated Unit Tests
```bash
python -m unittest tests/test_6_models.py -v
```

---

## 📁 Directory Architecture

```
PD LAB/
├── app.py                          # Streamlit Interactive Web Application
├── requirements.txt                # Environment specifications
├── README.md                       # Documentation
│
├── core/                           # Core Algorithm Modules
│   ├── cnn_da_emd_olsb.py         # CNN-DA-EMD-OLSB embed/extract engine (dual-stego)
│   ├── encryption.py               # AES-256-GCM & PBKDF2 key derivation
│   ├── payload.py                  # Header packing, zlib compression, serialization
│   └── metrics.py                  # PSNR, SSIM, MSE, BPP, BER quality calculations
│
├── cnn/                            # CNN Module
│   ├── model.py                    # DistortionCNN re-export
│   └── distortion_cnn.py          # Multi-scale DistortionCNN architecture
│
├── models/                         # 6 Research Model Implementations
│   ├── mpeh_rdh.py                 # Model 1: MPEH-RDH
│   ├── mcsh_rdh.py                 # Model 2: MCSH-RDH
│   ├── cnn_rdh.py                  # Model 3: CNN-RDH Predictor
│   ├── srdnn_stego.py              # Model 4: SRDNN-Stego
│   ├── emd_olsb.py                 # Model 5: EMD-OLSB RDH
│   ├── cnn_da_emd_olsb_model.py   # Model 6: CNN-DA-EMD-OLSB (Proposed) wrapper
│   └── adapters.py                 # Standard adapter framework for all 6 models
│
├── benchmark/                      # Benchmarking Engine
│   ├── runner.py                   # 6-model comparative benchmark runner
│   ├── metrics.py                  # Unified metrics (PSNR, SSIM, BER, Carrier Recovery)
│   └── experiments.py              # Multi-payload experiment runner & graph generator
│
├── experiments/                    # Attack Testing
│   └── attacks.py                  # JPEG, noise, cropping, resizing attack suite
│
├── tests/                          # Unit Tests
│   └── test_6_models.py            # Automated 6-model validation suite
│
├── utils/                          # Utility Modules
│   ├── image_utils.py              # Image loading, resizing, color conversion
│   └── payload_utils.py            # Bit/byte conversion helpers
│
└── results/                        # Output Results
    ├── csv/                        # Benchmark CSV files
    ├── graphs/                     # Research publication graphs
    └── outputs/                    # Generated stego image outputs
```

---

## 📊 Research Metrics Legend

| Metric | Full Name | Research Significance |
| :--- | :--- | :--- |
| **PSNR** | Peak Signal-to-Noise Ratio | Measures perceptual image quality in dB (higher is better, >40 dB is imperceptible). |
| **SSIM** | Structural Similarity Index | Quantifies structural similarity to human visual perception (1.0 = identical). |
| **MSE** | Mean Squared Error | Average squared pixel difference between cover and stego images. |
| **BPP** | Bits Per Pixel | Embedding payload capacity density (bits embedded per pixel). |
| **BER** | Bit Error Rate | Ratio of extracted bit errors (0.0 = perfect exact recovery). |
| **Carrier Recovery** | Cover Image Recovery Accuracy | 100.0% = bit-exact cover reconstruction (RDH models). |

---

## 🔬 Proposed Algorithm: CNN-DA-EMD-OLSB

### Embedding Pipeline
1. **CNN Distortion Maps**: DistortionCNN produces per-channel D_r, D_g, D_b from upper bitplanes (& 0xF8)
2. **Classification**: Pixels routed to EMD (Class 0/1) or OLSB (Class 2) based on thresholds T1, T2
3. **EMD Embedding**: R-G pairs embed base-5 digits via `f(p1,p2) = (p1 + 2·p2) mod 5`, generating dual candidates (S1, S2)
4. **OLSB Embedding**: Blue channel at Class 2 pixels gets 3-bit LSB substitution (identical in S1 and S2)
5. **Output**: Two stego images (S1, S2) + AES-256-GCM encrypted payload

### Extraction & Recovery
1. Recompute distortion class maps from stego upper bitplanes (& 0xF8)
2. Extract EMD digits from S1 R-G pairs: `s = (p1*1 + p2*2) mod 5`
3. Extract OLSB bits from S1 Blue channel LSBs
4. Recover cover: `p_orig = round((S1 + S2) / 2)` for all pixels
5. Decrypt AES-256-GCM payload and verify CRC32 integrity
