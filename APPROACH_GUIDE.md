# 🎯 Merchant Document Forgery & Verification Approach Guide

This document outlines the complete strategy, architecture, dataset usage, and step-by-step approach for solving the **Governed Merchant Onboarding Document Verification and Forensics** problem.

---

## 📌 1. The Core Problem Statement

During merchant onboarding, payment aggregators (e.g. NPCI, Stripe, Razorpay) collect KYC/KYB documents (PAN Card, Aadhaar, GST Registration, Bank Statements). 

Fraudsters try to onboard illegal or fake businesses using **7 Technical Forgery Types**:

1. **Text / Font Tampering**: Editing PAN/GST digits, dates, or account numbers.
2. **Splicing**: Pasting someone else's photo or company seal onto a document.
3. **Copy-Move Forgery**: Duplicating stamps, signatures, or digits within the same page.
4. **Retouching / Inpainting**: Erasing text or white-outing watermarks using image editing software.
5. **Template Counterfeiting**: Generating fake PAN/GST certificates using unofficial layout templates.
6. **Re-digitization (Screen Recapture)**: Taking a photo of a tampered document on a monitor to hide editing traces.
7. **Metadata Forgery**: Stripping or altering EXIF/PDF creation tags.

---

## 📊 2. The Dataset Strategy (How Your Datasets Solve Each Scenario)

Your local `./data` folder contains 4 datasets. Here is how each dataset targets a specific fraud type:

| Dataset | Format / Location | Target Forgery Scenario | What the Neural Network Learns |
| :--- | :--- | :--- | :--- |
| **DocTamper** | LMDB Database (`120,000` images) <br>`data/Doctamper/DocTamperV1-TrainingSet` | **Text & Field Tampering (#1, #3, #4)** | Learns character stroke anomalies, font baseline shifts ($1\text{--}2\text{ px}$ off), and micro JPEG compression grids around edited text. |
| **COCOGLIDE (TruFor)** | PNG Images + Masks (`512` images) <br>`data/COCOGLIDE_trufor` | **Splicing & AI Inpainting (#2, #4)** | Learns sensor noise discontinuities (Noiseprint) and boundary artifacts where external images/seals were pasted. |
| **Splicing Dataset** | JPG Images + Masks (`101` images) <br>`data/Image Forgery detecion dataset (splicing)` | **Copy-Move & Region Splicing (#2, #3)** | Learns duplicated region keypoint matches and edge blending discrepancies. |
| **MIDV-500** | ID Images & Documents <br>`data/Midv500` | **Template Counterfeiting (#5)** | Compares structural layout geometry, official logo placements, and boundary ratios against master templates. |

---

## ⚙️ 3. The Technical Algorithm & Model Suite

We use a **4-Layer Algorithmic Stack** combining Signal Processing, Frequency Analysis, and Deep Learning:

### Layer 1: Dual-Branch RGB + DCT Segmentation U-Net (Neural Engine)
* **RGB Branch**: Extracts visual spatial textures, edge continuity, and color distribution.
* **DCT (Discrete Cosine Transform) Frequency Branch**: Computes high-frequency Laplacian residual maps to catch JPEG double-compression boundaries around edited text.
* **GroupNormalization (`num_groups=32`)**: Replaces BatchNorm to ensure zero internal covariate drift when training on small batch sizes (e.g. batch size 16 on your 6GB RTX 3050).
* **Spatial Dropout2D ($p=0.2$)**: Prevents spatial feature co-adaptation and neural net overfitting.
* **Focal + Dice Loss**: Combines focal loss (focuses on rare tampered pixels) and dice loss (handles overlap) to prevent background non-edited pixels from dominating.

### Layer 2: 2D FFT Moiré Frequency Analysis (Screen Recapture - #6)
* Detects high-frequency lattice interference patterns created when a camera takes a photo of a laptop/mobile screen (0 MB dataset overhead!).

### Layer 3: SSIM & Layout Geometry Matcher (Template Counterfeiting - #5)
* Measures Structural Similarity Index (SSIM) and bounding box ratios against official master templates (PAN, GST, Aadhaar).

### Layer 4: EXIF & PDF Metadata Tag Inspector (Metadata Forgery - #7)
* Scans EXIF/XMP/PDF structure headers for editing software tags (*Photoshop*, *Canva*, *GIMP*).

---

## 🚀 4. Step-by-Step Execution Plan

### Step 1: Train the Dual-Branch Model
Open your terminal and run:

```bash
# 1. Train on DocTamper (Text & Field Forgery - Primary Model)
uv run python train.py --dataset doctamper --epochs 25 --batch-size 16

# 2. Train on COCOGLIDE (Splicing & AI Inpainting Model)
uv run python train.py --dataset cocoglide --epochs 25 --batch-size 16

# 3. Train on Splicing Dataset
uv run python train.py --dataset splicing --epochs 20 --batch-size 16
```

### Step 2: Auto-Generated Outputs
After training finishes, check your project folders:
* **`savedmodels/`**:
  * `best_model_doctamper.pth`: PyTorch checkpoint.
  * `production_model_doctamper.pt`: **Standalone TorchScript model** ready for zero-dependency production deployment!
* **`charts/`**:
  * `loss_chart.png`: Loss curves (Train vs Validation).
  * `metrics_chart.png`: Pixel IoU (%) and F1-Score (%) localization progress curves.

### Step 3: Benchmarking
Run the evaluation script to test pixel-level IoU and F1 score against ground truth:

```bash
uv run python evaluate.py --dataset doctamper
```

---

## 📁 Related Project Files to Open in Code Editor

* **[config.yaml](file:///e:/merchant/config.yaml)** — Hyperparameters & path setup.
* **[train.py](file:///e:/merchant/train.py)** — Main CLI training launcher.
* **[evaluate.py](file:///e:/merchant/evaluate.py)** — Benchmark test evaluation script.
* **[src/models/dual_branch_unet.py](file:///e:/merchant/src/models/dual_branch_unet.py)** — Core Dual-Branch U-Net architecture.
* **[src/dataset/base_dataset.py](file:///e:/merchant/src/dataset/base_dataset.py)** — Universal Dataset Loader (supports LMDB, COCOGLIDE, Splicing).
