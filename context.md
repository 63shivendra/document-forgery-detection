# 📄 Governed Merchant Document Forgery & Tampering Framework

Welcome to the complete, modular, and production-ready PyTorch training framework for **Governed Merchant Onboarding Document Verification and Forensics**.

---

## 📂 Project Directory Structure

```
merchant_fina/
├── context.md                    # Detailed documentation & operational guide (this file)
├── APPROACH_GUIDE.md             # High-level architecture & algorithmic strategy
├── config.yaml                   # Complete hyperparameters, regularization & dataset paths
├── verify_document.py            # Main CLI for Single Document Forgery & Tampering Inspection
├── verify_merchant_onboarding.py # Main CLI for Full Merchant Dossier KYB/KYC Onboarding Audit
├── train.py                      # Main CLI entrypoint for full-dataset training
├── evaluate.py                   # Benchmarking script for test set evaluation
├── savedmodels/                  # Export target directory for PyTorch checkpoints & TorchScript models
├── bigpower/                     # High-performance trained checkpoints & production models
│   ├── best_model_splicing_combined_efficientnet-b2_...pth
│   ├── production_model_splicing_combined_efficientnet-b2_...pt (TorchScript)
│   └── best_model_splicing_combined_efficientnet-b2_...onnx (ONNX Export)
├── charts/                       # Auto-generated loss curves and IoU / F1 metric charts
├── reports/                      # Generated forensic reports and visual evidence
│   └── forensics/                # Annotated images, 3-panel audit boards, crops & JSON reports
├── data/                         # Folder containing downloaded dataset zips/folders
└── src/
    ├── dataset/
    │   ├── augmentations.py      # Albumentations anti-overfitting image pipeline
    │   └── base_dataset.py       # Universal PyTorch Dataset loader
    ├── models/
    │   ├── blocks.py             # Residual blocks, SRM filters, and feature projection
    │   ├── dual_branch_unet.py   # Dual-Branch (RGB + DCT Frequency) Segmentation U-Net
    │   └── forgery_net.py        # Production ForgeryNet (EfficientNet-B2 + U-Net + SRM)
    ├── forensics/
    │   ├── document_classifier.py# Zero-Training Multi-Document Type Classifier (QR + OCR Anchors)
    │   ├── entity_cross_validator.py# Government Entity Rules (PAN 4th char, GSTIN match, Name similarity)
    │   ├── merchant_verifier.py  # End-to-End Onboarding Dossier & Compliance Auditor Engine
    │   ├── document_verifier.py  # Document-Level Decision & Part Localization Engine
    │   └── visualizer.py         # Bounding Box, Thermal Heatmap & 3-Panel Evidence Generator
    └── utils/
        ├── metrics.py            # Focal + Dice Loss & Pixel IoU, F1, Precision, Recall
        ├── logger.py             # Console logger
        ├── plotter.py            # Matplotlib Loss & Accuracy chart generator
        └── trainer.py            # AMP fp16 Trainer Engine with Early Stopping
```

---

## 🎯 Dataset Verification & Links

The official datasets integrated into this framework correspond directly to research benchmarks:

1. **DocTamper Dataset**: [https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data](https://www.kaggle.com/datasets/dinmkeljiame/doctamper/data)
   - *Target Forgeries*: Text field insertion, number replacement, font tampering, character edits.
2. **TruFor / CocoGlide Dataset**: [https://github.com/grip-unina/TruFor](https://github.com/grip-unina/TruFor)
   - *Target Forgeries*: Image splicing, photo retouching, AI inpainting.
3. **IDNet Dataset**: [https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis](https://www.kaggle.com/datasets/chitreshkr/idnet-identity-document-analysis)
   - *Target Forgeries*: Identity card counterfeiting, template structural divergence.
4. **FHDMi Dataset**: [https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5](https://drive.google.com/drive/folders/1IJSeBXepXFpNAvL5OyZ2Y1yu4KPvDxN5)
   - *Target Forgeries*: Screen-recaptured photos, re-digitization, Moiré frequency patterns.

---

## ⚙️ How Anti-Overfitting & Internal Drift Are Handled

1. **GroupNormalization (`num_groups=32`)**:
   - Replaces standard BatchNorm. BatchNorm degrades when batch sizes are small ($<16$). GroupNorm stabilizes activations per feature channel group, completely resolving internal covariate shift on 6GB VRAM GPUs.
2. **Spatial Dropout2D ($p=0.2$)**:
   - Drops entire $2\text{D}$ feature maps instead of random pixels, forcing the network to learn robust contextual cues rather than memorizing synthetic forgery edges.
3. **AdamW Weight Decay ($1\times 10^{-4}$)**:
   - Decouples L2 regularization from gradient updates to maintain smooth optimization.
4. **Albumentations Augmentation Pipeline**:
   - Applies random JPEG compression re-saving ($Q \in [50, 95]$), Gaussian noise, subtle color jittering, and affine warping on the fly.
5. **Focal + Dice Loss Combination (`pos_weight=10.0`)**:
   - Prevents class imbalance collapse where background non-edited pixels dominate ($>95\%$ of document area). 10x penalty for missing forged pixels.

---

## 🚀 How to Run Full-Dataset Training

### Step 1: Organize Downloaded Datasets
Unzip dataset files into `./data/`:
```
data/doctamper/train/images/  <-- put image files here (.jpg, .png)
data/doctamper/train/masks/   <-- put binary mask files here
```

### Step 2: Launch Training
Run `train.py` specifying target dataset:

```bash
# Train on DocTamper (Full Dataset)
python train.py --dataset doctamper --epochs 25 --batch-size 16

# Train on Splicing Combined
python train.py --dataset splicing_combined --epochs 25 --batch-size 16
```

---

## 📊 Auto-Generated Outputs & Production Export

When training completes, the system automatically creates:

1. **Production Models inside `bigpower/`**:
   - `best_model_splicing_combined_...pth`: PyTorch `state_dict` checkpoint.
   - `production_model_splicing_combined_...pt`: **Standalone TorchScript serialized model** ready for zero-dependency production deployment.
   - `best_model_splicing_combined_...onnx`: **ONNX Export** for C++/TensorRT inference engines.
2. **Loss & Metric Charts inside `charts/`**:
   - `loss_chart.png`: Loss trajectory comparison (Train vs Validation).
   - `metrics_chart.png`: Pixel IoU (%) and F1-Score (%) localization progress curves.

---

## 🔍 Single Document Verification & Forensic Localization Engine

Transforms raw pixel-level segmentation maps into governed document-level decisions and part-level forensic evidence.

```bash
# Verify a single document
python verify_document.py --input "WhatsApp Image 2026-08-23 at 4.25.55 PM.jpeg"
```

Generates:
* **Annotated Image (`_annotated.jpg`)**: Original document with color-coded bounding boxes and executive HUD banner.
* **3-Panel Audit Panel (`_audit_panel.jpg`)**: Panel 1: Original + BBoxes | Panel 2: Neural Heatmap | Panel 3: Zoomed Part Crops.
* **JSON Audit Certificate (`_forensic_report.json`)**: Bounding box coordinates, confidence scores, and distortion diagnosis.

---

## 🏛️ Governed Merchant Onboarding KYB/KYC Verification Engine

The framework features an automated **Merchant Onboarding Compliance Engine** ([merchant_verifier.py](file:///e:/project%20final/merchant_fina/src/forensics/merchant_verifier.py)) that validates full onboarding dossiers for 5 legal entity categories:

### 1. Supported Merchant Entity Categories & Checklists

1. **Sole Proprietorship (`proprietorship`)**:
   - Proprietor PAN (4th char MUST be `P`)
   - Proprietor ID Proof (Aadhaar / Passport / Voter ID / DL)
   - Business Proof (Shop & Establishment / Udyam Registration / GST)
   - Bank Proof (Cancelled Cheque or Bank Statement matching Proprietor Name)
2. **Partnership / LLP (`partnership_llp`)**:
   - Partnership Deed / LLP Deed (or COI for LLP)
   - Firm PAN (4th char MUST be `F` or `L`)
   - Authorised Signatory KYC ID Proof
   - Bank Proof (in Firm's Name)
   - GST Certificate
3. **Private / Public Limited Company (`pvt_ltd`)**:
   - Certificate of Incorporation (COI from MCA)
   - MOA & AOA
   - Company PAN (4th char MUST be `C`)
   - Board Resolution (authorising signatory)
   - Authorised Signatory KYC ID Proof
   - Bank Proof (in Company's Name)
   - GST Certificate
4. **Trust / Society / NGO (`trust_society`)**:
   - Trust Deed / Society Bye-Laws
   - Entity PAN (4th char MUST be `T`, `A`, or `B`)
   - Authorised Signatory KYC ID Proof
   - Bank Proof (in Entity's Name)
   - 80G / 12A Tax Exempt Certificate (Optional)
5. **Small Merchant Light KYC Route (`small_merchant` ≤ ₹40L turnover)**:
   - PAN
   - One OVD ID Proof (Aadhaar / Passport / DL / Voter ID)
   - One Business KYC Document

---

### 2. Zero-Training Document Classifier Strategy

Classifies uploaded documents into 15 KYC/KYB types **WITHOUT needing model retraining**:
1. **Deterministic QR Code Decoding (`zxingcpp`)**: Decodes UIDAI Secure QR / XML (Aadhaar), NSDL/UTIITSL payload (PAN), and GSTIN QR (GST Certificate) with **100% precision**.
2. **MICR Line & IFSC Syntax Matching**: Scans bottom band for 9-digit MICR code (`\d{9}`) and 11-char IFSC code (`[A-Z]{4}0[A-Z0-9]{6}`) on Cancelled Cheques.
3. **Structural Anchor Rules & Checksums**: Matches Government spec keypoint anchors (e.g. `INCOME TAX DEPARTMENT` for PAN, `FORM GST REG-06` for GST, 21-char CIN for COI, and **Verhoeff Dihedral Group $D_5$ algorithm** for Aadhaar numbers).

---

### 3. Government Entity Rules & Cross-Document Validation

1. **PAN 4th Character Enforcement**:
   - Checks if the 4th letter of the PAN matches the legal entity type (`P` for Proprietorship, `F` for Partnership, `L` for LLP, `C` for Company, `T`/`A`/`B` for Trust/Society).
2. **GSTIN Embedded PAN Cross-Match**:
   - Verifies characters 3–12 of the 15-digit GSTIN match the Entity PAN.
3. **Cross-Document Name Fuzzy Similarity Graph**:
   - Computes token-sort fuzzy string similarity ($\ge 85\%$ threshold) across PAN, Cancelled Cheque, GST Certificate, and COI.

---

### 4. Deep Learning Forgery Integration (`bigpower` Model)

Once a document passes classification, it is automatically inspected by `ForgeryNet` (loaded from `bigpower/`) to catch text modifications, digit edits, white-out erasures, or spliced stamps/signatures.

---

### 5. CLI Execution Guide for Merchant Onboarding

Run [verify_merchant_onboarding.py](file:///e:/project%20final/merchant_fina/verify_merchant_onboarding.py):

```bash
# Verify a Sole Proprietorship Dossier
python verify_merchant_onboarding.py \
  --entity-type proprietorship \
  --pan "data/samples/proprietor_pan.jpg" \
  --id-proof "data/samples/aadhaar.jpg" \
  --business-proof "data/samples/udyam.jpg" \
  --bank-proof "data/samples/cheque.jpg" \
  --gst "data/samples/gst_reg06.jpg"

# Verify a Private Limited Company Dossier
python verify_merchant_onboarding.py \
  --entity-type pvt_ltd \
  --coi "data/samples/coi.jpg" \
  --pan "data/samples/company_pan.jpg" \
  --board-resolution "data/samples/board_resolution.pdf" \
  --id-proof "data/samples/signatory_aadhaar.jpg" \
  --bank-proof "data/samples/company_cheque.jpg" \
  --gst "data/samples/gst_reg06.jpg"
```

#### Sample JSON Onboarding Certificate Output:
```json
{
  "entity_type": "proprietorship",
  "entity_label": "Sole Proprietorship",
  "onboarding_verdict": "APPROVED",
  "risk_score": 0.0215,
  "checklist_audit": [
    {
      "slot": "proprietor_pan",
      "classified_as": "PAN_CARD",
      "confidence": 1.0,
      "passed": true
    },
    {
      "slot": "bank_proof",
      "classified_as": "CANCELLED_CHEQUE",
      "confidence": 0.95,
      "passed": true
    }
  ],
  "forgery_reports": [
    {
      "slot": "proprietor_pan",
      "verdict": "AUTHENTIC",
      "is_fake": false,
      "fraud_score": 0.015,
      "risk_level": "LOW"
    }
  ],
  "cross_validation_rules": [
    {
      "valid": true,
      "reason": "PAN 4th character 'P' correctly matches expected entity type 'proprietorship'."
    },
    {
      "valid": true,
      "reason": "GSTIN embedded PAN 'ABCDE1234F' perfectly matches Entity PAN 'ABCDE1234F'."
    }
  ]
}
```
