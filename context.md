# 📄 Governed Merchant Document Forgery & Tampering Framework

Welcome to the complete, modular, and production-ready PyTorch training framework for **Governed Merchant Onboarding Document Verification and Forensics**.

---

## 📂 Project Directory Structure

```
e:/merchant/
├── context.md                    # Detailed documentation & operational guide (this file)
├── config.yaml                   # Complete hyperparameters, regularization & dataset paths
├── train.py                      # Main CLI entrypoint for full-dataset training
├── evaluate.py                   # Benchmarking script for test set evaluation
├── savedmodels/                  # Export target directory for PyTorch checkpoints & TorchScript models
├── charts/                       # Auto-generated loss curves and IoU / F1 metric charts
├── data/                         # Folder containing downloaded dataset zips/folders
│   ├── doctamper/                # DocTamper (170K text/field tampering images)
│   ├── cocoglide/                # CocoGlide (TruFor image splicing & AI inpainting)
│   ├── idnet/                    # IDNet (Identity document template analysis)
│   └── fhdmi/                    # FHDMi (Screen recaptured Moiré pattern dataset)
└── src/
    ├── dataset/
    │   ├── augmentations.py      # Albumentations anti-overfitting image pipeline
    │   └── base_dataset.py       # Universal PyTorch Dataset loader
    ├── models/
    │   ├── blocks.py             # Residual blocks + GroupNorm (num_groups=32) + Spatial Dropout2D
    │   └── dual_branch_unet.py   # Dual-Branch (RGB + DCT Frequency) Segmentation U-Net
    └── utils/
        ├── metrics.py            # Focal + Dice Loss & Pixel IoU, F1, Precision, Recall
        ├── logger.py             # Console logger
        ├── plotter.py            # Matplotlib Loss & Accuracy chart generator
        └── trainer.py            # AMP fp16 Trainer Engine with Early Stopping & TorchScript Export
```

---

## 🎯 Dataset Verification & Links

The 4 official datasets integrated into this framework correspond directly to your research papers:

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
5. **Focal + Dice Loss Combination**:
   - Prevents class imbalance collapse where background non-edited pixels dominate ($>95\%$ of document area).

---

## 🚀 How to Run Full-Dataset Training

### Step 1: Organize Your Downloaded Datasets
Unzip your downloaded dataset files into the `./data/` folder following this structure:
```
data/doctamper/train/images/  <-- put image files here (.jpg, .png)
data/doctamper/train/masks/   <-- put binary mask files here
data/doctamper/val/images/
data/doctamper/val/masks/
```

### Step 2: Launch Training
Run `train.py` specifying your dataset:

```bash
# Train on DocTamper (Full Dataset)
python train.py --dataset doctamper --epochs 25 --batch-size 16

# Train on CocoGlide (TruFor benchmark)
python train.py --dataset cocoglide --epochs 25 --batch-size 16

# Train on IDNet
python train.py --dataset idnet --epochs 20 --batch-size 16

# Train on FHDMi
python train.py --dataset fhdmi --epochs 20 --batch-size 16
```

---

## 📊 Auto-Generated Outputs & Production Export

When training completes, the system automatically creates:

1. **Production Models inside `savedmodels/`**:
   - `best_model_doctamper.pth`: PyTorch `state_dict` checkpoint with hyperparameter metadata.
   - `production_model_doctamper.pt`: **Standalone TorchScript serialized model** ready for zero-dependency production deployment!
2. **Loss & Metric Charts inside `charts/`**:
   - `loss_chart.png`: Loss trajectory comparison (Train vs Validation).
   - `metrics_chart.png`: Pixel IoU (%) and F1-Score (%) localization progress curves.

---

## 🧪 How to Benchmark Test Set

To run benchmark evaluation against test set ground truth:

```bash
python evaluate.py --dataset doctamper
```

This will print the official benchmark metrics:
- **Pixel IoU (%)**
- **Pixel F1-Score (%)**
- **Pixel Precision & Recall (%)**
