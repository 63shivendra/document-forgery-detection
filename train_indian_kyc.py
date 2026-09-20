"""
High-Throughput Indian KYC/KYB ForgeryNet Fine-Tuning Pipeline
Target Hardware: NVIDIA GeForce RTX 5080 (16GB VRAM, CUDA 12.8, Blackwell Architecture)
"""

import os
import sys
import glob
import time
import random
import argparse
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from src.models.forgery_net import ForgeryNet


# ==============================================================================
# 1. Dataset: Paired Indian KYC Dataset (Clean Negative + Synthetic Forgery)
# ==============================================================================
class IndianKYCFineTuneDataset(Dataset):
    """
    Dataset combining authentic Indian documents (Clean y=0 negatives)
    and dynamically synthesized Indian document forgeries (y=1 with exact pixel masks).
    """
    def __init__(self, workspace_root, img_size=(384, 384), split="train", total_samples=1200):
        self.img_size = img_size
        self.split = split
        self.total_samples = total_samples
        self.samples = []

        # Resolve path to Real_dataset_pan_adhaar
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_final_dir = os.path.dirname(script_dir)
        base_data = os.path.join(project_final_dir, "Real_dataset_pan_adhaar")
        pan_imgs = glob.glob(os.path.join(base_data, "pan_card", split, "images", "*.*"))
        adh_imgs = glob.glob(os.path.join(base_data, "adhaar_card", split, "images", "*.*"))
        chk_imgs = glob.glob(os.path.join(base_data, "bank_cheque", "*", "*.*"))
        gst_imgs = glob.glob(os.path.join(base_data, "gst_certificate", "*.*"))

        self.clean_pool = [p for p in (pan_imgs + adh_imgs + chk_imgs + gst_imgs) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if not self.clean_pool:
            # Fallback to whatever images are available
            self.clean_pool = glob.glob(os.path.join(base_data, "**", "*.jpg"), recursive=True)

        print(f"[{split.upper()}] Initialized pool with {len(self.clean_pool)} real Indian documents.")

    def __len__(self):
        return min(self.total_samples, len(self.clean_pool) * 2)

    def _synthesize_tamper(self, img_bgr):
        """Creates realistic Indian document forgery with 100% exact difference mask."""
        h, w = img_bgr.shape[:2]
        tampered = img_bgr.copy()
        mask = np.zeros((h, w), dtype=np.uint8)

        tamper_mode = random.choice(["digit_alter", "spliced_patch", "text_inpaint", "stamp_duplicate"])

        def safe_rand(low, high):
            return random.randint(low, max(low, high))

        if tamper_mode == "digit_alter":
            # Crop a small number patch and paste with offset
            pw, ph = safe_rand(25, 70), safe_rand(15, 35)
            x1, y1 = safe_rand(10, w - pw - 10), safe_rand(int(h * 0.4), h - ph - 10)
            src_patch = tampered[y1:y1+ph, x1:x1+pw].copy()
            dx = random.randint(-40, 40)
            target_x = max(5, min(w - pw - 5, x1 + dx))
            tampered[y1:y1+ph, target_x:target_x+pw] = src_patch
            mask[y1:y1+ph, target_x:target_x+pw] = 255

        elif tamper_mode == "spliced_patch":
            # Photo or signature replacement box
            pw, ph = safe_rand(50, 110), safe_rand(50, 110)
            x1, y1 = safe_rand(10, w - pw - 10), safe_rand(10, h - ph - 10)
            noise_patch = cv2.GaussianBlur(tampered[y1:y1+ph, x1:x1+pw], (11, 11), 0)
            tampered[y1:y1+ph, x1:x1+pw] = noise_patch
            mask[y1:y1+ph, x1:x1+pw] = 255

        elif tamper_mode == "text_inpaint":
            # White-out or background smudge
            pw, ph = safe_rand(40, 90), safe_rand(12, 25)
            x1, y1 = safe_rand(10, w - pw - 10), safe_rand(int(h * 0.2), h - ph - 10)
            bg_color = np.median(tampered[max(0, y1-5):y1, x1:x1+pw], axis=(0, 1))
            tampered[y1:y1+ph, x1:x1+pw] = bg_color
            mask[y1:y1+ph, x1:x1+pw] = 255

        else: # stamp_duplicate
            pw, ph = safe_rand(40, 80), safe_rand(40, 80)
            x1, y1 = safe_rand(10, w - pw - 10), safe_rand(10, h - ph - 10)
            patch = tampered[y1:y1+ph, x1:x1+pw].copy()
            tx, ty = safe_rand(10, w - pw - 10), safe_rand(10, h - ph - 10)
            tampered[ty:ty+ph, tx:tx+pw] = patch
            mask[ty:ty+ph, tx:tx+pw] = 255

        return tampered, mask

    def __getitem__(self, idx):
        clean_path = self.clean_pool[idx % len(self.clean_pool)]
        img_bgr = cv2.imread(clean_path)
        if img_bgr is None:
            img_bgr = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)

        # Ensure image is resized to standard training resolution first
        img_bgr = cv2.resize(img_bgr, self.img_size, interpolation=cv2.INTER_LINEAR)

        # 50% Clean Negative (y=0) vs 50% Tampered (y=1)
        is_tampered = (idx % 2 == 1)

        if is_tampered:
            img_bgr, mask = self._synthesize_tamper(img_bgr)
        else:
            mask = np.zeros(self.img_size, dtype=np.uint8)

        # Convert to RGB Float Tensor [3, H, W] normalized [-1, 1]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        # Normalize with ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0
        return img_tensor, mask_tensor, os.path.basename(clean_path)


# ==============================================================================
# 2. Combined Focal + Dice Loss
# ==============================================================================
class CombinedFocalDiceLoss(nn.Module):
    def __init__(self, pos_weight=5.0, focal_weight=0.4, dice_weight=0.6):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device)
        )
        probs = torch.sigmoid(logits)
        smooth = 1e-5
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        return self.focal_weight * bce + self.dice_weight * dice


# ==============================================================================
# 3. Main Training Routine (Blackwell RTX 5080 Optimized)
# ==============================================================================
def train_indian_kyc():
    parser = argparse.ArgumentParser(description="RTX 5080 Indian KYC ForgeryNet Fine-Tuning")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for RTX 5080 16GB")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate for fine-tuning")
    parser.add_argument("--img-size", type=int, default=384, help="Square image resolution")
    args = parser.parse_args()

    # Hardware & Performance Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bigpower_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bigpower")
    base_checkpoint = os.path.join(bigpower_dir, "best_model_splicing_combined_efficientnet-b2_20260908_1523.pth")

    print("==========================================================================")
    print("  [*] RTX 5080 Governed Indian KYC ForgeryNet Fine-Tuning Engine")
    print(f"  Device         : {device} ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM Available : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print(f"  Batch Size     : {args.batch_size} (AMP FP16 Enabled)")
    print(f"  Image Size     : {args.img_size} x {args.img_size}")
    print(f"  Learning Rate  : {args.lr}")
    print(f"  Base Checkpoint: {os.path.basename(base_checkpoint)}")
    print("==========================================================================")

    # 1. Datasets & Loaders
    train_dataset = IndianKYCFineTuneDataset(workspace_root, img_size=(args.img_size, args.img_size), split="train", total_samples=1600)
    val_dataset   = IndianKYCFineTuneDataset(workspace_root, img_size=(args.img_size, args.img_size), split="valid", total_samples=320)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # 2. Model Initialization & Weights Transfer
    model = ForgeryNet(out_channels=1, encoder_name="efficientnet-b2", dropout_prob=0.2).to(device)

    if os.path.exists(base_checkpoint):
        print(f"\n[1/3] Loading existing weights from {os.path.basename(base_checkpoint)}...")
        ckpt = torch.load(base_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print("      Successfully transferred existing forgery detection weights!")
    else:
        print("\n[!] Warning: Base checkpoint not found. Training from scratch.")

    # 3. Optimizer, Loss & Scaler
    criterion = CombinedFocalDiceLoss(pos_weight=6.0, focal_weight=0.4, dice_weight=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda')

    # 4. Training Loop
    best_val_iou = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = os.path.join(bigpower_dir, f"best_model_indian_kyc_enhanced_{timestamp}.pth")

    print(f"\n[2/3] Starting {args.epochs} Fine-Tuning Epochs on NVIDIA RTX 5080...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        for step, (images, masks, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        epoch_sec = time.time() - start_time

        # Validation Pass
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        val_batches = 0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, masks)

                val_loss += loss.item()

                # Calculate IoU
                preds = (torch.sigmoid(logits) > 0.35).float()
                intersection = (preds * masks).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3)) - intersection
                iou = (intersection / (union + 1e-6)).mean().item()
                total_iou += iou
                val_batches += 1

        val_loss /= max(1, len(val_loader))
        val_iou = (total_iou / max(1, val_batches)) * 100.0

        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "timestamp": timestamp
            }, save_path)

        flag = "[+] BEST MODEL SAVED!" if is_best else ""
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_sec:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:5.2f}% {flag}")

    print("\n==========================================================================")
    print(f"  Fine-Tuning Finished Successfully!")
    print(f"  Best Val IoU     : {best_val_iou:.2f}%")
    print(f"  Saved Checkpoint : {save_path}")
    print("==========================================================================")


if __name__ == "__main__":
    train_indian_kyc()
