import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss with pos_weight for severe class imbalance.

    Forgery regions are typically <5% of total pixels. Without pos_weight, the model
    learns to predict 'all authentic' and still gets low loss — completely ignoring forgery.

    pos_weight=10.0 means: missing a forged pixel is penalized 10x more than missing an
    authentic pixel. This forces the model to actively detect forgery regions.
    """
    def __init__(self, focal_weight=0.3, dice_weight=0.7, gamma=2.0, pos_weight=10.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # pos_weight tensor on same device
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)

        # 1. Weighted Focal Loss (with pos_weight for class imbalance)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw,
            reduction='none'
        )
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = (bce * ((1 - p_t) ** self.gamma)).mean()

        # 2. Dice Loss (geometry-aware, handles imbalance naturally)
        smooth = 1e-6
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = dice_loss.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


def calculate_pixel_metrics(pred_probs, targets, threshold=0.35, min_forged_pixels=15):
    """
    Computes comprehensive Forensic Evaluation Metrics:
    - Pixel Accuracy: (TP + TN) / (TP + TN + FP + FN)
    - Pixel Balanced Accuracy: (Sensitivity + Specificity) / 2
    - Pixel Specificity (TNR): TN / (TN + FP)
    - Pixel IoU (Jaccard Index): TP / (TP + FP + FN)
    - Pixel F1-Score (Dice): 2*TP / (2*TP + FP + FN)
    - Pixel Precision: TP / (TP + FP)
    - Pixel Recall (Sensitivity / TPR): TP / (TP + FN)
    - Mean Absolute Error (MAE): mean(|P - Y|)
    - Image-Level Document Classification Accuracy: Document Tampered vs Authentic
    """
    pred_bin = (pred_probs >= threshold).float()
    batch_size = pred_probs.size(0)

    acc_stats = {
        'iou': 0.0,
        'f1': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'pixel_accuracy': 0.0,
        'balanced_accuracy': 0.0,
        'specificity': 0.0,
        'mae': 0.0,
        'image_accuracy': 0.0,
        'image_tp': 0.0,
        'image_fp': 0.0,
        'image_fn': 0.0,
        'image_tn': 0.0
    }

    for b in range(batch_size):
        p = pred_bin[b, 0]
        t = targets[b, 0]
        raw_p = pred_probs[b, 0]

        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        total_px = tp + fp + fn + tn

        # 1. Pixel Accuracy
        pixel_acc = (tp + tn) / (total_px + 1e-7)

        # 2. Specificity (Background / Non-tampered accuracy)
        specificity = tn / (tn + fp + 1e-7)

        # 3. Precision & Recall
        if (tp + fn) == 0 and (tp + fp) == 0:
            iou = 1.0
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        elif (tp + fn) == 0 and (tp + fp) > 0:
            iou = 0.0
            precision = 0.0
            recall = 1.0
            f1 = 0.0
        else:
            union = tp + fp + fn
            iou = tp / (union + 1e-7)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)

        # 4. Balanced Accuracy
        balanced_acc = (recall + specificity) / 2.0

        # 5. Mean Absolute Error
        mae = torch.abs(raw_p - t).mean().item()

        # 6. Image-Level Document Classification (Tampered vs Authentic)
        gt_tampered = 1 if (tp + fn) > 0 else 0
        pred_tampered = 1 if (tp + fp) >= min_forged_pixels else 0
        img_correct = 1.0 if (gt_tampered == pred_tampered) else 0.0

        acc_stats['iou'] += iou
        acc_stats['f1'] += f1
        acc_stats['precision'] += precision
        acc_stats['recall'] += recall
        acc_stats['pixel_accuracy'] += pixel_acc
        acc_stats['balanced_accuracy'] += balanced_acc
        acc_stats['specificity'] += specificity
        acc_stats['mae'] += mae
        acc_stats['image_accuracy'] += img_correct

        if gt_tampered == 1 and pred_tampered == 1:
            acc_stats['image_tp'] += 1.0
        elif gt_tampered == 0 and pred_tampered == 1:
            acc_stats['image_fp'] += 1.0
        elif gt_tampered == 1 and pred_tampered == 0:
            acc_stats['image_fn'] += 1.0
        else:
            acc_stats['image_tn'] += 1.0

    return {k: v / batch_size for k, v in acc_stats.items()}
