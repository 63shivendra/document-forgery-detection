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


def calculate_pixel_metrics(pred_probs, targets, threshold=0.35):
    """
    Computes Sample-Averaged Pixel-Level IoU, F1-Score, Precision, and Recall.

    threshold=0.35: EfficientNet encoders with pretrained weights tend to output
    conservative probabilities (0.2-0.45 range for forgery). Using 0.35 instead
    of 0.5 recovers true positive detections that would otherwise be counted as zero.

    pred_probs: [B, 1, H, W] tensor of float probabilities in [0, 1]
    targets   : [B, 1, H, W] ground truth binary mask
    """
    pred_bin = (pred_probs >= threshold).float()
    batch_size = pred_probs.size(0)

    total_iou, total_f1, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0

    for b in range(batch_size):
        p = pred_bin[b, 0]
        t = targets[b, 0]

        intersection = (p * t).sum().item()
        total_pred = p.sum().item()
        total_gt = t.sum().item()

        if total_gt == 0 and total_pred == 0:
            # Both empty: perfect score (image is authentic and correctly predicted)
            iou = 1.0
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        elif total_gt == 0 and total_pred > 0:
            # False positive: predicted forgery where none exists
            iou = 0.0
            precision = 0.0
            recall = 1.0
            f1 = 0.0
        else:
            union = total_pred + total_gt - intersection
            iou = intersection / (union + 1e-7)
            precision = intersection / (total_pred + 1e-7)
            recall = intersection / (total_gt + 1e-7)
            f1 = (2.0 * precision * recall) / (precision + recall + 1e-7)

        total_iou += iou
        total_f1 += f1
        total_prec += precision
        total_rec += recall

    return {
        'iou': total_iou / batch_size,
        'f1': total_f1 / batch_size,
        'precision': total_prec / batch_size,
        'recall': total_rec / batch_size
    }
