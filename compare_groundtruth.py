import os
import sys
import argparse
import cv2
import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.forensics.document_verifier import DocumentVerifier


def evaluate_prediction_vs_groundtruth(verifier, img_path, mask_path, output_dir="reports/forensics/comparisons"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # 1. Load image and mask
    orig_bgr = cv2.imread(img_path)
    if orig_bgr is None:
        raise ValueError(f"Failed to read image at {img_path}")
    orig_h, orig_w = orig_bgr.shape[:2]

    gt_mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if gt_mask_raw is None:
        raise ValueError(f"Failed to read mask at {mask_path}")

    # Resize mask to image size if slightly different
    if gt_mask_raw.shape[:2] != (orig_h, orig_w):
        gt_mask_raw = cv2.resize(gt_mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    gt_bin = (gt_mask_raw > 127).astype(np.uint8) if gt_mask_raw.max() > 1 else (gt_mask_raw > 0).astype(np.uint8)
    gt_tampered_pixels = int(np.sum(gt_bin))
    gt_is_tampered = gt_tampered_pixels >= 15

    # 2. Run model verification
    result = verifier.verify(img_path)
    pred_probs = result["raw_probs"]  # (H, W) float
    pred_bin = (pred_probs >= verifier.pixel_threshold).astype(np.uint8)
    pred_tampered_pixels = int(np.sum(pred_bin))
    pred_is_tampered = result["is_fake"]

    # 3. Calculate mathematical overlap metrics
    intersection = int(np.sum(gt_bin & pred_bin))
    union = int(np.sum(gt_bin | pred_bin))
    iou = (intersection / float(union + 1e-7)) * 100.0

    precision = (intersection / float(pred_tampered_pixels + 1e-7)) * 100.0
    recall = (intersection / float(gt_tampered_pixels + 1e-7)) * 100.0
    f1_score = (2.0 * precision * recall) / (precision + recall + 1e-7)

    # Document-Level Decision Match
    doc_match = (gt_is_tampered == pred_is_tampered)

    # 4. Extract Ground Truth Bounding Boxes
    num_gt, _, stats_gt, _ = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)
    gt_boxes = []
    for i in range(1, num_gt):
        if stats_gt[i, cv2.CC_STAT_AREA] >= 10:
            gx = stats_gt[i, cv2.CC_STAT_LEFT]
            gy = stats_gt[i, cv2.CC_STAT_TOP]
            gw = stats_gt[i, cv2.CC_STAT_WIDTH]
            gh = stats_gt[i, cv2.CC_STAT_HEIGHT]
            gt_boxes.append([gx, gy, gx + gw, gy + gh])

    pred_boxes = [r["bbox"] for r in result["tampered_regions"]]

    # 5. Print Forensic Scorecard
    print("\n" + "=" * 78)
    print(f"  VERIFICATION AUDIT: Model Prediction vs. Ground Truth Mask")
    print("=" * 78)
    print(f"  * Document Image       : {os.path.basename(img_path)} ({orig_w}x{orig_h} px)")
    print(f"  * Ground Truth Mask    : {os.path.basename(mask_path)}")
    print("-" * 78)
    print(f"  [DOCUMENT-LEVEL VERDICT]")
    print(f"  * Ground Truth Status  : {'TAMPERED / FAKE' if gt_is_tampered else 'AUTHENTIC'}")
    print(f"  * Model Prediction     : {result['document_verdict']} (Confidence: {result['confidence']*100:.1f}%)")
    print(f"  * Verdict Correctness  : {'MATCH (100% Correct)' if doc_match else 'MISMATCH'}")
    print("-" * 78)
    print(f"  [PIXEL-LEVEL LOCALIZATION ACCURACY]")
    print(f"  * Overlap IoU (Jaccard): {iou:.2f}%")
    print(f"  * Dice Score (F1-Score): {f1_score:.2f}%")
    print(f"  * Precision            : {precision:.2f}% (How clean the detected area is)")
    print(f"  * Recall (Capture Rate): {recall:.2f}% (How much of the real edit was captured)")
    print("-" * 78)
    print(f"  [BOUNDING BOX LOCALIZATION]")
    print(f"  * Ground Truth Parts   : {len(gt_boxes)} true tampered regions")
    print(f"  * Model Detected Parts : {len(pred_boxes)} predicted tampered regions")
    for idx, gb in enumerate(gt_boxes, 1):
        print(f"    - True Region {idx} : BBox={gb}")
    for idx, pb in enumerate(pred_boxes, 1):
        print(f"    - Pred Region {idx} : BBox={pb} | Conf={result['tampered_regions'][idx-1]['peak_confidence']*100:.1f}%")

    # 6. Render 4-Panel Side-by-Side Diagnostic Board
    target_h = 450
    target_w = int(orig_w * (target_h / float(orig_h)))

    # Panel 1: Original
    p1 = cv2.resize(orig_bgr, (target_w, target_h))
    cv2.putText(p1, "1. ORIGINAL DOCUMENT", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # Panel 2: Ground Truth with Green Boxes
    p2 = cv2.resize(orig_bgr.copy(), (target_w, target_h))
    sx = target_w / float(orig_w)
    sy = target_h / float(orig_h)
    for gb in gt_boxes:
        cv2.rectangle(p2, (int(gb[0]*sx), int(gb[1]*sy)), (int(gb[2]*sx), int(gb[3]*sy)), (0, 220, 0), 2)
    cv2.putText(p2, f"2. GROUND TRUTH ({len(gt_boxes)} Parts)", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)

    # Panel 3: Model Prediction with Red Boxes + Heatmap
    thermal = cv2.applyColorMap((pred_probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(orig_bgr, 0.6, thermal, 0.4, 0)
    p3 = cv2.resize(blended, (target_w, target_h))
    for pb in pred_boxes:
        cv2.rectangle(p3, (int(pb[0]*sx), int(pb[1]*sy)), (int(pb[2]*sx), int(pb[3]*sy)), (0, 0, 255), 2)
    cv2.putText(p3, f"3. MODEL PREDICTION (IoU: {iou:.1f}%)", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 180, 255), 1, cv2.LINE_AA)

    # Panel 4: Color-Coded Error Map
    # Green = True Positive (Hit), Red = False Positive, Blue = False Negative (Miss)
    error_map = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    # TP (Hit) -> Bright Green
    tp_mask = (gt_bin == 1) & (pred_bin == 1)
    error_map[tp_mask] = (0, 255, 0)
    # FP (False alarm) -> Red
    fp_mask = (gt_bin == 0) & (pred_bin == 1)
    error_map[fp_mask] = (0, 0, 255)
    # FN (Missed) -> Cyan/Blue
    fn_mask = (gt_bin == 1) & (pred_bin == 0)
    error_map[fn_mask] = (255, 180, 0)

    p4_blended = cv2.addWeighted(orig_bgr, 0.5, error_map, 0.5, 0)
    p4 = cv2.resize(p4_blended, (target_w, target_h))
    cv2.putText(p4, "4. OVERLAP MAP (Green=Hit, Red=FP)", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    comparison_board = np.hstack([p1, p2, p3, p4])
    save_img_path = os.path.join(output_dir, f"{base_name}_vs_groundtruth.jpg")
    cv2.imwrite(save_img_path, comparison_board)

    print("-" * 78)
    print(f"[+] Saved 4-Panel Verification Evidence to: {save_img_path}")
    print("=" * 78 + "\n")

    return {
        "iou": round(iou, 2),
        "f1_score": round(f1_score, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "verdict_match": doc_match,
        "saved_evidence": save_img_path
    }


def main():
    parser = argparse.ArgumentParser(description="Direct Mathematical & Visual Verification: Model vs Ground Truth Mask")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to test image")
    parser.add_argument("--mask", "-m", type=str, required=True, help="Path to ground truth mask")
    parser.add_argument("--output-dir", "-o", type=str, default="reports/forensics/comparisons")
    args = parser.parse_args()

    verifier = DocumentVerifier()
    evaluate_prediction_vs_groundtruth(verifier, args.image, args.mask, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
