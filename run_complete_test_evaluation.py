import os
import sys
import json
import argparse
import time
import cv2
import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.forensics.document_verifier import DocumentVerifier
from src.dataset.base_dataset import get_dataset


def denormalize_image(tensor):
    """Denormalizes PyTorch tensor from ImageNet stats back to [0, 255] RGB numpy array."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = tensor.cpu().numpy() * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # H, W, C
    return arr


def run_evaluation(datasets_config, output_root="reports/forensics/complete_eval", limit_per_dataset=None):
    os.makedirs(output_root, exist_ok=True)
    
    print("\n" + "=" * 92)
    print("       COMPREHENSIVE TEST DATASET FORENSIC EVALUATION & AUDIT")
    print("       (Ground Truth Mask Comparison, Visual Evidence Boards & JSON Reports)")
    print("=" * 92)
    
    verifier = DocumentVerifier()
    
    overall_all_datasets = {}
    total_start_time = time.time()
    
    for ds_name, title, root in datasets_config:
        ds_out_dir = os.path.join(output_root, ds_name)
        boards_dir = os.path.join(ds_out_dir, "comparison_boards")
        os.makedirs(boards_dir, exist_ok=True)
        
        print(f"\n[{ds_name.upper()}] - {title}")
        print("-" * 92)
        
        try:
            split_name = "test"
            ds = get_dataset(ds_name, data_root=root, split=split_name, img_size=(384, 384))
            total_samples = len(ds)
            print(f"  * Available Test Samples in Dataset : {total_samples}")
            
            num_to_run = total_samples if limit_per_dataset is None else min(limit_per_dataset, total_samples)
            print(f"  * Running Complete Evaluation On    : {num_to_run} samples")
            
            results_list = []
            tp_count = 0
            tn_count = 0
            fp_count = 0
            fn_count = 0
            
            iou_list = []
            f1_list = []
            precision_list = []
            recall_list = []
            
            for idx in range(num_to_run):
                img_tensor, mask_tensor, sample_name = ds[idx]
                clean_name = os.path.splitext(os.path.basename(sample_name))[0]
                
                # Ground truth mask
                gt_mask = mask_tensor.squeeze().cpu().numpy()
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                gt_tampered_pixels = int(np.sum(gt_bin))
                gt_is_tampered = gt_tampered_pixels >= 25
                
                # Original Image
                rgb_img = denormalize_image(img_tensor)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                h, w = bgr_img.shape[:2]
                
                # Write temp file for verifier
                tmp_img_path = os.path.join(ds_out_dir, f"_temp_{idx}.png")
                cv2.imwrite(tmp_img_path, bgr_img)
                
                # Model Inference
                t0 = time.time()
                res = verifier.verify(tmp_img_path)
                inference_time_ms = round((time.time() - t0) * 1000.0, 1)
                
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
                
                pred_probs = res["raw_probs"]
                pred_bin = (pred_probs >= verifier.pixel_threshold).astype(np.uint8)
                pred_tampered_pixels = int(np.sum(pred_bin))
                pred_is_tampered = res["is_fake"]
                
                # Verdict Match
                doc_verdict_match = (gt_is_tampered == pred_is_tampered)
                if gt_is_tampered and pred_is_tampered:
                    tp_count += 1
                elif not gt_is_tampered and not pred_is_tampered:
                    tn_count += 1
                elif not gt_is_tampered and pred_is_tampered:
                    fp_count += 1
                elif gt_is_tampered and not pred_is_tampered:
                    fn_count += 1
                
                # Pixel Overlap Metrics
                intersection = int(np.sum(gt_bin & pred_bin))
                union = int(np.sum(gt_bin | pred_bin))
                
                if gt_is_tampered:
                    iou = (intersection / float(union + 1e-7)) * 100.0
                    prec = (intersection / float(pred_tampered_pixels + 1e-7)) * 100.0
                    rec = (intersection / float(gt_tampered_pixels + 1e-7)) * 100.0
                    f1 = (2.0 * prec * rec) / (prec + rec + 1e-7)
                    
                    iou_list.append(iou)
                    f1_list.append(f1)
                    precision_list.append(prec)
                    recall_list.append(rec)
                else:
                    iou = 100.0 if pred_tampered_pixels == 0 else 0.0
                    prec = 100.0 if pred_tampered_pixels == 0 else 0.0
                    rec = 100.0
                    f1 = 100.0 if pred_tampered_pixels == 0 else 0.0
                
                # Ground truth parts
                num_gt_labels, _, stats_gt, _ = cv2.connectedComponentsWithStats(gt_bin, connectivity=8)
                gt_boxes = []
                for g_id in range(1, num_gt_labels):
                    g_area = stats_gt[g_id, cv2.CC_STAT_AREA]
                    if g_area >= 10:
                        gx = stats_gt[g_id, cv2.CC_STAT_LEFT]
                        gy = stats_gt[g_id, cv2.CC_STAT_TOP]
                        gw = stats_gt[g_id, cv2.CC_STAT_WIDTH]
                        gh = stats_gt[g_id, cv2.CC_STAT_HEIGHT]
                        gt_boxes.append([int(gx), int(gy), int(gx + gw), int(gy + gh)])
                
                # Model detected regions
                regions_data = []
                for r in res["tampered_regions"]:
                    regions_data.append({
                        "region_id": r["region_id"],
                        "bounding_box": r["bbox"],
                        "document_region": r.get("document_region", "Unknown"),
                        "distortion_type": r["distortion_type"],
                        "peak_confidence": r["peak_confidence"],
                        "area_pixels": r["area_pixels"],
                        "extracted_text": r.get("extracted_text", "")
                    })
                
                # 4-Panel Side-by-Side Diagnostic Board
                # Panel 1: Original
                p1 = bgr_img.copy()
                cv2.putText(p1, f"1. ORIGINAL ({w}x{h})", (12, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(p1, clean_name[:25], (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Panel 2: Ground Truth with Green Boxes
                p2 = bgr_img.copy()
                gt_color = cv2.applyColorMap((gt_bin * 255).astype(np.uint8), cv2.COLORMAP_JET)
                p2 = cv2.addWeighted(p2, 0.65, gt_color, 0.35, 0)
                for gb in gt_boxes:
                    cv2.rectangle(p2, (gb[0], gb[1]), (gb[2], gb[3]), (0, 255, 0), 2)
                cv2.putText(p2, f"2. GROUND TRUTH ({len(gt_boxes)} Parts)", (12, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(p2, f"{'TAMPERED' if gt_is_tampered else 'AUTHENTIC'} ({gt_tampered_pixels} px)", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Panel 3: Model Prediction Heatmap + BBoxes + OCR Text
                p3 = bgr_img.copy()
                thermal = cv2.applyColorMap((pred_probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
                p3 = cv2.addWeighted(p3, 0.55, thermal, 0.45, 0)
                for r in res["tampered_regions"]:
                    bx1, by1, bx2, by2 = r["bbox"]
                    cv2.rectangle(p3, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    ocr_snippet = r.get("extracted_text", "")
                    label = f"OCR: {ocr_snippet[:15]}" if ocr_snippet else r["distortion_type"][:18]
                    cv2.putText(p3, label, (bx1, max(15, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(p3, f"3. MODEL PREDICTION", (12, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.putText(p3, f"{res['document_verdict']} ({res['confidence']*100:.1f}%)", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                
                # Panel 4: Overlap Diagnostic Map (Green=Hit, Red=FP, Cyan=FN)
                err_map = np.zeros_like(bgr_img)
                err_map[(gt_bin == 1) & (pred_bin == 1)] = (0, 255, 0)     # Hit (TP)
                err_map[(gt_bin == 0) & (pred_bin == 1)] = (0, 0, 255)     # False alarm (FP)
                err_map[(gt_bin == 1) & (pred_bin == 0)] = (255, 180, 0)   # Missed (FN)
                p4 = cv2.addWeighted(bgr_img, 0.5, err_map, 0.5, 0)
                cv2.putText(p4, f"4. OVERLAP MAP", (12, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(p4, f"IoU: {iou:.1f}% | Rec: {rec:.1f}%", (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                
                board = np.hstack([p1, p2, p3, p4])
                board_filename = f"{clean_name}_comparison.jpg"
                board_save_path = os.path.join(boards_dir, board_filename)
                cv2.imwrite(board_save_path, board)
                
                sample_record = {
                    "index": idx,
                    "sample_name": sample_name,
                    "clean_name": clean_name,
                    "resolution": [w, h],
                    "inference_time_ms": inference_time_ms,
                    "ground_truth": {
                        "is_tampered": gt_is_tampered,
                        "tampered_pixels": gt_tampered_pixels,
                        "bounding_boxes": gt_boxes
                    },
                    "model_prediction": {
                        "verdict": res["document_verdict"],
                        "is_fake": res["is_fake"],
                        "confidence": round(res["confidence"], 4),
                        "fraud_score": round(res["fraud_score"], 4),
                        "risk_level": res["risk_level"],
                        "tampered_pixels_detected": pred_tampered_pixels,
                        "detected_regions_count": len(regions_data),
                        "detected_regions": regions_data
                    },
                    "metrics": {
                        "verdict_match": doc_verdict_match,
                        "iou": round(iou, 2),
                        "dice_f1": round(f1, 2),
                        "precision": round(prec, 2),
                        "recall": round(rec, 2)
                    },
                    "evidence_board_path": board_save_path
                }
                results_list.append(sample_record)
                
                # Log progress
                status_icon = "[MATCH]" if doc_verdict_match else "[MISMATCH]"
                ocr_preview = f" | OCR: '{regions_data[0]['extracted_text'][:15]}'" if (regions_data and regions_data[0]['extracted_text']) else ""
                print(f"    [{idx+1:03d}/{num_to_run:03d}] {clean_name[:24]:<24} | {status_icon} GT={'TAMPERED' if gt_is_tampered else 'CLEAN'} | Pred={res['document_verdict'][:8]} | IoU={iou:>5.1f}% | Rec={rec:>5.1f}%{ocr_preview}")
            
            # Dataset Summary Stats
            total_evaluated = len(results_list)
            doc_accuracy = round(((tp_count + tn_count) / float(total_evaluated + 1e-7)) * 100.0, 2)
            mean_iou = round(float(np.mean(iou_list)) if iou_list else 0.0, 2)
            mean_f1 = round(float(np.mean(f1_list)) if f1_list else 0.0, 2)
            mean_rec = round(float(np.mean(recall_list)) if recall_list else 0.0, 2)
            mean_prec = round(float(np.mean(precision_list)) if precision_list else 0.0, 2)
            
            dataset_summary = {
                "dataset_name": ds_name,
                "dataset_title": title,
                "total_samples_evaluated": total_evaluated,
                "document_level_accuracy": doc_accuracy,
                "confusion_matrix": {
                    "true_positives": tp_count,
                    "true_negatives": tn_count,
                    "false_positives": fp_count,
                    "false_negatives": fn_count
                },
                "mean_pixel_metrics": {
                    "mean_iou": mean_iou,
                    "mean_dice_f1": mean_f1,
                    "mean_recall": mean_rec,
                    "mean_precision": mean_prec
                },
                "samples": results_list
            }
            
            # Save Dataset JSON Report
            ds_json_path = os.path.join(ds_out_dir, "evaluation_report.json")
            with open(ds_json_path, "w", encoding="utf-8") as f:
                json.dump(dataset_summary, f, indent=2)
                
            print(f"\n  [+] Saved {total_evaluated} Comparison Boards to : {boards_dir}")
            print(f"  [+] Saved Dataset JSON Report to       : {ds_json_path}")
            print(f"  --> Verdict Accuracy: {doc_accuracy}% | Mean Recall: {mean_rec}% | Mean IoU: {mean_iou}%")
            
            overall_all_datasets[ds_name] = {
                "title": title,
                "samples_evaluated": total_evaluated,
                "document_accuracy": doc_accuracy,
                "mean_recall": mean_rec,
                "mean_iou": mean_iou,
                "mean_dice_f1": mean_f1,
                "mean_precision": mean_prec,
                "json_report": ds_json_path,
                "boards_dir": boards_dir
            }
            
        except Exception as e:
            print(f"  [Error] Processing {ds_name} failed: {e}")
            import traceback
            traceback.print_exc()

    total_time = round(time.time() - total_start_time, 2)
    
    # Save Master Global Summary
    global_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_time,
        "datasets": overall_all_datasets
    }
    
    global_json_path = os.path.join(output_root, "full_benchmark_summary.json")
    with open(global_json_path, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)
        
    # Generate Global Markdown Summary
    md_path = os.path.join(output_root, "SUMMARY_REPORT.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Forensic Verification Benchmark Report\n\n")
        f.write(f"- **Execution Time**: {total_time}s\n")
        f.write(f"- **Total Datasets Evaluated**: {len(overall_all_datasets)}\n\n")
        f.write("| Dataset | Samples | Verdict Accuracy | Mean Recall | Mean Pixel IoU | Mean Dice F1 |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for k, v in overall_all_datasets.items():
            f.write(f"| **{v['title']}** | {v['samples_evaluated']} | **{v['document_accuracy']}%** | {v['mean_recall']}% | {v['mean_iou']}% | {v['mean_dice_f1']}% |\n")
        f.write("\n## Output Directories\n\n")
        for k, v in overall_all_datasets.items():
            f.write(f"- **{k.upper()}**: [JSON Report]({v['json_report']}) | [Comparison Boards]({v['boards_dir']})\n")
            
    print("\n" + "=" * 92)
    print(f"  COMPLETE BENCHMARK FINISHED in {total_time}s")
    print(f"  [+] Master Global Summary JSON : {global_json_path}")
    print(f"  [+] Markdown Summary Document  : {md_path}")
    print("=" * 92 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run complete test dataset verification and ground truth audit.")
    parser.add_argument("--dataset", "-d", type=str, default="all",
                        help="Specific dataset to evaluate: 'splicing', 'casia1', 'cocoglide', 'doctamper', 'doctamper_scd', or 'all'")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limit number of test samples per dataset (default: all)")
    parser.add_argument("--output-dir", "-o", type=str, default="reports/forensics/complete_eval",
                        help="Output directory for reports and images")
    args = parser.parse_args()
    
    all_configs = [
        ("splicing", "Splicing & Copy-Move Forgery Dataset", "data"),
        ("casia1", "CASIA 1.0 Tampered Benchmark", "test_datasets"),
        ("cocoglide", "COCOGLIDE / TruFor Splicing & Inpainting", "data"),
        ("doctamper", "DocTamper Document Text Tampering", "data"),
        ("doctamper_scd", "DocTamper Single Character / Digit Tampering", "data")
    ]
    
    if args.dataset != "all":
        selected = [c for c in all_configs if c[0] == args.dataset.lower()]
        if not selected:
            print(f"[Error] Dataset '{args.dataset}' not found. Available: {[c[0] for c in all_configs]}")
            return
        configs = selected
    else:
        configs = all_configs
        
    run_evaluation(configs, output_root=args.output_dir, limit_per_dataset=args.limit)


if __name__ == "__main__":
    main()
