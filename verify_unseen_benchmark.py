import os
import sys
import json
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

def denorm(tensor):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = tensor.cpu().numpy() * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    return arr

def run_benchmark_on_unseen(num_samples_per_ds=2, output_dir="reports/forensics/unseen_eval"):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "boards"), exist_ok=True)
    
    verifier = DocumentVerifier()
    
    datasets_to_test = [
        ("doctamper", "Official DocTamper (Document Text Tampering)", "data"),
        ("doctamper_scd", "DocTamper-SCD (Single Character / Digit Tampering)", "data"),
        ("splicing", "Splicing Dataset (Document & Image Splicing)", "data"),
        ("cocoglide", "COCOGLIDE (AI Inpainting & Object Removal)", "data"),
        ("casia1", "CASIA 1.0 (Tampered Benchmark)", "test_datasets"),
    ]
    
    overall_summary = []
    
    print("\n" + "=" * 90)
    print("       FORENSIC VERIFICATION AUDIT ON UNSEEN TEST DATASETS")
    print("       (Localization + Distortion Classification + Text Extraction + GT Evaluation)")
    print("=" * 90)
    
    for ds_name, title, root in datasets_to_test:
        print(f"\n[{ds_name.upper()}] - {title}")
        print("-" * 90)
        
        try:
            ds = get_dataset(ds_name, data_root=root, split="test", img_size=(384, 384))
            total = len(ds)
            print(f"  * Loaded {total} total unseen test samples.")
            
            # Pick diverse samples: find ones that have ground truth tampering
            tested_count = 0
            for idx in range(min(50, total)):
                if tested_count >= num_samples_per_ds:
                    break
                    
                img_tensor, mask_tensor, sample_name = ds[idx]
                gt_mask = mask_tensor.squeeze().cpu().numpy()
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                gt_pixels = int(np.sum(gt_bin))
                
                # We want to test samples that have tampered regions
                if gt_pixels < 25 and total > 5 and idx < 45:
                    continue  # skip clean ones until we tested tampered ones
                    
                rgb_img = denorm(img_tensor)
                bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                
                # Save temp image for verifier
                tmp_img_path = os.path.join(output_dir, f"temp_{ds_name}_{idx}.png")
                cv2.imwrite(tmp_img_path, bgr_img)
                
                # Run Verification (Detection + Localization + OCR)
                res = verifier.verify(tmp_img_path)
                
                pred_probs = res["raw_probs"]
                pred_bin = (pred_probs >= verifier.pixel_threshold).astype(np.uint8)
                pred_pixels = int(np.sum(pred_bin))
                
                # Calculate GT Overlap
                gt_is_tampered = gt_pixels >= 25
                pred_is_tampered = res["is_fake"]
                match = (gt_is_tampered == pred_is_tampered)
                
                inter = int(np.sum(gt_bin & pred_bin))
                union = int(np.sum(gt_bin | pred_bin))
                iou = (inter / float(union + 1e-7)) * 100.0
                prec = (inter / float(pred_pixels + 1e-7)) * 100.0
                rec = (inter / float(gt_pixels + 1e-7)) * 100.0
                dice = (2.0 * prec * rec) / (prec + rec + 1e-7)
                
                # Regions info
                regions = res["tampered_regions"]
                extracted_texts = [f"'{r['extracted_text']}'" for r in regions if r.get('extracted_text')]
                ocr_str = ", ".join(extracted_texts) if extracted_texts else "N/A"
                
                print(f"\n  Sample: {sample_name} (Index #{idx})")
                print(f"    - Ground Truth Status : {'TAMPERED' if gt_is_tampered else 'AUTHENTIC'} ({gt_pixels} tampered px)")
                print(f"    - Model Prediction    : {res['document_verdict']} (Conf: {res['confidence']*100:.1f}%, Score: {res['fraud_score']:.3f})")
                print(f"    - Verdict Match       : {'[CORRECT]' if match else '[MISMATCH]'}")
                print(f"    - Overlap Metrics     : IoU = {iou:.1f}% | Recall = {rec:.1f}% | Dice = {dice:.1f}%")
                print(f"    - Detected Regions    : {len(regions)} region(s) localized")
                for r_i, r in enumerate(regions[:3], 1):
                    ocr_info = f" | OCR: '{r['extracted_text']}'" if r.get('extracted_text') else ""
                    print(f"        * Region {r_i}: BBox={r['bbox']} | Zone={r.get('document_region', 'Unknown')} | Type={r['distortion_type']}{ocr_info}")
                    
                # Render 4-Panel Forensic Evidence Board
                p1 = bgr_img.copy()
                cv2.putText(p1, f"ORIGINAL: {sample_name[:22]}", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Ground truth mask overlay
                p2 = bgr_img.copy()
                gt_overlay = cv2.applyColorMap((gt_bin * 255).astype(np.uint8), cv2.COLORMAP_JET)
                p2 = cv2.addWeighted(p2, 0.6, gt_overlay, 0.4, 0)
                cv2.putText(p2, f"GT MASK ({gt_pixels} px)", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Model Prediction Heatmap + BBoxes
                p3 = bgr_img.copy()
                thermal = cv2.applyColorMap((pred_probs * 255).astype(np.uint8), cv2.COLORMAP_JET)
                p3 = cv2.addWeighted(p3, 0.55, thermal, 0.45, 0)
                for r in regions:
                    bx1, by1, bx2, by2 = r['bbox']
                    cv2.rectangle(p3, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                    lbl = r.get('extracted_text', '')[:12] or r['distortion_type']
                    cv2.putText(p3, lbl, (bx1, max(15, by1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(p3, f"PRED: {res['document_verdict']} ({res['confidence']*100:.0f}%)", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
                
                # Error Map (Hit=Green, FP=Red, Miss=Blue)
                err_map = np.zeros_like(bgr_img)
                err_map[(gt_bin == 1) & (pred_bin == 1)] = (0, 255, 0)   # Hit
                err_map[(gt_bin == 0) & (pred_bin == 1)] = (0, 0, 255)   # FP
                err_map[(gt_bin == 1) & (pred_bin == 0)] = (255, 180, 0) # Miss
                p4 = cv2.addWeighted(bgr_img, 0.5, err_map, 0.5, 0)
                cv2.putText(p4, f"OVERLAP (IoU: {iou:.1f}%)", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                
                board = np.hstack([p1, p2, p3, p4])
                board_path = os.path.join(output_dir, "boards", f"{ds_name}_{idx}_board.jpg")
                cv2.imwrite(board_path, board)
                
                # Cleanup temp image
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
                    
                overall_summary.append({
                    "dataset": ds_name,
                    "sample": sample_name,
                    "gt_status": "TAMPERED" if gt_is_tampered else "AUTHENTIC",
                    "pred_verdict": res["document_verdict"],
                    "fraud_score": res["fraud_score"],
                    "confidence": res["confidence"],
                    "verdict_match": match,
                    "iou": round(iou, 1),
                    "recall": round(rec, 1),
                    "dice": round(dice, 1),
                    "detected_regions": len(regions),
                    "extracted_text": ocr_str,
                    "board_path": board_path
                })
                
                tested_count += 1
                
        except Exception as e:
            print(f"  [Error] Testing {ds_name} failed: {e}")
            import traceback
            traceback.print_exc()

    # Save summary JSON
    sum_path = os.path.join(output_dir, "unseen_benchmark_results.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)
        
    print("\n" + "=" * 90)
    print("                     FINAL UNSEEN BENCHMARK AUDIT SCORECARD")
    print("=" * 90)
    print(f"{'Dataset':<15} | {'Sample':<25} | {'GT':<9} | {'Pred':<12} | {'IoU':<6} | {'Recall':<6} | {'OCR Text':<15}")
    print("-" * 90)
    for s in overall_summary:
        print(f"{s['dataset']:<15} | {s['sample'][:24]:<25} | {s['gt_status']:<9} | {s['pred_verdict']:<12} | {s['iou']:>4.1f}% | {s['recall']:>4.1f}% | {s['extracted_text'][:15]}")
    print("=" * 90)
    print(f"[+] All visual forensic boards saved to : {os.path.join(output_dir, 'boards')}")
    print(f"[+] Full JSON results saved to           : {sum_path}\n")

if __name__ == "__main__":
    run_benchmark_on_unseen()
