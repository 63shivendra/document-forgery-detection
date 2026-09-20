import os
import json
import glob
import cv2
import numpy as np
from src.forensics.document_classifier import ZeroTrainingDocumentClassifier
from src.forensics.document_verifier import DocumentVerifier


def evaluate_live_testing_dataset(testing_dir=None, model_path=None, output_report=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if testing_dir is None:
        testing_dir = os.path.join(script_dir, "testing_live")
    if output_report is None:
        output_report = os.path.join(testing_dir, "live_evaluation_report.json")
    config_file = os.path.join(script_dir, "config.yaml")

    manifest_path = os.path.join(testing_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest file '{manifest_path}' not found. Run generate_live_tampered_dataset.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data["dataset_manifest"]
    classifier = ZeroTrainingDocumentClassifier()
    verifier = DocumentVerifier(model_path=model_path, config_path=config_file)

    print("==========================================================================")
    print("  Evaluating ForgeryNet Model on Live Testing Benchmark")
    print(f"  Model Loaded    : {os.path.basename(verifier.model_path)}")
    print(f"  Model Path      : {verifier.model_path}")
    print(f"  Dataset Folder  : {testing_dir} ({len(samples)} samples)")
    print("==========================================================================")

    results = []
    tp = fp = fn = tn = 0

    for idx, sample in enumerate(samples, start=1):
        img_path = os.path.join(testing_dir, "images", sample["image_filename"])
        mask_path = os.path.join(testing_dir, "masks", sample["mask_filename"])

        if not os.path.exists(img_path):
            continue

        # 1. Classification Check
        class_res = classifier.classify(img_path)
        doc_type = class_res["document_type"]

        # 2. Forgery Inspection Check
        forgery_res = verifier.verify(img_path)

        gt_tampered = sample["is_tampered"]
        pred_tampered = forgery_res["is_fake"]

        if gt_tampered and pred_tampered:
            tp += 1
            status = "TRUE_POSITIVE (Detected Forgery)"
        elif not gt_tampered and pred_tampered:
            fp += 1
            status = "FALSE_POSITIVE (False Alarm)"
        elif gt_tampered and not pred_tampered:
            fn += 1
            status = "FALSE_NEGATIVE (Missed Forgery)"
        else:
            tn += 1
            status = "TRUE_NEGATIVE (Correct Authentic)"

        print(f" [{idx:2d}/{len(samples)}] {sample['image_filename']:45s} | Class: {doc_type:18s} | GT: {'FAKE' if gt_tampered else 'AUTH':4s} | Pred: {forgery_res['document_verdict']:15s} | Status: {status}")

        results.append({
            "sample_id": sample["sample_id"],
            "filename": sample["image_filename"],
            "forgery_type": sample["forgery_type"],
            "ground_truth_tampered": gt_tampered,
            "predicted_verdict": forgery_res["document_verdict"],
            "predicted_fraud_score": forgery_res["fraud_score"],
            "risk_level": forgery_res["risk_level"],
            "status": status
        })

    total = tp + fp + fn + tn
    acc = (tp + tn) / float(total) if total > 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n==========================================================================")
    print("  LIVE EVALUATION BENCHMARK METRICS SUMMARY")
    print("==========================================================================")
    print(f"  Accuracy           : {acc*100.0:.2f}% ({tp+tn}/{total})")
    print(f"  Precision          : {precision*100.0:.2f}%")
    print(f"  Recall (Sensitivity): {recall*100.0:.2f}%")
    print(f"  F1-Score           : {f1*100.0:.2f}%")
    print("--------------------------------------------------------------------------")
    print(f"  True Positives (TP)  : {tp:2d} | False Positives (FP) : {fp:2d}")
    print(f"  True Negatives (TN)  : {tn:2d} | False Negatives (FN) : {fn:2d}")

    # Category breakdown
    breakdown = {}
    for r in results:
        ft = r["forgery_type"]
        if ft not in breakdown:
            breakdown[ft] = {"total": 0, "correct": 0}
        breakdown[ft]["total"] += 1
        if "TRUE" in r["status"]:
            breakdown[ft]["correct"] += 1

    print("--------------------------------------------------------------------------")
    print("  PER-CATEGORY ACCURACY BREAKDOWN:")
    for ft, st in sorted(breakdown.items()):
        cat_acc = (st["correct"] / st["total"]) * 100.0 if st["total"] > 0 else 0.0
        print(f"    - {ft:20s}: {st['correct']:2d}/{st['total']:2d} ({cat_acc:5.1f}%)")

    summary_payload = {
        "dataset_dir": testing_dir,
        "model_used": verifier.model_path,
        "total_samples": total,
        "metrics": {
            "accuracy_pct": round(acc * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "recall_pct": round(recall * 100.0, 2),
            "f1_score_pct": round(f1 * 100.0, 2),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        },
        "breakdown": breakdown,
        "sample_details": results
    }

    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print(f"\n Saved Evaluation Report to: {output_report}")
    return summary_payload


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model on live testing dataset")
    parser.add_argument("--testing_dir", type=str, default=None, help="Path to testing_live directory")
    parser.add_argument("--model", type=str, default=None, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default=None, help="Output JSON report path")
    args = parser.parse_args()

    evaluate_live_testing_dataset(
        testing_dir=args.testing_dir,
        model_path=args.model,
        output_report=args.output
    )
