import os
import glob
import json
import csv
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.forensics.document_classifier import ZeroTrainingDocumentClassifier
from src.forensics.document_verifier import DocumentVerifier




def main():
    mix_dir = r"E:\project final\panadhargst\mix"
    output_json = os.path.join(mix_dir, "classification_and_forgery_report.json")
    output_csv = os.path.join(mix_dir, "classification_and_forgery_report.csv")

    if not os.path.exists(mix_dir):
        print(f"Error: Directory '{mix_dir}' does not exist.")
        return

    print("==========================================================================")
    print(f"  Governed Document Classification & Forgery Inspection Pipeline")
    print(f"  Target Folder : {mix_dir}")
    print("==========================================================================")

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    image_paths = []
    for ext in valid_exts:
        image_paths.extend(glob.glob(os.path.join(mix_dir, ext)))

    # Filter out any generated report images if present
    image_paths = [p for p in image_paths if not os.path.basename(p).startswith("report_")]
    image_paths.sort()
    total_images = len(image_paths)

    print(f" Found [{total_images}] image files to process in folder.\n")

    print("[1/2] Initializing Zero-Training Document Classifier...")
    classifier = ZeroTrainingDocumentClassifier()

    print("[2/2] Loading Production ForgeryNet Model from bigpower...")
    try:
        verifier = DocumentVerifier(config_path="config.yaml")
        has_verifier = True
        print(f"      Successfully loaded model: {os.path.basename(verifier.model_path)}")
    except Exception as e:
        print(f"      Warning: Could not load ForgeryNet model ({e}). Proceeding with classification only.")
        has_verifier = False

    results = []
    summary_stats = {
        "total_images": total_images,
        "classified_passed": 0,
        "classified_unknown": 0,
        "authentic_count": 0,
        "tampered_count": 0,
        "doc_breakdown": {}
    }

    start_time = time.time()

    print("\n--- PROCESSING DOSSIER IMAGES ---")
    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)

        # STAGE 1: Document Classification
        try:
            class_res = classifier.classify(img_path)
            doc_type = class_res["document_type"]
            class_conf = class_res["confidence"]
            class_method = class_res["classification_method"]
            metadata = class_res.get("metadata", {})
        except Exception as e:
            doc_type = "UNKNOWN_DOCUMENT"
            class_conf = 0.0
            class_method = f"ERROR: {str(e)}"
            metadata = {}

        if doc_type not in summary_stats["doc_breakdown"]:
            summary_stats["doc_breakdown"][doc_type] = 0
        summary_stats["doc_breakdown"][doc_type] += 1

        stage1_passed = (doc_type != "UNKNOWN_DOCUMENT" and class_conf >= 0.30)
        if stage1_passed:
            summary_stats["classified_passed"] += 1
        else:
            summary_stats["classified_unknown"] += 1

        # STAGE 2: Deep Learning Forgery Inspection (If Stage 1 Passed)
        forgery_info = {
            "forgery_stage_executed": False,
            "verdict": "SKIPPED_UNKNOWN_DOCUMENT" if not stage1_passed else "NOT_RUN",
            "is_fake": False,
            "fraud_score": 0.0,
            "risk_level": "N/A",
            "tampered_regions_count": 0
        }

        if stage1_passed and has_verifier:
            try:
                forgery_res = verifier.verify(img_path)
                forgery_info = {
                    "forgery_stage_executed": True,
                    "verdict": forgery_res["document_verdict"],
                    "is_fake": forgery_res["is_fake"],
                    "fraud_score": forgery_res["fraud_score"],
                    "risk_level": forgery_res["risk_level"],
                    "tampered_regions_count": forgery_res["total_tampered_regions"]
                }
                if forgery_res["is_fake"]:
                    summary_stats["tampered_count"] += 1
                else:
                    summary_stats["authentic_count"] += 1
            except Exception as e:
                forgery_info["verdict"] = f"FORGERY_ERROR: {str(e)}"

        record = {
            "index": idx,
            "filename": filename,
            "stage1_classification": {
                "passed": stage1_passed,
                "document_type": doc_type,
                "confidence_score_pct": round(class_conf * 100.0, 1),
                "method": class_method,
                "metadata": metadata
            },
            "stage2_forgery_inspection": forgery_info
        }
        results.append(record)

        # Log to Console
        c_pct = f"{class_conf*100.0:5.1f}%"
        verdict_str = forgery_info["verdict"]
        print(f" [{idx:3d}/{total_images}] {filename:15s} | Class: {doc_type:20s} ({c_pct}) | Forgery Check: {verdict_str}")

    elapsed = time.time() - start_time

    # Output Summary
    print("\n==========================================================================")
    print(f"  Processing Complete in {elapsed:.2f} seconds ({elapsed/total_images:.3f} s/image)")
    print("==========================================================================")
    print(f"  Classified Successfully : {summary_stats['classified_passed']} / {total_images}")
    print(f"  Unknown Documents       : {summary_stats['classified_unknown']} / {total_images}")
    if has_verifier:
        print(f"  Authentic Documents     : {summary_stats['authentic_count']}")
        print(f"  Tampered / Fake Docs    : {summary_stats['tampered_count']}")

    print("\n  DOCUMENT TYPE BREAKDOWN:")
    for dt, cnt in summary_stats["doc_breakdown"].items():
        print(f"   - {dt:25s} : {cnt:4d} files")

    # Save JSON Report directly inside E:\project final\panadhargst\mix
    final_output_payload = {
        "folder_path": mix_dir,
        "total_images_processed": total_images,
        "summary": summary_stats,
        "results": results
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(final_output_payload, f, indent=2)

    # Save CSV Report directly inside E:\project final\panadhargst\mix
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Filename", "Stage1 Passed", "Document Type", "Class Confidence (%)", "Stage2 Verdict", "Is Fake", "Fraud Score", "Risk Level", "Tampered Regions"])
        for r in results:
            s1 = r["stage1_classification"]
            s2 = r["stage2_forgery_inspection"]
            writer.writerow([
                r["index"],
                r["filename"],
                s1["passed"],
                s1["document_type"],
                s1["confidence_score_pct"],
                s2["verdict"],
                s2["is_fake"],
                s2["fraud_score"],
                s2["risk_level"],
                s2["tampered_regions_count"]
            ])

    print(f"\n Saved Full JSON Report to: {output_json}")
    print(f" Saved Full CSV Report to : {output_csv}")


if __name__ == "__main__":
    main()
