import os, glob, json
from src.forensics.document_classifier import ZeroTrainingDocumentClassifier
from src.forensics.document_verifier import DocumentVerifier
from src.forensics.visualizer import ForensicVisualizer

def analyze_whatsapp_images():
    base_dir = r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina"
    whatsapp_files = sorted(glob.glob(os.path.join(base_dir, "WhatsApp Image*.jpeg")))

    print("=" * 80)
    print(f"ANALYZING {len(whatsapp_files)} WHATSAPP IMAGES WITH NEW ENHANCED MODEL")
    print("=" * 80)

    classifier = ZeroTrainingDocumentClassifier()
    verifier = DocumentVerifier()
    output_reports_dir = os.path.join(base_dir, "reports", "forensics_whatsapp")
    os.makedirs(output_reports_dir, exist_ok=True)

    analysis_results = []

    for idx, img_path in enumerate(whatsapp_files, start=1):
        filename = os.path.basename(img_path)
        print(f"\n[{idx}/{len(whatsapp_files)}] Processing: {filename}")
        
        # 1. Classification
        class_res = classifier.classify(img_path)
        doc_type = class_res["document_type"]
        confidence_class = class_res["confidence"]
        
        # 2. Verification
        verif_res = verifier.verify(img_path)
        
        verdict = verif_res["document_verdict"]
        fraud_score = verif_res["fraud_score"]
        risk = verif_res["risk_level"]
        num_regions = verif_res["total_tampered_regions"]
        
        print(f"  - Classified As  : {doc_type} (conf: {confidence_class:.2f})")
        print(f"  - Verdict        : {verdict}")
        print(f"  - Fraud Score    : {fraud_score:.4f} ({risk} RISK)")
        print(f"  - Regions Flagged: {num_regions}")
        
        # List regions & what was done
        region_details = []
        for r in verif_res["tampered_regions"]:
            r_info = {
                "id": r["region_id"],
                "box": r["bbox"],
                "zone": r["document_region"],
                "distortion": r["distortion_type"],
                "confidence": r["peak_confidence"],
                "area_percentage": r["area_percentage"],
                "ocr_text": r["extracted_text"]
            }
            region_details.append(r_info)
            print(f"    * Region #{r['region_id']}: [{r['distortion_type']}] in [{r['document_region']}] | Area: {r['area_percentage']}% | Conf: {r['peak_confidence']:.3f} | OCR: '{r['extracted_text']}'")

        # 3. Save Visual Artifacts
        try:
            saved = ForensicVisualizer.save_all_artifacts(verif_res, output_dir=output_reports_dir)
            audit_panel = saved.get("audit_panel")
        except Exception as e:
            audit_panel = str(e)

        analysis_results.append({
            "index": idx,
            "filename": filename,
            "document_type": doc_type,
            "classification_confidence": confidence_class,
            "verdict": verdict,
            "fraud_score": fraud_score,
            "risk_level": risk,
            "num_regions": num_regions,
            "top_regions": region_details,
            "audit_panel": audit_panel
        })

    out_json = os.path.join(output_reports_dir, "whatsapp_analysis_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Completed analysis! Summary saved to: {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_whatsapp_images()
