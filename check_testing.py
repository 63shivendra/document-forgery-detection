import cv2, os, glob
from src.forensics.document_verifier import DocumentVerifier

def main():
    verifier = DocumentVerifier()
    print("Loaded Model:", os.path.basename(verifier.model_path))
    print(f"Pixel threshold: {verifier.pixel_threshold}, min_region_pixels: {verifier.min_region_pixels}")

    pan_files = glob.glob(r"d:\shivendra pratap singh\onwardsmrechant\project final\Real_dataset_pan_adhaar\pan_card\**\*.jpg", recursive=True)
    aadhaar_files = glob.glob(r"d:\shivendra pratap singh\onwardsmrechant\project final\Real_dataset_pan_adhaar\adhaar_card\**\*.jpg", recursive=True)
    cheque_files = glob.glob(r"d:\shivendra pratap singh\onwardsmrechant\project final\Real_dataset_pan_adhaar\bank_cheque\**\*.jpg", recursive=True)
    gst_files = glob.glob(r"d:\shivendra pratap singh\onwardsmrechant\project final\Real_dataset_pan_adhaar\gst_certificate\**\*.jpg", recursive=True)

    print(f"\n--- Testing Genuine Documents from Real_dataset_pan_adhaar ---")
    for name, flist in [("PAN", pan_files), ("Aadhaar", aadhaar_files), ("Cheque", cheque_files), ("GST", gst_files)]:
        print(f"\nEvaluating {name} ({len(flist)} available):")
        fake_count = 0
        auth_count = 0
        scores = []
        for p in flist[:10]:
            res = verifier.verify(p)
            scores.append(res["fraud_score"])
            if res["is_fake"]:
                fake_count += 1
            else:
                auth_count += 1
            print(f"  [{name}] {os.path.basename(p)[:30]:30s} | Verdict: {res['document_verdict']:15s} | Fraud Score: {res['fraud_score']:.4f} | Regions: {res['total_tampered_regions']}")
        print(f"  --> {name} Sample Summary: {auth_count} Authentic, {fake_count} False Alarms | Mean Score: {sum(scores)/len(scores):.4f}")

if __name__ == "__main__":
    main()
