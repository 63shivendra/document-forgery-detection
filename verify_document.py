import os
import sys
import argparse
import glob
from src.forensics.document_verifier import DocumentVerifier
from src.forensics.visualizer import ForensicVisualizer

# Ensure UTF-8 output on Windows terminal
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def print_single_result(result, saved_files):
    print("\n" + "=" * 76)
    print(f"  DOCUMENT FORENSIC AUDIT: {result['document_name']}")
    print("=" * 76)
    print(f"  * Resolution           : {result['document_dimensions'][0]} x {result['document_dimensions'][1]} px")
    print(f"  * Document Verdict     : {result['document_verdict']}")
    print(f"  * Fraud Risk Score     : {result['fraud_score'] * 100:.2f}%  [{result['risk_level']} RISK]")
    print(f"  * Confidence           : {result['confidence'] * 100:.2f}%")
    print(f"  * Tampered Regions     : {result['total_tampered_regions']}")
    print(f"  * Tampered Area %      : {result['tampered_area_percentage']:.3f}% of total document")
    print("-" * 76)

    if result["is_fake"] and result["tampered_regions"]:
        print("  LOCALIZED FORGED / DISTORTED PARTS:")
        print(f"  {'ID':<4} | {'Bounding Box (x1,y1,x2,y2)':<28} | {'Document Field / Zone':<26} | {'Distortion Type':<24} | {'Conf'}")
        print("  " + "-" * 95)
        for reg in result["tampered_regions"]:
            bbox_str = f"[{reg['bbox'][0]}, {reg['bbox'][1]}, {reg['bbox'][2]}, {reg['bbox'][3]}]"
            print(f"  {reg['region_id']:<4} | {bbox_str:<28} | {reg['document_region'][:26]:<26} | {reg['distortion_type'][:24]:<24} | {reg['peak_confidence']*100:.1f}%")
            if reg.get("extracted_text"):
                print(f"       └──> [OCR Extracted Text]: \"{reg['extracted_text']}\"")
        print("-" * 76)
    else:
        print("  No tampered, spliced, or distorted regions detected.")
        print("  Document passed all forensic consistency checks.")
        print("-" * 76)

    print("  EXPORTED EVIDENCE ARTIFACTS:")
    if saved_files.get("annotated_image"):
        print(f"  [+] Annotated Image  : {saved_files['annotated_image']}")
    if saved_files.get("audit_panel"):
        print(f"  [+] 3-Panel Audit    : {saved_files['audit_panel']}")
    if saved_files.get("crop_images"):
        print(f"  [+] Part Crops ({len(saved_files['crop_images'])}) : {os.path.dirname(saved_files['crop_images'][0])}")
    if saved_files.get("json_report"):
        print(f"  [+] JSON Report      : {saved_files['json_report']}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Document-Level Verification & Forensic Localization")
    parser.add_argument("--input", "-i", type=str, default=None, help="Path to single document image to verify")
    parser.add_argument("--input-dir", "-d", type=str, default=None, help="Directory of document images to batch verify")
    parser.add_argument("--output-dir", "-o", type=str, default="reports/forensics", help="Directory where annotated artifacts will be saved")
    parser.add_argument("--model-path", "-m", type=str, default=None, help="Path to .pth checkpoint (default: auto-detect best)")
    parser.add_argument("--threshold", "-t", type=float, default=0.35, help="Pixel tampering probability threshold (default: 0.35)")
    parser.add_argument("--min-pixels", "-p", type=int, default=25, help="Minimum connected pixels for a valid tampered part (default: 25)")
    parser.add_argument("--max-samples", "-n", type=int, default=None, help="Maximum number of images to verify in batch mode (default: all)")
    args = parser.parse_args()

    # Collect images
    image_paths = []
    if args.input:
        if not os.path.exists(args.input):
            print(f"[Error] File not found: {args.input}")
            sys.exit(1)
        image_paths.append(args.input)
    elif args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"[Error] Directory not found: {args.input_dir}")
            sys.exit(1)
        exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp")
        collected = []
        for ext in exts:
            collected.extend(glob.glob(os.path.join(args.input_dir, "**", ext), recursive=True))
            collected.extend(glob.glob(os.path.join(args.input_dir, "**", ext.upper()), recursive=True))
            collected.extend(glob.glob(os.path.join(args.input_dir, ext)))
            collected.extend(glob.glob(os.path.join(args.input_dir, ext.upper())))

        # Deduplicate
        collected = sorted(list(set(collected)))

        # Automatically filter out ground-truth mask files
        for p in collected:
            norm_p = p.replace("\\", "/").lower()
            base_p = os.path.basename(p).lower()
            if "/masks/" in norm_p or "/ground truth/" in norm_p or "/mask/" in norm_p:
                continue
            if base_p.endswith(("_gt.png", "_mask.png", "_gt.jpg", "_mask.jpg", "_gt.tif", "_mask.tif")):
                continue
            image_paths.append(p)

        if args.max_samples and len(image_paths) > args.max_samples:
            print(f"[Info] Limiting batch verification to {args.max_samples} of {len(image_paths)} found images.")
            image_paths = image_paths[:args.max_samples]
    else:
        # Default: verify WhatsApp images in the current folder if present
        whatsapp_imgs = sorted(glob.glob("WhatsApp Image*.jpeg"))
        if whatsapp_imgs:
            print(f"[Info] No input specified. Found {len(whatsapp_imgs)} WhatsApp test images in workspace. Processing them:")
            image_paths.extend(whatsapp_imgs[:4])
        else:
            parser.print_help()
            sys.exit(1)

    if not image_paths:
        print("[Error] No valid document images found to verify.")
        sys.exit(1)

    print(f"\nInitializing Document-Level Verification Engine (Target Images: {len(image_paths)})...")
    verifier = DocumentVerifier(
        model_path=args.model_path,
        pixel_threshold=args.threshold,
        min_region_pixels=args.min_pixels
    )

    fake_count = 0
    authentic_count = 0

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"\nProcessing [{idx}/{len(image_paths)}]: {os.path.basename(img_path)} ...")
        result = verifier.verify(img_path)
        saved_files = ForensicVisualizer.save_all_artifacts(result, output_dir=args.output_dir)
        print_single_result(result, saved_files)

        if result["is_fake"]:
            fake_count += 1
        else:
            authentic_count += 1

    if len(image_paths) > 1:
        print("\n" + "#" * 76)
        print("  BATCH VERIFICATION SUMMARY:")
        print(f"  * Total Documents Analyzed : {len(image_paths)}")
        print(f"  * Tampered / Fake Detected : {fake_count} ({(fake_count / len(image_paths)) * 100:.1f}%)")
        print(f"  * Authentic Verified       : {authentic_count} ({(authentic_count / len(image_paths)) * 100:.1f}%)")
        print(f"  * All Visual Reports Saved : {os.path.abspath(args.output_dir)}")
        print("#" * 76 + "\n")


if __name__ == "__main__":
    main()
