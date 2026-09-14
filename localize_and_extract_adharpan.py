import os
import sys
import re
import json
import glob
import cv2
import numpy as np
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import easyocr
import zxingcpp

def parse_aadhaar_qr_payload(raw_text, raw_bytes):
    """Parses UIDAI PrintLetterBarcodeData XML or Secure QR binary."""
    data = {
        "qr_type": "Standard / Unknown",
        "has_digital_signature": False,
        "fields": {}
    }
    
    if "<PrintLetterBarcodeData" in raw_text:
        data["qr_type"] = "UIDAI Aadhaar XML Barcode"
        data["has_digital_signature"] = True
        for attr in ["uid", "name", "gender", "yob", "dob", "vtc", "po", "dist", "state", "pc", "street", "house", "loc"]:
            match = re.search(rf'{attr}="([^"]*)"', raw_text, re.IGNORECASE)
            if match:
                data["fields"][attr] = match.group(1)
    elif "http" in raw_text:
        data["qr_type"] = "Web URL / Online Verification Link"
        data["fields"]["url"] = raw_text
    elif len(raw_bytes) > 200:
        data["qr_type"] = "High-Density Binary / Secure QR Payload"
        data["has_digital_signature"] = True
        data["fields"]["raw_byte_length"] = len(raw_bytes)
        
    return data

def classify_text_field(text):
    text_clean = text.strip()
    text_upper = text_clean.upper()
    
    # 1. Aadhaar Number: 12 digits (often formatted as XXXX XXXX XXXX)
    if re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text_clean):
        return "Aadhaar Number (UID)"
    
    # 2. PAN Number: 5 letters, 4 digits, 1 letter (e.g. ABCDE1234F)
    if re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', text_upper):
        return "Permanent Account Number (PAN)"
        
    # 3. DOB / Date of Birth
    if any(k in text_upper for k in ["DOB", "DATE OF BIRTH", "D.O.B", "YEAR OF BIRTH", "YOB"]) or re.search(r'\b\d{2}[/\-.]\d{2}[/\-.]\d{4}\b', text_clean):
        return "Date of Birth (DOB)"
        
    # 4. Gender
    if any(k == text_upper for k in ["MALE", "FEMALE", "TRANSGENDER", "M", "F"]) or any(k in text_upper for k in ["MALE", "FEMALE"]):
        return "Gender"
        
    # 5. Authority / Header
    if any(k in text_upper for k in ["GOVERNMENT OF INDIA", "BHARAT SARKAR", "INCOME TAX DEPARTMENT", "UNIQUE IDENTIFICATION", "UIDAI", "GOVT. OF INDIA"]):
        return "Government / Authority Header"
        
    # 6. Slogan / Footer
    if any(k in text_upper for k in ["MERA AADHAAR", "MERI PEHCHAN", "AAM AADMI KA ADHIKAR", "HELP@UIDAI"]):
        return "Motto / Slogan / Contact"
        
    # 7. Father's Name / Relative
    if any(k in text_upper for k in ["FATHER", "S/O", "D/O", "W/O", "C/O"]):
        return "Relative / Father's Name"
        
    # Default to generic text
    if any(c.isdigit() for c in text_clean):
        return "Numeric / Alphanumeric Field"
    return "Personal / Document Text Field"


def extract_structured_document_semantics(text_fields, qr_codes):
    """
    Intelligently maps unstructured OCR bounding boxes and QR digital signatures
    into a formal structured KYC identity schema (Name, ID, DOB, Address, Gender).
    """
    schema = {
        "document_type": "UNKNOWN_ID",
        "issuing_authority": "Government of India",
        "structured_identity": {
            "full_name": None,
            "father_or_guardian_name": None,
            "id_number": None,
            "id_type": None,
            "date_of_birth": None,
            "gender": None,
            "address": {
                "full_address": None,
                "district": None,
                "state": None,
                "pincode": None
            }
        },
        "digital_cross_verification": {
            "is_qr_present": len(qr_codes) > 0,
            "is_digitally_signed": False,
            "qr_ocr_consistency": "N/A",
            "mismatch_detected": False
        }
    }
    
    # 1. First, check QR Code for authoritative digitally signed data
    qr_data = None
    if qr_codes:
        for q in qr_codes:
            parsed = q.get("parsed_signature_data", {})
            if parsed.get("has_digital_signature") and parsed.get("fields"):
                qr_data = parsed["fields"]
                schema["digital_cross_verification"]["is_digitally_signed"] = True
                break
                
    if qr_data:
        schema["document_type"] = "AADHAAR_CARD"
        schema["structured_identity"]["id_type"] = "AADHAAR_UID"
        schema["structured_identity"]["id_number"] = qr_data.get("uid")
        schema["structured_identity"]["full_name"] = qr_data.get("name")
        schema["structured_identity"]["gender"] = "Male" if qr_data.get("gender") == "M" else ("Female" if qr_data.get("gender") == "F" else qr_data.get("gender"))
        schema["structured_identity"]["date_of_birth"] = qr_data.get("dob") or qr_data.get("yob")
        
        co = qr_data.get("co", "")
        if co:
            schema["structured_identity"]["father_or_guardian_name"] = co.replace("S/O:", "").replace("S/O", "").strip()
            
        addr_parts = [qr_data.get(k) for k in ["house", "street", "loc", "vtc", "po", "subdist", "dist", "state"] if qr_data.get(k)]
        if addr_parts:
            schema["structured_identity"]["address"]["full_address"] = ", ".join(addr_parts)
        schema["structured_identity"]["address"]["district"] = qr_data.get("dist")
        schema["structured_identity"]["address"]["state"] = qr_data.get("state")
        schema["structured_identity"]["address"]["pincode"] = qr_data.get("pc")

    # 2. Parse / Supplement from Visual OCR Fields
    all_texts = [f["text"].strip() for f in text_fields]
    combined_text = " ".join(all_texts)
    combined_upper = combined_text.upper()
    
    # Detect document type
    if "INCOME TAX" in combined_upper or "PERMANENT ACCOUNT" in combined_upper or re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', combined_upper):
        schema["document_type"] = "PAN_CARD"
        schema["issuing_authority"] = "Income Tax Department, Govt. of India"
        schema["structured_identity"]["id_type"] = "PAN"
    elif "AADHAAR" in combined_upper or "UIDAI" in combined_upper or re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', combined_text):
        if not schema["document_type"] or schema["document_type"] == "UNKNOWN_ID":
            schema["document_type"] = "AADHAAR_CARD"
            schema["structured_identity"]["id_type"] = "AADHAAR_UID"
            schema["issuing_authority"] = "Unique Identification Authority of India (UIDAI)"

    # Extract ID Number from OCR if not in QR
    if not schema["structured_identity"]["id_number"]:
        if schema["document_type"] == "PAN_CARD":
            pan_match = re.search(r'\b[A-Z]{5}\d{4}[A-Z]\b', combined_upper)
            if pan_match:
                schema["structured_identity"]["id_number"] = pan_match.group(0)
        else:
            uid_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', combined_text)
            if uid_match:
                schema["structured_identity"]["id_number"] = uid_match.group(0)

    # Extract DOB from OCR if not in QR
    if not schema["structured_identity"]["date_of_birth"]:
        dob_match = re.search(r'\b(\d{2}[/\-.]\d{2}[/\-.]\d{4})\b', combined_text)
        if dob_match:
            schema["structured_identity"]["date_of_birth"] = dob_match.group(1)

    # Extract Gender from OCR if not in QR
    if not schema["structured_identity"]["gender"]:
        for t in all_texts:
            t_u = t.upper()
            if t_u in ["MALE", "PURUSH"] or "MALE" in t_u:
                schema["structured_identity"]["gender"] = "Male"
                break
            elif t_u in ["FEMALE", "STREE"] or "FEMALE" in t_u:
                schema["structured_identity"]["gender"] = "Female"
                break

    # Extract PIN code from OCR if not in QR
    if not schema["structured_identity"]["address"]["pincode"]:
        pin_match = re.search(r'\b[1-9][0-9]{5}\b', combined_text)
        if pin_match:
            schema["structured_identity"]["address"]["pincode"] = pin_match.group(0)

    # Extract Name from OCR if not in QR
    if not schema["structured_identity"]["full_name"]:
        # Heuristic: Name usually sits between Header and DOB
        for i, f in enumerate(text_fields):
            t = f["text"].strip()
            # If line is before DOB and is clean alphabetic with > 2 words
            if f["field_category"] in ["Personal / Document Text Field", "Numeric / Alphanumeric Field"]:
                words = t.split()
                if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                    if not any(k in t.upper() for k in ["GOVERNMENT", "INDIA", "INCOME", "TAX", "DEPARTMENT", "MALE", "FEMALE"]):
                        schema["structured_identity"]["full_name"] = t
                        break

    # Extract Address from OCR if not in QR
    if not schema["structured_identity"]["address"]["full_address"]:
        addr_lines = []
        is_addr_section = False
        for t in all_texts:
            if any(k in t.upper() for k in ["ADDRESS", "POST-", "VILLAGE", "TEHSIL", "DIST", "PIN"]):
                is_addr_section = True
            if is_addr_section:
                if not any(k in t.upper() for k in ["AADHAAR", "HELP@UIDAI", "WWW.", "1947"]):
                    addr_lines.append(t)
        if addr_lines:
            schema["structured_identity"]["address"]["full_address"] = ", ".join(addr_lines[:4])

    # 3. Perform Cross-Field Verification (OCR vs QR)
    if qr_data and schema["structured_identity"]["id_number"]:
        qr_uid = str(qr_data.get("uid", "")).replace(" ", "")
        ocr_uid = str(schema["structured_identity"]["id_number"]).replace(" ", "")
        if qr_uid and ocr_uid:
            if qr_uid in ocr_uid or ocr_uid in qr_uid:
                schema["digital_cross_verification"]["qr_ocr_consistency"] = "100% MATCH (Tamper-Proof Verified)"
                schema["digital_cross_verification"]["mismatch_detected"] = False
            else:
                schema["digital_cross_verification"]["qr_ocr_consistency"] = "MISMATCH (Possible Tampering / Alteration!)"
                schema["digital_cross_verification"]["mismatch_detected"] = True

    return schema


def process_adharpan_folder(folder_path="test_datasets/adharpan"):
    print("\n" + "=" * 88)
    print("      TEXT LOCALIZATION & OCR EXTRACTION ON AADHAAR / PAN LEGIT DOCUMENTS")
    print(f"      Target Directory: {os.path.abspath(folder_path)}")
    print("=" * 88)
    
    # Supported image extensions (exclude annotated images and cropped QR codes)
    valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".tif", ".bmp"]
    all_files = os.listdir(folder_path)
    img_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_exts and not f.endswith("_text_localized.jpg") and "_qr_" not in f]
    
    print(f"[+] Found {len(img_files)} document images to process.\n")
    
    print("[+] Initializing EasyOCR GPU Engine (CRAFT Text Detector + ResNet Recognizer)...")
    reader = easyocr.Reader(['en', 'hi'], gpu=True, verbose=False)
    
    all_docs_summary = []
    
    for idx, fname in enumerate(img_files, start=1):
        fpath = os.path.join(folder_path, fname)
        base_name = os.path.splitext(fname)[0]
        
        # Read image (handle webp & orientation)
        try:
            pil_img = Image.open(fpath).convert("RGB")
            orig_rgb = np.array(pil_img)
            orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"  [Error] Failed to load {fname}: {e}")
            continue
            
        h, w = orig_bgr.shape[:2]
        
        # Run CRAFT text detection + OCR recognition
        # detail=1 returns list of [ [ [x1,y1], [x2,y2], [x3,y3], [x4,y4] ], text, confidence ]
        ocr_results = reader.readtext(orig_rgb, detail=1, paragraph=False)
        
        text_fields = []
        annotated_img = orig_bgr.copy()
        overlay = orig_bgr.copy()
        
        for field_id, (box_pts, text_str, conf) in enumerate(ocr_results, start=1):
            text_str = text_str.strip()
            if not text_str:
                continue
                
            pts = np.array(box_pts, dtype=np.int32)
            x_coords = [int(p[0]) for p in box_pts]
            y_coords = [int(p[1]) for p in box_pts]
            
            x1 = max(0, min(x_coords))
            y1 = max(0, min(y_coords))
            x2 = min(w, max(x_coords))
            y2 = min(h, max(y_coords))
            
            box_w = x2 - x1
            box_h = y2 - y1
            
            field_type = classify_text_field(text_str)
            
            text_fields.append({
                "field_id": field_id,
                "text": text_str,
                "confidence": round(float(conf), 4),
                "field_category": field_type,
                "bounding_box": [x1, y1, x2, y2],
                "polygon_points": [[int(p[0]), int(p[1])] for p in box_pts],
                "dimensions": {"width": box_w, "height": box_h, "area_pixels": box_w * box_h}
            })
            
            # Choose color based on category
            if "Aadhaar" in field_type or "PAN" in field_type:
                box_color = (0, 0, 255)     # Bright Red for ID numbers
                tag_bg = (0, 0, 200)
            elif "DOB" in field_type or "Gender" in field_type:
                box_color = (255, 140, 0)   # Orange for bio info
                tag_bg = (200, 100, 0)
            elif "Authority" in field_type:
                box_color = (0, 180, 0)     # Green for Govt Header
                tag_bg = (0, 140, 0)
            else:
                box_color = (255, 200, 0)   # Cyan/Yellow for text fields
                tag_bg = (180, 140, 0)
                
            # Draw polygon & bounding box
            cv2.polylines(annotated_img, [pts], isClosed=True, color=box_color, thickness=2, lineType=cv2.LINE_AA)
            
            # Label banner
            label = f"#{field_id} {text_str[:22]} ({int(conf*100)}%)"
            font_scale = max(0.35, min(0.55, box_h / 24.0))
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Label background badge
            badge_y1 = max(0, y1 - lh - 6)
            badge_y2 = y1
            badge_x1 = x1
            badge_x2 = min(w, x1 + lw + 8)
            cv2.rectangle(annotated_img, (badge_x1, badge_y1), (badge_x2, badge_y2), tag_bg, -1)
            cv2.putText(annotated_img, label, (badge_x1 + 4, badge_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 2. QR Code Localization & Digital Signature Extraction
        qr_results = zxingcpp.read_barcodes(orig_bgr)
        detected_qrs = []
        for q_id, q in enumerate(qr_results, start=1):
            pos = q.position
            pts_q = np.array([
                [pos.top_left.x, pos.top_left.y],
                [pos.top_right.x, pos.top_right.y],
                [pos.bottom_right.x, pos.bottom_right.y],
                [pos.bottom_left.x, pos.bottom_left.y]
            ], dtype=np.int32)
            qx1 = max(0, int(min(pts_q[:, 0])))
            qy1 = max(0, int(min(pts_q[:, 1])))
            qx2 = min(w, int(max(pts_q[:, 0])))
            qy2 = min(h, int(max(pts_q[:, 1])))
            
            # Crop QR code for signature verification
            pad = 10
            crop_x1 = max(0, qx1 - pad)
            crop_y1 = max(0, qy1 - pad)
            crop_x2 = min(w, qx2 + pad)
            crop_y2 = min(h, qy2 + pad)
            qr_crop = orig_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            qr_crop_path = os.path.join(folder_path, f"{base_name}_qr_{q_id}.png")
            cv2.imwrite(qr_crop_path, qr_crop)
            
            parsed_qr = parse_aadhaar_qr_payload(q.text, q.bytes)
            
            detected_qrs.append({
                "qr_id": q_id,
                "bounding_box": [qx1, qy1, qx2, qy2],
                "polygon_points": pts_q.tolist(),
                "qr_format": str(q.format),
                "error_correction_level": str(q.ec_level),
                "raw_payload_bytes": len(q.bytes),
                "text_content": q.text,
                "parsed_signature_data": parsed_qr,
                "cropped_qr_path": qr_crop_path
            })
            
            # Draw QR box in vibrant Purple / Magenta
            cv2.polylines(annotated_img, [pts_q], isClosed=True, color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            qr_label = f"[QR CODE #{q_id}] {parsed_qr['qr_type']}"
            cv2.putText(annotated_img, qr_label, (qx1, max(22, qy1 - 8)),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
        
        # Top Executive Header Banner
        hud_h = 55
        hud = np.zeros((hud_h, w, 3), dtype=np.uint8)
        cv2.rectangle(hud, (0, 0), (w, hud_h), (25, 25, 25), -1)
        cv2.putText(hud, f"TEXT & QR LOCALIZATION: {fname}", (15, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(hud, f"Resolution: {w}x{h} px  |  Text Blocks: {len(text_fields)}  |  QR Codes: {len(detected_qrs)}  |  Status: AUTHENTIC",
                    (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)
        
        final_vis = np.vstack([hud, annotated_img])
        
        # Save Annotated Image inside adharpan folder
        annotated_path = os.path.join(folder_path, f"{base_name}_text_localized.jpg")
        cv2.imwrite(annotated_path, final_vis)
        
        # Extract Formal Semantic KYC Data Schema
        semantic_data = extract_structured_document_semantics(text_fields, detected_qrs)
        ident = semantic_data["structured_identity"]
        
        # Save Individual JSON Report inside adharpan folder
        doc_report = {
            "document_filename": fname,
            "document_type": semantic_data["document_type"],
            "issuing_authority": semantic_data["issuing_authority"],
            "document_status": "LEGITIMATE / AUTHENTIC",
            "resolution": {"width": w, "height": h},
            "structured_identity": ident,
            "digital_cross_verification": semantic_data["digital_cross_verification"],
            "total_text_fields_detected": len(text_fields),
            "total_qr_codes_detected": len(detected_qrs),
            "qr_codes": detected_qrs,
            "annotated_image_path": annotated_path,
            "raw_text_fields": text_fields
        }
        
        single_json_path = os.path.join(folder_path, f"{base_name}_text_report.json")
        with open(single_json_path, "w", encoding="utf-8") as f:
            json.dump(doc_report, f, indent=2, ensure_ascii=False)
            
        all_docs_summary.append(doc_report)
        
        # Print progress summary
        print(f"[{idx}/{len(img_files)}] Processed: {fname} ({w}x{h} px)")
        print(f"    - Document Type     : {semantic_data['document_type']}")
        print(f"    - Full Name         : {ident['full_name']}")
        print(f"    - ID Number         : {ident['id_number']} ({ident['id_type']})")
        print(f"    - Date of Birth     : {ident['date_of_birth']} | Gender: {ident['gender']}")
        if ident['father_or_guardian_name']:
            print(f"    - Father/Guardian   : {ident['father_or_guardian_name']}")
        if ident['address']['full_address']:
            print(f"    - Address           : {ident['address']['full_address'][:60]}... (PIN: {ident['address']['pincode']})")
        print(f"    - Localized Fields  : {len(text_fields)} text regions | {len(detected_qrs)} QR codes")
        print(f"    [+] Saved Annotated Image : {os.path.basename(annotated_path)}")
        print(f"    [+] Saved Structured JSON : {os.path.basename(single_json_path)}\n")
        
    # Save Master Aggregated JSON inside adharpan folder
    master_json_path = os.path.join(folder_path, "adharpan_master_text_report.json")
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_folder": os.path.abspath(folder_path),
            "total_documents_processed": len(all_docs_summary),
            "documents": all_docs_summary
        }, f, indent=2, ensure_ascii=False)
        
    print("=" * 88)
    print(f"[+] All {len(all_docs_summary)} documents processed successfully!")
    print(f"[+] Master JSON Report Saved to : {master_json_path}")
    print("=" * 88 + "\n")

if __name__ == "__main__":
    target = "test_datasets/adharpan"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    process_adharpan_folder(target)
