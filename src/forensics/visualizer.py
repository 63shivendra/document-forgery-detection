import os
import json
import cv2
import numpy as np


class ForensicVisualizer:
    """
    Visual evidence generator for Document-Level Verification.
    Renders:
    1. Annotated Document with Bounding Boxes & Status Banner
    2. Multi-Panel Forensic Audit Canvas (Detection + Thermal Heatmap + Cropped Evidence Gallery)
    3. Individual cropped evidence patches
    4. Structured JSON audit report
    """

    COLOR_CRITICAL = (30, 30, 220)    # Red (BGR)
    COLOR_HIGH     = (20, 100, 240)   # Orange-Red
    COLOR_MEDIUM   = (0, 180, 255)    # Amber / Yellow
    COLOR_AUTHENTIC= (40, 170, 50)    # Emerald Green
    COLOR_TEXT_BG  = (20, 20, 20)     # Dark Slate

    @classmethod
    def get_risk_color(cls, risk_level):
        if risk_level == "CRITICAL":
            return cls.COLOR_CRITICAL
        elif risk_level == "HIGH":
            return cls.COLOR_HIGH
        elif risk_level == "MEDIUM":
            return cls.COLOR_MEDIUM
        else:
            return cls.COLOR_AUTHENTIC

    @classmethod
    def annotate_document(cls, result):
        """Draws bounding boxes, part tags, and a top status banner on the original document."""
        img = result["original_bgr"].copy()
        h, w = img.shape[:2]
        is_fake = result["is_fake"]
        risk_level = result["risk_level"]
        primary_color = cls.get_risk_color(risk_level)

        # 1. Draw Top Status Banner
        banner_h = max(55, int(h * 0.065))
        banner = np.full((banner_h, w, 3), cls.COLOR_TEXT_BG, dtype=np.uint8)

        # Status accent bar
        cv2.rectangle(banner, (0, 0), (int(w * 0.015), banner_h), primary_color, -1)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.55, min(1.0, w / 1200.0))
        thickness = max(1, int(round(font_scale * 1.8)))

        if is_fake:
            title_text = f"STATUS: FAKE / TAMPERED  |  Score: {result['fraud_score']*100:.1f}% [{risk_level}]"
            subtitle_text = f"Parts Altered: {result['total_tampered_regions']}  |  Tampered Area: {result['tampered_area_percentage']:.2f}%"
        else:
            title_text = f"STATUS: AUTHENTIC DOCUMENT  |  Confidence: {result['confidence']*100:.1f}%"
            subtitle_text = "Integrity Verified: No Tampering or Splicing Artifacts Detected"

        cv2.putText(banner, title_text, (int(w * 0.03), int(banner_h * 0.45)), font, font_scale, primary_color, thickness, cv2.LINE_AA)
        cv2.putText(banner, subtitle_text, (int(w * 0.03), int(banner_h * 0.85)), font, font_scale * 0.8, (210, 210, 210), max(1, thickness - 1), cv2.LINE_AA)

        # Attach banner on top
        annotated = np.vstack([banner, img])

        # 2. Draw Bounding Boxes around tampered regions
        box_thickness = max(2, int(round(min(w, h) / 400.0)))

        for reg in result["tampered_regions"]:
            x1, y1, x2, y2 = reg["bbox"]
            # Shift Y by banner height
            adj_y1 = y1 + banner_h
            adj_y2 = y2 + banner_h

            color = cls.get_risk_color("CRITICAL" if reg["peak_confidence"] >= 0.80 else "HIGH" if reg["peak_confidence"] >= 0.60 else "MEDIUM")

            # Outer box
            cv2.rectangle(annotated, (x1, adj_y1), (x2, adj_y2), color, box_thickness)

            # Label badge
            txt_tag = f" \"{reg['extracted_text'][:14]}\"" if reg.get("extracted_text") else ""
            label_text = f"Part {reg['region_id']}: {reg['distortion_type']}{txt_tag} ({reg['peak_confidence']*100:.0f}%)"
            (lbl_w, lbl_h), baseline = cv2.getTextSize(label_text, font, font_scale * 0.65, 1)

            lbl_y1 = max(banner_h, adj_y1 - lbl_h - 8)
            lbl_y2 = max(banner_h + lbl_h + 8, adj_y1)
            lbl_x2 = min(w, x1 + lbl_w + 12)

            cv2.rectangle(annotated, (x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
            cv2.putText(annotated, label_text, (x1 + 6, lbl_y2 - 5), font, font_scale * 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return annotated

    @classmethod
    def generate_heatmap_overlay(cls, original_bgr, raw_probs, alpha=0.45):
        """Generates a color-mapped thermal overlay showing anomaly probability."""
        norm_probs = np.clip(raw_probs * 255.0, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(norm_probs, cv2.COLORMAP_JET)

        # Blend with original image
        blended = cv2.addWeighted(original_bgr, 1.0 - alpha, color_map, alpha, 0)
        return blended

    @classmethod
    def generate_forensic_audit_panel(cls, result, max_evidence_crops=3):
        """
        Creates a high-level 3-Panel Forensic Audit Board:
        [Panel 1: Annotated Document] | [Panel 2: Thermal Heatmap] | [Panel 3: Evidence Crops]
        """
        annotated = cls.annotate_document(result)
        orig_bgr = result["original_bgr"]
        raw_probs = result["raw_probs"]
        heatmap_overlay = cls.generate_heatmap_overlay(orig_bgr, raw_probs)

        # Standardize height for horizontal stitching
        target_h = 768
        h_ann, w_ann = annotated.shape[:2]
        w_ann_resized = int(w_ann * (target_h / float(h_ann)))
        p1 = cv2.resize(annotated, (w_ann_resized, target_h))

        h_hm, w_hm = heatmap_overlay.shape[:2]
        w_hm_resized = int(w_hm * (target_h / float(h_hm)))
        p2 = cv2.resize(heatmap_overlay, (w_hm_resized, target_h))

        # Add title header to heatmap panel
        hm_header_h = 45
        hm_canvas = np.zeros((target_h + hm_header_h, w_hm_resized, 3), dtype=np.uint8)
        hm_canvas[:hm_header_h, :] = (35, 35, 35)
        cv2.putText(hm_canvas, "NEURAL FORENSIC HEATMAP", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 220, 255), 1, cv2.LINE_AA)
        hm_canvas[hm_header_h:, :] = p2

        # Create Panel 3: Evidence Patch Column
        evidence_w = 400
        p3_canvas = np.full((target_h + hm_header_h, evidence_w, 3), 25, dtype=np.uint8)
        cv2.rectangle(p3_canvas, (0, 0), (evidence_w, hm_header_h), (35, 35, 35), -1)
        cv2.putText(p3_canvas, "TAMPER EVIDENCE CROPS", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        regions = result["tampered_regions"][:max_evidence_crops]
        if regions:
            slot_h = (target_h) // max_evidence_crops
            for idx, reg in enumerate(regions):
                patch = reg["patch_bgr"]
                y_offset = hm_header_h + (idx * slot_h)

                if patch is not None and patch.size > 0:
                    ph, pw = patch.shape[:2]
                    # Resize patch keeping aspect ratio within available slot
                    avail_pw = evidence_w - 30
                    avail_ph = slot_h - 55
                    scale = min(avail_pw / float(pw), avail_ph / float(ph))
                    npw = max(10, int(pw * scale))
                    nph = max(10, int(ph * scale))
                    resized_patch = cv2.resize(patch, (npw, nph))

                    # Center patch
                    px = 15 + (avail_pw - npw) // 2
                    py = y_offset + 30 + (avail_ph - nph) // 2
                    p3_canvas[py:py + nph, px:px + npw] = resized_patch

                    # Border around patch
                    pcolor = cls.get_risk_color("CRITICAL" if reg["peak_confidence"] >= 0.80 else "HIGH")
                    cv2.rectangle(p3_canvas, (px - 1, py - 1), (px + npw + 1, py + nph + 1), pcolor, 2)

                # Info text
                info_line1 = f"Part {reg['region_id']}: {reg['distortion_type'][:22]}"
                if reg.get("extracted_text"):
                    info_line2 = f"Text: \"{reg['extracted_text'][:18]}\" | {reg['peak_confidence']*100:.0f}%"
                else:
                    info_line2 = f"Conf: {reg['peak_confidence']*100:.1f}% | {reg['document_region'][:18]}"
                cv2.putText(p3_canvas, info_line1, (15, y_offset + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
                cv2.putText(p3_canvas, info_line2, (15, y_offset + slot_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)
        else:
            cv2.putText(p3_canvas, "No tampered regions found", (30, target_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 50), 1, cv2.LINE_AA)
            cv2.putText(p3_canvas, "Document is authentic.", (30, target_h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

        # Adjust p1 with header to match height
        p1_canvas = np.zeros((target_h + hm_header_h, w_ann_resized, 3), dtype=np.uint8)
        p1_canvas[:hm_header_h, :] = (35, 35, 35)
        cv2.putText(p1_canvas, f"INSPECTION: {result['document_name'][:30]}", (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        p1_canvas[hm_header_h:, :] = p1

        # Combine panels horizontally
        audit_board = np.hstack([p1_canvas, hm_canvas, p3_canvas])
        return audit_board

    @classmethod
    def save_all_artifacts(cls, result, output_dir="reports/forensics"):
        """Saves annotated image, audit panel board, individual crops, and JSON report."""
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(result["document_name"]))[0]

        saved_files = {}

        # 1. Save Annotated Document
        annotated_img = cls.annotate_document(result)
        annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated_img)
        saved_files["annotated_image"] = annotated_path

        # 2. Save Multi-Panel Audit Board
        audit_panel = cls.generate_forensic_audit_panel(result)
        panel_path = os.path.join(output_dir, f"{base_name}_audit_panel.jpg")
        cv2.imwrite(panel_path, audit_panel)
        saved_files["audit_panel"] = panel_path

        # 3. Save Individual Part Crops
        crop_paths = []
        for reg in result["tampered_regions"]:
            patch = reg["patch_bgr"]
            if patch is not None and patch.size > 0:
                crop_path = os.path.join(output_dir, f"{base_name}_part_{reg['region_id']}.jpg")
                cv2.imwrite(crop_path, patch)
                crop_paths.append(crop_path)
        saved_files["crop_images"] = crop_paths

        # 4. Save JSON Report (excluding numpy arrays)
        json_report = {
            "document_name": result["document_name"],
            "dimensions": result["document_dimensions"],
            "verdict": result["document_verdict"],
            "is_fake": result["is_fake"],
            "fraud_score": result["fraud_score"],
            "confidence": result["confidence"],
            "risk_level": result["risk_level"],
            "total_tampered_regions": result["total_tampered_regions"],
            "tampered_area_percentage": result["tampered_area_percentage"],
            "tampered_regions": [
                {
                    "region_id": r["region_id"],
                    "bbox": r["bbox"],
                    "normalized_bbox": r["normalized_bbox"],
                    "width": r["width"],
                    "height": r["height"],
                    "aspect_ratio": r["aspect_ratio"],
                    "area_pixels": r["area_pixels"],
                    "area_percentage": r["area_percentage"],
                    "peak_confidence": r["peak_confidence"],
                    "mean_confidence": r["mean_confidence"],
                    "document_region": r["document_region"],
                    "distortion_type": r["distortion_type"],
                    "extracted_text": r.get("extracted_text", ""),
                }
                for r in result["tampered_regions"]
            ],
            "exported_files": {
                "annotated_image": annotated_path,
                "audit_panel": panel_path,
                "crops": crop_paths
            }
        }

        json_path = os.path.join(output_dir, f"{base_name}_forensic_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_report, f, indent=2)
        saved_files["json_report"] = json_path

        return saved_files
