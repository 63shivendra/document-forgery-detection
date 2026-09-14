import os
import glob
import cv2
import numpy as np
import yaml
from PIL import Image

try:
    import torch
    import torchvision.transforms.functional as TF
    from src.models.forgery_net import ForgeryNet
    HAS_TORCH = True
except Exception as e:
    HAS_TORCH = False
    torch = None
    TF = None
    ForgeryNet = None



class DocumentVerifier:
    """
    Production Document-Level Verification & Forensic Localization Engine.

    Transitions from raw pixel-level segmentation maps to:
    1. Document-Level Verdict: Is document FAKE or AUTHENTIC? (with fraud score & confidence)
    2. Part-Level Localization: Exact bounding boxes [x1, y1, x2, y2] mapped to original resolution.
    3. Document Spatial Zoning: Identifies which document field is affected (Header, Text Body, Photo, Footer).
    4. Distortion / Tampering Diagnosis: Categorizes the manipulation type (Text Tampering, Splicing, Inpainting).
    """

    def __init__(
        self,
        model_path=None,
        config_path="config.yaml",
        device=None,
        pixel_threshold=None,
        min_region_pixels=None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch dependency is not available or blocked by system Application Control policy.")

        # Load configuration if available
        self.config = {}
        if os.path.exists(config_path):

            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        forensics_cfg = self.config.get("forensics", {})
        self.pixel_threshold = pixel_threshold if pixel_threshold is not None else forensics_cfg.get("pixel_threshold", 0.35)
        self.min_region_pixels = min_region_pixels if min_region_pixels is not None else forensics_cfg.get("min_region_pixels", 25)

        # Determine compute device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.img_size = tuple(self.config.get("training", {}).get("image_size", [384, 384]))
        encoder_name = self.config.get("model", {}).get("encoder", "efficientnet-b2")

        # Auto-resolve checkpoint if not provided
        if not model_path:
            model_path = self._auto_resolve_checkpoint()

        self.model_path = model_path
        self.model = ForgeryNet(
            out_channels=1,
            encoder_name=encoder_name,
            dropout_prob=0.0
        ).to(self.device)

        self._load_weights(model_path)
        self.model.eval()

        # Initialize EasyOCR engine
        self.enable_ocr = forensics_cfg.get("enable_ocr", True)
        self.ocr_reader = None
        if self.enable_ocr:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(["en"], gpu=(self.device == "cuda"), verbose=False)
                print("[DocumentVerifier] EasyOCR initialized (GPU accelerated).")
            except Exception as e:
                print(f"[DocumentVerifier] OCR note: {e} (continuing without OCR)")
                self.ocr_reader = None

    def _auto_resolve_checkpoint(self):
        candidate_dirs = [
            self.config.get("paths", {}).get("saved_models_dir", "./bigpower"),
            "./bigpower",
            "./savedmodels"
        ]
        found_paths = []
        for cdir in candidate_dirs:
            if os.path.exists(cdir):
                found_paths.extend(glob.glob(os.path.join(cdir, "best_model_*.pth")))
                found_paths.extend(glob.glob(os.path.join(cdir, "*.pth")))

        if not found_paths:
            raise FileNotFoundError("No trained .pth checkpoint found in ./bigpower or ./savedmodels")

        # Sort by modification time, newest first
        found_paths.sort(key=os.path.getmtime, reverse=True)
        return found_paths[0]

    def _load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[DocumentVerifier] Loaded checkpoint: {os.path.basename(path)} (epoch {checkpoint.get('epoch', '?')})")
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
            print(f"[DocumentVerifier] Loaded weights: {os.path.basename(path)}")
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

    def _determine_spatial_zone(self, norm_x1, norm_y1, norm_x2, norm_y2):
        """Classify spatial location within standard document geometry."""
        cx = (norm_x1 + norm_x2) / 2.0
        cy = (norm_y1 + norm_y2) / 2.0

        if cy < 0.22:
            return "Header / Issuer / Document Title"
        elif cy > 0.78:
            if cx < 0.50:
                return "Footer / Terms / Authority Seal"
            else:
                return "Footer / Authorized Signature Field"
        else:
            if cx > 0.68 and cy < 0.55:
                return "Photo / ID Portrait / Stamp Area"
            elif cx < 0.32:
                return "Document Body (Left Field / Label Column)"
            else:
                return "Document Body (Text / Number / Value Field)"

    def _diagnose_distortion_type(self, patch_bgr, aspect_ratio, area_px, peak_conf):
        """
        Heuristic forensic analysis to identify manipulation category:
        - Text/Number Tampering: High aspect ratio or compact character dimensions
        - Splicing/Copy-Paste: High boundary gradient step
        - Inpainting/Erasure: Abnormally low texture variance
        """
        if patch_bgr is None or patch_bgr.size == 0:
            return "Unspecified Manipulation"

        h, w = patch_bgr.shape[:2]
        gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Texture variance via Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = float(laplacian.var())

        # 2. Aspect ratio cues for text
        is_elongated = (aspect_ratio >= 2.0) or (aspect_ratio <= 0.45)
        is_small = area_px < 6000

        if is_elongated and is_small:
            return "Text / Number Field Alteration"
        elif texture_var < 20.0 and min(h, w) > 20:
            return "Inpainting / Content Erasure / White-Out"
        elif aspect_ratio >= 0.7 and aspect_ratio <= 1.4 and area_px > 3000:
            return "Photo / Seal / Region Splicing"
        else:
            if peak_conf > 0.85:
                return "Copy-Move / Region Splicing"
            return "Localized Pixel Manipulation"

    def verify(self, image_input, threshold=None, min_pixels=None):
        """
        Run document-level verification and part localization.

        Args:
            image_input: File path (str), PIL.Image, or numpy.ndarray (BGR or RGB)
            threshold: Pixel probability cutoff (default: self.pixel_threshold)
            min_pixels: Minimum pixels in cluster to form a valid part (default: self.min_region_pixels)

        Returns:
            dict containing:
                - document_verdict: 'TAMPERED / FAKE' or 'AUTHENTIC'
                - is_fake: bool
                - fraud_score: float (0.0 to 1.0)
                - confidence: float (0.0 to 1.0)
                - risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'
                - total_tampered_regions: int
                - tampered_area_percentage: float
                - tampered_regions: list of dicts (bbox, zone, distortion_type, conf, etc.)
                - raw_probs: np.ndarray (H_orig, W_orig) probability map
                - original_bgr: np.ndarray original image in BGR
        """
        th = threshold if threshold is not None else self.pixel_threshold
        min_px = min_pixels if min_pixels is not None else self.min_region_pixels

        # 1. Ingest image
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            original_bgr = cv2.imread(image_input)
            if original_bgr is None:
                raise ValueError(f"Failed to read image at {image_input}")
            pil_img = Image.open(image_input).convert("RGB")
            doc_name = os.path.basename(image_input)
        elif isinstance(image_input, np.ndarray):
            original_bgr = image_input.copy()
            pil_img = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
            doc_name = "in_memory_document"
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            original_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            doc_name = getattr(image_input, "filename", "in_memory_document")
        else:
            raise TypeError("Unsupported image input type. Use file path, PIL.Image, or numpy array.")

        orig_w, orig_h = pil_img.size

        # 2. Preprocess for ForgeryNet
        resized_img = pil_img.resize(self.img_size, Image.BILINEAR)
        img_tensor = TF.to_tensor(resized_img)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ).unsqueeze(0).to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs_384 = torch.sigmoid(logits).squeeze().cpu().numpy()

        # 4. Upscale probability map to original resolution
        raw_probs_orig = cv2.resize(probs_384, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # 5. Connected Component Analysis & Denoising
        bin_mask_384 = (probs_384 >= th).astype(np.uint8)
        # Morphological opening to remove 1-3 px sensor noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_bin_384 = cv2.morphologyEx(bin_mask_384, cv2.MORPH_OPEN, kernel_open)

        # Morphological closing to group fragmented character/word strokes into cohesive field blocks
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        grouped_bin_384 = cv2.morphologyEx(clean_bin_384, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grouped_bin_384, connectivity=8)

        detected_regions = []
        valid_cluster_pixels_384 = 0

        scale_x = orig_w / float(self.img_size[0])
        scale_y = orig_h / float(self.img_size[1])

        for label_id in range(1, num_labels):
            area_384 = stats[label_id, cv2.CC_STAT_AREA]
            if area_384 < min_px:
                continue

            valid_cluster_pixels_384 += area_384
            x_384 = stats[label_id, cv2.CC_STAT_LEFT]
            y_384 = stats[label_id, cv2.CC_STAT_TOP]
            w_384 = stats[label_id, cv2.CC_STAT_WIDTH]
            h_384 = stats[label_id, cv2.CC_STAT_HEIGHT]

            # Project coordinates back to original resolution
            x1_orig = max(0, int(round(x_384 * scale_x)))
            y1_orig = max(0, int(round(y_384 * scale_y)))
            x2_orig = min(orig_w, int(round((x_384 + w_384) * scale_x)))
            y2_orig = min(orig_h, int(round((y_384 + h_384) * scale_y)))

            box_w = max(1, x2_orig - x1_orig)
            box_h = max(1, y2_orig - y1_orig)
            aspect_ratio = round(box_w / float(box_h), 3)

            # Extract component probabilities
            mask_component = (labels == label_id)
            comp_probs = probs_384[mask_component]
            peak_conf = float(np.max(comp_probs))
            mean_conf = float(np.mean(comp_probs))

            # Crop patch from original image
            patch_bgr = original_bgr[y1_orig:y2_orig, x1_orig:x2_orig]

            # Semantic Zone & Distortion Diagnosis
            norm_x1 = round(x1_orig / orig_w, 4)
            norm_y1 = round(y1_orig / orig_h, 4)
            norm_x2 = round(x2_orig / orig_w, 4)
            norm_y2 = round(y2_orig / orig_h, 4)

            doc_zone = self._determine_spatial_zone(norm_x1, norm_y1, norm_x2, norm_y2)
            distortion_type = self._diagnose_distortion_type(patch_bgr, aspect_ratio, box_w * box_h, peak_conf)

            # OCR Text Extraction on the altered region
            extracted_text = ""
            if self.ocr_reader is not None and patch_bgr is not None and patch_bgr.size > 0:
                pad = 8
                x1_p = max(0, x1_orig - pad)
                y1_p = max(0, y1_orig - pad)
                x2_p = min(orig_w, x2_orig + pad)
                y2_p = min(orig_h, y2_orig + pad)
                ocr_patch = original_bgr[y1_p:y2_p, x1_p:x2_p]
                try:
                    ocr_res = self.ocr_reader.readtext(ocr_patch, detail=0)
                    extracted_text = " ".join([t.strip() for t in ocr_res if t.strip()])
                except Exception:
                    extracted_text = ""

            detected_regions.append({
                "region_id": len(detected_regions) + 1,
                "bbox": [x1_orig, y1_orig, x2_orig, y2_orig],
                "normalized_bbox": [norm_x1, norm_y1, norm_x2, norm_y2],
                "width": box_w,
                "height": box_h,
                "aspect_ratio": aspect_ratio,
                "area_pixels": box_w * box_h,
                "area_percentage": round(((box_w * box_h) / float(orig_w * orig_h)) * 100.0, 3),
                "peak_confidence": round(peak_conf, 4),
                "mean_confidence": round(mean_conf, 4),
                "document_region": doc_zone,
                "distortion_type": distortion_type,
                "extracted_text": extracted_text,
                "patch_bgr": patch_bgr
            })

        # Sort regions by peak confidence descending
        detected_regions.sort(key=lambda r: r["peak_confidence"], reverse=True)
        for idx, r in enumerate(detected_regions, start=1):
            r["region_id"] = idx

        total_regions = len(detected_regions)
        is_fake = total_regions > 0

        # Calculate document-level scores
        if is_fake:
            peak_overall = max(r["peak_confidence"] for r in detected_regions)
            top_mean = detected_regions[0]["mean_confidence"]
            area_ratio = min(1.0, (valid_cluster_pixels_384 / float(self.img_size[0] * self.img_size[1])) * 5.0)

            raw_fraud_score = (0.60 * peak_overall) + (0.30 * top_mean) + (0.10 * area_ratio)
            fraud_score = float(np.clip(raw_fraud_score, 0.0, 0.999))
            confidence = fraud_score
            verdict = "TAMPERED / FAKE"

            if fraud_score >= 0.80:
                risk_level = "CRITICAL"
            elif fraud_score >= 0.60:
                risk_level = "HIGH"
            elif fraud_score >= 0.40:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
        else:
            background_peak = float(np.max(probs_384))
            fraud_score = float(background_peak * 0.4)
            confidence = float(1.0 - fraud_score)
            verdict = "AUTHENTIC"
            risk_level = "LOW"

        total_area_pct = sum(r["area_percentage"] for r in detected_regions)

        return {
            "document_name": doc_name,
            "document_dimensions": [orig_w, orig_h],
            "document_verdict": verdict,
            "is_fake": is_fake,
            "fraud_score": round(fraud_score, 4),
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "total_tampered_regions": total_regions,
            "tampered_area_percentage": round(total_area_pct, 3),
            "tampered_regions": detected_regions,
            "raw_probs": raw_probs_orig,
            "original_bgr": original_bgr
        }
