"""
Agent 1: Document Extraction & Structural Agent (DocumentExtractionAgent)
Uses DocumentRegistry for Open/Closed Principle (OCP) Zero-Code Document Extension.
"""

import re
import cv2
import numpy as np
from typing import Dict, Any, Optional
import zxingcpp

from .base_agent import BaseAgent, BaseDocumentParser, DocumentRegistry


# ------------------------------------------------------------------------------
# CONCRETE DOCUMENT PARSERS (Registered with DocumentRegistry)
# ------------------------------------------------------------------------------

class PANCardParser(BaseDocumentParser):
    def document_type(self) -> str:
        return "PAN_CARD"

    def match_score(self, text: str, qr_data: Optional[str] = None, aspect_ratio: float = 1.0) -> float:
        score = 0.0
        if re.search(r"[A-Z]{5}\d{4}[A-Z]", text):
            score += 0.6
        if "INCOME TAX" in text.upper() or "GOVT OF INDIA" in text.upper():
            score += 0.3
        if 1.40 <= aspect_ratio <= 1.75:
            score += 0.1
        return min(1.0, score)

    def parse_fields(self, text: str, qr_data: Optional[str] = None) -> Dict[str, Any]:
        match = re.search(r"([A-Z]{5}\d{4}[A-Z])", text)
        pan_num = match.group(1) if match else ""
        return {"pan_number": pan_num}

    def get_pii_fields(self) -> list:
        return ["pan_number"]


class AadhaarCardParser(BaseDocumentParser):
    @staticmethod
    def _verhoeff_checksum(num_str: str) -> bool:
        """Verhoeff Dihedral Group D5 algorithm checksum validation."""
        clean = re.sub(r"\D", "", num_str)
        if len(clean) != 12:
            return False
        
        d = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
             [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
             [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
             [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
             [5, 9, 8, 7, 6, 0, 1, 2, 3, 4],
             [6, 5, 9, 8, 7, 1, 2, 3, 4, 0],
             [7, 6, 5, 9, 8, 2, 3, 4, 0, 1],
             [8, 7, 6, 5, 9, 3, 4, 0, 1, 2],
             [9, 8, 7, 6, 5, 4, 0, 1, 2, 3]]
        
        p = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1, 5, 7, 6, 2, 8, 3, 4, 9, 0],
             [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
             [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
             [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
             [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
             [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
             [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]]
        
        c = 0
        for i, item in enumerate(reversed(clean)):
            c = d[c][p[i % 8][int(item)]]
        return c == 0

    def document_type(self) -> str:
        return "AADHAAR_CARD"

    def match_score(self, text: str, qr_data: Optional[str] = None, aspect_ratio: float = 1.0) -> float:
        score = 0.0
        if qr_data and ("uidai" in qr_data.lower() or "xml" in qr_data.lower()):
            return 1.0
        if "UNIQUE IDENTIFICATION" in text.upper() or "UIDAI" in text.upper() or "GOVERNMENT OF INDIA" in text.upper():
            score += 0.5
        match = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", text)
        if match and self._verhoeff_checksum(match.group(0)):
            score += 0.4
        return min(1.0, score)

    def parse_fields(self, text: str, qr_data: Optional[str] = None) -> Dict[str, Any]:
        match = re.search(r"\b(\d{4}\s?\d{4}\s?\d{4})\b", text)
        uid = match.group(1) if match else ""
        valid_checksum = self._verhoeff_checksum(uid) if uid else False
        return {"aadhaar_number": uid, "verhoeff_checksum_valid": valid_checksum}

    def get_pii_fields(self) -> list:
        return ["aadhaar_number"]


class GSTCertParser(BaseDocumentParser):
    def document_type(self) -> str:
        return "GST_REG06"

    def match_score(self, text: str, qr_data: Optional[str] = None, aspect_ratio: float = 1.0) -> float:
        score = 0.0
        if "FORM GST REG-06" in text.upper() or "REGISTRATION CERTIFICATE" in text.upper():
            score += 0.6
        if re.search(r"\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]", text):
            score += 0.4
        return min(1.0, score)

    def parse_fields(self, text: str, qr_data: Optional[str] = None) -> Dict[str, Any]:
        match = re.search(r"(\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9])", text)
        gstin = match.group(1) if match else ""
        return {"gstin": gstin}

    def get_pii_fields(self) -> list:
        return []


# Register Parsers in DocumentRegistry
DocumentRegistry.register(PANCardParser())
DocumentRegistry.register(AadhaarCardParser())
DocumentRegistry.register(GSTCertParser())


# ------------------------------------------------------------------------------
# EXTRACTION AGENT IMPLEMENTATION
# ------------------------------------------------------------------------------

class DocumentExtractionAgent(BaseAgent):
    def __init__(self):
        super().__init__("DocumentExtractionAgent")

    def decode_qr(self, image: np.ndarray) -> Optional[str]:
        """Decodes QR payload using native C++ zxingcpp."""
        try:
            results = zxingcpp.read_barcodes(image)
            for r in results:
                if r.text:
                    return r.text
        except Exception:
            pass
        return None

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        image = payload.get("image_np")
        raw_text = payload.get("raw_text", "")
        
        qr_data = None
        if image is not None:
            qr_data = self.decode_qr(image)
            h, w = image.shape[:2]
            aspect_ratio = float(w) / float(h) if h > 0 else 1.0
        else:
            aspect_ratio = 1.0

        # Classify document zero-shot using DocumentRegistry (OCP)
        cls_result = DocumentRegistry.classify(raw_text, qr_data, aspect_ratio)
        doc_type = cls_result["document_type"]
        confidence = cls_result["confidence"]

        extracted_fields = {}
        parser = DocumentRegistry.get_parser(doc_type)
        if parser:
            extracted_fields = parser.parse_fields(raw_text, qr_data)

        updated_payload = payload.copy()
        updated_payload.update({
            "document_type": doc_type,
            "classification_confidence": confidence,
            "qr_decoded": qr_data is not None,
            "qr_data": qr_data,
            "extracted_fields": extracted_fields
        })
        
        # Merge parsed fields into top-level for downstream agents
        updated_payload.update(extracted_fields)
        return updated_payload
