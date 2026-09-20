"""
Standalone Edge Verification Entrypoint & CLI Engine
Runs 5-Agent Governed Pipeline strictly on Edge / Mobile ONNX Engine.
Zero PyTorch & Zero CUDA dependency.
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np

# UTF-8 stdout fix for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents.extraction_agent import DocumentExtractionAgent
from agents.forensic_agent import ForensicDetectionAgent
from agents.privacy_agent import PrivacyGovernanceAgent
from agents.validation_agent import EntityValidationAgent
from agents.reasoning_agent import ReasoningAgent


class GovernedEdgeAuditEngine:
    """
    Main Orchestrator for the 5-Agent Governed Edge Pipeline.
    Runs completely in-memory on local edge devices.
    """

    def __init__(self):
        print("⚡ Initializing Governed 5-Agent Edge Verification Engine...")
        t0 = time.time()
        self.agent_extraction = DocumentExtractionAgent()
        self.agent_forensic = ForensicDetectionAgent()
        self.agent_privacy = PrivacyGovernanceAgent()
        self.agent_validation = EntityValidationAgent()
        self.agent_reasoning = ReasoningAgent()
        print(f"✅ Engine initialized in {(time.time() - t0)*1000:.1f} ms!")

    def audit_document(self, image_path: str, merchant_entity_type: str = "proprietorship") -> dict:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        # Load raw bytes & OpenCV image matrix into volatile RAM
        with open(image_path, "rb") as f:
            raw_bytes = f.read()

        image_np = cv2.imread(image_path)
        if image_np is None:
            raise ValueError(f"Failed to decode image matrix: {image_path}")

        # OCR / Raw text heuristic extraction for demo
        raw_text = self._basic_text_extract(image_path)

        # Initial Payload
        payload = {
            "image_path": image_path,
            "raw_image_bytes": raw_bytes,
            "image_np": image_np,
            "raw_text": raw_text,
            "merchant_entity_type": merchant_entity_type
        }

        t_start = time.time()

        # Execute 5-Agent Pipeline
        payload = self.agent_extraction.process(payload)
        payload = self.agent_forensic.process(payload)
        payload = self.agent_privacy.process(payload)
        payload = self.agent_validation.process(payload)
        payload = self.agent_reasoning.process(payload)

        latency_ms = (time.time() - t_start) * 1000.0

        receipt = payload.get("trust_receipt", {})
        receipt["edge_execution_latency_ms"] = round(latency_ms, 2)

        return receipt

    def _basic_text_extract(self, path: str) -> str:
        """Basic text extraction fallback."""
        filename = os.path.basename(path)
        return filename.upper()


def main():
    parser = argparse.ArgumentParser(description="Governed Edge & Mobile Verification Engine")
    parser.add_argument("--input", required=True, help="Path to input document image")
    parser.add_argument("--entity-type", default="proprietorship", help="Merchant entity type (proprietorship, pvt_ltd, etc.)")
    parser.add_argument("--output-json", default=None, help="Optional output JSON path for trust receipt")

    args = parser.parse_args()

    engine = GovernedEdgeAuditEngine()
    receipt = engine.audit_document(args.input, merchant_entity_type=args.entity_type)

    print("\n" + "="*80)
    print("📜 OFFICIAL NPCI DOCUMENT TRUST RECEIPT (EMITTED UPSTREAM)")
    print("="*80)
    print(json.dumps(receipt, indent=2))
    print("="*80)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"✅ Saved Trust Receipt JSON to: {args.output_json}")


if __name__ == "__main__":
    main()
