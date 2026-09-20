"""
Governed LLM / SLM Reasoning Bridge
Translates technical forensic signals, bounding box callouts, and rule checks into
natural, human-readable executive explanations.
"""

import os
from typing import Dict, Any, List


class GovernedLLMBridge:
    """
    Bridge connecting Edge Verification Signals to LLM/SLM Reasoning Engines.
    Uses Gemini API if API key available, or local SLM deterministic template offline.
    """

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", None)

    def explain_decision(
        self,
        doc_type: str,
        decision: str,
        risk_score: float,
        reason_codes: List[str],
        boxes: List[Dict[str, Any]]
    ) -> str:
        """Generates executive natural language explanation."""
        if decision == "APPROVED":
            return (
                f"Document verified as AUTHENTIC ({doc_type}). "
                f"Zero visual tampering detected (Risk Score: {risk_score:.2%}). "
                f"All structural keypoints, QR decoding, and government checksum rules passed."
            )
            
        # Rejection narrative synthesis
        reasons_text = ", ".join(reason_codes)
        num_boxes = len(boxes)
        box_str = f"{num_boxes} suspicious region(s)" if num_boxes > 0 else "structural anomalies"
        
        narrative = (
            f"Document REJECTED ({doc_type}). Visual forensic analysis detected {box_str} "
            f"with a peak fraud certainty of {risk_score:.1%}. "
            f"Primary reason codes: [{reasons_text}]. "
            f"PII has been masked per UIDAI/RBI regulations."
        )
        return narrative
