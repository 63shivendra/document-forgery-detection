"""
Governed Local Evidence Reasoning Engine
Translates technical forensic signals, bounding box callouts, and rule checks into
natural, human-readable executive explanations.
100% Pure Local On-Device Execution. Zero External API Dependencies.
"""

from typing import Dict, Any, List


class GovernedLLMBridge:
    """
    100% Local On-Device Evidence Reasoning Engine.
    Translates structured forensic signals into clear, human-readable executive audit summaries.
    Operates completely offline with zero API keys or cloud dependencies.
    """

    def explain_decision(
        self,
        doc_type: str,
        decision: str,
        risk_score: float,
        reason_codes: List[str],
        boxes: List[Dict[str, Any]]
    ) -> str:
        """Generates executive natural language explanation 100% locally."""
        if decision == "APPROVED":
            return (
                f"Document verified as AUTHENTIC ({doc_type}). "
                f"Zero visual tampering detected (Risk Score: {risk_score:.1%}). "
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
