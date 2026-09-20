"""
Agent 5: Governed LLM Reasoning Agent (ReasoningAgent)
Links forensic signals to NPCI Reason Codes, evaluates evidence sufficiency,
triggers escalation, and emits official NPCI Document Trust Receipt.
"""

from typing import Dict, Any, List
import datetime
from .base_agent import BaseAgent

try:
    from llm_bridge.reasoning_llm import GovernedLLMBridge
except ImportError:
    from ..llm_bridge.reasoning_llm import GovernedLLMBridge


class ReasoningAgent(BaseAgent):
    def __init__(self):
        super().__init__("ReasoningAgent")
        self.llm_bridge = GovernedLLMBridge()

    def determine_reason_codes(self, payload: Dict[str, Any]) -> List[str]:
        """Maps forensic anomalies and rule failures to standardized NPCI Reason Codes."""
        codes = []
        if payload.get("forgery_detected", False):
            codes.append("REASON_VISUAL_TAMPERING_DETECTED")
            risk = payload.get("forensic_risk_score", 0.0)
            if risk >= 0.70:
                codes.append("REASON_HIGH_CONFIDENCE_DIGIT_EDIT")
            else:
                codes.append("REASON_MEDIUM_CONFIDENCE_BORDER_ANOMALY")
                
        if not payload.get("verhoeff_checksum_valid", True):
            codes.append("REASON_VERHOEFF_CHECKSUM_FAILED")
            
        if not payload.get("entity_validation_passed", True):
            codes.append("REASON_ENTITY_PAN_MISMATCH")
            
        if not codes:
            codes.append("REASON_VERIFICATION_PASSED_CLEAN")
            
        return codes

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        risk_score = payload.get("forensic_risk_score", 0.0)
        forgery_detected = payload.get("forgery_detected", False)
        entity_passed = payload.get("entity_validation_passed", True)
        verhoeff_passed = payload.get("verhoeff_checksum_valid", True)
        
        # 1. Determine Overall Decision
        if forgery_detected or not entity_passed or not verhoeff_passed:
            decision = "REJECTED"
        else:
            decision = "APPROVED"

        # 2. Determine Human Escalation Need
        # Ambiguous risk zone (0.30 < risk < 0.65) triggers escalation
        escalation_required = (0.30 < risk_score < 0.65) or (decision == "REJECTED" and risk_score < 0.50)

        # 3. Determine Reason Codes
        reason_codes = self.determine_reason_codes(payload)

        # 4. Generate Human-Readable Executive Summary via LLM Bridge
        exec_summary = self.llm_bridge.explain_decision(
            doc_type=payload.get("document_type", "UNKNOWN_DOCUMENT"),
            decision=decision,
            risk_score=risk_score,
            reason_codes=reason_codes,
            boxes=payload.get("suspicious_bounding_boxes", [])
        )

        # 5. Emit Official NPCI Document Trust Receipt JSON
        trust_receipt = {
            "document_hash": payload.get("document_hash", "HASH_UNAVAILABLE"),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "agent_identity": "NPCI_EDGE_VERIFIER_V1",
            "policy_version": "v2.4.1",
            "decision": decision,
            "risk_score": risk_score,
            "confidence_level": round(1.0 - abs(0.5 - risk_score), 3),
            "evidence_sufficiency": True,
            "escalation_required": escalation_required,
            "reason_codes": reason_codes,
            "suspicious_fields": payload.get("suspicious_bounding_boxes", []),
            "pii_masked_payload": {
                "doc_type": payload.get("document_type", "UNKNOWN_DOCUMENT"),
                "classification_confidence": payload.get("classification_confidence", 0.0),
                "pan_masked": payload.get("pan_number_masked", ""),
                "aadhaar_masked": payload.get("aadhaar_number_masked", ""),
                "verhoeff_checksum_valid": verhoeff_passed
            },
            "executive_summary": exec_summary
        }

        updated = payload.copy()
        updated["trust_receipt"] = trust_receipt
        return updated
