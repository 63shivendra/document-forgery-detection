"""
Agent 3: Privacy & PII Governance Agent (PrivacyGovernanceAgent)
Acts as an in-memory PII Privacy Firewall before data leaves local RAM.
"""

from typing import Dict, Any
from .base_agent import BaseAgent

try:
    from privacy.pii_masker import PIIMasker
except ImportError:
    from ..privacy.pii_masker import PIIMasker


class PrivacyGovernanceAgent(BaseAgent):
    def __init__(self):
        super().__init__("PrivacyGovernanceAgent")

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        raw_image_bytes = payload.get("raw_image_bytes")
        image_np = payload.get("image_np")
        
        # 1. Apply PII Sanitization
        sanitized = PIIMasker.sanitize_payload(payload, raw_image_bytes)
        
        # 2. Blur visual face box if detected
        face_box = payload.get("face_box")
        if image_np is not None and face_box:
            sanitized_image = PIIMasker.blur_sensitive_region(image_np, face_box)
            sanitized["sanitized_image_np"] = sanitized_image

        # 3. Purge raw image pointers from payload memory
        sanitized.pop("raw_image_bytes", None)
        sanitized.pop("image_np", None)
        sanitized.pop("raw_text", None)
        
        sanitized["privacy_firewall_passed"] = True
        return sanitized
