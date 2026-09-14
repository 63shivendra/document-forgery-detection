"""
Governed Document Verification & Forensic Localization Engine
Provides document-level authenticity decisioning, part localization, and distortion diagnosis.
"""

try:
    from .document_verifier import DocumentVerifier
except Exception as e:
    DocumentVerifier = None

try:
    from .visualizer import ForensicVisualizer
except Exception as e:
    ForensicVisualizer = None

from .document_classifier import ZeroTrainingDocumentClassifier
from .entity_cross_validator import EntityCrossValidator
from .merchant_verifier import MerchantOnboardingVerifier

__all__ = [
    "DocumentVerifier",
    "ForensicVisualizer",
    "ZeroTrainingDocumentClassifier",
    "EntityCrossValidator",
    "MerchantOnboardingVerifier",
]
