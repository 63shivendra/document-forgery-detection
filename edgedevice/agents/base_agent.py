"""
Base Agent Abstract Classes & Document Registry (SOLID Principles Architecture)
Follows Open/Closed Principle (OCP) - System is open for extension, closed for modification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List


class BaseAgent(ABC):
    """Abstract Base Class for all 5 Governed Agents."""
    
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming payload and return updated payload."""
        pass


class BaseDocumentParser(ABC):
    """
    Abstract Base Class for Document Parsers.
    To add a NEW document type (e.g. FSSAI License, Passport), simply subclass this interface
    and register it with DocumentRegistry without modifying existing core code.
    """
    
    @abstractmethod
    def document_type(self) -> str:
        """Returns unique document type identifier string (e.g. 'PAN_CARD', 'AADHAAR_CARD')."""
        pass

    @abstractmethod
    def match_score(self, text: str, qr_data: Optional[str] = None, aspect_ratio: float = 1.0) -> float:
        """Returns confidence score [0.0 - 1.0] that given document matches this document type."""
        pass

    @abstractmethod
    def parse_fields(self, text: str, qr_data: Optional[str] = None) -> Dict[str, Any]:
        """Parses document-specific fields into structured dictionary."""
        pass

    @abstractmethod
    def get_pii_fields(self) -> List[str]:
        """Returns list of field names that contain sensitive PII requiring masking."""
        pass


class DocumentRegistry:
    """
    Dynamic Document Registry following Open/Closed Principle.
    Allows registering new document parsers dynamically at runtime or import time.
    """
    _registry: Dict[str, BaseDocumentParser] = {}

    @classmethod
    def register(cls, parser: BaseDocumentParser):
        """Register a document parser instance."""
        doc_type = parser.document_type().upper()
        cls._registry[doc_type] = parser

    @classmethod
    def get_parser(cls, doc_type: str) -> Optional[BaseDocumentParser]:
        """Retrieve parser by document type."""
        return cls._registry.get(doc_type.upper())

    @classmethod
    def classify(cls, text: str, qr_data: Optional[str] = None, aspect_ratio: float = 1.0) -> Dict[str, Any]:
        """
        Classify document across all registered parsers zero-shot.
        Returns best matching document type and confidence score.
        """
        best_doc_type = "UNKNOWN_DOCUMENT"
        best_score = 0.0
        
        for doc_type, parser in cls._registry.items():
            score = parser.match_score(text, qr_data, aspect_ratio)
            if score > best_score:
                best_score = score
                best_doc_type = doc_type
                
        if best_score < 0.30:
            return {"document_type": "UNKNOWN_DOCUMENT", "confidence": best_score}
            
        return {"document_type": best_doc_type, "confidence": round(best_score, 3)}
