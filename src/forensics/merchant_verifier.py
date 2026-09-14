import os
import json
import numpy as np
from src.forensics.document_classifier import ZeroTrainingDocumentClassifier
from src.forensics.entity_cross_validator import EntityCrossValidator

try:
    from src.forensics.document_verifier import DocumentVerifier
    HAS_VERIFIER = True
except Exception:
    DocumentVerifier = None
    HAS_VERIFIER = False


MERCHANT_CHECKLISTS = {
    "proprietorship": {
        "entity_label": "Sole Proprietorship",
        "mandatory": [
            {"slot": "proprietor_pan", "allowed_types": ["PAN_CARD"]},
            {"slot": "proprietor_id_proof", "allowed_types": ["AADHAAR_CARD", "PASSPORT", "VOTER_ID", "DRIVING_LICENSE"]},
            {"slot": "business_proof", "allowed_types": ["SHOP_ESTABLISHMENT", "UDYAM_REGISTRATION", "GST_REG06"]},
            {"slot": "bank_proof", "allowed_types": ["CANCELLED_CHEQUE", "BANK_STATEMENT"]}
        ],
        "optional": [
            {"slot": "tax_registration", "allowed_types": ["GST_REG06"]}
        ]
    },
    "partnership_llp": {
        "entity_label": "Partnership Firm / LLP",
        "mandatory": [
            {"slot": "firm_deed", "allowed_types": ["PARTNERSHIP_DEED", "CERTIFICATE_OF_INCORPORATION"]},
            {"slot": "firm_pan", "allowed_types": ["PAN_CARD"]},
            {"slot": "signatory_kyc", "allowed_types": ["AADHAAR_CARD", "PAN_CARD", "PASSPORT", "VOTER_ID"]},
            {"slot": "bank_proof", "allowed_types": ["CANCELLED_CHEQUE", "BANK_STATEMENT"]},
            {"slot": "tax_registration", "allowed_types": ["GST_REG06"]}
        ],
        "optional": []
    },
    "pvt_ltd": {
        "entity_label": "Private / Public Limited Company",
        "mandatory": [
            {"slot": "coi", "allowed_types": ["CERTIFICATE_OF_INCORPORATION"]},
            {"slot": "moa_aoa", "allowed_types": ["MOA_AOA", "CERTIFICATE_OF_INCORPORATION"]},
            {"slot": "company_pan", "allowed_types": ["PAN_CARD"]},
            {"slot": "board_resolution", "allowed_types": ["BOARD_RESOLUTION"]},
            {"slot": "signatory_kyc", "allowed_types": ["AADHAAR_CARD", "PAN_CARD", "PASSPORT"]},
            {"slot": "bank_proof", "allowed_types": ["CANCELLED_CHEQUE", "BANK_STATEMENT"]},
            {"slot": "tax_registration", "allowed_types": ["GST_REG06"]}
        ],
        "optional": []
    },
    "trust_society": {
        "entity_label": "Trust / Society / NGO",
        "mandatory": [
            {"slot": "trust_deed", "allowed_types": ["TRUST_DEED"]},
            {"slot": "entity_pan", "allowed_types": ["PAN_CARD"]},
            {"slot": "signatory_kyc", "allowed_types": ["AADHAAR_CARD", "PAN_CARD", "PASSPORT"]},
            {"slot": "bank_proof", "allowed_types": ["CANCELLED_CHEQUE", "BANK_STATEMENT"]}
        ],
        "optional": []
    },
    "small_merchant": {
        "entity_label": "Small Merchant (Light KYC Route ≤ ₹40L)",
        "mandatory": [
            {"slot": "pan", "allowed_types": ["PAN_CARD"]},
            {"slot": "ovd_identity", "allowed_types": ["AADHAAR_CARD", "PASSPORT", "VOTER_ID", "DRIVING_LICENSE"]},
            {"slot": "business_kyc", "allowed_types": ["SHOP_ESTABLISHMENT", "UDYAM_REGISTRATION", "GST_REG06", "CERTIFICATE_OF_INCORPORATION", "PARTNERSHIP_DEED"]}
        ],
        "optional": []
    }
}


class MerchantOnboardingVerifier:
    def __init__(self, config_path="config.yaml"):
        self.classifier = ZeroTrainingDocumentClassifier()
        if HAS_VERIFIER and DocumentVerifier is not None:
            try:
                self.forgery_verifier = DocumentVerifier(config_path=config_path)
            except Exception:
                self.forgery_verifier = None
        else:
            self.forgery_verifier = None

    def verify_merchant_dossier(self, entity_type, document_slots):
        entity_key = entity_type.lower().strip()
        if entity_key not in MERCHANT_CHECKLISTS:
            raise ValueError(f"Invalid entity_type: '{entity_type}'. Must be one of {list(MERCHANT_CHECKLISTS.keys())}")

        checklist = MERCHANT_CHECKLISTS[entity_key]
        audit_results = {
            "entity_type": entity_key,
            "entity_label": checklist["entity_label"],
            "onboarding_verdict": "APPROVED",
            "risk_score": 0.0,
            "checklist_audit": [],
            "forgery_reports": [],
            "cross_validation_rules": [],
            "summary_flags": []
        }

        extracted_metadata = {}

        # 1. Checklist Audit & Classification
        for req in checklist["mandatory"]:
            slot = req["slot"]
            allowed = req["allowed_types"]

            if slot not in document_slots or not os.path.exists(document_slots[slot]):
                audit_results["checklist_audit"].append({
                    "slot": slot,
                    "status": "MISSING_MANDATORY_DOCUMENT",
                    "allowed_types": allowed,
                    "passed": False
                })
                audit_results["summary_flags"].append(f"Missing mandatory document for slot '{slot}'")
                audit_results["onboarding_verdict"] = "REJECTED"
                continue

            img_path = document_slots[slot]
            class_res = self.classifier.classify(img_path)
            doc_type = class_res["document_type"]

            is_correct_type = (doc_type in allowed)
            if not is_correct_type:
                audit_results["summary_flags"].append(f"Slot '{slot}': Uploaded document classified as '{doc_type}', expected one of {allowed}.")
                audit_results["onboarding_verdict"] = "FLAGGED_FOR_MANUAL_REVIEW"

            audit_results["checklist_audit"].append({
                "slot": slot,
                "file_path": img_path,
                "classified_as": doc_type,
                "confidence": class_res["confidence"],
                "classification_method": class_res["classification_method"],
                "allowed_types": allowed,
                "passed": is_correct_type
            })

            if class_res.get("metadata"):
                extracted_metadata.update(class_res["metadata"])

        # 2. Forgery Inspection
        total_fraud_score = 0.0
        num_docs = 0

        if self.forgery_verifier is not None:
            for slot, img_path in document_slots.items():
                if not os.path.exists(img_path):
                    continue

                forgery_res = self.forgery_verifier.verify(img_path)
                num_docs += 1
                total_fraud_score += forgery_res["fraud_score"]

                if forgery_res["is_fake"]:
                    audit_results["summary_flags"].append(
                        f"Forgery detected in document '{slot}' ({os.path.basename(img_path)}): Risk {forgery_res['risk_level']} [{forgery_res['fraud_score']*100:.1f}%]"
                    )
                    if forgery_res["risk_level"] in ["HIGH", "CRITICAL"]:
                        audit_results["onboarding_verdict"] = "REJECTED"
                    elif audit_results["onboarding_verdict"] != "REJECTED":
                        audit_results["onboarding_verdict"] = "FLAGGED_FOR_MANUAL_REVIEW"

                audit_results["forgery_reports"].append({
                    "slot": slot,
                    "file_name": os.path.basename(img_path),
                    "verdict": forgery_res["document_verdict"],
                    "is_fake": forgery_res["is_fake"],
                    "fraud_score": forgery_res["fraud_score"],
                    "risk_level": forgery_res["risk_level"],
                    "tampered_regions_count": forgery_res["total_tampered_regions"]
                })

        # 3. Government & Cross-Validation Rules
        pan_no = extracted_metadata.get("pan")
        if pan_no:
            pan_val = EntityCrossValidator.validate_pan_entity_type(pan_no, entity_key)
            audit_results["cross_validation_rules"].append(pan_val)
            if not pan_val["valid"]:
                audit_results["summary_flags"].append(pan_val["reason"])
                audit_results["onboarding_verdict"] = "REJECTED"

        gstin_no = extracted_metadata.get("gstin")
        if gstin_no and pan_no:
            gst_val = EntityCrossValidator.validate_gstin_pan_crossmatch(gstin_no, pan_no)
            audit_results["cross_validation_rules"].append(gst_val)
            if not gst_val["valid"]:
                audit_results["summary_flags"].append(gst_val["reason"])
                audit_results["onboarding_verdict"] = "REJECTED"

        avg_fraud = (total_fraud_score / float(num_docs)) if num_docs > 0 else 0.0
        audit_results["risk_score"] = round(avg_fraud, 4)

        return audit_results
