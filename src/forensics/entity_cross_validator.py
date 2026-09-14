import re
from difflib import SequenceMatcher


def fuzzy_string_similarity(str1, str2):
    """Computes similarity ratio (0.0 to 1.0) between two names."""
    if not str1 or not str2:
        return 0.0
    
    s1 = re.sub(r'[^A-Z0-9\s]', '', str1.upper()).strip()
    s2 = re.sub(r'[^A-Z0-9\s]', '', str2.upper()).strip()
    
    if s1 == s2:
        return 1.0

    tokens1 = sorted(s1.split())
    tokens2 = sorted(s2.split())
    
    s1_sorted = " ".join(tokens1)
    s2_sorted = " ".join(tokens2)
    
    return SequenceMatcher(None, s1_sorted, s2_sorted).ratio()


class EntityCrossValidator:
    """
    Cross-Document Identity & Government Specification Rule Validator.
    """

    @staticmethod
    def validate_pan_entity_type(pan_number, entity_type):
        if not pan_number or len(pan_number) != 10:
            return {
                "valid": False,
                "reason": f"Invalid PAN format: '{pan_number}'."
            }

        pan_upper = pan_number.upper()
        fourth_char = pan_upper[3]

        allowed_map = {
            "proprietorship": ["P"],
            "partnership_llp": ["F", "L"],
            "pvt_ltd": ["C"],
            "trust_society": ["T", "A", "B", "G", "J"],
            "small_merchant": ["P", "F", "C", "L"]
        }

        entity_key = entity_type.lower().strip()
        allowed_chars = allowed_map.get(entity_key, ["P", "F", "C", "T", "A", "B", "L"])

        if fourth_char in allowed_chars:
            return {
                "valid": True,
                "fourth_char": fourth_char,
                "reason": f"PAN 4th character '{fourth_char}' correctly matches expected entity type '{entity_type}'."
            }
        else:
            return {
                "valid": False,
                "fourth_char": fourth_char,
                "reason": f"PAN entity mismatch! 4th character '{fourth_char}' is invalid for '{entity_type}' (expected one of {allowed_chars})."
            }

    @staticmethod
    def validate_gstin_pan_crossmatch(gstin_number, pan_number):
        if not gstin_number or len(gstin_number) != 15:
            return {
                "valid": False,
                "reason": f"Invalid GSTIN format: '{gstin_number}'."
            }
        if not pan_number or len(pan_number) != 10:
            return {
                "valid": False,
                "reason": f"Invalid PAN format: '{pan_number}'."
            }

        embedded_pan = gstin_number[2:12].upper()
        target_pan = pan_number.upper()

        if embedded_pan == target_pan:
            return {
                "valid": True,
                "embedded_pan": embedded_pan,
                "reason": f"GSTIN embedded PAN '{embedded_pan}' perfectly matches Entity PAN '{target_pan}'."
            }
        else:
            return {
                "valid": False,
                "embedded_pan": embedded_pan,
                "target_pan": target_pan,
                "reason": f"GSTIN embedded PAN mismatch! GSTIN contains '{embedded_pan}' but provided PAN is '{target_pan}'."
            }

    @staticmethod
    def validate_name_cross_consistency(extracted_names, threshold=0.85):
        results = []
        valid_items = [(doc, name) for doc, name in extracted_names.items() if name]

        if len(valid_items) < 2:
            return {
                "valid": True,
                "similarity_score": 1.0,
                "matches": [],
                "reason": "Insufficient document names provided for cross-matching."
            }

        all_passed = True
        scores = []

        for i in range(len(valid_items)):
            for j in range(i + 1, len(valid_items)):
                doc1, name1 = valid_items[i]
                doc2, name2 = valid_items[j]

                sim = fuzzy_string_similarity(name1, name2)
                scores.append(sim)
                passed = sim >= threshold

                if not passed:
                    all_passed = False

                results.append({
                    "doc1": doc1,
                    "name1": name1,
                    "doc2": doc2,
                    "name2": name2,
                    "similarity": round(sim, 4),
                    "passed": passed
                })

        avg_score = round(sum(scores) / len(scores), 4) if scores else 1.0

        return {
            "valid": all_passed,
            "average_similarity": avg_score,
            "comparisons": results,
            "reason": "All document names match consistently." if all_passed else f"Name discrepancy detected across documents (avg similarity: {avg_score*100:.1f}%)."
        }
