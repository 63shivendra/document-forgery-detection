# 📊 REPORTEDGE: Standalone Governed Edge Architecture & Implementation Plan

**Module Name:** Standalone Governed Edge & Mobile Verification Engine (`edgedevice/`)  
**Target Execution Profile:** Mobile Devices (Android/iOS), Edge Gateways, On-Device Microservices  
**Core Dependencies:** `onnxruntime` (INT8), `opencv-python-headless`, `zxingcpp`, `pyyaml`, `numpy`  

---

## 📑 Technical Implementation Roadmap

### Phase 1: Model Compression & INT8 Mobile Export
* **File:** `export_mobile_model.py`
* **Input Checkpoint:** `bigpower/best_model_splicing_combined_efficientnet-b2_20260908_1523.onnx` or `bigpower/best_model_indian_kyc_enhanced_20260917_1601.pth`
* **Quantization Method:** ONNX Runtime `quantize_dynamic` (FP32 $\rightarrow$ INT8 symmetric per-channel quantization).
* **Target Output:** `edgedevice/models/model_forgery_int8.onnx`
* **Target Metrics:**
  - Binary file size: **~10–12 MB** (reduced from 38.5 MB)
  - Inference Latency: **< 25 ms** on mobile NPU / CPU
  - RAM Footprint: **< 30 MB** (reduced from 3.5 GB PyTorch)

---

### Phase 2: Privacy Firewall & PII Redaction Engine
* **File:** `privacy/pii_masker.py`
* **Implementation Features:**
  1. **Aadhaar 8-Digit Masking**: Replaces first 8 digits of UID with 'X' (`XXXX-XXXX-9012`).
  2. **PAN Number Masking**: Obfuscates name characters (`XXXXX1234F`).
  3. **Visual Crop Anonymization**: Applies Gaussian blur ($k=21$) over cardholder face photo and signature strip.
  4. **SHA-256 Hashing & RAM Purge**: Computes `sha256(raw_bytes)` and deallocates raw image memory.

---

### Phase 3: The 5 Governed Agent Modules
* **Directory:** `agents/`

| Agent Module | Primary File | Responsibilities | Key Technologies |
| :--- | :--- | :--- | :--- |
| **Extraction Agent** | `agents/extraction_agent.py` | Document type classification (16 types), QR decoding, MICR/IFSC syntax, Verhoeff $D_5$ Aadhaar checksum. | `zxingcpp`, OpenCV, Math $D_5$ |
| **Forensic Agent** | `agents/forensic_agent.py` | **Wraps `bigpower/` INT8 ONNX Model**. Calculates pixel forgery heatmap, thresholding (0.35), bounding box callouts. | `onnxruntime` INT8 |
| **Privacy Agent** | `agents/privacy_agent.py` | Intercepts payload, masks Aadhaar/PAN PII, blurs faces, hashes file with SHA-256, purges raw RAM. | SHA-256, OpenCV Blur |
| **Validation Agent** | `agents/validation_agent.py` | Validates PAN 4th character entity rules (`P`, `C`, `F`, `L`, `T`), GSTIN embedded PAN digits, fuzzy name matching ($\ge 85\%$). | Regex, String Matching |
| **Reasoning Agent** | `agents/reasoning_agent.py` | Links forensic bounding boxes and rule outputs to NPCI **Reason Codes**; triggers LLM bridge for human-readable summaries. | Reason Code Linker |

---

### Phase 4: Governed LLM / SLM Explanation Bridge
* **File:** `llm_bridge/reasoning_llm.py`
* **Functionality:**
  - Consumes sanitized, PII-masked JSON outputs from Agents 1–4.
  - Constructs structured prompt for Google Gemini API or local Gemma/Llama SLM.
  - Generates natural, human-readable executive audit explanations.
  - Formats and emits official **NPCI Document Trust Receipt JSON**.

---

### Phase 5: Standalone Single-Entrypoint CLI & Python API
* **File:** `run_edge_audit.py`
* **CLI Command:**
  ```bash
  python edgedevice/run_edge_audit.py --input "path/to/sample.jpg" --entity-type proprietorship
  ```
* **Sample Standardized Output Payload (NPCI Document Trust Receipt):**
  ```json
  {
    "document_hash": "a8f5f167f44f4964e6c998dee827110c",
    "timestamp": "2026-09-20T19:20:00Z",
    "agent_identity": "NPCI_EDGE_VERIFIER_V1",
    "policy_version": "v2.4.1",
    "decision": "REJECTED",
    "risk_score": 0.824,
    "confidence_level": 0.978,
    "evidence_sufficiency": true,
    "escalation_required": false,
    "reason_codes": ["REASON_DOB_SPLICING_EDIT"],
    "pii_masked_payload": {
      "doc_type": "AADHAAR_CARD",
      "uid_masked": "XXXX-XXXX-1098",
      "verhoeff_valid": true
    },
    "executive_summary": "Document rejected due to visual font tampering in the Date of Birth field (Risk: 82.4%). Verhoeff UID checksum passed. PII has been masked per UIDAI regulations."
  }
  ```

---

## 🎯 Verification & Deliverables Checklist

- [x] Create standalone folder `edgedevice/` with subdirectories `models/`, `agents/`, `privacy/`, `llm_bridge/`.
- [x] Create `CONTEXT.md` and `REPORTEDGE.md` detailing architecture & implementation plan.
- [ ] Export INT8 Quantized Mobile ONNX Engine to `edgedevice/models/model_forgery_int8.onnx`.
- [ ] Build `privacy/pii_masker.py` PII redaction firewall.
- [ ] Implement 5 Governed Agents in `agents/`.
- [ ] Build `llm_bridge/reasoning_llm.py` Gemini/SLM explanation bridge.
- [ ] Build `run_edge_audit.py` standalone CLI & API launcher.
