# 📄 CONTEXT: Standalone Governed Edge & Mobile Verification Framework

Welcome to the **Edge & Mobile Verification Engine** (`edgedevice/`), a standalone, privacy-preserving, and lightweight execution framework for **Governed Merchant Onboarding Document Verification and Forensics**.

---

## 🎯 Executive Summary & Philosophy

This framework implements **NPCI's Governed Merchant Onboarding Mandate**:
1. **Edge-First / Device-First Processing**: Processing executes on-device (mobile app, edge gateway, microservice). Raw document images remain strictly local in volatile RAM. Zero raw images are sent upstream.
2. **Strict PII Compliance**: Automatically redacts sensitive Personal Identifiable Information (Aadhaar 8-digit masking `XXXX-XXXX-9012`, PAN obfuscation `XXXXX1234F`, face photo/signature blurring) before emitting output.
3. **Synergy with `bigpower/` Model**: Wraps around the trained deep learning model in `bigpower/` (`best_model_splicing_combined_efficientnet-b2_...onnx`) converted to an **INT8 Quantized Mobile ONNX Engine** (~10 MB, <25 ms latency, zero PyTorch dependency).
4. **Evidence-Backed NPCI Document Trust Receipt**: Produces standardized, explainable JSON receipts with SHA-256 document hashes, risk scores, suspicious field bounding boxes, NPCI reason codes, and natural LLM reasoning summaries.

---

## 📂 Directory Structure of `edgedevice/`

```
edgedevice/
├── CONTEXT.md                    # Detailed operational context & architecture guide (this file)
├── REPORTEDGE.md                 # Technical implementation plan & benchmark specification
├── README.md                     # Integration overview & mobile usage manual
├── requirements_edge.txt         # Lightweight runtime requirements (onnxruntime, opencv, zxingcpp) - ZERO PyTorch!
├── export_mobile_model.py        # Quantizes bigpower/ PyTorch checkpoint -> INT8 Mobile ONNX Engine
├── run_edge_audit.py             # Single CLI entrypoint & Python API for standalone edge verification
├── models/
│   └── model_forgery_int8.onnx   # ~10 MB INT8 Quantized Mobile Forgery Engine
├── privacy/
│   └── pii_masker.py             # Privacy Firewall: Aadhaar 8-digit, PAN, & visual PII anonymizer
├── agents/
│   ├── extraction_agent.py       # Agent 1: QR Decoding, MICR Regex, Verhoeff D5 Checksum
│   ├── forensic_agent.py         # Agent 2: INT8 ONNX Forgery Segmentation (Wraps bigpower model)
│   ├── privacy_agent.py          # Agent 3: PII Masking, Face Blur, SHA-256 Hashing, RAM Purge
│   ├── validation_agent.py       # Agent 4: Entity Rules (PAN 4th char, GSTIN match, Fuzzy Name)
│   └── reasoning_agent.py        # Agent 5: Evidence Linker & Trust Receipt Generator
└── llm_bridge/
    └── reasoning_llm.py          # Governed LLM / SLM Explanation Bridge (Gemini / Gemma / Local SLM)
```

---

## 🏛️ The 5-Agent Governed Architecture

```
                       Raw Document Upload (Mobile RAM)
                                      │
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ 1. Document Extraction Agent (extraction_agent.py)             │
     │    • Zero-shot 16 document type classification                 │
     │    • Native QR decoding (UIDAI Aadhaar XML, NSDL PAN, GST)     │
     │    • Verhoeff D5 Dihedral Checksum & MICR / IFSC verification │
     └────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ 2. Edge Forensic Detection Agent (forensic_agent.py)           │
     │    ⚡ WRAPS DIRECTLY AROUND YOUR bigpower/ MODEL ⚡             │
     │    • Executes INT8 Quantized ONNX Model (<25 ms on NPU/CPU)   │
     │    • Outputs bounding box callouts & forgery risk score (0.85) │
     └────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ 3. Privacy & PII Redaction Agent (privacy_agent.py)            │
     │    🛡️ THE PRIVACY FIREWALL 🛡️                                 │
     │    • Masks Aadhaar 8 digits (XXXX-XXXX-9012) & PAN (XXXXX1234F)│
     │    • Blurs cardholder face photo & signature strip in crops    │
     │    • Computes SHA-256 document hash & purges raw RAM image     │
     └────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ 4. Entity Cross-Validation Agent (validation_agent.py)         │
     │    • Validates PAN 4th char (P=Proprietor, C=Company, F=Firm) │
     │    • Cross-matches GSTIN embedded PAN digits 3–12              │
     │    • Computes multi-document fuzzy name matching (>=85%)       │
     └────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
     ┌────────────────────────────────────────────────────────────────┐
     │ 5. Governed LLM Reasoning Agent (reasoning_agent.py)           │
     │    • Links signals to reason codes (REASON_DIGIT_EDIT, etc.)   │
     │    • Synthesizes human-readable LLM explanation summary        │
     │    • Emits standardized NPCI Document Trust Receipt JSON       │
     └────────────────────────────────────────────────────────────────┘
```

---

## 🔒 PII Compliance & Security Guarantees

1. **In-Memory Volatile Processing**: Document pixels are processed purely in temporary RAM.
2. **SHA-256 Cryptographic Hash**: Uniquely identifies original uploads without storing raw pixel files.
3. **Automated RAM Purge**: Once the Trust Receipt is signed, raw image memory is immediately zeroed and deallocated.
4. **UIDAI / RBI Compliance**: Ensures full adherence to UIDAI Aadhaar Circulars and RBI Master Directions.

---

## 🛠️ Threshold Calibration & False Alarm Mitigation

To eliminate false alarms on clean authentic customer documents (which contain high-contrast photo frames, micro-print borders, and scanner glare):

1. **Minimum Tamper Region Size (`min_area = 200` pixels)**:
   - Filters out tiny 5x5 pixel edge noise (<100 pixels) on clean cards.
   - Real forged text or photo swaps (>1,000 pixels) remain 100% captured.
2. **Fraud Confidence Cutoff (`threshold = 0.50`)**:
   - Requires $\ge 50\%$ neural probability before flagging visual tampering.
3. **Connected Component Region Risk Calculation**:
   - Computes risk strictly across connected regions $\ge 200$ pixels, suppressing background noise on authentic cards so clean documents receive **`APPROVED`** verdicts.
