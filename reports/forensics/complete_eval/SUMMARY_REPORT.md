# Comprehensive Forensic Verification Benchmark Report

- **Execution Time**: 12.49s
- **Total Datasets Evaluated**: 5

| Dataset | Samples | Verdict Accuracy | Mean Recall | Mean Pixel IoU | Mean Dice F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Splicing & Copy-Move Forgery Dataset** | 21 | **100.0%** | 95.02% | 19.54% | 25.36% |
| **CASIA 1.0 Tampered Benchmark** | 36 | **100.0%** | 66.29% | 5.95% | 10.77% |
| **COCOGLIDE / TruFor Splicing & Inpainting** | 36 | **100.0%** | 92.89% | 20.37% | 30.07% |
| **DocTamper Document Text Tampering** | 36 | **61.11%** | 62.71% | 37.27% | 44.97% |
| **DocTamper Single Character / Digit Tampering** | 36 | **97.22%** | 37.24% | 16.89% | 25.23% |

## Output Directories

- **SPLICING**: [JSON Report](reports/forensics/complete_eval\splicing\evaluation_report.json) | [Comparison Boards](reports/forensics/complete_eval\splicing\comparison_boards)
- **CASIA1**: [JSON Report](reports/forensics/complete_eval\casia1\evaluation_report.json) | [Comparison Boards](reports/forensics/complete_eval\casia1\comparison_boards)
- **COCOGLIDE**: [JSON Report](reports/forensics/complete_eval\cocoglide\evaluation_report.json) | [Comparison Boards](reports/forensics/complete_eval\cocoglide\comparison_boards)
- **DOCTAMPER**: [JSON Report](reports/forensics/complete_eval\doctamper\evaluation_report.json) | [Comparison Boards](reports/forensics/complete_eval\doctamper\comparison_boards)
- **DOCTAMPER_SCD**: [JSON Report](reports/forensics/complete_eval\doctamper_scd\evaluation_report.json) | [Comparison Boards](reports/forensics/complete_eval\doctamper_scd\comparison_boards)
