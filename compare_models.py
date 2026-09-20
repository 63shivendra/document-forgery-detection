import json
import numpy as np

with open(r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\live_evaluation_report.json", encoding="utf-8") as f:
    d_new = json.load(f)
with open(r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\live_evaluation_old_model.json", encoding="utf-8") as f:
    d_old = json.load(f)

auth_new = [s["predicted_fraud_score"] for s in d_new["sample_details"] if not s["ground_truth_tampered"]]
auth_old = [s["predicted_fraud_score"] for s in d_old["sample_details"] if not s["ground_truth_tampered"]]

fake_new = [s["predicted_fraud_score"] for s in d_new["sample_details"] if s["ground_truth_tampered"]]
fake_old = [s["predicted_fraud_score"] for s in d_old["sample_details"] if s["ground_truth_tampered"]]

print("=" * 60)
print("  MODEL COMPARISON ON LIVE TESTING DATASET (80 SAMPLES)")
print("=" * 60)
print(f"AUTHENTIC SAMPLES (20 items):")
print(f"  Old Base Model  : Mean Fraud Score = {np.mean(auth_old):.4f} | Min = {np.min(auth_old):.4f} | Max = {np.max(auth_old):.4f}")
print(f"  New KYC Model   : Mean Fraud Score = {np.mean(auth_new):.4f} | Min = {np.min(auth_new):.4f} | Max = {np.max(auth_new):.4f}")

print("\nTAMPERED SAMPLES (60 items):")
print(f"  Old Base Model  : Mean Fraud Score = {np.mean(fake_old):.4f} | Min = {np.min(fake_old):.4f} | Max = {np.max(fake_old):.4f}")
print(f"  New KYC Model   : Mean Fraud Score = {np.mean(fake_new):.4f} | Min = {np.min(fake_new):.4f} | Max = {np.max(fake_new):.4f}")
print("=" * 60)
