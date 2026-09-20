import os, json, glob, cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from src.models.forgery_net import ForgeryNet

def analyze_calibration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\bigpower\best_model_indian_kyc_enhanced_20260917_1601.pth"
    model = ForgeryNet(out_channels=1, encoder_name="efficientnet-b2").to(device)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()

    test_manifest = r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\manifest.json"
    with open(test_manifest, "r", encoding="utf-8") as f:
        m = json.load(f)

    samples = m["dataset_manifest"]
    auth_samples = [s for s in samples if not s["is_tampered"]][:10]
    tamp_samples = [s for s in samples if s["is_tampered"]][:10]

    def get_stats(img_path):
        pil_img = Image.open(img_path).convert("RGB").resize((384, 384), Image.BILINEAR)
        t = TF.normalize(TF.to_tensor(pil_img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
        
        # Pixels > 0.5, > 0.7, > 0.9
        p50 = (prob >= 0.50).sum()
        p70 = (prob >= 0.70).sum()
        p90 = (prob >= 0.90).sum()
        max_p = prob.max()
        mean_p = prob.mean()
        return max_p, mean_p, p50, p70, p90

    print(f"{'TYPE':12s} | {'FILE':35s} | {'MAX':6s} | {'MEAN':6s} | {'>0.50':7s} | {'>0.70':7s} | {'>0.90':7s}")
    print("-" * 85)
    for s in auth_samples:
        p = os.path.join(r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\images", s["image_filename"])
        mx, mn, c50, c70, c90 = get_stats(p)
        print(f"AUTHENTIC    | {s['image_filename'][:35]:35s} | {mx:6.3f} | {mn:6.3f} | {c50:7d} | {c70:7d} | {c90:7d}")

    print("-" * 85)
    for s in tamp_samples:
        p = os.path.join(r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina\testing_live\images", s["image_filename"])
        mx, mn, c50, c70, c90 = get_stats(p)
        print(f"TAMPERED     | {s['image_filename'][:35]:35s} | {mx:6.3f} | {mn:6.3f} | {c50:7d} | {c70:7d} | {c90:7d}")

if __name__ == "__main__":
    analyze_calibration()
