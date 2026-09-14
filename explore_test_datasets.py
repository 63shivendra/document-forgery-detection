import os
import sys
import json
import cv2
import numpy as np
import torch
from PIL import Image

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.dataset.base_dataset import get_dataset


def extract_mask_components(mask_uint8):
    """Calculates ground truth tampered parts, bounding boxes, and area from mask."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    parts = []
    total_tampered_pixels = 0

    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area >= 5:  # filter noise
            total_tampered_pixels += area
            x = stats[lid, cv2.CC_STAT_LEFT]
            y = stats[lid, cv2.CC_STAT_TOP]
            w = stats[lid, cv2.CC_STAT_WIDTH]
            h = stats[lid, cv2.CC_STAT_HEIGHT]
            parts.append({
                "part_id": len(parts) + 1,
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area_pixels": int(area),
                "aspect_ratio": round(float(w) / float(max(1, h)), 2)
            })

    total_px = mask_uint8.size
    tampered_pct = round((total_tampered_pixels / float(total_px)) * 100.0, 3)

    return {
        "tampered_pixels": int(total_tampered_pixels),
        "tampered_percentage": float(tampered_pct),
        "is_tampered": bool(len(parts) > 0 and total_tampered_pixels >= 15),
        "total_parts": int(len(parts)),
        "parts": parts
    }


def denormalize_image(tensor):
    """Denormalizes PyTorch tensor from ImageNet stats back to [0, 255] RGB numpy array."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    arr = tensor.cpu().numpy() * std + mean
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))  # H, W, C
    return arr


def explore_all_unseen_test_datasets():
    datasets_to_explore = [
        ("doctamper", "Official DocTamper Text Tampering Test Set", "data"),
        ("doctamper_scd", "DocTamper Single-Character Digit Tampering Test Set", "data"),
        ("cocoglide", "COCOGLIDE / TruFor Splicing & AI Inpainting Benchmark", "data"),
        ("splicing", "Splicing & Copy-Move Forgery Dataset", "data"),
        ("casia", "CASIA 2.0 Tampered & Authentic Test Benchmark", "data"),
        ("casia1", "CASIA 1.0 Tampered & Copy-Move Test Benchmark", "test_datasets"),
    ]

    print("\n" + "=" * 78)
    print("      DEEP EXPLORATION & AUDIT OF UNSEEN TEST DATASETS AND MASKS")
    print("=" * 78)

    inventory = {}
    visual_rows = []

    os.makedirs("reports/forensics", exist_ok=True)

    for name, title, root in datasets_to_explore:
        print(f"\n[{name.upper()}] - {title}")
        print("-" * 78)

        try:
            ds = get_dataset(name, data_root=root, split="test", img_size=(384, 384))
            total_samples = len(ds)
            print(f"  * Available Test Samples in Split : {total_samples}")

            if total_samples == 0:
                print("  [Warning] No test samples found in this split.")
                continue

            # Analyze first 5 samples
            sample_analyses = []
            num_to_inspect = min(5, total_samples)

            for i in range(num_to_inspect):
                img_tensor, mask_tensor, sample_name = ds[i]
                mask_np = mask_tensor.squeeze().cpu().numpy()
                mask_bin = (mask_np > 0.5).astype(np.uint8)

                meta = extract_mask_components(mask_bin)
                meta["sample_name"] = sample_name
                meta["resolution"] = list(mask_bin.shape)
                sample_analyses.append(meta)

                if i == 0:
                    # Render preview row for sample 0
                    rgb_img = denormalize_image(img_tensor)
                    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    # Thermal Mask
                    mask_vis = (mask_bin * 255).astype(np.uint8)
                    mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)

                    # Overlay with GT BBoxes
                    overlay = bgr_img.copy()
                    for p in meta["parts"]:
                        x1, y1, x2, y2 = p["bbox"]
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(overlay, f"Part {p['part_id']}", (x1, max(15, y1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

                    # Header banners
                    target_w = 384
                    target_h = 384
                    p1 = cv2.resize(bgr_img, (target_w, target_h))
                    p2 = cv2.resize(mask_color, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    p3 = cv2.resize(overlay, (target_w, target_h))

                    cv2.putText(p1, f"{name.upper()} Original", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(p2, f"GT Mask: {meta['tampered_percentage']}% Tampered", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(p3, f"GT BBoxes: {meta['total_parts']} Parts", (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

                    row = np.hstack([p1, p2, p3])
                    visual_rows.append(row)

            # Print stats
            print(f"  * Inspected {num_to_inspect} samples:")
            for s in sample_analyses:
                verdict_str = "TAMPERED" if s["is_tampered"] else "AUTHENTIC"
                print(f"    - {s['sample_name']:<30} | {verdict_str:<10} | Tampered: {s['tampered_percentage']:>6.2f}% ({s['tampered_pixels']} px) | Parts: {s['total_parts']}")
                for part in s["parts"][:2]:
                    print(f"        Part {part['part_id']}: BBox={part['bbox']}, Area={part['area_pixels']}px, AspectRatio={part['aspect_ratio']}")

            inventory[name] = {
                "dataset_title": title,
                "total_test_samples": total_samples,
                "inspected_samples": sample_analyses
            }

        except Exception as e:
            print(f"  [Error] Failed to load {name}: {e}")

    # Save visual comparison
    if visual_rows:
        composite = np.vstack(visual_rows)
        preview_path = "reports/forensics/unseen_test_dataset_exploration.jpg"
        cv2.imwrite(preview_path, composite)
        print("\n" + "=" * 78)
        print(f"[+] Saved Visual Ground-Truth Evidence Board to: {preview_path}")

    # Save JSON Inventory
    json_path = "reports/forensics/test_dataset_inventory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)
    print(f"[+] Saved Complete Dataset & Mask Inventory to  : {json_path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    explore_all_unseen_test_datasets()
