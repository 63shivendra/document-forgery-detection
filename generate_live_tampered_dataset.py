import os
import glob
import json
import random
import cv2
import numpy as np


def apply_splicing_forgery(target_img, source_img):
    """
    Splicing Forgery: Pastes a patch from source_img onto target_img.
    Returns: (tampered_image, binary_mask)
    """
    h_t, w_t = target_img.shape[:2]
    h_s, w_s = source_img.shape[:2]

    # Crop random patch from source
    pw = random.randint(min(50, w_s // 4), min(200, w_s // 2))
    ph = random.randint(min(30, h_s // 4), min(120, h_s // 2))
    sx = random.randint(0, max(0, w_s - pw))
    sy = random.randint(0, max(0, h_s - ph))
    patch = source_img[sy:sy + ph, sx:sx + pw]

    # Target position
    tx = random.randint(0, max(0, w_t - pw))
    ty = random.randint(0, max(0, h_t - ph))

    tampered = target_img.copy()
    mask = np.zeros((h_t, w_t), dtype=np.uint8)

    tampered[ty:ty + ph, tx:tx + pw] = patch
    mask[ty:ty + ph, tx:tx + pw] = 255

    return tampered, mask


def apply_inpainting_erasure(img):
    """
    Inpainting / White-out / Content Erasure Forgery.
    Fills a text field region with surrounding background color/blur.
    Returns: (tampered_image, binary_mask)
    """
    h, w = img.shape[:2]
    pw = random.randint(min(60, w // 4), min(250, w // 2))
    ph = random.randint(min(25, h // 8), min(80, h // 4))
    tx = random.randint(0, max(0, w - pw))
    ty = random.randint(0, max(0, h - ph))

    tampered = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    # White-out / Background color fill or Telea Inpainting
    patch_area = tampered[ty:ty + ph, tx:tx + pw]
    avg_color = np.mean(patch_area, axis=(0, 1)).astype(np.uint8)
    
    # 50% chance white-out, 50% chance cv2.inpaint
    if random.random() > 0.5:
        tampered[ty:ty + ph, tx:tx + pw] = avg_color
    else:
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[ty:ty + ph, tx:tx + pw] = 255
        tampered = cv2.inpaint(tampered, inpaint_mask, 3, cv2.INPAINT_TELEA)

    mask[ty:ty + ph, tx:tx + pw] = 255
    return tampered, mask


def apply_copy_move_forgery(img):
    """
    Copy-Move Forgery: Copies a region within the same document and pastes it elsewhere.
    Returns: (tampered_image, binary_mask)
    """
    h, w = img.shape[:2]
    pw = random.randint(min(40, w // 5), min(180, w // 3))
    ph = random.randint(min(20, h // 6), min(90, h // 3))

    sx = random.randint(0, max(0, w - pw))
    sy = random.randint(0, max(0, h - ph))
    patch = img[sy:sy + ph, sx:sx + pw].copy()

    # Destination position
    tx = random.randint(0, max(0, w - pw))
    ty = random.randint(0, max(0, h - ph))

    tampered = img.copy()
    mask = np.zeros((h, w), dtype=np.uint8)

    tampered[ty:ty + ph, tx:tx + pw] = patch
    mask[ty:ty + ph, tx:tx + pw] = 255

    return tampered, mask


def generate_live_testing_dataset(
    source_dir=r"E:\project final\panadhargst\mix",
    output_dir="testing_live",
    num_tampered_samples=60,
    num_authentic_samples=20
):
    """
    Generates a live testing benchmark dataset in `testing_live/` by applying:
    1. Splicing Forgery
    2. Inpainting / Content Erasure
    3. Copy-Move Forgery
    """
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found.")
        return

    images_out = os.path.join(output_dir, "images")
    masks_out = os.path.join(output_dir, "masks")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(masks_out, exist_ok=True)

    valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    source_files = []
    for ext in valid_exts:
        source_files.extend(glob.glob(os.path.join(source_dir, ext)))

    # Filter out report images
    source_files = [f for f in source_files if not os.path.basename(f).startswith("report_")]
    if len(source_files) < 2:
        print(f"Error: Not enough images in {source_dir} to perform dataset generation.")
        return

    print("==========================================================================")
    print("  Generating Synthetic Live Tampered Benchmark Dataset")
    print(f"  Source Folder : {source_dir} ({len(source_files)} available files)")
    print(f"  Output Folder : {output_dir}")
    print("==========================================================================")

    manifest = []
    sample_id = 1

    # 1. Generate Authentic Samples
    random.shuffle(source_files)
    authentic_sources = source_files[:min(num_authentic_samples, len(source_files))]

    print(f"\n[1/2] Processing {len(authentic_sources)} Authentic Baseline Samples...")
    for src_path in authentic_sources:
        img = cv2.imread(src_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_name = f"live_sample_{sample_id:04d}_authentic.jpg"
        mask_name = f"live_sample_{sample_id:04d}_authentic_mask.png"

        save_img_path = os.path.join(images_out, img_name)
        save_mask_path = os.path.join(masks_out, mask_name)

        mask = np.zeros((h, w), dtype=np.uint8)

        cv2.imwrite(save_img_path, img)
        cv2.imwrite(save_mask_path, mask)

        manifest.append({
            "sample_id": sample_id,
            "image_filename": img_name,
            "mask_filename": mask_name,
            "is_tampered": False,
            "forgery_type": "AUTHENTIC",
            "original_source": os.path.basename(src_path)
        })
        sample_id += 1

    # 2. Generate Tampered Samples
    print(f"\n[2/2] Generating {num_tampered_samples} Synthetic Tampered Samples (Splicing, Inpainting, Copy-Move)...")
    for i in range(num_tampered_samples):
        src_path = random.choice(source_files)
        target_img = cv2.imread(src_path)
        if target_img is None:
            continue

        forgery_choice = random.choice(["SPLICING", "INPAINTING_ERASURE", "COPY_MOVE"])

        if forgery_choice == "SPLICING":
            # Pick a second distinct image as source
            donor_path = random.choice(source_files)
            while donor_path == src_path and len(source_files) > 1:
                donor_path = random.choice(source_files)
            donor_img = cv2.imread(donor_path)
            if donor_img is None:
                donor_img = target_img.copy()
            tampered_img, mask = apply_splicing_forgery(target_img, donor_img)

        elif forgery_choice == "INPAINTING_ERASURE":
            tampered_img, mask = apply_inpainting_erasure(target_img)

        else:  # COPY_MOVE
            tampered_img, mask = apply_copy_move_forgery(target_img)

        img_name = f"live_sample_{sample_id:04d}_tampered_{forgery_choice.lower()}.jpg"
        mask_name = f"live_sample_{sample_id:04d}_tampered_{forgery_choice.lower()}_mask.png"

        save_img_path = os.path.join(images_out, img_name)
        save_mask_path = os.path.join(masks_out, mask_name)

        cv2.imwrite(save_img_path, tampered_img)
        cv2.imwrite(save_mask_path, mask)

        manifest.append({
            "sample_id": sample_id,
            "image_filename": img_name,
            "mask_filename": mask_name,
            "is_tampered": True,
            "forgery_type": forgery_choice,
            "original_source": os.path.basename(src_path)
        })
        sample_id += 1

    # Save Manifest JSON
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(manifest),
            "authentic_samples": len(authentic_sources),
            "tampered_samples": num_tampered_samples,
            "dataset_manifest": manifest
        }, f, indent=2)

    print("\n==========================================================================")
    print(f"  Live Tampered Testing Dataset Created Successfully!")
    print(f"  Total Samples Generated : {len(manifest)}")
    print(f"  Image Directory         : {images_out}")
    print(f"  Mask Directory          : {masks_out}")
    print(f"  Manifest File           : {manifest_path}")
    print("==========================================================================")


if __name__ == "__main__":
    generate_live_testing_dataset()
