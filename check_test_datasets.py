import os
import glob

def check():
    print("=== DEEP INSPECTION OF test_datasets ===")
    for folder in os.listdir('test_datasets'):
        fp = os.path.join('test_datasets', folder)
        if os.path.isdir(fp):
            all_files = glob.glob(os.path.join(fp, '**', '*.*'), recursive=True)
            imgs = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif', '.bmp'))]
            non_imgs = [f for f in all_files if f not in imgs]
            print(f"\nFolder: {folder}")
            print(f"  Total items: {len(all_files)}")
            print(f"  Image/Mask files: {len(imgs)}")
            for ni in non_imgs:
                print(f"  Non-image file: {os.path.basename(ni)} ({os.path.getsize(ni)/1024:.2f} KB)")
                # check if zip
                if ni.endswith('.zip'):
                    import zipfile
                    try:
                        with zipfile.ZipFile(ni, 'r') as z:
                            print(f"    -> Valid ZIP file! Entries: {len(z.namelist())}")
                    except Exception as e:
                        print(f"    -> NOT a valid zip file ({e})")

    # Check CASIA 1.0 specifically
    print("\n--- CASIA 1.0 Detailed Analysis ---")
    casia1_imgs = glob.glob('test_datasets/CASIA1.0/**/*.png', recursive=True)
    print(f"Found {len(casia1_imgs)} PNG files in CASIA1.0.")
    print("Sample filenames:")
    for f in casia1_imgs[:5]:
        print(" ", os.path.relpath(f, 'test_datasets/CASIA1.0'))

    # Check if they are all masks or if any are RGB images
    from PIL import Image
    is_mask = []
    for f in casia1_imgs[:20]:
        im = Image.open(f)
        is_mask.append(im.mode)
    print(f"Image modes of sample files: {set(is_mask)} (L/1 = binary mask, RGB = photo)")

    print("\n--- CASIA 1.0 Masks Analysis ---")
    mask_files = glob.glob('test_datasets/CASIA1.0/masks/**/*.*', recursive=True)
    print(f"Total files in test_datasets/CASIA1.0/masks: {len(mask_files)}")
    for f in mask_files[:10]:
        print(" ", os.path.relpath(f, 'test_datasets/CASIA1.0/masks'))

if __name__ == '__main__':
    check()
