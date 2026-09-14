import os
import shutil
import kagglehub

def download():
    print("=" * 65)
    print("  Downloading: recodai-luc-scientific-image-forgery-detection")
    print("=" * 65)

    # Check if credentials are set
    has_creds = os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")) or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )

    if not has_creds:
        print("\n[Notice] Kaggle API credentials not found.")
        print("Please enter your Kaggle username and API key below.")
        print("(You can generate a key at: https://www.kaggle.com/settings -> 'Create New Token')\n")
        try:
            kagglehub.login()
        except Exception as e:
            print(f"[Error] Login failed: {e}")
            return

    target_dir = os.path.abspath("data/recodai_scientific")
    os.makedirs(target_dir, exist_ok=True)

    print("\nStarting download from Kaggle...")
    try:
        path = kagglehub.competition_download('recodai-luc-scientific-image-forgery-detection')
        print(f"\n[Success] Dataset downloaded to: {path}")

        # Summary of files
        print("\nContents of downloaded dataset:")
        for item in os.listdir(path):
            ip = os.path.join(path, item)
            if os.path.isdir(ip):
                count = len(os.listdir(ip))
                print(f"  [Folder] {item} ({count} items)")
            else:
                sz = os.path.getsize(ip) / (1024 * 1024)
                print(f"  [File]   {item} ({sz:.2f} MB)")

        # Save metadata pointer
        pointer_file = os.path.join(target_dir, "kagglehub_path.txt")
        with open(pointer_file, "w") as f:
            f.write(path)
        print(f"\nTarget folder configured: {target_dir}")

    except Exception as e:
        print(f"\n[Download Error]: {e}")
        print("\nCommon Solutions:")
        print("1. Have you accepted the competition rules on Kaggle?")
        print("   Visit: https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection/rules")
        print("   Click 'I Understand and Accept'.")
        print("2. Ensure your Kaggle username & API key are valid.")

if __name__ == '__main__':
    download()
