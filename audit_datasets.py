import os
import glob
import lmdb

def audit():
    print("=" * 65)
    print("           COMPREHENSIVE DATASET AUDIT")
    print("=" * 65)

    print("\n>>> 1. AUDIT OF ./data/ FOLDER:")
    if os.path.exists('data'):
        for item in sorted(os.listdir('data')):
            p = os.path.join('data', item)
            if os.path.isdir(p):
                print(f"\n[Dataset]: {item}")
                for sub in sorted(os.listdir(p)):
                    sub_p = os.path.join(p, sub)
                    if os.path.isdir(sub_p):
                        # check if lmdb
                        if os.path.exists(os.path.join(sub_p, 'data.mdb')):
                            try:
                                env = lmdb.open(sub_p, readonly=True, lock=False)
                                with env.begin() as txn:
                                    entries = (txn.stat()['entries'] - 1) // 2
                                env.close()
                                print(f"  * LMDB Database: '{sub}' -> {entries} image-mask sample pairs")
                            except Exception as e:
                                print(f"  * LMDB Database: '{sub}' -> error reading: {e}")
                        else:
                            count = len(glob.glob(os.path.join(sub_p, '**', '*.*'), recursive=True))
                            img_count = len([f for f in glob.glob(os.path.join(sub_p, '**', '*.*'), recursive=True) if f.lower().endswith(('.png','.jpg','.jpeg','.tif','.bmp'))])
                            print(f"  * Subfolder    : '{sub}' -> {count} total files ({img_count} images/masks)")
                    else:
                        sz = os.path.getsize(sub_p) / (1024 * 1024)
                        print(f"  * File         : '{sub}' ({sz:.2f} MB)")

    print("\n" + "=" * 65)
    print(">>> 2. AUDIT OF ./test_datasets/ FOLDER:")
    if os.path.exists('test_datasets'):
        for item in sorted(os.listdir('test_datasets')):
            p = os.path.join('test_datasets', item)
            if os.path.isdir(p):
                files = glob.glob(os.path.join(p, '**', '*.*'), recursive=True)
                img_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg', '.bmp'))]
                other_files = [f for f in files if not f.lower().endswith(('.jpg', '.png', '.tif', '.jpeg', '.bmp'))]
                print(f"\n[Folder]: {item}")
                print(f"  * Total Files       : {len(files)}")
                print(f"  * Images / Masks    : {len(img_files)}")
                if other_files:
                    print(f"  * Other Files       : {[os.path.basename(x) for x in other_files[:5]]}")
                    for of in other_files[:3]:
                        print(f"    - {os.path.basename(of)}: {os.path.getsize(of)/1024:.1f} KB")

    print("\n" + "=" * 65)
    print(">>> 3. TESTING COMPATIBILITY OF DATASET LOADERS:")
    from src.dataset.base_dataset import get_dataset
    for name in ['doctamper', 'doctamper_scd', 'cocoglide', 'splicing', 'casia']:
        try:
            ds = get_dataset(name, data_root='data', split='test')
            print(f"  * [{name:14s}] Ready for testing -> {len(ds)} test samples available")
        except Exception as e:
            print(f"  * [{name:14s}] Error: {e}")

if __name__ == '__main__':
    audit()
