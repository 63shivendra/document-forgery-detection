import os, glob, easyocr

reader = easyocr.Reader(['en'], gpu=True, verbose=False)
base_dir = r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina"
files = sorted(glob.glob(os.path.join(base_dir, "WhatsApp Image*.jpeg")))

for idx, f in enumerate(files, 1):
    fn = os.path.basename(f)
    print(f"\n=======================================================")
    print(f"IMAGE #{idx}: {fn}")
    print(f"=======================================================")
    texts = reader.readtext(f, detail=0)
    print("\n".join(texts))
