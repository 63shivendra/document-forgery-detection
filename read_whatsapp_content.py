import os, glob, easyocr

reader = easyocr.Reader(['en'], gpu=True, verbose=False)
base_dir = r"d:\shivendra pratap singh\onwardsmrechant\project final\merchant_fina"
files = sorted(glob.glob(os.path.join(base_dir, "WhatsApp Image*.jpeg")))

for f in files:
    fn = os.path.basename(f)
    print("=" * 70)
    print(f"FILE: {fn}")
    texts = reader.readtext(f, detail=0)
    full_text = " ".join([t.strip() for t in texts if t.strip()])
    print(f"TEXT CONTENT: {full_text[:400]}...")
