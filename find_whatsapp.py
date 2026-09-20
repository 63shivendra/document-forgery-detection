import os

search_roots = [
    r"d:\shivendra pratap singh\onwardsmrechant",
    r"E:\project final" if os.path.exists(r"E:\project final") else None
]

found = []
for root in search_roots:
    if not root:
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        if "whatsapp" in dirpath.lower():
            found.append(dirpath)
        for f in filenames:
            if "whatsapp" in f.lower():
                found.append(os.path.join(dirpath, f))

print(f"Found {len(found)} WhatsApp related items:")
for p in found[:50]:
    print(" ", p)
